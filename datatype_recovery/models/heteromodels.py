import torch
from torch import nn
from torch_geometric.nn import HGTConv

from .model_repo import register_model
from .dataset.encoding import *
from .structural_model import create_linear_stack, get_node0_indices

class DragonHGT(torch.nn.Module):
    def __init__(self,
                num_hops:int,
                heads:int,
                hc_graph:int,
                hc_linear:int,
                hc_task:int,
                num_shared_layers:int,
                num_task_specific_layers:int,
                confidence:bool) -> None:
        super().__init__()

        self.num_hops = num_hops
        self.heads = heads
        self.hc_graph = hc_graph
        self.hc_linear = hc_linear
        self.hc_task = hc_task
        self.num_shared_layers = num_shared_layers
        self.num_task_specific_layers = num_task_specific_layers

        self.gnn_layers = nn.ModuleList([])

        meta = HeteroNodeEncoder.get_metadata()

        self.gnn_layers.append(HGTConv(-1, hc_graph, meta, heads))
        for i in range(1, num_hops):
            self.gnn_layers.append(HGTConv(hc_graph*heads, hc_graph, meta, heads))

        self.shared_linear = create_linear_stack(num_shared_layers, hc_graph*heads, hc_linear)

        # task-specific
        self.ptr_l1_head = create_linear_stack(num_task_specific_layers-1, hc_linear, hc_task)
        self.ptr_l2_head = create_linear_stack(num_task_specific_layers-1, hc_linear, hc_task)
        self.ptr_l3_head = create_linear_stack(num_task_specific_layers-1, hc_linear, hc_task)

        self.leaf_category_head = create_linear_stack(num_task_specific_layers-1, hc_linear, hc_task)
        self.leaf_signed_head = create_linear_stack(num_task_specific_layers-1, hc_linear, hc_task)
        self.leaf_floating_head = create_linear_stack(num_task_specific_layers-1, hc_linear, hc_task)
        self.leaf_size_head = create_linear_stack(num_task_specific_layers-1, hc_linear, hc_task)

        # final output layers
        self.ptr_l1_head.append(nn.Linear(hc_task, 3))
        self.ptr_l2_head.append(nn.Linear(hc_task, 3))
        self.ptr_l3_head.append(nn.Linear(hc_task, 3))

        self.leaf_category_head.append(nn.Linear(hc_task, len(LeafType.valid_categories())))
        self.leaf_signed_head.append(nn.Linear(hc_task, 1))
        self.leaf_floating_head.append(nn.Linear(hc_task, 1))
        self.leaf_size_head.append(nn.Linear(hc_task, len(LeafType.valid_sizes())))

        self.confidence = nn.Linear(hc_linear, 1) if confidence else None

        # save the GNN node type for DeclRefExpr (right now it's NodeWithType)
        self.declref_group = HeteroNodeEncoder.get_node_group('DeclRefExpr')
        self.is_hetero = True       # for TrainContext class

    def forward(self, x_dict, edge_index_dict, batch_dict):
        n0_idxs = get_node0_indices(batch_dict[self.declref_group])

        final_gnn_idx = len(self.gnn_layers) - 1

        for i, hgt in enumerate(self.gnn_layers):
            x_dict = hgt(x_dict, edge_index_dict)

            # don't compute relu after final GNN layer
            if i < final_gnn_idx:
                x_dict = {k: x.relu() for k, x in x_dict.items()}

        target_node_embs = x_dict[self.declref_group][n0_idxs]      # grab embeddings for target nodes only
        x = self.shared_linear(target_node_embs)

        ptr_l1_logits = self.ptr_l1_head(x)
        ptr_l2_logits = self.ptr_l2_head(x)
        ptr_l3_logits = self.ptr_l3_head(x)

        leaf_category_logits = self.leaf_category_head(x)
        leaf_signed_logit = self.leaf_signed_head(x)
        leaf_floating_logit = self.leaf_floating_head(x)
        leaf_size_logits = self.leaf_size_head(x)

        # NOTE: return predictions IN THE SAME ORDER as the encoded data type
        # vector: [ptr_levels (9)][leaf type (13)]
        # ...that way to convert it into a single vector, the caller just has
        # to call torch.cat(model_out_tuple)...but it's already separated for
        # loss purposes

        # PTR LEVELS: [L1 ptr_type (3)][L2 ptr_type (3)][L3 ptr_type (3)]
        # LEAF TYPE: [category (5)][sign (1)][float (1)][size (6)]

        pred = (ptr_l1_logits, ptr_l2_logits, ptr_l3_logits,
                leaf_category_logits, leaf_signed_logit, leaf_floating_logit, leaf_size_logits)

        if self.confidence:
            return (pred, self.confidence(x))

        return pred

    @staticmethod
    def create_model(**kwargs):
        def get_arg(key, default_value):
            return kwargs[key] if key in kwargs else default_value

        num_hops = int(get_arg('num_hops', 5))
        heads = int(get_arg('heads', 1))
        hc_graph = int(get_arg('hc_graph', 64))
        hc_linear = int(get_arg('hc_linear', 64))
        hc_task = int(get_arg('hc_task', 64))
        num_shared = int(get_arg('num_shared', 3))
        num_task = int(get_arg('num_task', 2))
        confidence = bool(get_arg('confidence', False))

        return DragonHGT(num_hops, heads, hc_graph, hc_linear, hc_task, num_shared, num_task, confidence)

register_model('DragonHGT', DragonHGT.create_model)
