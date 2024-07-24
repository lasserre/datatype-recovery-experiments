import torch
from torch import nn
from torch_geometric.nn import HGTConv

from .model_repo import register_model
from .dataset.encoding import *
from .structural_model import create_linear_stack

class DragonHGT(torch.nn.Module):
    def __init__(self,
                num_hops:int,
                heads:int,
                hc_graph:int,
                hc_linear:int,
                hc_task:int,
                num_shared_layers:int,
                num_task_specific_layers:int) -> None:
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

    def forward(self, x_dict, edge_index_dict):

        # TODO: I think we can customize the params to forward() to include batch
        # TODO: I think we can still implement get_node0_indices() for this, BUT
        # ...this time our target node (node0) isn't globally node 0 - I think (hope) it
        # will be node 0 for the data['Default'].x group (or maybe x_dict['Default'])


        ################# from example -----------------
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.lin(x_dict['author'])

        # TODO: finish implementing DragonHGT...
        # TODO: build a dataset and kick this off for training
        #   - train Dragon vs. DragonHGT on same training dataset, similar params
        #   - eval on same binary
        #   - eval TyGR on same binary
        # TODO: go back (while that is running) and look at creating a good/deduplicated training dataset
        # that would be of the same caliber as TyDA-min's deduplicated training split

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

        return DragonHGT(num_hops, heads, hc_graph, hc_linear, hc_task, num_shared, num_task)

register_model('DragonHGT', DragonHGT.create_model)
