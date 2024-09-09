import torch
from torch import nn
from torch_geometric.data import Dataset
from torch_geometric.nn import GATConv, Linear
from typing import List

from .model_repo import register_model
from .dataset.encoding import *

def split_node_index_by_graph(batch:torch.tensor, batch_size:int) -> List[torch.tensor]:
    '''
    Returns the node_index for each individual graph within the batch as a list of tensors.

    Each list entry (node_index) is a tensor of node indices for an individual graph within the
    batch, where each of the node indices will index into the containing DataBatch.x.
    '''
    return [(batch==i).nonzero(as_tuple=True) for i in range(batch_size)]

def get_node0_indices(batch:torch.tensor) -> List[int]:
    '''
    Returns the indices of all "node 0" nodes in the batch
    '''
    batch_size = batch.max().item()+1
    node_index_by_graph = split_node_index_by_graph(batch, batch_size)
    return [x[0][0].item() for x in node_index_by_graph]

def create_linear_stack(N:int, first_dim:int, hidden_dim:int) -> nn.Sequential:
    '''
    Stacks N linear layers together followed by ReLU activation layers

    first_dim: First input dimensionality
    hidden_dim: Dimensionality of each layer other than first input dimension
    '''
    linear_stack = nn.Sequential()
    for i in range(N):
        input_dim = first_dim if i == 0 else hidden_dim
        linear_stack.append(nn.Linear(input_dim, hidden_dim))
        linear_stack.append(nn.ReLU())
    return linear_stack

class BaseHomogenousModel(torch.nn.Module):
    def __init__(self, num_hops:int,
                hc_graph:int,
                num_node_features:int,
                edge_dim:int=None,
                heads:int=1,
                num_shared_layers:int=1,
                num_task_layers:int=2,
                hc_task:int=64,
                hc_linear:int=64,
                confidence:bool=False,
                dropout:float=0.0,
                num_leafsize_layers:int=None,
                hc_leafsize:int=None):
        super(BaseHomogenousModel, self).__init__()

        # if we go with fewer layers than the # hops in our dataset
        # that may be fine for experimenting, but eventually we are wasting
        # time/space and can cut our dataset down to match (# hops = # layers)

        self.gat_layers = nn.ModuleList([])
        self.num_hops = num_hops
        self.hc_graph = hc_graph
        self.edge_dim = edge_dim
        self.num_heads = heads
        self.num_shared_layers = num_shared_layers
        self.num_task_layers = num_task_layers
        self.hc_task = hc_task
        self.hc_linear = hc_linear
        self.dropout = dropout
        self.num_leafsize_layers = num_leafsize_layers if num_leafsize_layers is not None else num_task_layers
        self.hc_leafsize = hc_leafsize if hc_leafsize is not None else hc_task

        # ---------------------------
        # GNN layers
        # ---------------------------
        self.gat_layers.append(GATConv(num_node_features, hc_graph, edge_dim=edge_dim, heads=heads, dropout=dropout))
        for i in range(1, num_hops):
            self.gat_layers.append(GATConv(hc_graph*heads, hc_graph, edge_dim=edge_dim, heads=heads, dropout=dropout))

        # ---------------------------
        # shared linear layers
        # ---------------------------
        self.shared_linear_layers = nn.Sequential()
        for i in range(num_shared_layers):
            # concat node0 data type tensor with GNN hidden state
            input_dim = (hc_graph*heads + TypeEncoder.tensor_size()) if i == 0 else hc_linear
            self.shared_linear_layers.append(nn.Linear(input_dim, hc_linear))
            self.shared_linear_layers.append(nn.ReLU())

        # this is where we diverge into task-specific layers (everything above
        # are shared base layers)

        # ---------------------------
        # task-specific layers
        # ---------------------------
        leafcat_out_dim = len(LeafType.valid_categories())

        self.ptr_l1_head = create_linear_stack(num_task_layers-1, hc_linear, hc_task)
        self.ptr_l2_head = create_linear_stack(num_task_layers-1, hc_linear, hc_task)
        self.ptr_l3_head = create_linear_stack(num_task_layers-1, hc_linear, hc_task)

        self.leaf_category_head = create_linear_stack(num_task_layers-1, hc_linear, hc_task)
        self.leaf_signed_head   = create_linear_stack(num_task_layers-1, hc_linear, hc_task)
        self.leaf_floating_head = create_linear_stack(num_task_layers-1, hc_linear, hc_task)

        # leaf_bool input =
        #   CONCAT(shared_linear, ghidra_dtype_vec, leaf_cat_logits, leaf_signed_logit, leaf_floating_logit)
        leafbool_in_dim = hc_linear + TypeEncoder.tensor_size() + leafcat_out_dim + 2
        self.leaf_bool_head     = create_linear_stack(num_task_layers-1, leafbool_in_dim, hc_task)

        # leaf_size input =
        #   CONCAT(shared_linear, ghidra_dtype_vec, leaf_cat_logits, leaf_signed_logit, leaf_floating_logit, leaf_bool_logit)
        leafsize_in_dim = hc_linear + TypeEncoder.tensor_size() + leafcat_out_dim + 3
        self.leaf_size_head     = create_linear_stack(self.num_leafsize_layers-1, leafsize_in_dim, self.hc_leafsize)

        # ---------------------------
        # final output layers (no ReLU)
        # ---------------------------
        self.ptr_l1_head.append(nn.Linear(hc_task, 3))
        self.ptr_l2_head.append(nn.Linear(hc_task, 3))
        self.ptr_l3_head.append(nn.Linear(hc_task, 3))

        self.leaf_category_head.append(nn.Linear(hc_task, leafcat_out_dim))
        self.leaf_signed_head.append(nn.Linear(hc_task, 1))
        self.leaf_floating_head.append(nn.Linear(hc_task, 1))
        self.leaf_bool_head.append(nn.Linear(hc_task, 1))
        self.leaf_size_head.append(nn.Linear(self.hc_leafsize, len(LeafType.valid_sizes())))

        self.confidence = nn.Linear(hc_linear, 1) if confidence else None
        self.is_hetero = False       # for TrainContext class

    @property
    def uses_edge_features(self) -> bool:
        return self.edge_dim is not None

    def forward(self, x, edge_index, batch, edge_attr=None):
        node0_indices = get_node0_indices(batch)

        # extract node0 data types while we can still access unaltered x input
        # (which contains node0's data type input)
        _, dtype_vec, _ = NodeEncoder.split_node_vec(x[node0_indices])

        # GNN layers
        # ----------
        # NOTE: it's tempting to downselect to node0 indices here, but I think
        # that may be incorrect - we would not be passing ALL NODES through the
        # network, just node 0.

        # I think we WANT to compute the network on every node
        # for N hops and then simply make predictions based on the node 0 nodes
        #
        # yeah...if we did not compute this on ALL nodes, I think we are
        # making it effectively 1 hop only, and just going multiple rounds with 1 hop!

        final_gat_idx = len(self.gat_layers) - 1

        for i, gat in enumerate(self.gat_layers):
            x = gat(x, edge_index, edge_attr)

            # don't compute relu after final GAT layer
            if i < final_gat_idx:
                x = x.relu()

        # concatenate node0 data type with GNN hidden state before passing
        # to shared linear layers
        x = torch.cat([x[node0_indices], dtype_vec], dim=1)
        x = self.shared_linear_layers(x)

        ptr_l1_logits = self.ptr_l1_head(x)
        ptr_l2_logits = self.ptr_l2_head(x)
        ptr_l3_logits = self.ptr_l3_head(x)

        leaf_category_logits = self.leaf_category_head(x)
        leaf_signed_logit = self.leaf_signed_head(x)
        leaf_floating_logit = self.leaf_floating_head(x)

        # pass previous output into leaf_bool layers
        bool_input = torch.cat([x,
                dtype_vec,
                leaf_category_logits,
                leaf_signed_logit,
                leaf_floating_logit], dim=1)
        leaf_bool_logit = self.leaf_bool_head(bool_input)

        # pass previous outputs into leaf_size layers
        size_input = torch.cat([x,
                dtype_vec,
                leaf_category_logits,
                leaf_signed_logit,
                leaf_floating_logit,
                leaf_bool_logit], dim=1)

        leaf_size_logits = self.leaf_size_head(size_input)

        # NOTE: return predictions IN THE SAME ORDER as the encoded data type
        # vector: [ptr_levels (9)][leaf type (14)]
        # ...that way to convert it into a single vector, the caller just has
        # to call torch.cat(model_out_tuple)...but it's already separated for
        # loss purposes

        # PTR LEVELS: [L1 ptr_type (3)][L2 ptr_type (3)][L3 ptr_type (3)]
        # LEAF TYPE: [category (5)][sign (1)][float (1)][bool (1)][size (6)]

        pred = (ptr_l1_logits, ptr_l2_logits, ptr_l3_logits,
                leaf_category_logits, leaf_signed_logit, leaf_floating_logit, leaf_bool_logit, leaf_size_logits)

        if self.confidence:
            return (pred, self.confidence(x))

        return pred

class StructuralTypeSeqModel(BaseHomogenousModel):
    def __init__(self, num_hops:int, hc_graph:int=128):
        num_node_features = get_num_node_features(structural_model=True)
        super().__init__(num_hops, hc_graph, num_node_features)

    @staticmethod
    def create_model(**kwargs):
        num_hops = int(kwargs['num_hops'])
        hc_graph = bool(int(kwargs['hc_graph']))
        return StructuralTypeSeqModel(num_hops=num_hops, hc_graph=hc_graph)

register_model('StructuralTypeSeq', StructuralTypeSeqModel.create_model)
