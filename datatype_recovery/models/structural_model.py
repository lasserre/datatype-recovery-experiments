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
    def __init__(self, max_seq_len:int, num_hops:int, include_component:bool,
                hidden_channels:int,
                num_node_features:int,
                edge_dim:int=None,
                heads:int=1,
                num_shared_linear_layers:int=1,
                num_task_specific_layers:int=2,
                task_hidden_channels:int=64):
        super(BaseHomogenousModel, self).__init__()

        # New encoding: [Ptr hierarchy][Leaf type]
        # TODO: start by just predicting LEAF TYPE
        # - shared # of base layers (gnn layers + 0+ linear layers)
        # - task-specific # of layers (extra linear layers)
        # - feed certain outputs into other task-specific layers (e.g. category -> isSigned)

        # if we go with fewer layers than the # hops in our dataset
        # that may be fine for experimenting, but eventually we are wasting
        # time/space and can cut our dataset down to match (# hops = # layers)
        # self.max_seq_len = max_seq_len
        self.gat_layers = nn.ModuleList([])
        self.num_hops = num_hops
        self.hidden_channels = hidden_channels
        self.edge_dim = edge_dim
        self.num_heads = heads
        self.num_shared_linear_layers = num_shared_linear_layers
        self.num_task_specific_layers = num_task_specific_layers
        self.task_hidden_channels = task_hidden_channels

        # ---------------------------
        # GNN layers
        # ---------------------------
        self.gat_layers.append(GATConv(num_node_features, hidden_channels, edge_dim=edge_dim, heads=heads))
        for i in range(1, num_hops):
            self.gat_layers.append(GATConv(hidden_channels*heads, hidden_channels, edge_dim=edge_dim, heads=heads))

        # ---------------------------
        # shared linear layers
        # ---------------------------
        self.shared_linear_layers = nn.Sequential()
        for i in range(num_shared_linear_layers):
            input_dim = hidden_channels*heads if i == 0 else hidden_channels
            self.shared_linear_layers.append(nn.Linear(input_dim, hidden_channels))
            self.shared_linear_layers.append(nn.ReLU())

        # this is where we diverge into task-specific layers (everything above
        # are shared base layers)

        # TODO - add pred head layers below for each unique output
        # - leaf_size_head
        # - leaf_signed_head
        # - leaf_floating_head
        # - leaf_category_head
        # ...

        # ---------------------------
        # task-specific/output layers
        # ---------------------------
        self.leaf_category_head = create_linear_stack(num_task_specific_layers-1, hidden_channels, task_hidden_channels)

        # final output layers (no ReLU)
        self.leaf_category_head.append(nn.Linear(task_hidden_channels, len(LeafType.valid_categories())))

    @property
    def uses_edge_features(self) -> bool:
        return self.edge_dim is not None

    def forward(self, x, edge_index, batch, edge_attr=None):
        node0_indices = get_node0_indices(batch)

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

        x = self.shared_linear_layers(x[node0_indices])

        leaf_category_logits = self.leaf_category_head(x)

        # TODO: return tuple: out1, out2, ...
        return leaf_category_logits

        # OLD -------------
        # logits = self.pred_head(x[node0_indices])
        # batch_size = batch.max().item() + 1
        # return logits.view((batch_size, self.num_classes, self.max_seq_len))

        # return 3d tensor to match what CrossEntropy loss expects: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        # (also see https://discuss.pytorch.org/t/how-to-use-crossentropyloss-on-a-transformer-model-with-variable-sequence-length-and-constant-batch-size-1/157439)


class StructuralTypeSeqModel(BaseHomogenousModel):
    def __init__(self, max_seq_len:int, num_hops:int, include_component:bool, hidden_channels:int=128):
        num_node_features = get_num_node_features(structural_model=True, include_component=include_component)
        super().__init__(max_seq_len, num_hops, include_component, hidden_channels, num_node_features)

    @staticmethod
    def create_model(**kwargs):
        max_seq_len = int(kwargs['max_seq_len'])
        num_hops = int(kwargs['num_hops'])
        include_component = bool(int(kwargs['include_component']))
        return StructuralTypeSeqModel(max_seq_len=max_seq_len, num_hops=num_hops, include_component=include_component)

register_model('StructuralTypeSeq', StructuralTypeSeqModel.create_model)
