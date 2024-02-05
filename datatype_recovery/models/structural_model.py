import torch
from torch.nn import functional as F
from torch_geometric.data import Dataset
from torch_geometric.nn import GATConv, Linear
from typing import List

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

class StructuralTypeSeqModel(torch.nn.Module):
    def __init__(self, dataset:Dataset, hidden_channels:int):
        super(StructuralTypeSeqModel, self).__init__()

        # if we go with fewer layers than the # hops in our dataset
        # that may be fine for experimenting, but eventually we are wasting
        # time/space and can cut our dataset down to match (# hops = # layers)
        self.gat1 = GATConv(dataset.num_node_features, hidden_channels)
        self.gat2 = GATConv(hidden_channels, hidden_channels)
        self.gat3 = GATConv(hidden_channels, hidden_channels)

        # TODO - later on, add sequential layer(s) here

        self.pred_head = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        node0_indices = get_node0_indices(batch)

        # FIXME: if we need floats in the dataset, go back and save the data
        # this way in our Dataset class
        x = x.to(torch.float32)

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

        h = self.gat1(x, edge_index)
        h = h.relu()

        h = self.gat2(h, edge_index)
        h = h.relu()

        h = self.gat3(h, edge_index)
        # relu here?

        logits = self.pred_head(h[node0_indices])
        return logits
