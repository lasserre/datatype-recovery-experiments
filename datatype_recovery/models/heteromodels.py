import torch
from torch import nn
from torch_geometric.nn import HGTConv

from .dataset.encoding import HeteroNodeEncoder

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
        # TODO: did this work??

        self.gnn_layers.append(HGTConv(-1, hc_graph, meta, heads=heads))

        # TODO: finish implementing DragonHGT...
        # TODO: build a dataset and kick this off for training
        #   - train Dragon vs. DragonHGT on same training dataset, similar params
        #   - eval on same binary
        #   - eval TyGR on same binary
        # TODO: go back (while that is running) and look at creating a good/deduplicated training dataset
        # that would be of the same caliber as TyDA-min's deduplicated training split
