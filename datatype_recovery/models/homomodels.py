
from .structural_model import BaseHomogenousModel
from .model_repo import register_model
from .dataset.encoding import EdgeTypes, get_num_node_features

class WithEdgeTypesModel(BaseHomogenousModel):
    '''
    Just add an edge type to the structural model (to "prove" I'm doing this right? lol)

    I expect an incremental improvement as this adds information...but shouldn't be as
    much on its own as with the other node features (data type, opcode, etc)
    '''
    def __init__(self, max_seq_len:int, num_hops:int, include_component:bool, hidden_channels:int=128):

        num_node_features = get_num_node_features(structural_model=True)
        super().__init__(max_seq_len, num_hops, include_component, hidden_channels, num_node_features,
                        edge_dim=len(EdgeTypes.all_types()))

    @staticmethod
    def create_model(**kwargs):
        max_seq_len = int(kwargs['max_seq_len'])
        num_hops = int(kwargs['num_hops'])
        include_component = bool(kwargs['include_component'])
        if 'hidden_channels' in kwargs:
            hidden_channels = int(kwargs['hidden_channels'])
        else:
            hidden_channels = 128
        return WithEdgeTypesModel(max_seq_len=max_seq_len, num_hops=num_hops, include_component=include_component, hidden_channels=hidden_channels)

register_model('WithEdgeTypesTypeSeq', WithEdgeTypesModel.create_model)
