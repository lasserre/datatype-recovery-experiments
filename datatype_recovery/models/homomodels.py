
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

        num_node_features = get_num_node_features(structural_model=True, include_component=include_component)
        edge_dim = EdgeTypes.edge_dim()
        super().__init__(max_seq_len, num_hops, include_component, hidden_channels, num_node_features, edge_dim)

    @staticmethod
    def create_model(**kwargs):
        max_seq_len = int(kwargs['max_seq_len'])
        num_hops = int(kwargs['num_hops'])
        include_component = bool(int(kwargs['include_component']))
        if 'hidden_channels' in kwargs:
            hidden_channels = int(kwargs['hidden_channels'])
        else:
            hidden_channels = 128
        return WithEdgeTypesModel(max_seq_len=max_seq_len, num_hops=num_hops, include_component=include_component, hidden_channels=hidden_channels)

register_model('WithEdgeTypesTypeSeq', WithEdgeTypesModel.create_model)

class DragonModel(BaseHomogenousModel):
    '''
    Node features:
        - node type
        - data type (type sequence vector)
        - opcode

    Edge features:
        - edge type
    '''
    def __init__(self, max_seq_len:int, num_hops:int, include_component:bool, hidden_channels:int=128,
                node_typeseq_len:int=3, heads:int=1, num_linear_layers:int=1):

        # NOTE: if node_typeseq_len changes, it has to EXACTLY match the node_typeseq_len used to create
        # the dataset...
        # UPDATE: leaving node_typeseq_len alone in case I want to mess with it later, but RIGHT NOW
        # everything is hardcoded to 3 pointer levels (which really means sequence length of 4)

        num_node_features = get_num_node_features(structural_model=False, include_component=include_component, type_seq_len=node_typeseq_len)
        edge_dim = EdgeTypes.edge_dim()
        super().__init__(max_seq_len, num_hops, include_component, hidden_channels, num_node_features, edge_dim, heads, num_linear_layers)

    @staticmethod
    def create_model(**kwargs):
        # max_seq_len = int(kwargs['max_seq_len'])
        max_seq_len = 0     # NOTE: not using currently
        num_hops = int(kwargs['num_hops'])
        include_component = bool(int(kwargs['include_component']))
        heads = int(kwargs['heads'])
        num_linear = int(kwargs['num_linear'])

        if 'hidden_channels' in kwargs:
            hidden_channels = int(kwargs['hidden_channels'])
        else:
            hidden_channels = 128
        if 'node_typeseq_len' in kwargs:
            node_typeseq_len = int(kwargs['node_typeseq_len'])
        else:
            node_typeseq_len = 3
        return DragonModel(max_seq_len=max_seq_len, num_hops=num_hops, include_component=include_component,
                            hidden_channels=hidden_channels, node_typeseq_len=node_typeseq_len,
                            heads=heads, num_linear_layers=num_linear)

register_model('DRAGON', DragonModel.create_model)