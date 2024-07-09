import astlib
from typing import List, Tuple
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from varlib.datatype import DataType
from astlib import FindAllVarRefs, compute_var_ast_signature
from astlib.ast import *

from .model_repo import register_model
from .dataset.encoding import TypeEncoder, get_num_node_features, EdgeTypes, LeafTypeThresholds
from .dataset.variablegraphbuilder import VariableGraphBuilder
from .structural_model import BaseHomogenousModel

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

class VarPrediction:
    '''
    Helper class to encapsulate data associated with
    variable predictions
    '''
    def __init__(self, vardecl:VarDecl, pred_dt:DataType, varid:tuple=None) -> None:
        self.vardecl = vardecl
        self.pred_dt = pred_dt
        self.varid = varid

    @property
    def num_refs(self) -> int:
        return len(self.varid[2].split(',')) if self.varid else -1

    def __str__(self) -> str:
        return f'{self.vardecl.dtype} {self.vardecl.name} -> {self.pred_dt} varid={self.varid}'

    def __repr__(self) -> str:
        return str(self)

class DragonModel(BaseHomogenousModel):
    '''
    Node features:
        - node type
        - data type (type sequence vector)
        - opcode

    Edge features:
        - edge type
    '''
    def __init__(self, num_hops:int, hidden_channels:int=128,
                heads:int=1, num_linear_layers:int=1,
                leaf_thresholds:LeafTypeThresholds=None):

        # NOTE: if node_typeseq_len changes, it has to EXACTLY match the node_typeseq_len used to create
        # the dataset...
        # UPDATE: leaving node_typeseq_len alone in case I want to mess with it later, but RIGHT NOW
        # everything is hardcoded to 3 pointer levels (which really means sequence length of 4)

        num_node_features = get_num_node_features(structural_model=False)
        edge_dim = EdgeTypes.edge_dim()
        super().__init__(num_hops, hidden_channels, num_node_features, edge_dim, heads, num_linear_layers)
        self.leaf_thresholds = leaf_thresholds if leaf_thresholds else LeafTypeThresholds()

    def predict_type(self, ast:astlib.TranslationUnitDecl, varname:str, device:str='cpu') -> DataType:
        '''
        Predict the data type for a local or parameter with the given name in
        this function AST

        NOTE: caller should call model.eval() and model.to(device) first! Not calling them
              here to avoid unnecessary calls for every variable
        '''
        loader = DataLoader([
            VariableGraphBuilder(varname, ast, self.num_hops).build().to(device)
        ], batch_size=1)
        data = list(loader)[0]  # have to go through loader for batch to work
        out = self(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr)
        out_tensor = torch.cat(out,dim=1)
        return TypeEncoder.decode(out_tensor, self.leaf_thresholds)

    def predict_func_types(self, ast:astlib.TranslationUnitDecl,
                            device:str='cpu', bid:int=-1,
                            skip_unique_vars:bool=True,
                            skip_dup_sigs:bool=False,
                            skip_signatures:List[str]=None) -> List[VarPrediction]:
        '''
        Predict data types for each local and parameter variable within this function

        ast: Function AST
        device: Device where data objects should be moved (model should already be moved here)
        bid: Binary ID, if the varids will be used outside this function
        skip_unique_vars: Don't make predictions on unique variables
        skip_signatures: List of variable signatures which should be skipped

        NOTE: caller should call model.eval() and model.to(device) first! Not calling them
              here to avoid unnecessary calls for every variable
        '''
        skip_signatures = [] if skip_signatures is None else skip_signatures
        fdecl = ast.get_fdecl()

        # put all variables into one batch, run on the batch
        # NOTE: these are in the same order so we can locate the matching
        # VarDecl by DataLoader index
        func_vars = fdecl.local_vars + fdecl.params

        if skip_unique_vars:
            skip_loctypes = ['unique', '']      # sometimes we get empty loc_types for unique or hash vars
            func_vars = list(filter(lambda v: v.location.loc_type not in skip_loctypes, func_vars))

        # find and pass refs in to builder so we avoid visiting the
        # AST 2x to find the same references
        var_refs = [FindAllVarRefs(v.name).visit(fdecl.func_body) for v in func_vars]
        var_sigs = [compute_var_ast_signature(refs, fdecl.address) for refs in var_refs]

        # filter out empty/duplicate/unwanted signatures
        if skip_dup_sigs:
            dup_counts = {sig: 0 for sig in var_sigs}
            for sig in var_sigs:
                dup_counts[sig] += 1
            unique_sigs = [sig for sig in var_sigs if dup_counts[sig] < 2]
        else:
            unique_sigs = var_sigs      # duplicates are ok

        # keep only nonempty, unique, non-skipped signatures :)
        # -> we can only handle nonempty signatures since we have to have at least 1 reference
        #    to use the reference-graph-based DRAGON model
        keep_sig_idxs = [i for i, sig in enumerate(var_sigs) if sig and sig in unique_sigs and sig not in skip_signatures]

        # filter each list down to keep in sync (index-wise)
        func_vars = [func_vars[i] for i in keep_sig_idxs]
        var_refs = [var_refs[i] for i in keep_sig_idxs]
        var_sigs = [var_sigs[i] for i in keep_sig_idxs]

        data_objs = [
            VariableGraphBuilder(v.name, ast, self.num_hops)
                .build_from_refs(var_refs[i], bid)
                .to(device)
            for i, v in enumerate(func_vars)
        ]

        if not data_objs:
            return []   # no data to make predictions for

        loader = DataLoader(data_objs, batch_size=len(data_objs))
        data = list(loader)[0]  # have to go through loader for batch to work
        out = self(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr)
        out_tensor = torch.cat(out,dim=1)

        return [
            VarPrediction(
                vardecl=func_vars[i],
                pred_dt=TypeEncoder.decode(pred, self.leaf_thresholds),
                varid=data_objs[i].varid
            )
            for i, pred in enumerate(out_tensor)
        ]

    @staticmethod
    def create_model(**kwargs):
        num_hops = int(kwargs['num_hops'])
        heads = int(kwargs['heads'])
        num_linear = int(kwargs['num_linear'])

        if 'hidden_channels' in kwargs:
            hidden_channels = int(kwargs['hidden_channels'])
        else:
            hidden_channels = 128

        return DragonModel(num_hops=num_hops, hidden_channels=hidden_channels,
                            heads=heads, num_linear_layers=num_linear)

register_model('DRAGON', DragonModel.create_model)