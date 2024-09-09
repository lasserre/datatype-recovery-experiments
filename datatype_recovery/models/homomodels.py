import astlib
from typing import List, Tuple
import torch
from torch.nn import functional as F
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
    def __init__(self, num_hops:int, hc_graph:int=128):

        num_node_features = get_num_node_features(structural_model=True)
        edge_dim = EdgeTypes.edge_dim()
        super().__init__(num_hops, hc_graph, num_node_features, edge_dim)

    @staticmethod
    def create_model(**kwargs):
        num_hops = int(kwargs['num_hops'])
        if 'hc_graph' in kwargs:
            hc_graph = int(kwargs['hc_graph'])
        else:
            hc_graph = 128
        return WithEdgeTypesModel(num_hops=num_hops, hc_graph=hc_graph)

register_model('WithEdgeTypesTypeSeq', WithEdgeTypesModel.create_model)

class VarPrediction:
    '''
    Helper class to encapsulate data associated with
    variable predictions
    '''
    def __init__(self, vardecl:VarDecl, pred_dt:DataType, varid:tuple=None,
                 num_other_vars:int=-1, confidence:float=0.0, num_callers:int=0) -> None:
        self.vardecl = vardecl
        self.pred_dt = pred_dt
        self.varid = varid
        self.num_other_vars = num_other_vars
        self.confidence = confidence
        self.influence = self.calc_influence(num_callers)
        self.num_callers = num_callers

    @property
    def num_refs(self) -> int:
        return len(self.varid[2].split(',')) if self.varid else -1

    def calc_influence(self, num_callers:int) -> int:
        '''
        Returns a measure of the influence this variable has on other variables
        in terms of the potential for this variable's retyping to affect the
        type prediction of other variables
        '''
        if not self.varid:
            return -1
        # locals = # refs
        # params = # refs (internal to function) + # func refs (callsites)
        return self.num_refs if self.varid[-1] == 'l' else self.num_refs + num_callers

    def to_record(self) -> list:
        '''
        Convert this prediction to a record (list of values) suitable for conversion
        to a pandas DataFrame using DataFrame.from_records()
        '''
        return [*self.varid,
                self.vardecl.name,
                self.vardecl.location,
                self.pred_dt,
                self.pred_dt.to_json(),
                self.num_refs,
                self.num_other_vars,
                self.influence,
                self.confidence
            ]

    @staticmethod
    def record_columns() -> List[str]:
        '''Get the list of column names that go with the to_record() list of values'''
        return [
            'BinaryId','FunctionStart','Signature','Vartype',
            'Name',
            'Location',
            'Pred',
            'PredJson',
            'NumRefs',
            'NumOtherVars',
            'Influence',
            'Confidence'
        ]

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
    def __init__(self, num_hops:int,
                heads:int,
                hc_graph:int,
                hc_linear:int,
                hc_task:int,
                num_shared_layers:int,
                num_task_layers:int,
                confidence:bool,
                gnn_dropout:float,
                shared_dropout:float,
                task_dropout:float,
                num_leafsize_layers:int=None,
                hc_leafsize:int=None,
                leaf_thresholds:LeafTypeThresholds=None):

        # NOTE: if node_typeseq_len changes, it has to EXACTLY match the node_typeseq_len used to create
        # the dataset...
        # UPDATE: leaving node_typeseq_len alone in case I want to mess with it later, but RIGHT NOW
        # everything is hardcoded to 3 pointer levels (which really means sequence length of 4)

        num_node_features = get_num_node_features(structural_model=False)
        edge_dim = EdgeTypes.edge_dim()
        super().__init__(num_hops, hc_graph, num_node_features, edge_dim,
                        heads, num_shared_layers, num_task_layers, hc_task,
                        hc_linear, confidence, gnn_dropout, num_leafsize_layers, hc_leafsize,
                        shared_dropout, task_dropout)
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
                            skip_signatures:List[str]=None,
                            num_callers:int=0) -> List[VarPrediction]:
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
        pred = out[0] if self.confidence else out
        conf = F.sigmoid(out[1]) if self.confidence else None
        out_tensor = torch.cat(pred,dim=1)

        return [
            VarPrediction(
                vardecl=func_vars[i],
                pred_dt=TypeEncoder.decode(pred, self.leaf_thresholds),
                varid=data_objs[i].varid,
                num_other_vars=data_objs[i].num_other_vars,
                confidence=conf[i].item() if conf else 0.0,
                num_callers=num_callers,
            )
            for i, pred in enumerate(out_tensor)
        ]

    @staticmethod
    def load_model(model_path:Path, device:str='cpu', eval:bool=True) -> 'DragonModel':
        '''
        Load a saved DragonModel, move it to device, and set it to eval or train mode
        '''
        model_load = torch.load(model_path)
        model = DragonModel(num_hops=model_load.num_hops, heads=model_load.num_heads, hc_graph=model_load.hc_graph,
                            hc_linear=model_load.hc_linear, hc_task=model_load.hc_task, num_shared_layers=model_load.num_shared_layers,
                            num_task_layers=model_load.num_task_layers, confidence=bool(model_load.confidence),
                            gnn_dropout=model_load.gnn_dropout, shared_dropout=model_load.shared_dropout, task_dropout=model_load.task_dropout,
                            num_leafsize_layers=model_load.num_leafsize_layers, hc_leafsize=model_load.hc_leafsize)
        model.load_state_dict(model_load.state_dict())
        model.to(device)
        if eval:
            model.eval()
        else:
            model.train()
        return model

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
        gnn_dropout = float(get_arg('gnn_dropout', 0.0))
        shared_dropout = float(get_arg('shared_dropout', 0.0))
        task_dropout = float(get_arg('task_dropout', 0.0))

        num_leafsize = get_arg('num_leafsize', None)
        hc_leafsize = get_arg('hc_leafsize', None)
        if num_leafsize is not None:
            num_leafsize = int(num_leafsize)
        if hc_leafsize is not None:
            hc_leafsize = int(hc_leafsize)

        return DragonModel(num_hops, heads, hc_graph, hc_linear, hc_task,
                           num_shared, num_task, confidence,
                           gnn_dropout, shared_dropout, task_dropout,
                           num_leafsize, hc_leafsize)

register_model('DRAGON', DragonModel.create_model)