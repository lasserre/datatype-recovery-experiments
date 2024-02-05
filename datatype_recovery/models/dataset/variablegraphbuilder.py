import astlib
from astlib import ASTNode
from astlib.find_all_references import FindAllVarRefs
import varlib
from itertools import chain
import torch
from torch.nn import functional as F
from typing import Dict, Tuple, List

from .encoding import encode_astnode

# FIXME: later on, we will add these things, but for now:
# - No edge types
# - NODE TYPE is only node feature

# TODO: need to map truth data types to enum values (then we will then one-hot them)

class VariableGraphBuilder:
    '''
    Builds the variable graph that will be used as input to the model.

    Currently supports only locals and params
    '''
    def __init__(self, var_name:str, ast:ASTNode, sdb:varlib.StructDatabase=None):
        '''
        var_name: Name of the target variable
        ast: AST for the function this variable resides in
        sdb: Struct database for this AST
        '''
        self.__reset_state()
        self.var_name = var_name

        self.ast = ast
        self.sdb = sdb

    def __reset_state(self):
        self.ast_node_list = []
        self.edge_index = []

    def build_variable_graph(self, max_hops:int) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Generates the variable graph and returns it as a
        [node_list, edge_index] tuple of tensors
        '''
        self.__reset_state()

        fbody = self.ast.inner[-1].inner[-1]
        ref_exprs = FindAllVarRefs(self.var_name).visit(fbody)

        # each refexpr is an independent sample that needs to be merged
        # into our target node 0

        # 1. add node 0/merge ref_exprs into special target node
        for r in ref_exprs:
            r.pyg_idx = 0

        self.ast_node_list = [ref_exprs[0]]  # pick one and encode it, they are identical

        # edge index starts out as a list of strings of the form "<start_idx>,<stop_idx>"
        # so we can prevent adding duplicate edges. Then we convert to tensor form once finished
        self.edge_index = []

        # add other nodes by following edges up to MAX HOPS
        for r in ref_exprs:
            # collect subgraph connected to r (we've already got the reference node captured)
            self.collect_node_neighbors(r, max_hops)

        # collect edge indices into flat list, then reshape into (N, 2)
        flat_list = [int(idx) for edge_str in self.edge_index for idx in edge_str.split(',')]
        N = int(len(flat_list)/2)
        self.edge_index = torch.tensor(flat_list, dtype=torch.long).reshape((N, 2))

        # torch-ify :)
        node_list = [encode_astnode(n) for n in self.ast_node_list]
        node_list = torch.stack(node_list)
        self.edge_index = self.edge_index.t().contiguous()

        # reset all pyg_idx values so we can reuse this self.ast object
        # for other locals/params WITHOUT re-reading from json each time
        # (skip ast_node_list[0] as it also exists in ref_exprs)
        for n in chain(self.ast_node_list[1:], ref_exprs):
            delattr(n, 'pyg_idx')

        return node_list, self.edge_index

    def add_node(self, node:ASTNode):
        if not hasattr(node, 'pyg_idx'):
            # this is a new node - add it
            node.pyg_idx = len(self.ast_node_list)
            self.ast_node_list.append(node)

    def _get_edge_string(self, start_idx:int, stop_idx:int):
        return f'{start_idx},{stop_idx}'

    def add_edge(self, start_node:ASTNode, end_node:ASTNode, bidirectional:bool):
        fwd_edge = self._get_edge_string(start_node.pyg_idx, end_node.pyg_idx)
        if fwd_edge not in self.edge_index:
            self.edge_index.append(fwd_edge)

        if bidirectional:
            back_edge = self._get_edge_string(end_node.pyg_idx, start_node.pyg_idx)
            if back_edge not in self.edge_index:
                self.edge_index.append(back_edge)

    def collect_node_neighbors(self, node:ASTNode, k:int):
        '''
        Collect the k-hop neighborhood of node (not including node) staying
        within the current statement
        '''
        if k <= 0:
            return

        # if we are at a statement node, don't go up outside this statement
        if node.parent and not node.is_statement:
            self.add_node(node.parent)
            self.add_edge(node, node.parent, bidirectional=True)
            self.collect_node_neighbors(node.parent, k-1)

        for child in node.inner:
            self.add_node(child)
            self.add_edge(node, child, bidirectional=True)
            self.collect_node_neighbors(child, k-1)
