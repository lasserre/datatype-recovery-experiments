import astlib
from astlib import ASTNode, TranslationUnitDecl
from astlib.find_all_references import FindAllVarRefs
import varlib
from itertools import chain
import torch
from torch.nn import functional as F
from typing import Dict, Tuple, List, Callable, Any

from astlib import *
from .encoding import encode_astnode, EdgeTypes


class VariableGraphBuilder:
    '''
    Builds the variable graph that will be used as input to the model.

    Currently supports only locals and params
    '''
    def __init__(self, var_name:str, tudecl:TranslationUnitDecl, sdb:varlib.StructDatabase=None,
                node_kind_only:bool=True):
        '''
        var_name: Name of the target variable
        ast: AST for the function this variable resides in
        sdb: Struct database for this AST
        '''
        self.__reset_state()
        self.var_name = var_name

        self.tudecl = tudecl
        self.sdb = sdb
        self.node_kind_only = node_kind_only

    def __reset_state(self):
        self.ast_node_list = []
        self.ref_exprs = []
        self.edge_list:List[str] = []
        self.edge_type_list:List[str] = []  # edge type for each edge in edge_list

    def build_variable_graph(self, max_hops:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Generates the variable graph and returns it as a
        [node_list, edge_index, edge_attr] tuple of tensors
        '''
        self.__reset_state()

        fbody = self.tudecl.inner[-1].inner[-1]
        self.ref_exprs = FindAllVarRefs(self.var_name).visit(fbody)

        # each refexpr is an independent sample that needs to be merged
        # into our target node 0

        # 1. add node 0/merge ref_exprs into special target node
        for r in self.ref_exprs:
            r.pyg_idx = 0

        self.ast_node_list = [self.ref_exprs[0]]  # pick one and encode it, they are identical

        # edge index starts out as a list of strings of the form "<start_idx>,<stop_idx>"
        # so we can prevent adding duplicate edges. Then we convert to tensor form once finished
        self.edge_list = []

        # add other nodes by following edges up to MAX HOPS
        for r in self.ref_exprs:
            # collect subgraph connected to r (we've already got the reference node captured)
            self.collect_node_neighbors(r, max_hops)

        # collect edge indices into flat list, then reshape into (N, 2)
        flat_list = [int(idx) for edge_str in self.edge_list for idx in edge_str.split(',')]
        N = int(len(flat_list)/2)
        edge_index = torch.tensor(flat_list, dtype=torch.long).reshape((N, 2))

        # torch-ify :)
        node_list = [encode_astnode(n, self.node_kind_only) for n in self.ast_node_list]
        node_list = torch.stack(node_list)
        edge_index = edge_index.t().contiguous()

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.stack([EdgeTypes.encode(x) for x in self.edge_type_list])

        # reset all pyg_idx values so we can reuse this self.tudecl object
        # for other locals/params WITHOUT re-reading from json each time
        # (skip ast_node_list[0] as it also exists in ref_exprs)
        for n in chain(self.ast_node_list[1:], self.ref_exprs):
            delattr(n, 'pyg_idx')

        return node_list, edge_index, edge_attr

    def add_node(self, node:ASTNode):
        if not hasattr(node, 'pyg_idx'):
            # this is a new node - add it
            node.pyg_idx = len(self.ast_node_list)
            self.ast_node_list.append(node)

    def _get_edge_string(self, start_idx:int, stop_idx:int):
        return f'{start_idx},{stop_idx}'

    def add_edge(self, parent:ASTNode, child:ASTNode, bidirectional:bool, child_idx:int=None):
        # edge types are same for either direction
        edge_type_str = EdgeTypes.get_edge_type(parent, child, child_idx)

        fwd_edge = self._get_edge_string(parent.pyg_idx, child.pyg_idx)
        if fwd_edge not in self.edge_list:
            self.edge_list.append(fwd_edge)
            self.edge_type_list.append(edge_type_str)

        if bidirectional:
            back_edge = self._get_edge_string(child.pyg_idx, parent.pyg_idx)
            if back_edge not in self.edge_list:
                self.edge_list.append(back_edge)
                self.edge_type_list.append(edge_type_str)

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
            self.add_edge(node.parent, node, bidirectional=True)
            self.collect_node_neighbors(node.parent, k-1)

        for i, child in enumerate(node.inner):
            self.add_node(child)
            self.add_edge(node, child, bidirectional=True, child_idx=i)
            self.collect_node_neighbors(child, k-1)

class VariableGraphViewer(ASTViewer):
    def __init__(self, varname:str, tudecl:TranslationUnitDecl, max_hops:int, one_edge_only:bool=False,
                format_node:Callable[[ASTNode,NodeAttrs],Any]=None, node_kind_only:bool=False) -> None:
        '''
        vgraph_nodes: List of nodes in the variable graph, with node[0] being the
                      DeclRefExpr node
        render_khop: If > -1, render the khop neighborhood even if it goes outside
                     the variable graph (useful for context if format_node highlights
                     only the vgraph)
        '''
        super().__init__(format_node, node_kind_only)

        self.id_ctr = 0     # reset
        self.one_edge_only = one_edge_only

        # build the variable graph, save the outputs we need
        builder = VariableGraphBuilder(varname, tudecl)
        builder.build_variable_graph(max_hops)

        # "<start_idx>,<stop_idx>" corresponding to ast_node_list indices
        self.edge_list = builder.edge_list.copy()
        self.edge_type_list = builder.edge_type_list.copy()
        self.vgraph_nodes = builder.ast_node_list.copy()
        self.other_refs = builder.ref_exprs[1:]   # ref_expr[0] is already included

    @property
    def declref(self) -> DeclRefExpr:
        return self.vgraph_nodes[0] if self.vgraph_nodes else None

    @property
    def declref_varname(self) -> str:
        return self.declref.referencedDecl.name if self.declref else ''

    def get_all_declrefs(self) -> List[ASTNode]:
        return [self.vgraph_nodes[0], *self.other_refs]

    def render_vargraph(self, format:str='pdf', outfolder:Path=None, ast_name:str='',
                   fontname:str='Cascadia Code'):
        if not ast_name:
            ast_name = f'{self.declref_varname}_VarGraph'
        self.id_ctr = 0     # reset

        self.g = Graph(ast_name, format=format)
        self.g.attr('graph', rankdir='BT')
        self.g.attr('node', shape='plaintext')
        self.g.attr('node', fontname=fontname)

        self.assign_ids()

        for n in self.vgraph_nodes:
            self.add_node(n, self.visit(n))

        edge_indices = [[int(x) for x in edge_str.split(',')] for edge_str in self.edge_list]

        if self.one_edge_only:
            sorted_indices = [sorted(edge) for edge in edge_indices]
            edge_indices = list(set((x, y) for x, y in sorted_indices))

        for i, edge in enumerate(edge_indices):
            parent = self.vgraph_nodes[edge[0]]
            child = self.vgraph_nodes[edge[1]]

            # color each outgoing edge from target node
            ids = [parent._graph_id, child._graph_id]
            edge_color = 'red' if self.declref._graph_id in ids else 'black'
            self.add_edge(parent, child, color=edge_color, label=self.edge_type_list[i])

        if outfolder is not None:
            self.g.render(directory=outfolder, view=False)

        # clear ids
        self.clear_ids_in_tree(self.declref.find_root_node())

        return self.g

    def _assign_node_id(self, node:ASTNode):
        node._graph_id = str(self.id_ctr)
        self.id_ctr += 1

    def assign_ids(self):
        '''Assigns ids and returns the list of nodes to be rendered'''

        # assign declref ids first to all be the same
        self.declref._graph_id = str(self.id_ctr)
        for x in self.other_refs:
            x._graph_id = self.declref._graph_id
        self.id_ctr += 1

        for node in self.vgraph_nodes[1:]:
            self._assign_node_id(node)

    def get_edge_str(self, parent:ASTNode, child:ASTNode) -> str:
        return f'{parent._graph_id}:{child._graph_id}'

    def visit(self, node:'ASTNode'):
        visit_method = getattr(self, node.visitor_method_name, None)
        node_attrs = visit_method(node) if visit_method else NodeAttrs(f'TODO: {node.kind}')
        if self._format_node:
            # give the callback a chance to modify the formatting
            self._format_node(node, node_attrs)
        return node_attrs
