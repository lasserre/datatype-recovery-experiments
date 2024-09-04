import astlib
from astlib import ASTNode, TranslationUnitDecl
from astlib.find_all_references import FindAllVarRefs
import varlib
from itertools import chain

import torch
from torch.nn import functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data, HeteroData
from typing import Dict, Tuple, List, Callable, Any

from astlib import *
from .encoding import encode_astnode, EdgeTypes, HeteroNodeEncoder, HeteroEdgeTypes, NodeEncoder

class VariableGraphBuilder:
    '''
    Builds the variable graph that will be used as input to the model.

    Currently supports only locals and params
    '''
    def __init__(self, var_name:str, tudecl:TranslationUnitDecl, max_hops:int,
                sdb:varlib.StructDatabase=None,
                node_kind_only:bool=False):
        '''
        var_name: Name of the target variable
        ast: AST for the function this variable resides in
        max_hops: Size of the target node's neighborhood in hops
        sdb: Struct database for this AST
        '''
        self._reset_state()
        self.var_name = var_name
        self.var_signature = None   # we will fill this out when we run

        self.tudecl = tudecl
        self.max_hops = max_hops
        self.sdb = sdb
        self.node_kind_only = node_kind_only

    def _reset_state(self):
        self.ast_node_list = []
        self.ref_exprs = []
        self.edge_list:List[str] = []
        # each entry in edge_type_list: (EDGE_TYPE_STR, child_to_parent)
        self.edge_type_list:List[Tuple[str,bool]] = []  # edge type for each edge in edge_list

    def build(self, bid:int=-1) -> Data:
        '''
        Build the variable graph for the given variable inside this function AST, and return
        the resulting graph as a Data object.
        '''
        return self._build_variable_graph(bid=bid)

    def build_from_refs(self, var_refs:List[DeclRefExpr], bid:int=-1) -> Data:
        '''
        Build the variable graph for the given set of variable references (should be all refs within
        the function), and return the resulting graph as a Data object
        '''
        return self._build_variable_graph(var_refs, bid)

    def _convert_builder_outputs_to_data(self) -> Data:
        '''
        Converts the nodes/edges into a Data object
        '''

        # collect edge indices into flat list, then reshape into (N, 2)
        flat_list = [int(idx) for edge_str in self.edge_list for idx in edge_str.split(',')]
        N = int(len(flat_list)/2)
        edge_index = torch.tensor(flat_list, dtype=torch.long).reshape((N, 2))

        # torch-ify :)
        node_list = [encode_astnode(n, self.node_kind_only) for n in self.ast_node_list]
        node_list = torch.stack(node_list)
        edge_index = edge_index.t().contiguous()

        # NOTE: if edge_index needs special handling for no edges, use this (I don't think it will):
        # edge_index=torch.tensor([[], []], dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.stack([EdgeTypes.encode(etype, to_parent) for etype, to_parent in self.edge_type_list]) if self.edge_type_list else torch.tensor([])

        if node_list is None:
            return None     # this variable has no references, thus no graph

        return Data(x=node_list, edge_index=edge_index, edge_attr=edge_attr)

    def _calculate_num_other_vars(self) -> int:
        '''
        Calculate the number of other variables present in this variable graph
        '''

        # CallExpr has unrelated children (arguments) who don't "add value" to each other...
        EXCLUDE_CALL_CHILDREN = False

        var_ref_nodes = [x for x in self.ast_node_list if isinstance(x, DeclRefExpr) and x.referencedDecl.kind != 'FunctionDecl']

        if EXCLUDE_CALL_CHILDREN:
            var_ref_nodes = filter(lambda x: x.parent.kind != 'CallExpr', var_ref_nodes)

        num_other_vars = len(set([x.referencedDecl.name for x in var_ref_nodes])) - 1    # -1 to exclude target node
        num_other_vars = max(num_other_vars, 0)     # in case we went negative with -1

        return num_other_vars

    def _build_variable_graph(self, all_var_refs:List[DeclRefExpr]=None, bid:int=-1) -> Data:
        '''
        Generates the variable graph and returns it as a
        [node_list, edge_index, edge_attr] tuple of tensors

        If all_var_refs is supplied, this is used as-is instead of visiting the AST
        to go collect the variable reference expressions (this allows only collecting
        refs 1x if this has already been done)
        '''
        self._reset_state()

        fdecl = self.tudecl.get_fdecl()

        self.ref_exprs = FindAllVarRefs(self.var_name).visit(fdecl.func_body) if all_var_refs is None else all_var_refs

        # go ahead and compute the signature while we hold all the references
        # to avoid revisiting the AST for no reason
        self.var_signature = compute_var_ast_signature(self.ref_exprs, fdecl.address)

        if not self.ref_exprs:
            return None     # return None to indicate there are no references

        # each refexpr is an independent sample that needs to be merged
        # into our target node 0

        # 1. add node 0/merge ref_exprs into special target node
        self.add_node(self.ref_exprs[0])    # pick one and encode it, they are identical
        for r in self.ref_exprs[1:]:
            r.pyg_idx = 0   # set matching pyg_idx of 0

        # edge index starts out as a list of strings of the form "<start_idx>,<stop_idx>"
        # so we can prevent adding duplicate edges. Then we convert to tensor form once finished
        self.edge_list = []

        # add other nodes by following edges up to MAX HOPS
        for r in self.ref_exprs:
            # collect subgraph connected to r (we've already got the reference node captured)
            self.collect_node_neighbors(r, self.max_hops)

        data = self._convert_builder_outputs_to_data()

        # build signature/varid while we have all refs available
        fdecl = self.tudecl.get_fdecl()
        vartype = get_vartype_from_ref(self.ref_exprs[0])
        signature = compute_var_ast_signature(self.ref_exprs, fdecl.address)

        data.varid = build_varid(bid, fdecl.address, signature, vartype)
        data.num_other_vars = self._calculate_num_other_vars()

        # reset all pyg_idx values so we can reuse this self.tudecl object
        # for other locals/params WITHOUT re-reading from json each time
        # (skip ast_node_list[0] as it also exists in ref_exprs)
        for n in chain(self.ast_node_list[1:], self.ref_exprs):
            delattr(n, 'pyg_idx')

        return data

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
        child_to_parent = False     # fwd is parent->child
        if fwd_edge not in self.edge_list:
            self.edge_list.append(fwd_edge)
            self.edge_type_list.append((edge_type_str, child_to_parent))

        if bidirectional:
            back_edge = self._get_edge_string(child.pyg_idx, parent.pyg_idx)
            child_to_parent = True     # back is child->parent
            if back_edge not in self.edge_list:
                self.edge_list.append(back_edge)
                self.edge_type_list.append((edge_type_str, child_to_parent))

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

class VariableHeteroGraphBuilder(VariableGraphBuilder):
    '''
    Builds the heterogeneous variable graph that will be used as input to the model.

    Currently supports only locals and params
    '''
    def __init__(self, var_name:str, tudecl:TranslationUnitDecl, max_hops:int,
                sdb:varlib.StructDatabase=None):
        '''
        var_name: Name of the target variable
        ast: AST for the function this variable resides in
        max_hops: Size of the target node's neighborhood in hops
        sdb: Struct database for this AST
        '''
        super().__init__(var_name, tudecl, max_hops, sdb)

    def _reset_state(self):
        super()._reset_state()
        self.nodes_by_group = {}     # map node group -> node
        self.edges_by_tuple = {}    # map (start node kind, edge name, end node kind) -> (start_idx, end_idx)

    def build(self, bid:int=-1) -> HeteroData:
        '''
        Build the variable graph for the given variable inside this function AST, and return
        the resulting graph as a HeteroData object.
        '''
        return self._build_variable_graph(bid=bid)

    def build_from_refs(self, var_refs:List[DeclRefExpr], bid:int=-1) -> HeteroData:
        '''
        Build the variable graph for the given set of variable references (should be all refs within
        the function), and return the resulting graph as a Data object
        '''
        return self._build_variable_graph(var_refs, bid=bid)

    def add_node(self, node:ASTNode):
        if not hasattr(node, 'pyg_idx'):
            # this is a new node - add it
            group = HeteroNodeEncoder.get_node_group(node.kind)
            if group not in self.nodes_by_group:
                self.nodes_by_group[group] = []

            node.pyg_idx = len(self.nodes_by_group[group])
            self.nodes_by_group[group].append(node)

            # also maintain a flat list for deleting pyg_idx fields at the end
            self.ast_node_list.append(node)

    def add_edge(self, parent:ASTNode, child:ASTNode, bidirectional:bool, child_idx:int=None):
        # edge types are same for either direction
        edge_name = HeteroEdgeTypes.get_edge_type(parent, child, child_idx)

        parent_group = HeteroNodeEncoder.get_node_group(parent.kind)
        child_group = HeteroNodeEncoder.get_node_group(child.kind)
        edge_tuple = (parent_group, edge_name, child_group)

        if edge_tuple not in self.edges_by_tuple:
            self.edges_by_tuple[edge_tuple] = []

        fwd_edge = self._get_edge_string(parent.pyg_idx, child.pyg_idx)

        if fwd_edge not in self.edges_by_tuple[edge_tuple]:
            self.edges_by_tuple[edge_tuple].append(fwd_edge)

    def _convert_builder_outputs_to_data(self) -> HeteroData:
        '''
        Converts the nodes/edges into a Data object
        '''
        hetero_dict = {}    # map edge_tuple to dict containing edge_index
                            # (assigning to data[k[0], k[1], k[2]] does not work!)

        # nodes: [num_nodes, num_features]
        for kind, nodes in self.nodes_by_group.items():
            hetero_dict[kind] = {'x': torch.stack([NodeEncoder().visit(n) for n in nodes])}

        # edges: [2, num_edges]
        for k, edge_list in self.edges_by_tuple.items():
            flat_list = [int(idx) for edge_str in edge_list for idx in edge_str.split(',')]
            N = int(len(flat_list)/2)
            hetero_dict[k] = {'edge_index': torch.tensor(flat_list, dtype=torch.long).reshape((N, 2)).t().contiguous()}

        return T.ToUndirected()(HeteroData(hetero_dict))

        # NOTE: edge features if we need them
        # data['paper', 'cites', 'paper'].edge_attr = ... # [num_edges_cites, num_features_cites]
        # data['author', 'writes', 'paper'].edge_attr = ... # [num_edges_writes, num_features_writes]

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
        builder = VariableGraphBuilder(varname, tudecl, max_hops)
        builder.build()

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
            edge_type, to_parent = self.edge_type_list[i]
            self.add_edge(parent, child, color=edge_color, label=f'{edge_type} ({int(to_parent)})')

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
