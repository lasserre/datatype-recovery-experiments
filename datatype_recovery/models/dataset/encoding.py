from astlib import ASTNode
import torch
from torch.nn import functional as F
from torch_geometric.transforms import BaseTransform
from typing import List, Any

from varlib.datatype.datatypes import _builtin_floats_by_size, _builtin_ints_by_size, _builtin_uints_by_size
from varlib.datatype import *

node_kind_names = [
    'ArraySubscriptExpr',
    'BinaryOperator',
    'BreakStmt',
    'CallExpr',
    'CaseStmt',
    'CharacterLiteral',
    'CompoundStmt',
    'ConstantExpr',
    'CStyleCastExpr',
    'DeclRefExpr',
    'DeclStmt',
    'DefaultStmt',
    'DoStmt',
    'FloatingLiteral',
    'ForStmt',
    'FunctionDecl',
    'GotoStmt',
    'IfStmt',
    'IntegerLiteral',
    'LabelStmt',
    'MemberExpr',
    'NullNode',
    'ParenExpr',
    'ParmVarDecl',
    'ReturnStmt',
    'StringLiteral',
    'SwitchStmt',
    'UnaryOperator',
    'ValueDecl',
    'VarDecl',
    'WhileStmt',
]

# Unused/not-applicable node types
# 'BuiltinType',
# 'ArrayType',
# 'EnumDecl',
# 'EnumConstantDecl',
# 'EnumType',
# 'FieldDecl',
# 'FunctionType',
# 'PointerType',
# 'RecordDecl',
# 'StructField',
# 'StructType',
# 'TranslationUnitDecl',
# 'Type',
# 'TypedefDecl',
# 'TypedefType',
# 'VoidType',

_node_kind_ids = None

def node_kind_ids() -> dict:
    global _node_kind_ids
    if _node_kind_ids is None:
        _node_kind_ids = {name: idx for idx, name in enumerate(node_kind_names)}
    return _node_kind_ids

def encode_astnode(node:ASTNode) -> torch.Tensor:
    '''Encodes an ASTNode into a feature vector'''
    kind_to_id = node_kind_ids()
    return F.one_hot(torch.tensor(kind_to_id[node.kind]), len(kind_to_id.keys())).to(torch.float32)

def decode_astnode(encoded_node:torch.Tensor) -> 'str':
    '''Decodes a node into its kind string'''
    return node_kind_names[encoded_node.argmax()]

# these are the individual model output elements for type sequence prediction
model_type_elem_names = [
    *_builtin_floats_by_size.values(),
    *_builtin_ints_by_size.values(),
    *_builtin_uints_by_size.values(),
    'void',
    'PTR',
    'ARR',
    'STRUCT',
    'UNION',
    'ENUM',
    'FUNC',
    '<EMPTY>',  # indicates end of type sequence, N/A, etc.
    'COMP',
]

class EdgeTypes:
    # -------------------------------------
    # Multi-level dict mapping ParentNodeType->ChildIdx->EdgeName
    # ParentNodeType: {
    #       ChildIdx: EdgeName
    # }
    # -------------------------------------
    _edgetype_lookup = {
        'BinaryOperator': {
            0: 'BinaryLeft',
            1: 'BinaryRight',
        },
        'IfStmt': {
            0: 'IfCond',
            1: 'IfTrueEdge',
            2: 'IfFalseEdge'
        },
        'CallExpr': {
            0: 'CallTarget',
            1: 'Param1',
            2: 'Param2',
            3: 'Param3',
            4: 'Param4',
            5: 'Param5',
            6: 'Param6',   # per histogram of coreutils, # params drops WAY off after 6
        },
        'ArraySubscriptExpr': {
            0: 'ArrayVar',
            1: 'ArrayIdxExpr',
        },
        'CaseStmt': {
            0: 'CaseValue',
        },
        'DoStmt': {
            0: 'DoLoop',
            1: 'DoCond',
        },
        'ForStmt': {
            # my doc says some of these are optional, but in Ghidra I insert NullNodes
            # as placeholders if no real element exists
            0: 'ForInit',
            1: 'ForCond',
            2: 'ForIncr',
            3: 'ForBody',
        },
        'SwitchStmt': {
            0: 'SwitchExpr',
            1: 'SwitchCases',
        },
        'WhileStmt': {
            0: 'WhileCond',
            1: 'WhileBody',
        },
    }

    DefaultEdgeName = ''    # kind of want to just make default edge all zeros, but this simplifies encode/decode logic

    _edge_type_names = None
    _edge_type_ids = None

    @staticmethod
    def get_edge_type(parent:ASTNode, child:ASTNode, child_idx:int=None):
        '''
        Returns the edge type string for this AST edge.

        parent: Parent node
        child: Child node
        child_idx: Optional index of child in parent.inner. If not provided it will
                   be looked up (supply it if you are iterating through children with enumerate)
        '''
        if child_idx is None:
            child_idx = parent.inner.index(child)

        # if this edge type is mapped, return it
        if parent.kind in EdgeTypes._edgetype_lookup:
            nodetype_lookup = EdgeTypes._edgetype_lookup[parent.kind]
            if child_idx in nodetype_lookup:
                return nodetype_lookup[child_idx]

        return EdgeTypes.DefaultEdgeName

    @staticmethod
    def encode(edge_type:str) -> torch.Tensor:
        '''Encodes the specified edge type string into an edge type feature vector'''
        edge_type_ids = EdgeTypes.edge_type_ids()
        return F.one_hot(torch.tensor(edge_type_ids[edge_type]), len(edge_type_ids.keys())).to(torch.float32)

    @staticmethod
    def decode(encoded_edge_type:torch.Tensor) -> 'str':
        '''Decodes a node into its kind string'''
        if encoded_edge_type.dim() > 1:
            return [EdgeTypes.all_types()[x.argmax()] for x in encoded_edge_type]
        return EdgeTypes.all_types()[encoded_edge_type.argmax()]

    @staticmethod
    def edge_type_ids() -> dict:
        if EdgeTypes._edge_type_ids is None:
            EdgeTypes._edge_type_ids = {name: idx for idx, name in enumerate(EdgeTypes.all_types())}
        return EdgeTypes._edge_type_ids

    @staticmethod
    def all_types() -> List[str]:
        if EdgeTypes._edge_type_names is None:
            EdgeTypes._edge_type_names = [edge_name for nodeDict in EdgeTypes._edgetype_lookup.values() for edge_name in nodeDict.values()]
            EdgeTypes._edge_type_names.append(EdgeTypes.DefaultEdgeName)
        return EdgeTypes._edge_type_names

def get_num_node_features(structural_model:bool=True):
    # TODO: add other node feature length when we support homogenous/heterogeneous
    return len(node_kind_names)

def get_num_model_type_elements(include_component:bool) -> int:
    return len(model_type_elem_names) if include_component else len(model_type_elem_names) - 1


class TypeSequence:
    def __init__(self, include_comp:bool=False) -> None:
        self.include_comp = include_comp

    @property
    def type_element_names(self) -> List[str]:
        all_names = model_type_elem_names.copy()
        all_names.remove('<EMPTY>')
        if not self.include_comp:
            all_names.remove('COMP')
        return all_names

    @property
    def nonterminals(self) -> List[str]:
        return ['ARR','PTR']    # <EMPTY>?

    @property
    def terminals(self) -> List[str]:
        return [x for x in self.type_element_names if x not in self.nonterminals]

    @property
    def special(self) -> List[str]:
        special = []
        if self.include_comp:
            special.append('COMP')
        return special

    @property
    def aggregates(self) -> List[str]:
        return ['STRUCT', 'UNION', 'FUNC', *self.nonterminals]

    @property
    def non_primitives(self) -> List[str]:
        return [*self.aggregates, *self.special, *self.nonterminals]

    @property
    def primitives(self) -> List[str]:
        return [x for x in self.type_element_names if x not in self.non_primitives]

    def valid_type_sequences_for_len(self, seq_len:int) -> List[List[str]]:
        '''
        Generate a list of all valid type sequences for the given sequence length,
        and return them as a list of type sequences where each sequence itself is a list
        of type element strings
        '''
        all_classes = []
        prev_nts = []

        for level_idx in range(seq_len):
            current_level = [[x] for x in self.type_element_names]
            if prev_nts:
                for pnt in prev_nts:
                    # add this level to nonterminals from prev level
                    all_classes.extend([[*pnt, *x] for x in current_level])
            else:
                all_classes = current_level

            current_nts = []
            for x in all_classes:
                if x[-1] in self.nonterminals and len(x) == level_idx+1:
                    current_nts.append(x.copy())
            prev_nts = current_nts

        # filter out invalid FUNC combos
        filtered_classes = []
        for x in all_classes:
            if 'FUNC' in x:
                fidx = x.index('FUNC')
                # don't allow FUNC to appear first
                if fidx > 0:
                    # only allow FUNC to follow PTR
                    if x[fidx - 1] == 'PTR':
                        filtered_classes.append(x)
            else:
                filtered_classes.append(x)

        return filtered_classes

    # NOTE: I just realized, I think I can also just use this
    # for applying the "corrections" at the output :)

    @staticmethod
    def seq_to_datatype(typeseq:List[str]) -> DataType:
        return TypeSequence.element_to_datatype(typeseq[0], typeseq[1:])

    @staticmethod
    def element_to_datatype(type_element:str, remaining_seq:List[str]) -> DataType:
        std_names = BuiltinType.get_std_names()

        if type_element in std_names:
            return BuiltinType.from_standard_name(type_element)
        elif type_element == DataTypeCategories.Enum:
            return EnumType(DataTypeCategories.Enum)
        elif type_element == DataTypeCategories.Function:
            return FunctionType(None, [], name=DataTypeCategories.Function)
        elif type_element == DataTypeCategories.Struct:
            return StructType(db=None, name=DataTypeCategories.Struct)
        elif type_element == DataTypeCategories.Union:
            return UnionType(db=None, name=DataTypeCategories.Union)
        elif type_element == DataTypeCategories.Pointer:
            pointed_to = None
            if remaining_seq:
                pointed_to = TypeSequence.element_to_datatype(remaining_seq[0], remaining_seq[1:])
            return PointerType(pointed_to, pointer_size=8)      # pointer size doesn't matter for this, just defaulting to 8B
        elif type_element == DataTypeCategories.Array:
            element_type = None
            if remaining_seq:
                element_type = TypeSequence.element_to_datatype(remaining_seq[0], remaining_seq[1:])
            return ArrayType(element_type, num_elements=None)  # we don't have access to array dimension sizes here

_typeseq_ids = None

def typeseq_name_to_id() -> dict:
    global _typeseq_ids
    if _typeseq_ids is None:
        _typeseq_ids = {ts_name: i for i, ts_name in enumerate(model_type_elem_names)}
    return _typeseq_ids

def encode_typeseq(type_seq:List[str], batch_fmt:bool=True) -> torch.Tensor:
    '''
    Encodes a type sequence (list of type names) into a dataset-formatted feature vector
    '''
    # map individual type names to their ordinal
    name_to_id = typeseq_name_to_id()
    type_ids = torch.tensor([name_to_id[x] for x in type_seq])
    dataset_fmt = F.one_hot(type_ids, num_classes=len(name_to_id)).to(torch.float32)
    if batch_fmt:
        return dataset_to_batch_format(dataset_fmt)
    return dataset_fmt

def batch_to_dataset_format(batch_tensor:torch.Tensor) -> List[torch.Tensor]:
    '''
    Return a list of each batch-formatted tensor converted to a dataset-formatted
    tensor
    '''
    return [x.T for x in batch_tensor]

def dataset_to_batch_format(ds_tensor:torch.Tensor) -> torch.Tensor:
    '''
    Convert the dataset tensor (N, 22) to a batch tensor that has the extra batch
    dimension added to form a (1, 22, N) sized tensor
    '''
    # transpose to get (22, N), then use [None,:,:] to add batch dimension
    # result is: (1, 22, N) where N is length of the sequence
    return ds_tensor.T[None,:,:]

def decode_typeseq(typeseq_probabilities:torch.Tensor, drop_empty_elems:bool=False,
        force_valid_seq:bool=False) -> List[str]:
    '''Decodes a type sequence vector into a list of string type names'''
    index_seq = typeseq_probabilities.argmax(dim=typeseq_probabilities.dim()-2)
    typeseq = []

    if index_seq.numel() > 1:
        typeseq = [model_type_elem_names[i] for i in index_seq.squeeze()]
    else:
        typeseq = [model_type_elem_names[i] for i in index_seq]

    if force_valid_seq:
        orig_length = len(typeseq)

        # correct model outputs via heuristics
        # 1. replace <EMPTY> with void
        no_empty = [x if x != '<EMPTY>' else 'void' for x in typeseq]

        # 2. truncate sequence after first non-terminal
        terminals = TypeSequence(include_comp=True).terminals
        terminal_idxs = [i for i, x in enumerate(no_empty) if x in terminals]
        first_terminal_idx = terminal_idxs[0] if terminal_idxs else None
        truncated = no_empty[:first_terminal_idx+1] if first_terminal_idx is not None else no_empty

        # 3. ensure FUNC is not first element and always follows a PTR
        if truncated[0] == 'FUNC':
            truncated.insert(0, 'PTR')
        elif 'FUNC' in truncated:
            fidx = truncated.index('FUNC')
            if truncated[fidx-1] != 'PTR':
                truncated.insert(fidx, 'PTR')
                truncated = truncated[:orig_length]     # ensure we don't exceed original fixed length

        return truncated

    elif drop_empty_elems:
        isempty = [x == '<EMPTY>' for x in typeseq]
        first_nonempty_from_rear = isempty[-1::-1].index(False)

        # drop all trailing <EMPTY> elements
        typeseq = typeseq[:len(typeseq)-first_nonempty_from_rear]

    return typeseq


def extend_to_fixed_len(y:torch.tensor, fixed_len:int) -> torch.tensor:
    num_empty_slots = fixed_len - y.shape[-1]
    if num_empty_slots < 1:
        return y
    return torch.cat((y, encode_typeseq(['<EMPTY>']*num_empty_slots)), dim=2)

class ToFixedLengthTypeSeq(BaseTransform):
    '''
    According to this link, the normal NLP approach is just pad the sequence to max length
    for batching: https://github.com/pyg-team/pytorch_geometric/discussions/4226
    '''
    def __init__(self, fixed_len:int) -> None:
        super().__init__()
        self.fixed_len = fixed_len

    def forward(self, data: Any) -> Any:
        # blindly make data.y of fixed-length
        seq_len = data.y.shape[-1]
        if seq_len >= self.fixed_len:
            data.y = data.y[..., :self.fixed_len]
        else:
            data.y = extend_to_fixed_len(data.y, self.fixed_len)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.fixed_len})'

class ToBatchTensors(BaseTransform):
    '''
    We have to store tensors in the .pt files with the final dimension being
    always equal, so we use (N, 22) tensors. However, for batches/training we
    need (batch_size, 22, N) formatted tensors.

    This transform converts each data object from dataset format to batch
    format
    '''
    def __init__(self) -> None:
        super().__init__()

    def forward(self, data: Any) -> Any:
        data.y = dataset_to_batch_format(data.y)
        return data
