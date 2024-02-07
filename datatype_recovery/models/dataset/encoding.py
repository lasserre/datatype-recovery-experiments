from astlib import ASTNode
import torch
from torch.nn import functional as F
from torch_geometric.transforms import BaseTransform
from typing import List, Any

from varlib.datatype.datatypes import _standard_floats_by_size, _standard_ints_by_size, _standard_uints_by_size


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

# these are the output classes for type sequence prediction
type_seq_names = [
    *_standard_floats_by_size.values(),
    *_standard_ints_by_size.values(),
    *_standard_uints_by_size.values(),
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

def get_num_node_features(structural_model:bool=True):
    # TODO: add other node feature length when we support homogenous/heterogeneous
    return len(node_kind_names)

def get_num_classes(include_component:bool) -> int:
    return len(type_seq_names) if include_component else len(type_seq_names) - 1

_typeseq_ids = None

def typeseq_name_to_id() -> dict:
    global _typeseq_ids
    if _typeseq_ids is None:
        _typeseq_ids = {ts_name: i for i, ts_name in enumerate(type_seq_names)}
    return _typeseq_ids

def encode_typeseq(type_seq:List[str]) -> torch.Tensor:
    '''Encodes a type sequence (list of type names) into a feature vector'''
    # map individual type names to their ordinal
    name_to_id = typeseq_name_to_id()
    type_ids = torch.tensor([name_to_id[x] for x in type_seq])
    return F.one_hot(type_ids, num_classes=len(name_to_id)).to(torch.float32)

def decode_typeseq(typeseq_probabilities:torch.Tensor) -> List[str]:
    '''Decodes a type sequence vector into a list of string type names'''
    return [type_seq_names[i] for i in typeseq_probabilities.argmax(dim=1)]

def extend_to_fixed_len(y:torch.tensor, fixed_len:int) -> torch.tensor:
    num_empty_slots = fixed_len-len(y)
    if num_empty_slots < 1:
        return y
    return torch.cat((y, encode_typeseq(['<EMPTY>']*num_empty_slots)))

class ToFixedLengthTypeSeq(BaseTransform):
    def __init__(self, fixed_len:int) -> None:
        super().__init__()
        self.fixed_len = fixed_len

    def forward(self, data: Any) -> Any:
        # blindly make data.y of fixed-length
        seq_len = data.y.shape[0]
        if seq_len >= self.fixed_len:
            data.y = data.y[:self.fixed_len, ...]
        else:
            data.y = extend_to_fixed_len(data.y, self.fixed_len)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.fixed_len})'