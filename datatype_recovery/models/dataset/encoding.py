from astlib import ASTNode
import torch
from torch.nn import functional as F
from typing import List
from varlib.datatype import _standard_floats_by_size, _standard_ints_by_size, _standard_uints_by_size


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
    return F.one_hot(torch.tensor(kind_to_id[node.kind]), len(kind_to_id.keys()))

def decode_astnode(encoded_node:torch.Tensor) -> 'str':
    '''Decodes a node into its kind string'''
    return node_kind_names[encoded_node.argmax()]

# these are the output classes for type sequence prediction
type_seq_names = [
    *_standard_floats_by_size.values(),
    *_standard_ints_by_size.values(),
    *_standard_uints_by_size.values(),
    'PTR',
    'ARR',
    'STRUCT',
    'UNION',
    'ENUM',
    'FUNC'
]
print(f'TODO - predict COMP or not???')

_typeseq_ids = None

def typeseq_name_to_id() -> dict:
    global _typeseq_ids
    if _typeseq_ids is None:
        _typeseq_ids = {ts_name: i for i, ts_name in enumerate(type_seq_names)}
    return _typeseq_ids

def encode_typeseq(type_seq:List[str]) -> torch.Tensor:
    '''Encodes a type sequence (list of type names) into a feature vector'''
    name_to_id = typeseq_name_to_id()
    return F.one_hot(torch.tensor([name_to_id[x] for x in type_seq]), len(name_to_id.keys()))

def decode_typeseq(encoded_typeseq:torch.Tensor) -> List[str]:
    '''Decodes a type sequence vector into a list of string type names'''
    return [type_seq_names[i] for i in encoded_typeseq.argmax(dim=1)]
