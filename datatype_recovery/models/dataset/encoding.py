from astlib import *
import torch
from torch.nn import functional as F
from torch_geometric.transforms import BaseTransform
from typing import List, Any

from varlib.datatype.datatypes import _builtin_floats_by_size, _builtin_ints_by_size, _builtin_uints_by_size
from varlib.datatype import *

class Opcodes:
    # both unary and binary opcodes
    _all_opcodes = [
        '',     # default/empty opcode for non operator nodes
        ',',
        '=',
        '+',
        '-',
        '*',
        '/',
        '%',
        '<<',
        '>>',
        '==',
        '<',
        '>',
        '<=',
        '>=',
        '!=',
        '!',
        '&',
        '^',
        '~',
        '|',
        '&&',
        '||',
        '*=',
        '/=',
        '%=',
        '+=',
        '-=',
        '<<=',
        '>>=',
        '&=',
        '|=',
        '^=',
    ]

    _opcode_to_id = None

    @staticmethod
    def all_opcodes() -> List[str]:
        return Opcodes._all_opcodes

    @staticmethod
    def opcode_to_id() -> dict:
        if Opcodes._opcode_to_id is None:
            Opcodes._opcode_to_id = {name: idx for idx, name in enumerate(Opcodes.all_opcodes())}
        return Opcodes._opcode_to_id

    @staticmethod
    def encode(opcode:str) -> torch.Tensor:
        '''Encodes the specified opcode string into an opcode feature vector'''
        opcode_ids = Opcodes.opcode_to_id()
        return F.one_hot(torch.tensor(opcode_ids[opcode]), len(Opcodes.all_opcodes())).to(torch.float32)

    @staticmethod
    def decode(encoded_opcode:torch.Tensor) -> 'str':
        '''Decodes an opcode into its opcode string'''
        return Opcodes.all_opcodes()[encoded_opcode.argmax()]

class NodeKinds:
    _all_names = [
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

    @staticmethod
    def all_names() -> List[str]:
        return NodeKinds._all_names

    _name_to_id = None

    @staticmethod
    def name_to_id() -> dict:
        if NodeKinds._name_to_id is None:
            NodeKinds._name_to_id = {name: idx for idx, name in enumerate(NodeKinds.all_names())}
        return NodeKinds._name_to_id

    @staticmethod
    def encode(node_kind:str) -> torch.Tensor:
        '''Encodes the specified node kind string into a node kind feature vector'''
        return F.one_hot(torch.tensor(NodeKinds.name_to_id()[node_kind]), len(NodeKinds.all_names())).to(torch.float32)

    @staticmethod
    def decode(encoded_node_kind:torch.Tensor) -> 'str':
        '''Decodes a node into its kind string'''
        return NodeKinds.all_names()[encoded_node_kind.argmax()]

class EdgeTypes:
    # -------------------------------------
    # Multi-level dict mapping ParentNodeType->ChildIdx->EdgeName
    # ParentNodeType: {
    #       ChildIdx: EdgeName
    # }
    # NOTE: these ChildIdx values are NOT the edge ids I'm encoding
    # ...I generate edge ids based on a global list of all edge names.
    # Here, I need to map the idx of the child AST node to its edge name
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
        'DoStmt': {
            0: 'DoLoop',    # body
            1: 'DoCond',    # cond
        },
        'ForStmt': {
            # my doc says some of these are optional, but in Ghidra I insert NullNodes
            # as placeholders if no real element exists
            0: 'ForInit',
            1: 'ForCond',   # expr
            2: 'ForIncr',
            3: 'ForBody',   # body
        },
        'SwitchStmt': {
            0: 'SwitchExpr',    # expr
            1: 'SwitchCases',   # body
        },
        'WhileStmt': {
            0: 'WhileCond', # cond
            1: 'WhileBody', # body
        },
    }

    DefaultEdgeName = 'default'    # kind of want to just make default edge all zeros, but this simplifies encode/decode logic

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
    def encode(edge_type:str, child_to_parent:bool=None) -> torch.Tensor:
        '''
        Encodes the specified edge type string into an edge type feature vector, with
        a 1 appended if this is a child->parent edge, or a 0 if this is a parent->child edge
        '''
        edge_type_ids = EdgeTypes.edge_type_ids()
        edge_type_tensor = F.one_hot(torch.tensor(edge_type_ids[edge_type]), len(edge_type_ids.keys()))

        if child_to_parent is None:
            return edge_type_tensor.to(torch.float32)

        to_parent = torch.tensor(int(child_to_parent)).unsqueeze(0)
        return torch.cat((edge_type_tensor, to_parent)).to(torch.float32)

    @staticmethod
    def edge_dim() -> int:
        '''Returns the size of an encoded edge feature vector'''
        # +1 for to_parent
        return len(EdgeTypes.all_types()) + 1

    @staticmethod
    def decode(encoded_edge_type:torch.Tensor) -> 'str':
        '''Decodes a edge into its edge type string'''
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

# our new encoding uses LeafType + PointerLevels as opposed to the
# old TypeSequence. I was going to change/reuse TypeSequence, but then
# I realized since this class is all about the encoding many of the details
# have changed, so I am creating new classes for the new encodings

class LeafTypeThresholds:
    def __init__(self, signed_threshold:float=0.0, floating_threshold:float=0.0):
        '''
        signed_threshold: Threshold in logits for is_signed output of model
        floating_threshold: Threshold in logits for is_floating output of model
        '''
        self.signed_threshold = signed_threshold
        self.floating_threshold = floating_threshold

class LeafType:
    '''Encoding of leaf data type'''

    _valid_sizes = [
        0, 1, 2, 4, 8, 16
    ]

    _category_name_to_id = {
        'BUILTIN': 0,
        'STRUCT': 1,
        'UNION': 2,
        'FUNC': 3,
        'ENUM': 4,
    }

    _category_id_to_name = {v: k for k, v in _category_name_to_id.items()}

    def __init__(self, leaf_category:str, is_signed:bool, is_floating:bool, size:int):
        self.leaf_category = leaf_category
        self.is_signed = is_signed
        self.is_floating = is_floating
        self.size = size

    def __repr__(self) -> str:
        return f'{self.leaf_category},signed={int(self.is_signed)},float={int(self.is_floating)},size={self.size}'

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, LeafType):
            return False
        return value.leaf_category == self.leaf_category and \
            value.is_signed == self.is_signed and \
            value.is_floating == self.is_floating and \
            value.size == self.size

    @staticmethod
    def tensor_size() -> int:
        '''Size of an encoded LeafType tensor'''
        # +2 for floating, signed
        return len(LeafType._valid_sizes) + len(LeafType._category_name_to_id) + 2

    @staticmethod
    def valid_categories() -> List[str]:
        return list(LeafType._category_name_to_id.keys())

    @staticmethod
    def valid_sizes() -> List[int]:
        return LeafType._valid_sizes

    @staticmethod
    def from_datatype(dtype:DataType) -> 'LeafType':
        return LeafType(dtype.leaf_type.category,
                        dtype.leaf_type.is_signed,
                        dtype.leaf_type.is_floating,
                        dtype.leaf_type.primitive_size)

    def to_dtype(self) -> DataType:
        '''Convert this LeafType instance to its corresponding DataType'''
        if self.leaf_category == 'BUILTIN':
            size = self.size
            if self.is_floating and self.size not in [4, 8]:
                # need to fix float size to be valid
                # - we encode 10-B long double with size 16 (so now we need to flip it back)
                # - floats < 4B are invalid, so we arbitrarily correct the size to be min valid size (4B)
                size = 4 if self.size < 4 else 10
            return BuiltinType('', self.is_floating, self.is_signed, size)
        elif self.leaf_category == 'STRUCT':
            return StructType(db=None, name='STRUCT')
        elif self.leaf_category == 'UNION':
            return UnionType(db=None, name='UNION')
        elif self.leaf_category == 'FUNC':
            return FunctionType(return_dtype=BuiltinType.create_void_type(), params=[], name='FUNC')
        elif self.leaf_category == 'ENUM':
            return EnumType(name='ENUM')
        raise Exception(f'Unrecognized leaf category {self.leaf_category}')

    @staticmethod
    def decode(leaftype_tensor:torch.Tensor, thresholds:LeafTypeThresholds=None) -> 'LeafType':
        '''
        Decodes a leaf type vector into a LeafType object

        leaftype_tensor: The tensor encoded by LeafType.encode()
        thresholds: If specified, interpret binary items as logits and use thresholds to
                    convert to binary. Otherwise, assume they are already in binary
        '''
        category = LeafType._category_id_to_name[leaftype_tensor[0,:5].argmax().item()]
        is_signed = leaftype_tensor[0,5].item()
        is_floating = leaftype_tensor[0,6].item()
        if thresholds:
            is_signed = is_signed > thresholds.signed_threshold
            is_floating = is_floating > thresholds.floating_threshold
        size = LeafType._valid_sizes[leaftype_tensor[0,7:].argmax().item()]
        return LeafType(category, is_signed, is_floating, size)

    # NOTE: I don't think I will need batch_fmt vs dataset format...try just using 1 format
    # def encode(self, batch_fmt:bool=True) -> torch.Tensor:

    @property
    def encoded_tensor(self) -> torch.Tensor:
        '''
        Encodes the leaf type into a batch-formatted feature vector of shape (1, 13)

        Vector format (one-hot encoded):
        [category (5)][sign (1)][float (1)][size (6)]
        '''
        # category
        num_categories = len(LeafType._category_name_to_id)
        category_id = LeafType._category_name_to_id[self.leaf_category]
        category_tensor = F.one_hot(torch.tensor([category_id]), num_classes=num_categories)

        # is_signed/is_floating (use unsqueeze to make the shape [1,2])
        signfloat_tensor = torch.tensor([int(self.is_signed), int(self.is_floating)]).unsqueeze(0)

        # size
        size_idx = LeafType._valid_sizes.index(self.size)
        size_tensor = F.one_hot(torch.tensor([size_idx]), num_classes=len(LeafType._valid_sizes))

        return torch.cat([category_tensor, signfloat_tensor, size_tensor], dim=1).to(torch.float32)

class PointerLevels:
    '''Encoding of the pointer hierarchy portion of a data type'''

    _ptr_type_to_id = {
        'L': 0,
        'P': 1,
        'A': 2,
    }

    _ptr_type_to_typeseq = {
        'P': 'PTR',
        'A': 'ARR',
        'L': 'LEAF',    # this is invalid, but helps represent raw predictions
    }

    _ptr_id_to_typename = {v: k for k, v in _ptr_type_to_id.items()}

    def __init__(self, ptr_levels:str='LLL') -> None:
        '''
        ptr_levels: Is the string representation of the pointer levels, where each
                    character is one of:
                    P - pointer
                    A - array
                    L - leaf
        '''
        self.ptr_levels = ptr_levels

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, PointerLevels):
            return False
        return value.ptr_levels == self.ptr_levels

    def __repr__(self) -> str:
        return ','.join(self.ptr_levels)

    @staticmethod
    def tensor_size() -> int:
        '''Size of an encoded PointerLevels tensor'''
        # num ptr types * num levels
        return len(PointerLevels._ptr_type_to_id) * 3

    @property
    def type_sequence_str_raw(self) -> str:
        '''
        Return the raw (uncorrected) type sequence string representing
        this pointer hierarchy
        '''
        # leave non-trailing LEAF entries alone so we capture any invalid
        # raw predictions (e.g. PLP translates to PTR,LEAF,PTR which is invalid)

        # BUT, we still trim trailing LEAF entries for a raw prediction
        # (e.g. since PAL should correctly be translated to PTR,ARR)

        trimmed_ptrs = self.ptr_levels.rstrip('L')
        return ','.join([PointerLevels._ptr_type_to_typeseq[p] for p in trimmed_ptrs])

    def to_dtype(self, leaf_type:DataType=None) -> DataType:
        '''
        Convert this pointer hierarchy to a concrete DataType with the given
        leaf_type, or void if no leaf type is supplied
        '''
        if not leaf_type:
            leaf_type = BuiltinType.create_void_type()

        # wrap contained types for all ptr levels, working inside-out (from the end)
        dtype = leaf_type

        # take all pointer levels until we hit the first leaf type
        # (i.e. the first LEAF ends the pointer hierarchy, keeping the resulting
        # data type valid...PLP would give us PTR,leaf_type)

        first_leaf = self.ptr_levels.find('L')
        ptrs_only = self.ptr_levels[:first_leaf] if first_leaf > -1 else self.ptr_levels

        for ptr in ptrs_only[-1::-1]:
            if ptr == 'P':
                dtype = PointerType(dtype, pointer_size=8)  # assume x64
            else:
                dtype = ArrayType(dtype, num_elements=0)

        return dtype

    @staticmethod
    def decode(ptrlevels_tensor:torch.Tensor, force_valid_type:bool=False) -> 'PointerLevels':
        '''
        Decodes a pointer levels vector into a PointerLevels object

        ptrlevels_tensor: The tensor encoded by PointerLevels.encode()
        force_valid_type: Ensure the output PointerLevels is a valid data type (don't just blindly accept raw predictions
                          like PLP which isn't valid)
        '''
        l1_ptype = PointerLevels._ptr_id_to_typename[ptrlevels_tensor[0,:3].argmax().item()]
        l2_ptype = PointerLevels._ptr_id_to_typename[ptrlevels_tensor[0,3:6].argmax().item()]
        l3_ptype = PointerLevels._ptr_id_to_typename[ptrlevels_tensor[0,6:9].argmax().item()]

        if force_valid_type:
            if l1_ptype == 'L':
                l2_ptype = 'L'
            if l2_ptype == 'L':
                l3_ptype = 'L'

        return PointerLevels(f'{l1_ptype}{l2_ptype}{l3_ptype}')

    @property
    def encoded_tensor(self) -> torch.Tensor:
        '''
        Encodes the pointer levels into a batch-formatted feature vector of shape (1, 3*N)
        where N is the number of levels (right now I'm using 3 levels, so shape is (1, 9))

        Vector format (one-hot encoded):
        [L1 ptr_type (3)][L2 ptr_type (3)]...[LN ptr_type (3)]
        '''
        ptr_type_tensors = []
        num_ptypes = len(PointerLevels._ptr_type_to_id)     # should always be 3 (P, A, L)
        for ptype in self.ptr_levels:
            pid = PointerLevels._ptr_type_to_id[ptype]
            ptype_tensor = F.one_hot(torch.tensor([pid]), num_classes=num_ptypes)
            ptr_type_tensors.append(ptype_tensor)
        return torch.cat(ptr_type_tensors, dim=1).to(torch.float32)

class TypeEncoder:
    '''Implements the new data type encoding'''

    @staticmethod
    def tensor_size() -> int:
        '''Size of an encoded data type tensor'''
        return LeafType.tensor_size() + PointerLevels.tensor_size()

    @staticmethod
    def empty_tensor(onedim:bool=False) -> torch.Tensor:
        '''
        Encode an all-zero tensor to represent an "EMPTY" data type
        (specifically for AST nodes in our homogenous GNN where we have to
        include something but a data type isn't meaningful)
        '''
        type_tensor = torch.zeros(1,TypeEncoder.tensor_size()).to(torch.float32)
        return type_tensor if not onedim else type_tensor.view(TypeEncoder.tensor_size())

    @staticmethod
    def encode(dtype:DataType, onedim:bool=False) -> torch.Tensor:
        '''
        Encodes the DataType object into a feature vector of shape (1,P+L) where P is
        the length of the ptr-levels tensor and L is the length of the leaf type tensor
        Currently using 3 pointer levels with a tensor size of 9 and a leaf tensor size of 13
        resulting in tensor shapes of (1, 22)

        Vector format (one-hot encoded):
        [ptr_levels (9)][leaf type (13)]
        '''
        leaf_tensor = LeafType.from_datatype(dtype).encoded_tensor
        ptrlevels_tensor = PointerLevels(''.join(dtype.ptr_hierarchy(3))).encoded_tensor
        type_tensor = torch.cat([ptrlevels_tensor, leaf_tensor], dim=1).to(torch.float32)
        return type_tensor if not onedim else type_tensor.view(TypeEncoder.tensor_size())

    @staticmethod
    def decode_ptrlevels(type_tensor:torch.Tensor) -> PointerLevels:
        '''Extract the pointer levels sub-tensor and decode it'''
        return PointerLevels.decode(type_tensor[:1,:9])

    @staticmethod
    def decode_leaftype(type_tensor:torch.Tensor, thresholds:LeafTypeThresholds=None) -> LeafType:
        '''Extract the leaf type sub-tensor and decode it'''
        return LeafType.decode(type_tensor[:1,9:], thresholds)

    @staticmethod
    def decode(type_tensor:torch.Tensor, thresholds:LeafTypeThresholds=None) -> DataType:
        x = type_tensor[None,:] if type_tensor.ndim == 1 else type_tensor
        ptrs = TypeEncoder.decode_ptrlevels(x)
        leaf_type = TypeEncoder.decode_leaftype(x, thresholds)
        return ptrs.to_dtype(leaf_type.to_dtype())

    @staticmethod
    def decode_raw_typeseq(type_tensor:torch.Tensor, thresholds:LeafTypeThresholds=None) -> str:
        ptr_levels = TypeEncoder.decode_ptrlevels(type_tensor)
        leaf_type = TypeEncoder.decode_leaftype(type_tensor, thresholds)
        if ptr_levels.type_sequence_str_raw:
            return ','.join([
                ptr_levels.type_sequence_str_raw,
                leaf_type.to_dtype().type_sequence_str
            ])
        return leaf_type.to_dtype().type_sequence_str

class TypeSequence:
    # these are the individual model output elements for type sequence prediction
    _model_type_elem_names = [
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
        'COMP',     # keep this last since it is optional
    ]

    _element_name_to_id = None

    @staticmethod
    def typeseq_name_to_id() -> dict:
        if TypeSequence._element_name_to_id is None:
            TypeSequence._element_name_to_id = {ts_name: i for i, ts_name in enumerate(TypeSequence._model_type_elem_names)}
        return TypeSequence._element_name_to_id

    def __init__(self, include_comp:bool=False) -> None:
        self.include_comp = include_comp

    @property
    def model_type_elements(self) -> List[str]:
        '''
        Number of raw model type elements, including <EMPTY> and including
        COMP if applicable
        '''
        all_names = TypeSequence._model_type_elem_names.copy()
        if not self.include_comp:
            all_names.remove('COMP')
        return all_names

    @property
    def logical_type_elements(self) -> List[str]:
        '''
        List of the logical type elements
        (i.e. not including <EMPTY> and not including COMP if applicable)
        '''
        all_names = self.model_type_elements
        all_names.remove('<EMPTY>')
        return all_names

    @property
    def nonterminals(self) -> List[str]:
        return ['ARR','PTR']    # <EMPTY>?

    @property
    def terminals(self) -> List[str]:
        return [x for x in self.logical_type_elements if x not in self.nonterminals]

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
        return [x for x in self.logical_type_elements if x not in self.non_primitives]

    def valid_type_sequences_for_len(self, seq_len:int) -> List[List[str]]:
        '''
        Generate a list of all valid type sequences for the given sequence length,
        and return them as a list of type sequences where each sequence itself is a list
        of type element strings
        '''
        exclude_first_level = ['FUNC', 'ARR', 'PTR']

        # NOTE: if/when we support return types, this is valid as a standalone type (for return types only)
        exclude_first_level.append('void')

        all_classes = [[x] for x in self.logical_type_elements if x not in exclude_first_level]
        prev_nts = [['ARR'], ['PTR']]

        arr_terminals = [x for x in self.terminals if x != 'void']

        for i in range(1, seq_len):
            for nonterminal in prev_nts:
                if nonterminal[-1] == 'ARR':
                    all_classes.extend([*nonterminal, x] for x in arr_terminals)
                else:
                    all_classes.extend([[*nonterminal, x] for x in self.terminals])

            new_nts = []
            for nonterminal in prev_nts:
                new_nts.append([*nonterminal, 'ARR'])
                new_nts.append([*nonterminal, 'PTR'])
            prev_nts = new_nts

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


    def encode(self, type_seq:List[str], batch_fmt:bool=True) -> torch.Tensor:
        '''
        Encodes a type sequence (list of type names) into a dataset-formatted feature vector

        If batch_fmt, output tensor shape will be (1, 22, N) where N is sequence length
        Otherwise for the default/dataset format, output tensor is (N, 22)
        '''
        # map individual type names to their ordinal
        name_to_id = TypeSequence.typeseq_name_to_id()
        num_classes = len(self.model_type_elements)
        type_ids = torch.tensor([name_to_id[x] for x in type_seq])

        dataset_fmt = F.one_hot(type_ids, num_classes=num_classes).to(torch.float32)

        if batch_fmt:
            return TypeSequence.dataset_to_batch_format(dataset_fmt)
        return dataset_fmt

    @staticmethod
    def batch_to_dataset_format(batch_tensor:torch.Tensor) -> List[torch.Tensor]:
        '''
        Return a list of each batch-formatted tensor converted to a dataset-formatted
        tensor
        '''
        return [x.T for x in batch_tensor]

    @staticmethod
    def dataset_to_batch_format(ds_tensor:torch.Tensor) -> torch.Tensor:
        '''
        Convert the dataset tensor (N, 22) to a batch tensor that has the extra batch
        dimension added to form a (1, 22, N) sized tensor
        '''
        # transpose to get (22, N), then use [None,:,:] to add batch dimension
        # result is: (1, 22, N) where N is length of the sequence
        return ds_tensor.T[None,:,:]

    def decode(self, seq_tensor:torch.Tensor, drop_empty_elems:bool=False,
            force_valid_seq:bool=False, batch_fmt:bool=True) -> List[str]:
        '''Decodes a type sequence vector into a list of string type names'''

        argmax_dim = seq_tensor.dim()-2 if batch_fmt else 1
        index_seq = seq_tensor.argmax(dim=argmax_dim)
        typeseq = []

        elem_names = self.model_type_elements

        if index_seq.numel() > 1:
            typeseq = [elem_names[i] for i in index_seq.squeeze()]
        else:
            typeseq = [elem_names[i] for i in index_seq]

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

            # 4. convert trailing PTR|ARR to void
            if truncated[-1] in ['PTR','ARR']:
                truncated[-1] = 'void'

            # 5. convert trailing ARR->void to ARR->char (arbitrarily! just need to make it valid)
            if truncated[-2:] == ['ARR','void']:
                truncated[-1] = 'char'

            return truncated

        elif drop_empty_elems:
            isempty = [x == '<EMPTY>' for x in typeseq]
            first_nonempty_from_rear = isempty[-1::-1].index(False)

            # drop all trailing <EMPTY> elements
            typeseq = typeseq[:len(typeseq)-first_nonempty_from_rear]

        return typeseq

    def to_fixed_len_tensor(self, y:torch.tensor, fixed_len:int, batch_fmt:bool=True) -> torch.tensor:
        '''
        Convert the encoded type sequence into a fixed-length tensor

        Batch format: (1, 22, N)
        Dataset format: (N, 22)
        '''
        seq_len = y.shape[-1] if batch_fmt else y.shape[0]

        if seq_len >= fixed_len:
            return y[..., :fixed_len] if batch_fmt else y[:fixed_len, ...]

        # we need to extend by at least one <EMPTY> element
        num_empty_slots = fixed_len - seq_len
        cat_dim = 2 if batch_fmt else 0
        return torch.cat((y, self.encode(['<EMPTY>']*num_empty_slots, batch_fmt=batch_fmt)), dim=cat_dim)

def get_num_node_features(structural_model:bool=True):
    if structural_model:
        return len(NodeKinds.all_names())     # only node type
    else:
        # Node features:
        # - node type
        # - data type (type sequence vector)
        # - opcode
        node_type_features = len(NodeKinds.all_names())
        data_type_features = TypeEncoder.empty_tensor().numel()
        opcode_features = len(Opcodes.all_opcodes())
        return node_type_features + data_type_features + opcode_features

def get_num_model_type_elements(include_component:bool) -> int:
    return len(TypeSequence(include_component).model_type_elements)

class HeteroEdgeTypes:
    # -------------------------------------
    # Multi-level dict mapping ParentNodeType->ChildIdx->EdgeName
    # ParentNodeType: {
    #       ChildIdx: EdgeName
    # }
    # -------------------------------------
    _edgetype_lookup = {
        'BinaryOperator': {
            0: 'Left',
            1: 'Right',
        },
        'IfStmt': {
            0: 'Cond',          # IfCond    # CLS: don't care to differentiate if vs. else blocks
        },
        'CallExpr': {
            # 0: 'CallTarget',
            1: 'Param1',
            2: 'Param2',
            3: 'Param3',
            4: 'Param4',
            # 5: 'Param5',
            # 6: 'Param6',   # per histogram of coreutils, # params drops WAY off after 6
        },
        'ArraySubscriptExpr': {
            0: 'ArrayVar',
            1: 'ArrayIdxExpr',
        },
        'DoStmt': {
            0: 'Body',    # DoLoop
            1: 'Cond',    # DoCond
        },
        'ForStmt': {
            # my doc says some of these are optional, but in Ghidra I insert NullNodes
            # as placeholders if no real element exists
            0: 'ForInit',
            1: 'Cond',      # ForCond
            2: 'ForIncr',
            3: 'Body',      # ForBody
        },
        'SwitchStmt': {
            0: 'Cond',      # SwitchExpr
            1: 'Body',      # SwitchCases
        },
        'WhileStmt': {
            0: 'Cond',      # WhileCond
            1: 'Body',      # WhileBody
        },
    }

    DefaultEdgeName = 'Default'    # kind of want to just make default edge all zeros, but this simplifies encode/decode logic

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
        if parent.kind in HeteroEdgeTypes._edgetype_lookup:
            nodetype_lookup = HeteroEdgeTypes._edgetype_lookup[parent.kind]
            if child_idx in nodetype_lookup:
                return nodetype_lookup[child_idx]

        return HeteroEdgeTypes.DefaultEdgeName

    @staticmethod
    def all_types() -> List[str]:
        if HeteroEdgeTypes._edge_type_names is None:
            HeteroEdgeTypes._edge_type_names = list(set([edge_name for nodeDict in HeteroEdgeTypes._edgetype_lookup.values() for edge_name in nodeDict.values()]))
            HeteroEdgeTypes._edge_type_names.append(HeteroEdgeTypes.DefaultEdgeName)
        return HeteroEdgeTypes._edge_type_names

class HeteroNodeEncoder(ASTVisitor):
    '''
    Encode node-specific features for each node type in the AST
    '''
    def __init__(self):
        super().__init__(warn_missing_visits=False, get_default_return_value=self._encode_default)

    # Node groups
    # ------------
    # - Default [Cond, Body, Expr, Default]
    # - Literal (Char, Float, Int, String) [to_child edges: None]
    # - Operator (Binop, Unop) [to_child edges: Left, Right, Default]
    # - NodeWithType [Default] (TODO - nothing fits here yet unless we add MemberExpr or combine with Literal)
    # - CallExpr [Param1-4, Default]

    _outgoing_edges_for_group = {
        'Default': ['Cond', 'Body', 'ArrayVar', 'ArrayIdxExpr', 'ForInit', 'ForIncr', 'Default'],
        'Literal': [],
        'Operator': ['Left', 'Right', 'Default'],
        # 'NodeWithType': ['Default'],
        'CallExpr': ['Param1', 'Param2', 'Param3', 'Param4', 'Default']
    }

    _node_kind_to_group = {
        # map all non-default nodes here

        # operators
        'BinaryOperator': 'Operator',
        'UnaryOperator': 'Operator',

        # literals
        'CharacterLiteral': 'Literal',
        'FloatingLiteral': 'Literal',
        'IntegerLiteral': 'Literal',
        'StringLiteral': 'Literal',

        # call expression
        'CallExpr': 'CallExpr'
    }

    @staticmethod
    def get_node_group(kind:str) -> str:
        '''
        Returns the group name for this node (the group name
        is the node type for purposes of the GNN, but not the
        specific node kind attached to the ASTNode)
        '''
        if kind in HeteroNodeEncoder._node_kind_to_group:
            return HeteroNodeEncoder._node_kind_to_group[kind]
        return 'Default'

    @staticmethod
    def get_metadata() -> Tuple[List[str], List[Tuple[str,str,str]]]:
        '''
        Returns the metadata tuple suitable for use in initializing hetero GNN models
        '''
        node_types = [HeteroNodeEncoder.get_node_group(k) for k in NodeKinds.all_names()]
        edge_types = []

        # forward edges
        for ntype in node_types:
            for edge_name in HeteroNodeEncoder._outgoing_edges_for_group[ntype]:
                # just assume it could arrive at any other node
                edge_types.extend([(ntype, edge_name, dest) for dest in node_types])

        # add reverse edges
        edge_types.extend([(x, f'rev_{edge_name}', y) for x, edge_name, y in edge_types])

        return (node_types, edge_types)

    @staticmethod
    def _encode_default(node:ASTNode):
        # just node kind
        return NodeKinds.encode(node.kind)

    @staticmethod
    def _encode_node_with_type(kind:str, dtype:DataType) -> torch.Tensor:
        return torch.cat((
            NodeKinds.encode(kind),
            TypeEncoder.encode(dtype, True) if dtype else TypeEncoder.empty_tensor(True),
        ))

    @staticmethod
    def _encode_operator(kind:str, opcode:str) -> torch.Tensor:
        return torch.cat((
            NodeKinds.encode(kind),
            Opcodes.encode(opcode)
        ))

    @staticmethod
    def encode(node:ASTNode) -> torch.Tensor:
        encode_func = getattr(HeteroNodeEncoder, f'encode{node.kind}', None)
        if encode_func:
            return encode_func(node)
        return HeteroNodeEncoder._encode_default(node)

    @staticmethod
    def encodeBinaryOperator(binop:BinaryOperator) -> torch.Tensor:
        return HeteroNodeEncoder._encode_operator(binop.kind, binop.opcode)

    @staticmethod
    def encodeCallExpr(expr:CallExpr):
        declref:DeclRefExpr = expr.inner[0]
        fdecl:FunctionDecl = declref.referencedDecl

        # want 4 params + return type = 5
        dtypes = [
            fdecl.return_dtype,
            *[p.dtype for p in fdecl.params],
            *[None]*(4-len(fdecl.params))
        ] if isinstance(fdecl, FunctionDecl) else [None]*5

        return torch.cat((
            NodeKinds.encode(expr.kind),
            *[TypeEncoder.encode(dt,True) if dt else TypeEncoder.empty_tensor(True) for dt in dtypes]
        ))

    @staticmethod
    def encodeCharacterLiteral(lit:CharacterLiteral) -> torch.Tensor:
        return HeteroNodeEncoder._encode_node_with_type(lit.kind, lit.dtype)

    @staticmethod
    def encodeCStyleCastExpr(expr:CStyleCastExpr) -> torch.Tensor:
        return torch.cat((
            NodeKinds.encode(expr.kind),
            TypeEncoder.encode(expr.dtype, onedim=True)
        ))

    @staticmethod
    def encodeDeclRefExpr(declref:DeclRefExpr) -> torch.Tensor:
        return torch.cat((
            NodeKinds.encode(declref.kind),
            TypeEncoder.encode(declref.referencedDecl.dtype, onedim=True)
        ))

    @staticmethod
    def encodeFloatingLiteral(lit:FloatingLiteral) -> torch.Tensor:
        return HeteroNodeEncoder._encode_node_with_type(lit.kind, lit.dtype)

    @staticmethod
    def encodeIntegerLiteral(lit:IntegerLiteral) -> torch.Tensor:
        return HeteroNodeEncoder._encode_node_with_type(lit.kind, lit.dtype)


    # NOTE: if we want to encode MemberExpr's (with their data type) we have to
    # pull in the sdb in order to get the member's data type...
    #
    # @staticmethod
    # def encodeMemberExpr(expr:MemberExpr) -> torch.Tensor:
    #     return HeteroNodeEncoder._encode_node_with_type(expr.kind, expr.dtype)

    @staticmethod
    def encodeStringLiteral(lit:StringLiteral) -> torch.Tensor:
        return HeteroNodeEncoder._encode_node_with_type(lit.kind, lit.dtype)

    # NOTE: we don't normally encode decl's in the portions of AST data we
    # look at (sometimes decl info gets included by other node types like DeclRefExpr)
    # VarDecl, ParmVarDecl, FunctionDecl -> could be NodeWithType but I haven't seen
    # them in var graphs yet

    @staticmethod
    def encodeUnaryOperator(unop:UnaryOperator) -> torch.Tensor:
        return HeteroNodeEncoder._encode_operator(unop.kind, unop.opcode)

    def decode_node(self, node_tensor:torch.Tensor, node_kind:str) -> tuple:
        '''
        Decode the provided tensor and return its decoded data as a tuple
        '''
        # TODO:
        pass

        # kind_vec, dtype_vec, op_vec = NodeEncoder.split_node_vec(node_tensor)
        # return NodeKinds.decode(kind_vec), \
        #     TypeEncoder.decode(dtype_vec[None,:]), \
        #     Opcodes.decode(op_vec)

class NodeEncoder(ASTVisitor):
    '''
    Encode the following node features for each node type in the AST
    (default to generic encoding for any unspecified visit methods)

    Node features:
        - node type
        - data type (type sequence vector)
        - opcode
    '''
    NODE_KIND_STOP = len(NodeKinds.all_names())
    DTYPE_STOP = NODE_KIND_STOP + TypeEncoder.tensor_size()
    def __init__(self):
        super().__init__(warn_missing_visits=False, get_default_return_value=self.default_encoding)

    def _encode_node(self, node:ASTNode, data_type:DataType=None, opcode:str=''):
        kind_vec = NodeKinds.encode(node.kind)
        dtype_vec = TypeEncoder.encode(data_type) if data_type else TypeEncoder.empty_tensor()
        op_vec = Opcodes.encode(opcode)

        return torch.cat((kind_vec, dtype_vec.view(TypeEncoder.tensor_size()), op_vec))

    @property
    def typeseq_vec_len(self) -> int:
        single_type_len = len(self.typeseq.model_type_elements)
        return single_type_len*self.node_typeseq_len

    @staticmethod
    def split_node_vec(node_tensor:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Splits the encoded node vector into its three constituent vectors,
        returning them as a triple: (kind_vec, dtype_vec, opcode_vec)
        '''
        return node_tensor[...,:NodeEncoder.NODE_KIND_STOP], \
            node_tensor[...,NodeEncoder.NODE_KIND_STOP:NodeEncoder.DTYPE_STOP], \
            node_tensor[...,NodeEncoder.DTYPE_STOP:]

    def decode_node(self, node_tensor:torch.Tensor) -> Tuple[str, List[str], str]:
        '''
        Decode the provided tensor and return its node kind, type sequence, and opcode as a triple
        '''
        kind_vec, dtype_vec, op_vec = NodeEncoder.split_node_vec(node_tensor)
        return NodeKinds.decode(kind_vec), \
            TypeEncoder.decode(dtype_vec[None,:]), \
            Opcodes.decode(op_vec)

    def default_encoding(self, node:ASTNode):
        # just node kind
        return self._encode_node(node)

    def visit_BinaryOperator(self, binop:BinaryOperator):
        return self._encode_node(binop, opcode=binop.opcode)

    # NOTE: let the FunctionDecl return type
    # def visit_CallExpr(self, callexpr:CallExpr):
    #     return self._encode_node(callexpr, callexpr.inner[0].referencedDecl.return_dtype)

    def visit_CharacterLiteral(self, lit:CharacterLiteral):
        return self._encode_node(lit, lit.dtype)

    def visit_CStyleCastExpr(self, castexpr:CStyleCastExpr):
        return self._encode_node(castexpr, castexpr.dtype)

    def visit_DeclRefExpr(self, declref:DeclRefExpr):
        return self._encode_node(declref, declref.referencedDecl.dtype)

    def visit_FloatingLiteral(self, lit:FloatingLiteral):
        return self._encode_node(lit, lit.dtype)

    def visit_FunctionDecl(self, fdecl:FunctionDecl):
        return self._encode_node(fdecl, fdecl.return_dtype)

    def visit_IntegerLiteral(self, lit:IntegerLiteral):
        return self._encode_node(lit, lit.dtype)

    # TODO - later we should encode member dtype
    # def visit_MemberExpr(self, memexpr:MemberExpr):
    #     return self._encode_node(memexpr, memexpr.dtype)

    def visit_ParmVarDecl(self, pvdecl:ParmVarDecl):
        return self._encode_node(pvdecl, pvdecl.dtype)

    # TODO - later we should capture format string information
    # def visit_StringLiteral(self, lit:StringLiteral):
    #     return self._encode_node(lit, lit.dtype)

    def visit_UnaryOperator(self, unop:UnaryOperator):
        return self._encode_node(unop, opcode=unop.opcode)

    def visit_VarDecl(self, vdecl:VarDecl):
        return self._encode_node(vdecl, vdecl.dtype)

def encode_astnode(node:ASTNode, structural_model:bool=True) -> torch.Tensor:
    '''Encodes an ASTNode into a feature vector'''
    if structural_model:
        return NodeKinds.encode(node.kind)
    else:
        return NodeEncoder().visit(node)

def decode_astnode(encoded_node:torch.Tensor, structural_model:bool=True) -> 'str':
    '''Decodes a node into its kind string'''
    if structural_model:
        return NodeKinds.decode(encoded_node)
    else:
        return NodeEncoder().decode_node(encoded_node)

class ToFixedLengthTypeSeq(BaseTransform):
    '''
    According to this link, the normal NLP approach is just pad the sequence to max length
    for batching: https://github.com/pyg-team/pytorch_geometric/discussions/4226
    '''
    def __init__(self, fixed_len:int, include_comp:bool) -> None:
        super().__init__()
        self.fixed_len = fixed_len
        self.include_comp = include_comp
        self.typeseq = TypeSequence(include_comp)

    def forward(self, data: Any) -> Any:
        # blindly make data.y of fixed-length
        data.y = self.typeseq.to_fixed_len_tensor(data.y, self.fixed_len, batch_fmt=True)
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
        data.y = TypeSequence.dataset_to_batch_format(data.y)
        return data
