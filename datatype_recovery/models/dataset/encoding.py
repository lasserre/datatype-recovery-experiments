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
        all_classes = []
        prev_nts = []

        for level_idx in range(seq_len):
            current_level = [[x] for x in self.logical_type_elements]
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

def get_num_node_features(structural_model:bool=True, include_component:bool=False, type_seq_len:int=3):
    if structural_model:
        return len(NodeKinds.all_names())     # only node type
    else:
        # Node features:
        # - node type
        # - data type (type sequence vector)
        # - opcode
        node_type_features = len(NodeKinds.all_names())
        num_type_elems = len(TypeSequence(include_component).model_type_elements)
        data_type_features = num_type_elems * type_seq_len
        opcode_features = len(Opcodes.all_opcodes())
        return node_type_features + data_type_features + opcode_features

def get_num_model_type_elements(include_component:bool) -> int:
    return len(TypeSequence(include_component).model_type_elements)

class NodeEncoder(ASTVisitor):
    '''
    Encode the following node features for each node type in the AST
    (default to generic encoding for any unspecified visit methods)

    Node features:
        - node type
        - data type (type sequence vector)
        - opcode
    '''
    def __init__(self, node_typeseq_len:int):
        super().__init__(warn_missing_visits=False, get_default_return_value=self.default_encoding)
        self.node_typeseq_len = node_typeseq_len

        # we don't encode COMP into node type sequences (doesn't make sense here, there are no COMP types in AST)
        self.typeseq = TypeSequence(include_comp=False)

    def _encode_node(self, node:ASTNode, data_type:DataType=None, opcode:str=''):
        type_seq = data_type.type_sequence_str.split(',') if data_type else ['<EMPTY>']
        return self._encode_node_with_seq(node, type_seq, opcode)

    def _encode_node_with_seq(self, node:ASTNode, type_seq:List[str], opcode:str=''):
        kind_vec = NodeKinds.encode(node.kind)

        if len(type_seq) < self.node_typeseq_len:
            type_seq.extend(['<EMPTY>']*(self.node_typeseq_len-len(type_seq)))
        type_seq = type_seq[:self.node_typeseq_len]     # truncate at fixed length


        # encode ts_vec as 1-d vector
        ts_vec = self.typeseq.encode(type_seq, batch_fmt=False).view((self.typeseq_vec_len))
        op_vec = Opcodes.encode(opcode)

        return torch.cat((kind_vec, ts_vec, op_vec))

    @property
    def typeseq_vec_len(self) -> int:
        single_type_len = len(self.typeseq.model_type_elements)
        return single_type_len*self.node_typeseq_len

    def decode_node(self, node_tensor:torch.Tensor) -> Tuple[str, List[str], str]:
        '''
        Decode the provided tensor and return its node kind, type sequence, and opcode as a triple
        '''
        kind_stop = len(NodeKinds.all_names())
        ts_stop = kind_stop + self.typeseq_vec_len
        kind_vec = node_tensor[:kind_stop]
        ts_vec = node_tensor[kind_stop:ts_stop]
        op_vec = node_tensor[ts_stop:]
        unflatten_shape = (self.node_typeseq_len, len(self.typeseq.model_type_elements))
        return NodeKinds.decode(kind_vec), \
            self.typeseq.decode(ts_vec.view(unflatten_shape), batch_fmt=False), \
            Opcodes.decode(op_vec)

    def default_encoding(self, node:ASTNode):
        # just node kind
        return self._encode_node_with_seq(node, ['<EMPTY>'], '')

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

def encode_astnode(node:ASTNode, structural_model:bool=True, node_typeseq_len:int=3) -> torch.Tensor:
    '''Encodes an ASTNode into a feature vector'''
    if structural_model:
        return NodeKinds.encode(node.kind)
    else:
        return NodeEncoder(node_typeseq_len).visit(node)

def decode_astnode(encoded_node:torch.Tensor, structural_model:bool=True, node_typeseq_len:int=3) -> 'str':
    '''Decodes a node into its kind string'''
    if structural_model:
        return NodeKinds.decode(encoded_node)
    else:
        return NodeEncoder(node_typeseq_len).decode_node(encoded_node)

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
