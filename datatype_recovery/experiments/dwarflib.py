from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection
from elftools.dwarf.dwarfinfo import DWARFInfo, CompileUnit
from elftools.dwarf.die import DIE
from elftools.dwarf.descriptions import ExprDumper

from pathlib import Path
from typing import Generator, Iterator, Set, List, Any, Dict

GHIDRA_ELF_IMAGE_BASE_DEFAULT_x64 = 0x100000

def ghidra_to_dwarf_addr(ghidra_addr:int):
    '''Adjust for Ghidra's default image base'''
    return ghidra_addr - GHIDRA_ELF_IMAGE_BASE_DEFAULT_x64

def dwarf_to_ghidra_addr(dwarf_addr:int):
    '''Adjust for Ghidra's default image base'''
    return dwarf_addr + GHIDRA_ELF_IMAGE_BASE_DEFAULT_x64

class StructMember:
    def __init__(self, name:str, dtype:str, offset:int, size:int) -> None:
        self.name = name
        self.dtype = dtype
        self.offset = offset
        self.size = size

# TODO: build a non-relocatable version...do DST symbols go away??

    def __str__(self) -> str:
        return f'[0x{self.offset:x}] {self.dtype} {self.name}'

    def __repr__(self) -> str:
        return str(self)

class StructLayout:
    def __init__(self, name:str, members:List[StructMember]) -> None:
        self.name = name
        self.members = members

    def __str__(self) -> str:
        sep = '\n '
        return f'STRUCT {self.name}\n {sep.join([str(m) for m in self.members])}'.strip()

    def __repr__(self) -> str:
        return str(self)


# Foo.b = property(lambda self: self.a + 1)
def die_property(attr_name:str, default_value:Any):
    return property(lambda x: x.attributes[attr_name].value if attr_name in x.attributes else default_value)

def get_namespaces(self:DIE, recursing:bool=False) -> List[str]:
    parent = self.get_parent()
    if parent:
        if parent.tag == 'DW_TAG_namespace':
            if recursing:
                return [*get_namespaces(parent, recursing=True), parent.name]
            return [*get_namespaces(parent, recursing=True), parent.name, self.name]
            # return f'{get_namespaces(parent)}::{self.name}'
    return []

def get_namespace_str(self:DIE) -> str:
    return "::".join(get_namespaces(self))

def get_type_die(self:DIE):
    type_die:DIE
    if 'DW_AT_type' not in self.attributes:
        return None
    return self.get_DIE_from_attribute('DW_AT_type')

def get_die_typename(self:DIE):
    type_die:DIE
    if 'DW_AT_type' not in self.attributes:
        return ''
    type_die = self.get_DIE_from_attribute('DW_AT_type')
    if type_die.tag == 'DW_TAG_base_type':
        if 'DW_AT_type' in type_die.attributes:
            raise Exception(f'Found nested DW_AT_type in base type {type_die.name}??')
        return type_die.name
    elif type_die.tag == 'DW_TAG_class_type':
        return f'class {get_namespace_str(type_die)}'
    elif type_die.tag == 'DW_TAG_structure_type':
        return f'struct {get_namespace_str(type_die)}'
    elif type_die.tag == 'DW_TAG_union_type':
        size = type_die.attributes['DW_AT_byte_size'].value
        return f'union {get_namespace_str(type_die)} (size = 0x{size:x})'
    elif type_die.tag == 'DW_TAG_array_type':
        if not type_die.has_children:
            raise Exception(f'Array type has no children? {type_die}')
        else:
            type_str = f'{get_die_typename(type_die)}'
            for child_die in type_die.iter_children():
                num_elements = child_die.attributes['DW_AT_count'].value
                type_str += f'[{num_elements}]'
            return type_str
    elif type_die.tag == 'DW_TAG_const_type':
        return f'const {get_die_typename(type_die)}'
    elif type_die.tag == 'DW_TAG_typedef':
        # return f'{type_die.name}'     # USE TYPEDEF'D NAME
        return f'{get_die_typename(type_die)}'    # USE BASE/ORIGINAL NAME
        # return f'typedef {get_die_typename(type_die)} {type_die.name}'  # TYPEDEF FORM
    elif type_die.tag == 'DW_TAG_pointer_type':
        if 'DW_AT_type' not in type_die.attributes:
            return f'void*'
        return f'{get_die_typename(type_die)}*'
    elif type_die.tag == 'DW_TAG_enumeration_type':
        return f'enum {get_namespace_str(type_die)}'      # ENUM NAME
        # return f'{get_die_typename(type_die)}'  # UNDERLYING TYPE
    elif type_die.tag == 'DW_TAG_reference_type':
        return f'{type_die.type}&'
    elif type_die.tag == 'DW_TAG_subroutine_type':
        params = [x.type for x in type_die.iter_children() if x.tag == 'DW_TAG_formal_parameter']
        for x in type_die.iter_children():
            if x.tag == 'DW_TAG_unspecified_parameters':
                raise Exception('DW_TAG_unspecified_parameters')
        rtype = type_die.type if 'DW_AT_type' in type_die.attributes else 'void'
        return f'{rtype} {type_die.name}({",".join(params)})'
        import IPython; IPython.embed()
    raise Exception(f'TODO: {type_die.tag}')
    # return f'TODO: {type_die.tag}'

def get_die_location(self:DIE):
    d = ExprDumper(self.dwarfinfo.structs)
    if 'DW_AT_location' in self.attributes:
        loc = self.attributes['DW_AT_location'].value
        return d.dump_expr(loc)
    return ''

def find_struct_tag(self:DIE) -> DIE:
    if self.tag == 'DW_TAG_structure_type' or self.tag == 'DW_TAG_class_type':
        return self
    elif self.tag == 'DW_TAG_base_type' or self.tag == 'DW_TAG_union_type':
        return None
    elif 'DW_AT_type' not in self.attributes:
        return None

    type_die:DIE
    type_die = self.get_DIE_from_attribute('DW_AT_type')
    return find_struct_tag(type_die)

def get_struct_layout(self:DIE):
    # we could have const or typedef or w/e, so find struct part first
    # type_die:DIE
    # type_die = self.get_DIE_from_attribute('DW_AT_type')
    stag:DIE = find_struct_tag(self)
    if not stag:
        return None

    members = []
    # layout_str = f'STRUCT: {stag.name}'
    for m in [x for x in stag.iter_children() if x.tag == 'DW_TAG_member']:
        m:DIE
        if 'DW_AT_data_member_location' in m.attributes:
            offset = m.attributes['DW_AT_data_member_location'].value
        else:
            offset = 0
        # layout_str += f'\n  [OFFSET 0x{offset:x}] {m.type} {m.name}'
        members.append(StructMember(m.name, m.type, offset, m.size))
    # return layout_str
    return StructLayout(stag.name, members)

# DIE.name = die_property('DW_AT_name', b'')
DIE.name = property(lambda x: x.attributes['DW_AT_name'].value.decode() if 'DW_AT_name' in x.attributes else '')
DIE.namebytes = die_property('DW_AT_name', b'')
# external => visible outside its compilation unit
DIE.external = die_property('DW_AT_external', False)
# DIE.location = die_property('DW_AT_location', None)
DIE.location = property(lambda x: get_die_location(x))
DIE.low_pc = die_property('DW_AT_low_pc', None)
DIE.high_pc = die_property('DW_AT_high_pc', None)
DIE.type_die = property(get_type_die)
DIE.type_name = property(get_die_typename)
DIE.struct_layout = property(get_struct_layout)

class DwarfDebugInfo:
    def __init__(self, dwarf:DWARFInfo) -> None:
        self.dwarf = dwarf
        self.funcdies_by_addr:Dict[int,DIE] = {}

        self._build_funcdies_by_addr()

    def _build_funcdies_by_addr(self):
        for fdie in self.get_function_dies():
            self.funcdies_by_addr[fdie.low_pc] = fdie

    def get_function_dies(self, cu_list:List[CompileUnit]=None) -> List[DIE]:
        '''
        If cu_list is None then all cu's will be included
        '''
        if not cu_list:
            cu_list = list(self.dwarf.iter_CUs())
        return [die for cu in cu_list for die in cu.get_top_DIE().iter_children() if die.tag == 'DW_TAG_subprogram']

    def get_function_locals(self, func:DIE):
        return list(self.extract_variables_from_die_tree(func))

    def extract_variables_from_die_tree(self, die:DIE) -> Generator[DIE,None,None]:
        '''
        Recursively extracts all DW_TAG_variable instances nested below
        this DIE in the tree
        '''
        for x in die.iter_children():
            x:DIE
            if x.tag == 'DW_TAG_variable':
                yield x
            elif x.has_children:
                yield from self.extract_variables_from_die_tree(x)
        # return [x for x in func.iter_children() if x.tag == 'DW_TAG_variable']

    def find_function(self, function_name:str) -> DIE:
        matches = [f for f in self.get_function_dies() if f.namebytes.decode() == function_name]
        return matches[0] if matches else None

def init_pyelftools_from_dwarf(dwarf:DWARFInfo):
    '''
    Call this first to work aroudn a pyelftools bug

    CLS: this is a bug in pyelftools - the _MACHINE_ARCH is unset
    when we dump an expression (really this shouldn't be global at all)
    -> quick fix: just set this manually to match current DWARF info
    before dumping anything
    '''
    from elftools.dwarf.descriptions import _MACHINE_ARCH
    from elftools.dwarf.descriptions import set_global_machine_arch
    set_global_machine_arch(dwarf.config.machine_arch)

def old_main(ris):
    with open(ris, 'rb') as f:
        ef = ELFFile(f)
        dwarf = ef.get_dwarf_info()



        ddi = DwarfDebugInfo(dwarf)
        # ddi.get_functions()

        # TODO: start getting CU's (compilation units) and iterating through
        # their debug information entries (DIEs) to find the data we need
        # (refer to the DWARF standard,
        #  see https://developer.ibm.com/articles/au-dwarf-debug-format/)

        cu = list(dwarf.iter_CUs())[0]
        die = cu.get_top_DIE()

        ft = ddi.find_function('FileText')
        ft_vars = ddi.get_function_locals(ft)
        v0 = ft_vars[0]

        d = ExprDumper(dwarf.structs)
        # d.dump_expr(v0.location)
        # print(v0.location)
        # loc = d.expr_parser.parse_expr(v0.location)

        # for i, v in enumerate(ft_vars):
        #     print(f'Local var {i+1}: {v.namebytes.decode()}, location={v.location}')

        # fc = ddi.find_function('FutureCheck')
        # fcvars = ddi.get_function_locals(fc)

        # print(f'Variables for function {fc.name}:')
        # for x in fcvars:
        #     print(f'  {x.type} {x.name}\t@ {x.location}')

        named_funcs = sorted(set([f.name for f in ddi.get_function_dies() if f.name]))

        # take the first function:
        fdie = ddi.find_function(named_funcs[0])
        f_locals = ddi.get_function_locals(fdie)
        locations = [x.location for x in f_locals]