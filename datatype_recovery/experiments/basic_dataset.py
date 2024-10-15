from pathlib import Path
import hashlib
import json
import pandas as pd
from rich.console import Console
import shutil
from typing import List, Tuple

from wildebeest import Experiment, RunConfig, ProjectRecipe
from wildebeest import DockerBuildAlgorithm, DefaultBuildAlgorithm
from wildebeest.postprocessing import find_binaries, flatten_binaries, strip_binaries
from wildebeest.postprocessing import ghidra_import
from wildebeest.preprocessing.ghidra import start_ghidra_server, create_ghidra_repo
from wildebeest.preprocessing.cc_wrapper import install_cc_wrapper
from wildebeest import *
from wildebeest.run import Run
from wildebeest.utils import print_runtime, show_progress

from wildebeest.postprocessing.flatlayoutbinary import FlatLayoutBinary

import astlib
from astlib import build_var_ast_signature, read_json
from varlib.location import Location
from varlib.datatype import *
import dwarflib
from dwarflib import *
from ghidralib import export_asts

_use_tqdm = False

def render_ast(ast, ast_name:str, outfolder:Path, format:str='svg', highlight_kind:str=None, highlight_color:str='red'):
    '''
    Render's the AST
    '''
    def do_highlight_kind(node, attrs):
        if node.kind == highlight_kind:
            attrs.font_color = highlight_color
        elif hasattr(node, 'IS_MEMBER_ACCESS'):
            attrs.font_color = 'blue'
    ast.render(format=format, ast_name=ast_name, outfolder=outfolder, format_node=do_highlight_kind)

def separate_ast_fails(ast_paths:set) -> set: #-> Tuple[set, set]:
    '''Checks for AST export failures (based on .log file presence) among the
    given set of AST export Paths, and partitions the set into failures and no failures.

    Returns a set of AST paths that failed to export, removing this set from the ast_paths
    supplied as an argument.
    '''
    fails = set([x for x in ast_paths if read_json(x).logfile])
    ast_paths -= fails
    return fails

def extract_ast_nodes(kind:str, root:astlib.ASTNode) -> List[dict]:
    matches = []
    nodes_to_check = [root]

    while nodes_to_check:
        next_node = nodes_to_check.pop()
        if next_node.kind == kind:
            # if next_node.parent.kind == kind:
            #     next_node.
            matches.append(next_node)
        # if 'inner' in next_node:
        nodes_to_check.extend(next_node.inner)

    return matches

def find_member_offset_matches(intliterals:List[astlib.ASTNode], effective_offset:int):
    # matches = [l for l in intliterals if l.value == offset]
    matches = []
    for lit in intliterals:
        # check our assertions about the structure of member offsets
        try:
            assert lit.parent.kind == 'BinaryOperator'
            assert lit.parent.opcode == '+'
            declrefs = [x for x in lit.parent.inner if x.kind == 'DeclRefExpr']
            assert len(declrefs) == 1
        except AssertionError:
            continue    # not a match
        vdecl = declrefs[0].referencedDecl
        if vdecl.dtype.kind == 'PointerType':
            elem_size = vdecl.dtype.inner[0].size
            if lit.value*elem_size == effective_offset:
                # match!
                matches.append(lit)
        else:
            # not a pointer type - just check it
            if lit.value == effective_offset:
                matches.append(lit)
    return matches

# I need my own Location and Type classes for doing these experiments that work
# across DWARF, my AST, and any other source of information...a common set of
# classes we can use to talk about variable locations and data types
# ---
# Implementing this will allow us to:
#   -> compare locations and types across DWARF and the AST
#   -> implement the comparison in various ways (exact match, name, etc)
#   -> calculate % match, structural match metric, etc.

# so this is kind of a "third" typelib or something that I can make my dwarflib
# and astlib convert their types TO
#
# OPTION 1: just use the AST's way of defining this
# OPTION 2: define it separately
# OPTION 3: hybrid
#
# I like the hybrid idea...make the information match the AST defs as much as
# possible, but define the classes STATICALLY so that I can define additional
# properties/methods on the objects and make intellisense better/easier
#
# *** also! in my astlib, whenever I am generating the tree, whenever I have a
# Type ASTNode instance I can either create the Type class right there or MAKE
# THAT NODE a Type object instead of the "NewClass" thing

class DwarfTables:
    def __init__(self):
        self.locals_df = None
        self.funcs_df = None
        self.params_df = None

def build_dwarf_data_tables(debug_binary_file:Path, get_location:bool=False) -> DwarfTables:
    '''
    Build the DWARF local variables table for the given debug binary

    DWARF Function Addr | Var Name | Location | Type | Type Category

    debug_binary_file: The path to the DWARF file (typically a debug binary build)
    '''
    # pull in debug info for this binary
    ddi = DwarfDebugInfo.fromElf(debug_binary_file)
    sdb = StructDatabase()

    with dwarflib.UseStructDatabase(sdb):
        tables = build_dwarf_data_tables_from_ddi(ddi, get_location)

    print(f'Serializing DWARF sdb...')
    sdb.to_json(debug_binary_file.with_suffix('.dwarf.sdb'))
    return tables

def build_dwarf_data_tables_from_ddi(ddi:DwarfDebugInfo, get_location:bool=False) -> DwarfTables:
    # NOTE: I can't think of a particularly good reason for doing this one way over
    # another, so arbitrarily deciding to use Ghidra addresses for consistency
    # (if nothing else, I will be looking at Ghidra much more often than DWARF data)

    locals_dfs = []
    func_names = []
    func_starts = []
    params_df_list = []

    for dwarf_addr, fdie in show_progress(ddi.funcdies_by_addr.items(), total=len(ddi.funcdies_by_addr)):
        if fdie.artificial:
            # print(f'Skipping artificial function {fdie.name} (intrinsic?)')
            continue
        if fdie.inline:
            # print(f'Skipping inlined function {fdie.name}')
            continue

        # print(f'Extracting DWARF from {fdie.name}',flush=True)

        ### Locals
        locals = ddi.get_function_locals(fdie)

        if get_location:
            locations = [l.location_varlib for l in locals]
        else:
            # NOTE: we aren't matching vars up by location anymore - don't even pull them,
            # right now it causes issues when we see location lists and we don't need to
            # spend time to fix this if we don't use it
            # --> use Undefined location as placeholder
            locations = [Location(LocationType.Undefined) for l in locals]

        df = pd.DataFrame({
            'Name': [l.name for l in locals],
            'Type': [l.dtype_varlib for l in locals],
            'LocType': pd.array([l.loc_type for l in locations], dtype=pd.StringDtype()),
            'LocRegName': pd.array([l.reg_name for l in locations], dtype=pd.StringDtype()),
            'LocOffset': pd.array([l.offset for l in locations], dtype=pd.Int64Dtype()),
        })

        df['FunctionStart'] = dwarf_to_ghidra_addr(dwarf_addr)
        df['FunctionName'] = fdie.name
        df['TypeCategory'] = [t.category for t in df.Type]
        df['TypeSeq'] = [t.type_sequence_str for t in df.Type]

        locals_dfs.append(df)

        ### Functions
        func_names.append(fdie.name)
        func_starts.append(dwarf_to_ghidra_addr(dwarf_addr))

        ### Function prototype
        params = list(ddi.get_function_params(fdie))

        #############################################################################################
        # ASSUMPTION: Negative stack offsets for x86 never make sense for parameters. If we see this,
        #             it means GCC is spilling parameters to the stack frame and DWARF is reporting
        #             this location.
        #             --> If we see this, simply correct it by applying the assumed System V x64 ABI
        #                 calling convention
        #############################################################################################

        # NOTE: same as above
        # param_locs = [p.location_varlib for p in params]
        # --> use Undefined location as placeholder
        param_locs = [Location(LocationType.Undefined) for p in params]

        # no DW_AT_type attribute indicates return type is void
        rtype = fdie.dtype_varlib if fdie.type_die else BuiltinType.create_void_type()

        params_df = pd.DataFrame({
            'FunctionStart': pd.array([dwarf_to_ghidra_addr(dwarf_addr)] * len(params), dtype=pd.UInt64Dtype()),
            'Name': [p.name for p in params],
            'IsReturnType': pd.array([False] * len(params), dtype=pd.BooleanDtype()),
            'Type': [p.dtype_varlib for p in params],
            'LocType': pd.array([l.loc_type for l in param_locs], dtype=pd.StringDtype()),
            'LocRegName': pd.array([l.reg_name for l in param_locs], dtype=pd.StringDtype()),
            'LocOffset': pd.array([l.offset for l in param_locs], dtype=pd.Int64Dtype()),
        })

        params_df = pd.concat([params_df, pd.DataFrame({
            'FunctionStart': pd.array([dwarf_to_ghidra_addr(dwarf_addr)], dtype=pd.UInt64Dtype()),
            'Name': [None],
            'IsReturnType': pd.array([True], dtype=pd.BooleanDtype()),
            'Type': [rtype],
            # apparently there is currently no way to get the location of a return value in DWARF
            # outstanding proposal to add this here: https://dwarfstd.org/issues/221105.1.html
            'LocType': pd.array([None], dtype=pd.StringDtype()),
            'LocRegName': pd.array([None], dtype=pd.StringDtype()),
            'LocOffset': pd.array([None], dtype=pd.Int64Dtype()),
        })], ignore_index=True)

        params_df['TypeCategory'] = [t.category for t in params_df.Type]
        params_df['TypeSeq'] = [t.type_sequence_str for t in params_df.Type]
        params_df_list.append(params_df)

    tables = DwarfTables()
    tables.funcs_df = pd.DataFrame({
        'FunctionStart': func_starts,
        'FunctionName': func_names
    })
    tables.locals_df = pd.concat(locals_dfs).reset_index(drop=True)
    tables.params_df = pd.concat(params_df_list).reset_index(drop=True)

    return tables

class FunctionData:
    '''
    To hopefully streamline processing of the data, I'm packaging up function-related
    data that can be bundled together so the AST files can be fully processed while
    they are open (and this way we only have to parse 1x into the AST python class
    structure, which is slow)
    '''
    def __init__(self) -> None:
        self.name:str = ''
        self.address:int = -1
        self.ast_json_filename = None
        self.func_df:pd.DataFrame = None    # only 1 row expected, but ready to pd.concat()
        self.params_df:pd.DataFrame = None
        self.locals_df:pd.DataFrame = None
        self.globals_accessed_df:pd.DataFrame = None

def extract_funcdata_from_ast(ast:astlib.ASTNode, ast_json:Path) -> FunctionData:

    fdecl = ast.fdecl

    fd = FunctionData()
    fd.ast_json_filename = ast_json
    fd.name = fdecl.name
    fd.address = fdecl.address
    fd.locals_df = build_ast_locals_table(fdecl, fdecl.local_vars)
    fd.params_df = build_ast_func_params_table(fdecl, fdecl.params, fdecl.return_dtype)

    # TODO globals?

    return fd

def build_ast_func_params_table(fdecl:astlib.ASTNode, params:List[astlib.ASTNode], return_type:DataType):
    '''
    Build the function parameters table for the given AST function
    '''
    df = pd.DataFrame({
        'FunctionStart': pd.array([fdecl.address] * len(params), dtype=pd.UInt64Dtype()),
        'Name': [p.name for p in params],
        'Signature': [build_var_ast_signature(fdecl, p.name) for p in params],
        'IsReturnType': pd.array([False] * len(params), dtype=pd.BooleanDtype()),
        'Type': [p.dtype for p in params],
        'LocType': pd.array([p.location.loc_type if p.location else None for p in params], dtype=pd.StringDtype()),
        'LocRegName': pd.array([p.location.reg_name if p.location else None for p in params], dtype=pd.StringDtype()),
        'LocOffset': pd.array([p.location.offset if p.location else None for p in params],
                              dtype=pd.Int64Dtype()),
    })

    # add return type
    df = pd.concat([df, pd.DataFrame({
        'FunctionStart': pd.array([fdecl.address], dtype=pd.UInt64Dtype()),
        'Name': [None],
        # NOTE: special signature of "-1" for return types
        # - populate with a string to avoid getting dropped as an empty signature
        # - we don't really need a "signature" for return type - there is one and only
        #   one (and our signature code looks for references to named variables but
        #        this is not a named variable)
        'Signature': ['-1'],
        'IsReturnType': pd.array([True], dtype=pd.BooleanDtype()),
        'Type': [return_type],
        'LocType': pd.array([None], dtype=pd.StringDtype()),
        'LocRegName': pd.array([None], dtype=pd.StringDtype()),
        'LocOffset': pd.array([None], dtype=pd.Int64Dtype()),
    })], ignore_index=True)

    # fix anything Ghidra output as a FUNC type to be PTR->FUNC (eventually we need to rerun with updated AST exporter)
    df['Type'] = df.Type.apply(lambda x: PointerType(x, pointer_size=8) if isinstance(x, FunctionType) else x)

    df['TypeCategory'] = [t.category for t in df.Type]
    df['TypeSeq'] = [t.type_sequence_str for t in df.Type]

    return df

def build_ast_locals_table(fdecl:astlib.FunctionDecl, local_vars:List[astlib.ASTNode]):
    '''
    Build the local variables table for the given AST

    Ghidra Function Addr | Var Name? | Location | Type | Type Category
    '''
    fbody = fdecl.func_body

    if fbody.kind != 'CompoundStmt':
        # no function body -> no locals
        return pd.DataFrame()

    if not local_vars:
        return pd.DataFrame()   # no locals

    # consider leaving these as objects in the table...? I may break it out into columns but
    # for now I can access the objects! so this may still allow some utility VERY easily
    df = pd.DataFrame({
        'FunctionStart': [fdecl.address] * len(local_vars),
        'Name': [v.name for v in local_vars],
        'Signature': [build_var_ast_signature(fdecl, lv.name) for lv in local_vars],
        'Type': [v.dtype for v in local_vars],
        # 'Location': [v.location for v in local_vars]
        'LocType': pd.array([v.location.loc_type if v.location else None for v in local_vars], dtype=pd.StringDtype()),
        'LocRegName': pd.array([v.location.reg_name if v.location else None for v in local_vars], dtype=pd.StringDtype()),
        'LocOffset': pd.array([v.location.offset if v.location else None for v in local_vars],
                              dtype=pd.Int64Dtype()),
    })

    # fix anything Ghidra output as a FUNC type to be PTR->FUNC (eventually we need to rerun with updated AST exporter)
    df['Type'] = df.Type.apply(lambda x: PointerType(x, pointer_size=8) if isinstance(x, FunctionType) else x)

    df['TypeCategory'] = [t.category for t in df.Type]
    df['TypeSeq'] = [t.type_sequence_str for t in df.Type]

    return df

def read_hex_lines_from_file(filepath:Path) -> List[int]:
    '''
    Parse out a hex number from each line of the given file if it exists.
    If the file does not exist, an empty list is returned
    '''
    if filepath.exists():
        with open(filepath, 'r') as f:
            return [int(l.strip(), 16) for l in f.readlines()]
    return []

def read_failed_decomps_and_asts(ast_folder:Path) -> Tuple[list, list]:
    '''
    Returns a tuple of lists: (failed_decomps, failed_asts)
    where each list contains function addresses that failed to decompile or export
    the function ast, respectively.
    '''
    failed_decomp_file = list(ast_folder.glob('*failed_decompilations.txt'))
    failed_ast_file = list(ast_folder.glob('*failed_ast_exports.txt'))

    failed_decomps = read_hex_lines_from_file(failed_decomp_file[0]) if failed_decomp_file else []
    failed_asts = read_hex_lines_from_file(failed_ast_file[0]) if failed_ast_file else []

    return (failed_decomps, failed_asts)

def collect_exported_asts(fb:FlatLayoutBinary):
    '''Collect the debug and stripped ASTs that did not have failures'''
    debug_asts = fb.data['debug_asts']
    stripped_asts = fb.data['stripped_asts']

    # collect functions for this binary, partitioned into debug/stripped sets
    stripped_funcs = set(stripped_asts.glob('*.json'))
    debug_funcs = set(debug_asts.glob('*.json'))

    # log the # of functions that had errors (decompilation or ast export)
    failed_debug_decomps, failed_debug_asts = read_failed_decomps_and_asts(debug_asts)
    failed_stripped_decomps, failed_stripped_asts = read_failed_decomps_and_asts(stripped_asts)

    if failed_stripped_decomps:
        print(f'# stripped decomp fails = {len(failed_stripped_decomps)}')
    if failed_debug_decomps:
        print(f'# debug decomp fails = {len(failed_debug_decomps)}')

    if failed_stripped_asts:
        print(f'# stripped AST export fails = {len(failed_stripped_asts)}')
    if failed_debug_asts:
        print(f'# debug AST export fails = {len(failed_debug_asts)}')

    return (debug_funcs, stripped_funcs)

def extract_funcdata_from_ast_set(ast_funcs:Set[Path], bin_path:Path, is_debug:bool) -> List[FunctionData]:
    '''Extracts FunctionData content from each of the provided ASTs'''
    fdatas:List[FunctionData] = []
    num_funcs = len(ast_funcs)

    # NOTE: Struct types not included anymore in AST files - we will pull this 1x
    # from Ghidra using new DataTypeArchive export code (TODO...)
    # sdb = StructDatabase()
    # with astlib.UseStructDatabase(sdb):

    json_recursion_errors = []  # list of files that errored out in json load due to recursion depth

    for i, ast_json in show_progress(enumerate(sorted(ast_funcs)), total=len(ast_funcs)):
        try:
            ast = astlib.read_json(ast_json)
        except astlib.JsonRecursionError:
            json_recursion_errors.append(ast_json)
            continue

        fdatas.append(extract_funcdata_from_ast(ast, ast_json))

    if json_recursion_errors:
        console = Console()
        console.print(f'[bold yellow]{len(json_recursion_errors):,} functions dropped due to JSON RecursionErrors')
        for fpath in json_recursion_errors:
            console.print(fpath.name)

    # suffix = '.debug.sdb' if is_debug else '.sdb'
    # sdb.to_json(bin_path.with_suffix(suffix))

    return fdatas

def extract_data_tables(fb:FlatLayoutBinary):
    '''
    Build our master table of local variables
    '''
    # Function | Binary |
    # ... DWARF local var | Stripped AST local var | Debug AST local var | Stripped Function AST
    console = Console()
    if not DwarfDebugInfo.is_PIE_exe_or_sharedobj(fb.debug_binary_file):
        console.print(f'Binary file {fb.debug_binary_file} is not a PIE executable or shared object', style='yellow')
        console.print(f'Skipping {fb.debug_binary_file}', style='yellow')
        return

    debug_funcs, stripped_funcs = collect_exported_asts(fb)

    print(f'Extracting DWARF data for binary {fb.debug_binary_file.name}...')
    dwarf_tables = build_dwarf_data_tables(fb.debug_binary_file)

    # extract data from ASTs/DWARF debug symbols
    print(f'Extracting data from {len(debug_funcs):,} debug ASTs for binary {fb.debug_binary_file.name}...')
    debug_funcdata = extract_funcdata_from_ast_set(debug_funcs, fb.debug_binary_file, is_debug=True)

    print(f'Extracting data from {len(stripped_funcs):,} stripped ASTs for binary {fb.binary_file.name}...')
    stripped_funcdata = extract_funcdata_from_ast_set(stripped_funcs, fb.binary_file, is_debug=False)

    ### Functions
    funcs_df = build_funcs_table(debug_funcdata, stripped_funcdata, dwarf_tables.funcs_df)
    funcs_df.loc[:,'BinaryId'] = fb.id
    funcs_df.to_csv(fb.data_folder/'functions.csv', index=False)

    ### Locals
    locals_df = build_locals_table(funcs_df, debug_funcdata, stripped_funcdata,
                                                    dwarf_tables.locals_df, fb.data_folder)

    if not locals_df.empty:
        locals_df.loc[:,'BinaryId'] = fb.id
    locals_df.to_csv(fb.data_folder/'locals.csv', index=False)

    ### Function Parameters (prototype)
    params_df = build_params_table(funcs_df, debug_funcdata, stripped_funcdata, dwarf_tables.params_df)
    if not params_df.empty:
        params_df.loc[:,'BinaryId'] = fb.id
    params_df.to_csv(fb.data_folder/'function_params.csv', index=False)

def build_params_table(funcs_df:pd.DataFrame, debug_funcdata:List[FunctionData], strip_funcdata:List[FunctionData],
                      dwarf_df:pd.DataFrame):
    debug_df = pd.concat(fd.params_df for fd in debug_funcdata if fd.address in funcs_df.FunctionStart.values)
    strip_df = pd.concat(fd.params_df for fd in strip_funcdata if fd.address in funcs_df.FunctionStart.values)

    # divide params/return types, combine separately, then recombine
    # -> this allows us to NOT drop debug return types because they don't
    #    (and can't!) align with DWARF return types by name...bc there is no name
    debug_rtypes = debug_df.loc[debug_df.IsReturnType,:]
    debug_params = debug_df.loc[~debug_df.IsReturnType,:]
    strip_rtypes = strip_df.loc[strip_df.IsReturnType,:]
    strip_params = strip_df.loc[~strip_df.IsReturnType,:]
    dwarf_rtypes = dwarf_df.loc[dwarf_df.IsReturnType,:]
    dwarf_params = dwarf_df.loc[~dwarf_df.IsReturnType,:]

    console = Console()
    console.rule(f'[yellow] Processing params/return types...', align='left', style='grey', characters='.')
    params_df = build_var_table_by_signatures(debug_params, strip_params, dwarf_params, fill_comp=False)
    rtypes_df = build_var_table_by_signatures(debug_rtypes, strip_rtypes, dwarf_rtypes, fill_comp=False)

    # recombine params/rtypes
    return pd.concat([params_df, rtypes_df]).reset_index(drop=True)

def build_funcs_table(debug_funcdata:List[FunctionData], strip_funcdata:List[FunctionData],
                      dwarf_funcs:pd.DataFrame):

    # why build a table of functions when we have function info in locals table, etc?
    # -> the AST functions are limited to functions that have successfully decompiled
    # -> the DWARF functions are limited to "non-artificial" (e.g. non intrinsic) functions
    # -> if Ghidra happens to miss some functions, we don't want to assume they are present
    # -> I think we are excluding external functions in AST extraction also

    debug_df = pd.DataFrame({
        'FunctionStart': [f.address for f in debug_funcdata],
        'FunctionName': [f.name for f in debug_funcdata],
        'AstJson_Debug': [f.ast_json_filename for f in debug_funcdata],
    })

    strip_df = pd.DataFrame({
        'FunctionStart': [f.address for f in strip_funcdata],
        'FunctionName': [f.name for f in strip_funcdata],
        'AstJson_Strip': [f.ast_json_filename for f in strip_funcdata],
    })

    df = debug_df.merge(strip_df, on='FunctionStart', how='outer', suffixes=['_Debug','_Strip'])
    df = df.merge(dwarf_funcs, on='FunctionStart', how='outer', suffixes=[None, '_DWARF'])
    df.rename(columns={'FunctionName': 'FunctionName_DWARF'}, inplace=True)
    df = df.reset_index(drop=True)

    # drop functions that do not appear in DWARF
    # (these are few, but sometimes completely invalid - i.e. not code)
    num_nondwarf_funcs = len(df[df.FunctionName_DWARF.isna()])
    print(f'Dropped {num_nondwarf_funcs:,} functions that do not appear in DWARF')
    df = df.loc[~df.FunctionName_DWARF.isna(), :]

    return df.reset_index(drop=True)

def drop_duplicate_vars(df:pd.DataFrame) -> pd.DataFrame:
    '''
    Drop all occurrences of variables that have duplicate signatures (within the same function)
    '''
    # can't just drop_duplicates() since I don't want ANY rows left over that had a duplicate
    varcounts = df.groupby(['FunctionStart', 'Signature']).count()
    dupvar_idx = varcounts[varcounts.Name>1].index
    return df.set_index(['FunctionStart','Signature']).drop(index=dupvar_idx).reset_index()

def build_locals_table(funcs_df:pd.DataFrame, debug_funcdata:List[FunctionData], stripped_funcdata:List[FunctionData],
                       dwarf_locals:pd.DataFrame, data_folder:Path) -> pd.DataFrame:
    ## Locals table

    # combine into single df
    debug_locals = pd.concat([fd.locals_df for fd in debug_funcdata if fd.address in funcs_df.FunctionStart.values]).reset_index(drop=True)
    stripped_locals = pd.concat([fd.locals_df for fd in stripped_funcdata if fd.address in funcs_df.FunctionStart.values]).reset_index(drop=True)

    # TEMP: save off raw dataframes so I can figure out why nothing lined up...lol
    dwarf_locals.to_csv(data_folder/'_raw_dwarf_locals.csv', index=False)
    debug_locals.to_csv(data_folder/'_raw_debug_locals.csv', index=False)
    stripped_locals.to_csv(data_folder/'_raw_stripped_locals.csv', index=False)

    console = Console()
    console.rule(f'[yellow] Processing local vars...', align='left', style='grey', characters='.')
    return build_var_table_by_signatures(debug_locals, stripped_locals, dwarf_locals)

def build_var_table_by_signatures(debug_vars:pd.DataFrame, stripped_vars:pd.DataFrame,
        dwarf_vars:pd.DataFrame, fill_comp:bool=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Create a variable table using AST signatures to align debug/stripped AST variables
    and labeling debug variables with HasDWARF if they align by name
    '''
    DWARF_IDX_COLS = ['FunctionStart', 'Name']  # no var signature for dwarf, use var name
    AST_IDX_COLS = ['FunctionStart', 'Signature']

    #######################################################
    # drop all stripped functions that don't ever occur in debug

    sv = stripped_vars  # alias to make the next line shorter...lol
    if sv.empty:
        return pd.DataFrame()   # no variables

    extra_sv_funcs = sv[~sv.FunctionStart.isin(debug_vars.FunctionStart)]
    stripped_vars = sv.drop(index=extra_sv_funcs.index).reset_index(drop=True)
    print(f'Dropping stripped vars from {len(extra_sv_funcs):,} functions that don\'t appear in debug vars')

    #######################################################
    # drop empty and duplicate signatures before merge

    num_empty_debug = len(debug_vars[debug_vars.Signature==''])
    num_empty_stripped = len(stripped_vars[stripped_vars.Signature==''])

    print(f'Dropped {num_empty_debug:,} debug vars with empty signatures (no refs)')
    print(f'Dropped {num_empty_stripped:,} stripped vars with empty signatures (no refs)')

    debug_df = debug_vars.loc[debug_vars.Signature!='', :]
    stripped_df = stripped_vars.loc[stripped_vars.Signature!='', :]

    dcount = debug_df.groupby(AST_IDX_COLS).count()
    scount = stripped_df.groupby(AST_IDX_COLS).count()
    num_dup_debug = dcount[dcount.TypeCategory>1].TypeCategory.sum()
    num_dup_stripped = scount[scount.TypeCategory>1].TypeCategory.sum()

    print(f'Dropped {num_dup_debug:,} duplicate debug vars')
    print(f'Dropped {num_dup_stripped:,} duplicate stripped vars')

    debug_df = drop_duplicate_vars(debug_df)
    stripped_df = drop_duplicate_vars(stripped_df)

    #######################################################
    # Compute extra cols for debug_df

    # Label vars that align with DWARF by name
    ddf = debug_df.set_index(DWARF_IDX_COLS)
    wdf = dwarf_vars.set_index(DWARF_IDX_COLS)
    ddf['HasDWARF'] = ddf.index.isin(wdf.index)
    debug_df = ddf.reset_index()

    # dump debug types to json so we can reconstruct from CSV (+ sdb for structs)
    debug_df['TypeJson_Debug'] = debug_df.Type.apply(lambda dt: dt.to_json())

    #######################################################
    # Align debug/stripped vars

    # align our stripped variables with debug vars based on (FunctionStart, Signature)
    df = stripped_df.merge(debug_df, how='left', on=AST_IDX_COLS, suffixes=['_Strip','_Debug'])

    if fill_comp:
        # mark stripped vars that do not align with debug as <Component> (like DIRTY)
        df['TypeSeq_Debug'] = df.TypeSeq_Debug.fillna('COMP')
        df['TypeCategory_Debug'] = df.TypeCategory_Debug.fillna('COMP')
        df['HasDWARF'] = df.HasDWARF.fillna(False)

    # drop the remaining NaNs - since we've filled nans with COMP for locals, these correspond to
    # stripped params that don't align
    print(f'Dropping {len(df[df.TypeSeq_Debug.isna()]):,} stripped vars that don\'t align with debug')
    df = df.drop(index=df[df.TypeSeq_Debug.isna()].index).reset_index(drop=True)

    # fill in leaf/ptr levels encoding
    df['LeafCategory'] = df.Type_Debug.apply(lambda x: x.leaf_type.category if not isinstance(x, float) else 'COMP')
    df['LeafSigned'] = df.Type_Debug.apply(lambda x: x.leaf_type.is_signed if not isinstance(x, float) else False)
    df['LeafFloating'] = df.Type_Debug.apply(lambda x: x.leaf_type.is_floating if not isinstance(x, float) else False)
    df['LeafSize'] = df.Type_Debug.apply(lambda x: x.leaf_type.primitive_size if not isinstance(x, float) else 0)
    df['PtrLevels'] = df.Type_Debug.apply(lambda x: ''.join(x.ptr_hierarchy(3)) if not isinstance(x, float) else 'LLL')
    df['PtrL1'] = df.PtrLevels.apply(lambda x: x[0])
    df['PtrL2'] = df.PtrLevels.apply(lambda x: x[1])
    df['PtrL3'] = df.PtrLevels.apply(lambda x: x[2])

    return df

def combine_fb_tables_into_rundata(run:Run, bin_list:List[FlatLayoutBinary], csv_name:str):
    '''
    Read each of the pandas tables (in file csv_name) from the list of flat binary folders
    and combine them into a single run-level data frame, writing it to the run data folder
    '''
    csv_names = [fb.data_folder/csv_name for fb in bin_list]
    df_list = [pd.read_csv(csv) for csv in csv_names if csv.exists() and os.path.getsize(csv) > 25]    # even the header line is 261 characters long
    # if we pd.concat() with an empty dataframe it messes up the column data types
    df_list = [df for df in df_list if not df.empty]
    combined_df = pd.concat(df_list) if df_list else pd.DataFrame()
    combined_df.to_csv(run.data_folder/csv_name, index=False)

def do_extract_debuginfo_labels(run:Run, params:Dict[str,Any], outputs:Dict[str,Any]):
    console = Console()

    # use tqdm if we are logging to the console (instead of a file)
    global _use_tqdm
    _use_tqdm = params['debug_in_process']

    locals_dfs = []
    num_binaries = len(outputs['flatten_binaries'])

    for i, (bin_id, fb) in enumerate(outputs['flatten_binaries'].items()):
        fb:FlatLayoutBinary
        console.rule(f'Processing binary: [bold yellow]{fb.binary_file.name}[/] ' + \
            f'({i+1:,} of {num_binaries:,}) {i/num_binaries*100:.1f}%')

        with print_runtime():
            extract_data_tables(fb)
            # temp_member_expression_logic(fb)

    # combine into unified files
    flat_bins = outputs['flatten_binaries'].values()

    bins_df = pd.DataFrame([(fb.id,
                             fb.binary_file.name,
                             fb.debug_binary_file.resolve()
                            ) for fb in flat_bins],
                            columns=['BinaryId','Name','DebugBinary'])
    bins_df.to_csv(run.data_folder/'binaries.csv', index=False)

    combine_fb_tables_into_rundata(run, flat_bins, 'locals.csv')
    combine_fb_tables_into_rundata(run, flat_bins, 'functions.csv')
    combine_fb_tables_into_rundata(run, flat_bins, 'function_params.csv')

def temp_member_expression_logic(fb:FlatLayoutBinary):
    '''
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    CLS: I just threw a function wrapper around this code to move it out of the way
    for now....when I am ready to implement the Member Offset/Expression table,
    this is where all the logic is that I was testing/playing around with
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    '''
    console = Console()
    debug_asts = fb.data['debug_asts']
    stripped_asts = fb.data['stripped_asts']

    # pull in debug info for this binary
    ddi = DwarfDebugInfo.fromElf(fb.debug_binary_file)

    # collect functions for this binary, partitioned into debug/stripped sets
    stripped_funcs = set(stripped_asts.glob('*.json'))
    debug_funcs = set(debug_asts.glob('*.json'))

    # map ghidra addr -> json Path
    stripped_by_addr = {}
    for j in stripped_funcs:
        with open(j) as f:
            data = json.load(f)
            ghidra_addr = data['inner'][-1]['address']
            stripped_by_addr[ghidra_addr] = j

    # separate the functions that had errors (log files)
    stripped_fails = separate_ast_fails(stripped_funcs)
    debug_fails = separate_ast_fails(debug_funcs)
    print(f'# stripped fails = {len(stripped_fails)}')
    print(f'# debug fails = {len(debug_fails)}')

    for ast_json_debug in sorted(debug_funcs):

        print(f'converting {ast_json_debug.stem}...')

        # TEMP
        # if ast_json_debug.stem != 'r_batch_add':
        #     continue

        ast_debug, slib_debug = astlib.json_to_ast(ast_json_debug)
        # with open(ast_json_debug) as f:
        #     ast_debug_dict = json.load(f)

        funcdbg_addr_gh = ast_debug.inner[-1].address
        funcdbg_addr_dw = ghidra_to_dwarf_addr(funcdbg_addr_gh)

        if funcdbg_addr_dw not in ddi.funcdies_by_addr:
            # this is ok in general - startup functions like _DT_INIT and _DT_FINI don't have debug info
            console.print(f'No debug info for function @ 0x{funcdbg_addr_dw:x} (ghidra name = {ast_json_debug.stem}, ghidra addr = 0x{funcdbg_addr_gh:x})',
                        style='red')
            continue

        fdie = ddi.funcdies_by_addr[funcdbg_addr_dw]

        print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print(f'AST for function {ast_debug.inner[-1].name}')
        print(f'Function {fdie.name} at {funcdbg_addr_dw:x} (ghidra address = {funcdbg_addr_gh:x})')


        # TODO: start with an AST function -> read in JSON using astlib
        # 2) in DWARF data, map variable addresses -> variable DIEs

        locals = ddi.get_function_locals(fdie)
        params = list(ddi.get_function_params(fdie))

        ast_json = stripped_by_addr[funcdbg_addr_gh]
        print(f'converting {ast_json.stem}...')
        ast, slib = astlib.json_to_ast(ast_json)

        # slayout = get_struct_layout(params[0])

        # -----------------------------------------
        # NEED TO SUPPORT QUICK ANALYSIS, COMPUTING
        # ANSWERS ACROSS BINARY/EXP (pd.DataFrames...)
        # -----------------------------------------
        # TODO: - refactor pieces of this code to allow QUICKLY/easily
        # accessing:
        #   - a binary and its matching debug binary
        #   - a binary function and its matching debug function
        #   - ASTs for both
        #   - debug info for debug function
        # TODO: - start answering some of the questions I have...
        #   - find all functions that are MISSING parameters (vs. debug info)
        #   - find all functions IN DEBUG build that don't match DWARF debug info
        #   - ...
        # TODO: >>> start formulating the problem/scope SPECIFICALLY
        #   > it's ok to start with the "easy" case(s)...
        #   > some of these "quick calculation" numbers will help get a better
        #     feel for what we're dealing with... (e.g. only handles "easy case", but this
        #     occurs for 78% of functions in our dataset)

        # -----------------------------------------
        # MEMBER OFFSETS
        # -----------------------------------------
        # 1. Visit AST_DEBUG and find all MemberExpr nodes
        # 2. Extract the (instr_addr, sid, offset) for each one
        # 3. Using this data, visit AST and find all IntegerLiteral nodes at
        #    each instr_addr
        #       - Q1: are these the only AST node type that represent member offsets?
        #       - Q2: can we locate ALL of these?
        #       - Q2.2: if not, check if the reason is due to short* variables that
        #               have "offsets" with pointer arithmetic applied
        #               (where we see OFF/4 instead of OFF)
        #       - Q3: may need to look at whether or not the corresponding structure
        #             variable has been recovered to know this
        #             i.e. - some cases could be "dead code eliminated"
        # 4. If this approach works, then we build a labeled member offset dataset
        #    by simply labeling each matching node as a 1 or MEMBER (and all other nodes
        #    are a 0 or NON-MEMBER)
        member_nodes = extract_ast_nodes('MemberExpr', ast_debug.inner[-1])

        member_exprs = []
        for node in member_nodes:
            # for each chain of nested MemberExpr nodes, we only want to capture
            # the leaf-most node and look for a combined offset value of each
            # (non-isArrow) member in the chain...i.e. all its parent MemberExpr
            # as in: node.x.y.z
            #
            # which has this AST structure:
            # [node] --> [x] --> [y] --> [z]

            if not node.isArrow:
                # don't skip this if isArrow == True...
                #
                # isArrow "breaks the chain" of combined member offsets
                # since it can't be combined with any previous offsets (due to
                # the pointer indirection)
                #
                # so if we have var.x.y->node.z then node will have its own unique
                # member offset that can't be combined with x.y (but could be combined
                # with z)
                if node.inner[0].kind == 'MemberExpr':
                    # node is NOT an isArrow node (so we have parent.node not parent->node)
                    # AND it has a child member (parent.node.child) so skip node...
                    # we'll combine the offsets in the leaf-most child
                    continue

            x = node
            offset = node.offset
            while x.parent.kind == 'MemberExpr' and not x.parent.isArrow:
                x = x.parent
                offset += x.offset
            member_exprs.append((node, offset))
        # member_exprs = [(x.instr_addr, x.sid, x.offset) for x in member_nodes]

        for node, offset in member_exprs:
            sid = node.sid
            instr_addr = node.instr_addr
            if offset == 0:
                # CLS: only look for nonzero offsets
                # - there is always a member at 0 (we don't have to find the offset)
                # - the "0" is implicit and won't appear in the AST anyway!
                continue

            intliterals = [n for n in ast.nodes_at_addr(instr_addr) if n.kind == 'IntegerLiteral']
            # matches = [l for l in intliterals if l.value == offset]
            if ast_json_debug.stem == 'a_ctx_update':
                print('here')
            matches = find_member_offset_matches(intliterals, offset)
            sname = slib_debug[sid].name if sid in slib_debug else ''
            if sid in slib_debug and isinstance(slib_debug[sid], astlib.UnionDef):
                print(f'Skipping UNION type {sname}')
                continue
            # mname = slib_debug[sid].layout[offset].name if sid in slib_debug else ''
            mname = node.name

            if len(matches) > 1:
                print(f'{len(matches)} matches found for member access @ 0x{instr_addr:x} ({sname}.{mname})')
                for x in matches:
                    print(f'{x.kind}: value={x.value} parent={x.parent}')
                import IPython; IPython.embed()
            elif not matches:
                declref = node
                while declref.kind != 'DeclRefExpr':
                    declref = declref.inner[0]

                vtype = 'local'
                if declref.referencedDecl.kind == 'ParmVarDecl':
                    vtype = 'param'
                elif declref.referencedDecl.parent.kind == 'TranslationUnitDecl':
                    vtype = 'global'

                console.print(f'NO MATCH FOUND for {vtype} var "{node.name}" member offset! {sname}.{mname} @ 0x{instr_addr:x} in {ast_json_debug.stem}',
                        style='red')
                # if ast_json_debug.stem == '_a_get_gain':
                #     continue
                # import IPython; IPython.embed()

            # if ast_json_debug.stem == 'fonsCreateInternal':
            #     import IPython; IPython.embed()

            for m in matches:
                m.IS_MEMBER_ACCESS = True

        render_ast(ast_debug, ast_json_debug.stem, ast_json_debug.parent, 'svg', 'MemberExpr')
        render_ast(ast, ast_json.stem, ast_json.parent, 'svg', 'MemberExpr')

        # import IPython; IPython.embed()

def extract_debuginfo_labels() -> RunStep:
    return RunStep('extract_debuginfo_labels', do_extract_debuginfo_labels)

class BasicDatasetExp(Experiment):
    def __init__(self,
        exp_folder:Path=None,
        runconfigs:List[RunConfig]=None,
        projectlist:List[ProjectRecipe]=[],
        params={},
        opt:str='O0',
        compilers:str='gcc:g++',
        platforms:str='x64',
        keep_binaries:str='') -> None:
        '''
        exp_folder: The experiment folder
        projectlist: List of project recipes
        clang_dir: Root directory of Clang for extracting funcprotos
        funcproto_so_dir: Folder containing the Clang funcproto plugin shared object
        opt: csv of optimization levels to use
        compilers: CSV of colon-separated compiler names "<C>:<C++>", where each compiler name is
                   mapped by each platform used
        platforms: CSV of platform names to build against. Each platform should be a recognized platform
                   name (hardcoded for now below)
        keep_binaries: CSV of binary names to include (and discard the rest). If not specified all binaries
                         will be kept
        '''

        # console = Console()
        # console.print(f'[yellow]Warning: hardcoding aarch64-linux-gnu-gcc compiler for a test...')
        # compilers = 'aarch64-linux-gnu-gcc-9:aarch64-linux-gnu-g++-9:aarch64-linux-gnu-strip'

        supported_platforms = {

            # NOTE: if we add a 32-bit platform, need to check the logic in find_binaries which looks
            # for "ELF 64-bit" in the output of "file <exe>" to locate binaries

            'x64': {
                'gcc': 'gcc',
                'g++': 'g++',
                'strip': 'strip',
                'apt_arch': '',
            },
            'arm64': {
                'gcc': 'aarch64-linux-gnu-gcc-9',
                'g++': 'aarch64-linux-gnu-g++-9',
                'strip': 'aarch64-linux-gnu-strip',
                'apt_arch': 'arm64',
            }
        }

        # experiment runs
        platform_list = [x.strip() for x in platforms.split(',')]
        compiler_configs = [x.strip() for x in compilers.split(',')]
        opt_levels = [x.strip() for x in opt.split(',')]

        runconfigs = []
        for platform in platform_list:
            if platform not in supported_platforms:
                raise Exception(f'Unrecognized platform {platform}')

            platform_dict = supported_platforms[platform]

            for compiler_cfg in compiler_configs:
                # import IPython; IPython.embed()
                compiler_parts = compiler_cfg.split(':')
                cc_name = compiler_parts[0]
                cxx_name = compiler_parts[1]

                if cc_name not in platform_dict:
                    raise Exception(f'C compiler {cc_name} unmapped for platform {platform}')
                if cxx_name not in platform_dict:
                    raise Exception(f'C++ compiler {cxx_name} unmapped for platform {platform}')

                cc = platform_dict[cc_name]
                cxx = platform_dict[cxx_name]
                strip = platform_dict['strip']
                apt_arch = platform_dict['apt_arch']

                for opt_flag in opt_levels:
                    rc = RunConfig(f'{platform}-{cc_name}-{opt_flag}')
                    # set compiler paths, debug info
                    rc.c_options.compiler_path = cc
                    rc.cpp_options.compiler_path = cxx
                    rc.strip_executable = strip
                    rc.apt_arch = apt_arch
                    rc.c_options.enable_debug_info()
                    rc.cpp_options.enable_debug_info()
                    rc.opt_level = f'-{opt_flag}'

                    # NOTE: trying without lld since we no longer use linker-objects to locate binaries
                    # use our linker so we can find exes automatically
                    # rc.linker_flags.extend(['-fuse-ld=lld'])
                    # don't need -B for gcc native, but cross-compiling aarch64 wouldn't work without
                    # using -B and -fuse-ld, and it doesn't seem to hurt normal x64
                    # rc.linker_flags.extend(['-B', '/llvm-build/bin'])

                    # for C++ make sure we don't get prototypes because of
                    # mangled symbol names in DST
                    rc.cpp_options.compiler_flags.extend(['-Xlinker', '--no-export-dynamic'])
                    runconfigs.append(rc)

        exp_params = {
            'exp_docker_cmds': [
                # install ourselves into docker :)
                'RUN pip install --upgrade pip',
                'RUN --mount=type=ssh pip install git+ssh://git@github.com/lasserre/datatype-recovery-experiments.git ' \
                                                 'git+ssh://git@github.com/lasserre/astlib.git',
                'RUN apt update && apt install -y gcc g++ clang autoconf texinfo ' \
                    'gcc-9-aarch64-linux-gnu g++-9-aarch64-linux-gnu',
                'ENV WRAPPER_BIN="/wrapper_bin"',
                'ENV PATH="${WRAPPER_BIN}:${PATH}"',
                'RUN mkdir -p ${WRAPPER_BIN} && chmod 777 ${WRAPPER_BIN}',
                # add arm64 entries to sources.list
                'RUN sed -i "s/deb http/deb [arch=amd64,i386] http/g" /etc/apt/sources.list && ' \
                    'echo "deb [arch=arm64] http://ports.ubuntu.com focal main universe restricted multiverse" >> /etc/apt/sources.list && ' \
                    'echo "deb [arch=arm64] http://ports.ubuntu.com focal-updates main universe restricted multiverse" >> /etc/apt/sources.list && ' \
                    'echo "deb [arch=arm64] http://ports.ubuntu.com focal-security main universe restricted multiverse" >> /etc/apt/sources.list',
                'RUN dpkg --add-architecture arm64 && apt update',
            ],
            'GHIDRA_INSTALL': Path.home()/'software'/'ghidra_10.3_DEV',
            'keep_binaries': keep_binaries,
        }

        algorithm = DockerBuildAlgorithm(
            preprocess_steps=[
                start_ghidra_server(),
                create_ghidra_repo(),
            ],
            # pre_build_steps=[
            pre_configure_steps=[
                install_cc_wrapper(),
                # dump_dt_labels()
            ],
            post_build_steps = [
                find_binaries(),
                flatten_binaries(),
                strip_binaries(),
                ghidra_import(debug=False, prescript=astlib.set_analysis_options_script()),
                ghidra_import(debug=True, prescript=astlib.set_analysis_options_script()),
                export_asts(debug=False),
                export_asts(debug=True),
                extract_debuginfo_labels(),
            ],
            postprocess_steps = [
            ])

        super().__init__('basic-dataset', algorithm, runconfigs,
            projectlist, exp_folder=exp_folder, params=exp_params)
