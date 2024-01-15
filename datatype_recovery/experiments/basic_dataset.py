from pathlib import Path
import hashlib
from io import StringIO
from itertools import chain
import itertools
import json
import pandas as pd
from rich.console import Console
import shutil
import subprocess
import sys
from tqdm import tqdm
from typing import List, Tuple

from wildebeest import Experiment, RunConfig, ProjectRecipe
from wildebeest import DockerBuildAlgorithm, DefaultBuildAlgorithm
from wildebeest.postprocessing import find_binaries, flatten_binaries, strip_binaries, find_instrumentation_files
from wildebeest.postprocessing import ghidra_import
from wildebeest.preprocessing.ghidra import start_ghidra_server, create_ghidra_repo
from wildebeest import *
from wildebeest.run import Run
from wildebeest.utils import PrintRuntime

from wildebeest.postprocessing.flatlayoutbinary import FlatLayoutBinary

import astlib
from varlib.location import Location
from varlib.datatype import *
import dwarflib
from dwarflib import *

_use_tqdm = False

def show_progress(iterator, total:int, use_tqdm:bool=None, progress_period:int=500):
    '''
    Show a progress indicator - either using tqdm progress bar (ideal for console output)
    or a (much less frequent) periodic print statement showing how far we have come
    (ideal for log files)

    iterator: The object being iterated over (as long as it behaves like an iterator and
              you unpack the values properly it should work)
    total:    Total number of items in the iterator, this gives flexibility with the iterator
              not being required to support len()
    use_tqdm: Use tqdm if true, print statement if false. If not specified, the global _use_tqdm
              will be used instead.
    progress_period: How many items should be iterated over before a progress line is printed
    '''
    global _use_tqdm
    if use_tqdm is None:
        use_tqdm = _use_tqdm

    if use_tqdm:
        for x in tqdm(iterator, total=total):
            yield x
    else:
        ctr = 1
        for x in iterator:
            if ctr % progress_period == 0:
                print(f'{ctr}/{total} ({ctr/total*100:.1f}%)...', flush=True)
            ctr += 1
            yield x

# not a bad function, just probably not going to use it right now...
# NOTE: hang on to it for now, if I end up totally not needing to look at DWARF or dtlabels
# source code info then I can discard...
# ------------------------
def _print_source_OLD_(srcfile:Path, start_line:int, end_line:int, markers:List[Tuple[int,int]]):
    '''Print the given lines of the source code file, adding "highlight" lines with carets
    underneath to show specific line:column locations

    (as specified in markers, which is a list of (line, column) tuples)'''
    with open(srcfile, 'r') as f:
        lines = f.readlines()[start_line-1:end_line]

    sorted_markers = sorted(markers, key=lambda m: m[0])
    for i, line in enumerate(lines):
        print(line.strip())
        if sorted_markers and sorted_markers[0][0] == i+start_line:
            cols = []
            while sorted_markers and sorted_markers[0][0] == i+start_line:
                cols.append(sorted_markers[0][1])
                sorted_markers = sorted_markers[1:]
            cols = sorted(cols)
            markers_str = ''
            last_col = 0
            for col in cols:
                delta = col - last_col
                if delta < 1:
                    continue
                if col > -1:
                    markers_str += f'{"-"*(delta-1)}^'
                last_col = col
            markers_str += f" ({','.join(f'{start_line+i}:{col}' for col in cols)})"
            print(markers_str)

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
    fails = set([x for x in ast_paths if x.with_suffix('.log').exists()])
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

def build_dwarf_data_tables(debug_binary_file:Path) -> DwarfTables:
    '''
    Build the DWARF local variables table for the given debug binary

    DWARF Function Addr | Var Name | Location | Type | Type Category

    debug_binary_file: The path to the DWARF file (typically a debug binary build)
    '''
    # pull in debug info for this binary
    ddi = DwarfDebugInfo.fromElf(debug_binary_file)
    sdb = StructDatabase()

    with dwarflib.UseStructDatabase(sdb):
        tables = build_dwarf_data_tables_from_ddi(ddi)

    print(f'Serializing DWARF sdb...')
    sdb.to_json(debug_binary_file.with_suffix('.dwarf.sdb'))
    return tables

def build_dwarf_data_tables_from_ddi(ddi:DwarfDebugInfo) -> DwarfTables:
    # NOTE: I can't think of a particularly good reason for doing this one way over
    # another, so arbitrarily deciding to use Ghidra addresses for consistency
    # (if nothing else, I will be looking at Ghidra much more often than DWARF data)

    locals_dfs = []
    func_names = []
    func_starts = []
    params_df_list = []

    for dwarf_addr, fdie in show_progress(ddi.funcdies_by_addr.items(), total=len(ddi.funcdies_by_addr)):
        if fdie.artificial:
            print(f'Skipping artificial function {fdie.name} (intrinsic?)')
            continue
        if fdie.inline:
            print(f'Skipping inlined function {fdie.name}')
            continue

        # print(f'Extracting DWARF from {fdie.name}',flush=True)

        ### Locals
        locals = ddi.get_function_locals(fdie)

        # NOTE: we aren't matching vars up by location anymore - don't even pull them,
        # right now it causes issues when we see location lists and we don't need to
        # spend time to fix this if we don't use it
        # locations = [l.location_varlib for l in locals]
        # --> use Undefined location as placeholder
        locations = [Location(LocationType.Undefined) for l in locals]

        df = pd.DataFrame({
            'Name': [l.name for l in locals],
            'Type': [l.dtype_varlib for l in locals],
            'LocType': [l.loc_type for l in locations],
            'LocRegName': [l.reg_name for l in locations],
            'LocOffset': pd.array([l.offset for l in locations], dtype=pd.Int64Dtype()),
        })

        df['FunctionStart'] = dwarf_to_ghidra_addr(dwarf_addr)
        df['FunctionName'] = fdie.name
        df['TypeCategory'] = [t.category for t in df.Type]
        df['TypeSeq'] = [t.type_sequence for t in df.Type]

        locals_dfs.append(df)

        ### Functions
        func_names.append(fdie.name)
        func_starts.append(dwarf_to_ghidra_addr(dwarf_addr))

        ### Function prototype
        params = list(ddi.get_function_params(fdie))

        #############################################################################
        # TODO: add check here for PARAMETERS with negative stack offsets
        #############################################################################
        # --> whenever this happens, we are in the situation where DWARF reports the
        #     locations of stack copies instead of the actual (register) locations
        # --> Ghidra uses the true/register locations for params, so we need to
        #     correct this to be able to line up Ghidra params with DWARF params
        #############################################################################

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

        #############################################################################################
        # CLS: OLD code applying System V calling convention...
        #############################################################################################
        # TODO: if we do detect it, maybe add a sanity check that run_config.compiler_flags does NOT
        # contain -O1-O3?

        # negative stack offset -> apply system v calling convention
        # if any([l.loc_type == LocationType.Stack and l.offset < 0 for l in param_locs]):
        #     dwarf_param_locs = param_locs   # save these in case we want to compare
        #     print(f'Converting param_locs for function {fdie.name} @ {dwarf_to_ghidra_addr(dwarf_addr):x}')
        #     try:
        #         param_locs = get_sysv_calling_conv([p.dtype_varlib for p in params])    # use sysv convention
        #     except:
        #         print(f'FAILED conversion')
        #         import IPython; IPython.embed()

        # no DW_AT_type attribute indicates return type is void
        rtype = fdie.dtype_varlib if fdie.type_die else BuiltinType.create_void_type()

        params_df = pd.DataFrame({
            'FunctionStart': pd.array([dwarf_to_ghidra_addr(dwarf_addr)] * len(params), dtype=pd.UInt64Dtype()),
            'Name': [p.name for p in params],
            'IsReturnType': pd.array([False] * len(params), dtype=pd.BooleanDtype()),
            'Type': [p.dtype_varlib for p in params],
            'LocType': [l.loc_type for l in param_locs],
            'LocRegName': [l.reg_name for l in param_locs],
            'LocOffset': pd.array([l.offset for l in param_locs], dtype=pd.Int64Dtype()),
        })

        params_df = pd.concat([params_df, pd.DataFrame({
            'FunctionStart': pd.array([dwarf_to_ghidra_addr(dwarf_addr)], dtype=pd.UInt64Dtype()),
            'Name': [None],
            'IsReturnType': pd.array([True], dtype=pd.BooleanDtype()),
            'Type': [rtype],
            # apparently there is currently no way to get the location of a return value in DWARF
            # outstanding proposal to add this here: https://dwarfstd.org/issues/221105.1.html
            'LocType': [None],
            'LocRegName': [None],
            'LocOffset': [None],
        })], ignore_index=True)

        params_df['TypeCategory'] = [t.category for t in params_df.Type]
        params_df['TypeSeq'] = [t.type_sequence for t in params_df.Type]
        params_df_list.append(params_df)

    tables = DwarfTables()
    tables.funcs_df = pd.DataFrame({
        'FunctionStart': func_starts,
        'FunctionName': func_names
    })
    tables.locals_df = pd.concat(locals_dfs)
    tables.params_df = pd.concat(params_df_list)

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

def compute_var_ast_signature(fdecl:astlib.ASTNode, fbody:astlib.ASTNode, varname:str) -> str:
    '''
    Compute the DIRTY-style variable signature for the given variable.

    The signature will be a string containing the sorted list of decimal instruction offsets
    (relative to the start of the function) in CSV format, and uniquely identifies a variable
    '''
    var_refs = astlib.FindAllVarRefs(varname).visit(fbody)
    ref_instr_offsets = sorted([x.instr_addr - fdecl.address for x in var_refs])
    return ','.join(map(str, ref_instr_offsets))

def extract_funcdata_from_ast(ast:astlib.ASTNode, ast_json:Path) -> FunctionData:

    fdecl = ast.inner[-1]
    fbody = fdecl.inner[-1]

    # prototype
    params = fdecl.inner[:-1]
    return_type = fdecl.return_dtype

    # locals
    local_decls = itertools.takewhile(lambda node: node.kind == 'DeclStmt', fbody.inner)
    local_vars = [decl_stmt.inner[0] for decl_stmt in local_decls]

    fd = FunctionData()
    fd.ast_json_filename = ast_json
    fd.name = fdecl.name
    fd.address = fdecl.address
    fd.locals_df = build_ast_locals_table(fdecl, local_vars)
    fd.params_df = build_ast_func_params_table(fdecl, params, return_type)

    # TODO globals?

    return fd

def build_ast_func_params_table(fdecl:astlib.ASTNode, params:List[astlib.ASTNode], return_type:astlib.ASTNode):
    '''
    Build the function parameters table for the given AST function
    '''
    fbody = fdecl.inner[-1]

    df = pd.DataFrame({
        'FunctionStart': pd.array([fdecl.address] * len(params), dtype=pd.UInt64Dtype()),
        'Name': [p.name for p in params],
        'Signature': [compute_var_ast_signature(fdecl, fbody, p.name) for p in params],
        'IsReturnType': pd.array([False] * len(params), dtype=pd.BooleanDtype()),
        'Type': [p.dtype.dtype_varlib for p in params],
        'LocType': [p.location.loc_type if p.location else None for p in params],
        'LocRegName': [p.location.reg_name if p.location else None for p in params],
        'LocOffset': pd.array([p.location.offset if p.location else None for p in params],
                              dtype=pd.Int64Dtype()),
    })

    # add return type
    df = pd.concat([df, pd.DataFrame({
        'FunctionStart': pd.array([fdecl.address], dtype=pd.UInt64Dtype()),
        'Name': [None],
        # NOTE: special signature of "0" for return types
        # - populate with a string to avoid getting dropped as an empty signature
        # - we don't really need a "signature" for return type - there is one and only
        #   one (and our signature code looks for references to named variables but
        #        this is not a named variable)
        'Signature': ['0'],
        'IsReturnType': pd.array([True], dtype=pd.BooleanDtype()),
        'Type': [return_type.dtype_varlib],
        'LocType': [return_type.location.loc_type if return_type.location else None],
        'LocRegName': [return_type.location.reg_name if return_type.location else None],
        'LocOffset': pd.array([return_type.location.offset if return_type.location else None],
                              dtype=pd.Int64Dtype()),
    })], ignore_index=True)

    df['TypeCategory'] = [t.category for t in df.Type]
    df['TypeSeq'] = [t.type_sequence for t in df.Type]

    return df

def build_ast_locals_table(fdecl:astlib.ASTNode, local_vars:List[astlib.ASTNode]):
    '''
    Build the local variables table for the given AST

    Ghidra Function Addr | Var Name? | Location | Type | Type Category
    '''
    fbody = fdecl.inner[-1]

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
        'Signature': [compute_var_ast_signature(fdecl, fbody, lv.name) for lv in local_vars],
        'Type': [v.dtype.dtype_varlib for v in local_vars],
        # 'Location': [v.location for v in local_vars]
        'LocType': [v.location.loc_type if v.location else None for v in local_vars],
        'LocRegName': [v.location.reg_name if v.location else None for v in local_vars],
        'LocOffset': pd.array([v.location.offset if v.location else None for v in local_vars],
                              dtype=pd.Int64Dtype()),
    })

    df['TypeCategory'] = [t.category for t in df.Type]
    df['TypeSeq'] = [t.type_sequence for t in df.Type]

    return df

def read_failed_decompilation_addrs(failed_decomps_txt:Path) -> List[int]:
    '''
    Parse out the addresses of functions that completely failed to decompile
    and return the list of function addresses. If none failed returns an empty list
    '''
    if failed_decomps_txt.exists():
        with open(failed_decomps_txt, 'r') as f:
            return [int(l.strip(), 16) for l in f.readlines()]
    return []

def collect_passing_asts(fb:FlatLayoutBinary):
    '''Collect the debug and stripped ASTs that did not have failures'''
    debug_asts = fb.data['debug_asts']
    stripped_asts = fb.data['stripped_asts']

    # collect functions for this binary, partitioned into debug/stripped sets
    stripped_funcs = set(stripped_asts.glob('*.json'))
    debug_funcs = set(debug_asts.glob('*.json'))

    failed_debug_decomps = debug_asts/'failed_decompilations.txt'
    failed_stripped_decomps = stripped_asts/'failed_decompilations.txt'

    failed_debug_addrs = read_failed_decompilation_addrs(failed_debug_decomps)
    failed_stripped_addrs = read_failed_decompilation_addrs(failed_stripped_decomps)

    # separate the functions that had errors (log files)
    stripped_fails = separate_ast_fails(stripped_funcs)
    debug_fails = separate_ast_fails(debug_funcs)
    print(f'# stripped decomp fails = {len(failed_stripped_addrs)}')
    print(f'# debug decomp fails = {len(failed_debug_addrs)}')
    print(f'--------------------')
    print(f'# stripped AST export fails = {len(stripped_fails)}')
    print(f'# debug AST export fails = {len(debug_fails)}')
    return (debug_funcs, stripped_funcs)

def extract_funcdata_from_ast_set(ast_funcs:Set[Path], bin_path:Path, is_debug:bool) -> List[FunctionData]:
    '''Extracts FunctionData content from each of the provided ASTs'''
    fdatas:List[FunctionData] = []
    num_funcs = len(ast_funcs)

    sdb = StructDatabase()

    with astlib.UseStructDatabase(sdb):
        for i, ast_json in show_progress(enumerate(sorted(ast_funcs)), total=len(ast_funcs)):
            ast, slib = astlib.json_to_ast(ast_json)
            fdatas.append(extract_funcdata_from_ast(ast, ast_json))

    suffix = '.debug.sdb' if is_debug else '.sdb'
    sdb.to_json(bin_path.with_suffix(suffix))

    return fdatas

def extract_data_tables(fb:FlatLayoutBinary):
    '''
    Build our master table of local variables
    '''
    # Function | Binary |
    # ... DWARF local var | Stripped AST local var | Debug AST local var | Stripped Function AST

    debug_funcs, stripped_funcs = collect_passing_asts(fb)

    print(f'Extracting DWARF data for binary {fb.debug_binary_file.name}...')

    dwarf_tables = build_dwarf_data_tables(fb.debug_binary_file)

    # extract data from ASTs/DWARF debug symbols
    print(f'Extracting data from {len(debug_funcs):,} debug ASTs for binary {fb.debug_binary_file.name}...')
    debug_funcdata = extract_funcdata_from_ast_set(debug_funcs, fb.debug_binary_file, is_debug=True)

    print(f'Extracting data from {len(stripped_funcs):,} stripped ASTs for binary {fb.binary_file.name}...')
    stripped_funcdata = extract_funcdata_from_ast_set(stripped_funcs, fb.binary_file, is_debug=False)

    # NOTE when we need more DWARF data, extract it all at once while we have
    # the file with DWARF debug info opened

    ### Locals
    locals_df, locals_stats_df = build_locals_table(debug_funcdata, stripped_funcdata,
                                                    dwarf_tables.locals_df, fb.data_folder)

    if not locals_df.empty:
        locals_df.loc[:,'BinaryId'] = fb.id
    locals_stats_df.loc[:,'BinaryId'] = fb.id
    locals_df.to_csv(fb.data_folder/'locals.csv', index=False)
    locals_stats_df.to_csv(fb.data_folder/'locals.stats.csv', index=False)

    ### Functions
    funcs_df = build_funcs_table(debug_funcdata, stripped_funcdata, dwarf_tables.funcs_df)
    funcs_df.loc[:,'BinaryId'] = fb.id
    funcs_df.to_csv(fb.data_folder/'functions.csv', index=False)

    ### Function Parameters (prototype)
    params_df, params_stats_df = build_params_table(debug_funcdata, stripped_funcdata, dwarf_tables.params_df)
    if not params_df.empty:
        params_df.loc[:,'BinaryId'] = fb.id
    params_stats_df.loc[:,'BinaryId'] = fb.id
    params_df.to_csv(fb.data_folder/'function_params.csv', index=False)
    params_stats_df.to_csv(fb.data_folder/'function_params.stats.csv', index=False)

def build_params_table(debug_funcdata:List[FunctionData], strip_funcdata:List[FunctionData],
                      dwarf_df:pd.DataFrame):
    debug_df = pd.concat(fd.params_df for fd in debug_funcdata)
    strip_df = pd.concat(fd.params_df for fd in strip_funcdata)

    # TODO: divide params/return types, combine separately, then recombine
    # -> this allows us to NOT drop debug return types because they don't
    #    (and can't!) align with DWARF return types by name...bc there is no name
    debug_rtypes = debug_df.loc[debug_df.IsReturnType,:]
    debug_params = debug_df.loc[~debug_df.IsReturnType,:]
    strip_rtypes = strip_df.loc[strip_df.IsReturnType,:]
    strip_params = strip_df.loc[~strip_df.IsReturnType,:]
    dwarf_rtypes = dwarf_df.loc[dwarf_df.IsReturnType,:]
    dwarf_params = dwarf_df.loc[~dwarf_df.IsReturnType,:]

    print(f'Combining params/return types...')
    params_df, stats_df = build_var_table_by_signatures(debug_params, strip_params, dwarf_params,
                                                        drop_extra_debug_vars=True)
    rtypes_df, _ = build_var_table_by_signatures(debug_rtypes, strip_rtypes, dwarf_rtypes,
                                                        drop_extra_debug_vars=False)

    # recombine params/rtypes
    return pd.concat([params_df, rtypes_df]).reset_index(drop=True), stats_df

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

    return df.reset_index(drop=True)

def drop_duplicate_vars(df:pd.DataFrame) -> pd.DataFrame:
    '''
    Drop all occurrences of variables that have duplicate signatures (within the same function)
    '''
    # can't just drop_duplicates() since I don't want ANY rows left over that had a duplicate
    varcounts = df.groupby(['FunctionStart', 'Signature']).count()
    dupvar_idx = varcounts[varcounts.Name>1].index
    return df.set_index(['FunctionStart','Signature']).drop(index=dupvar_idx).reset_index()

def build_locals_table(debug_funcdata:List[FunctionData], stripped_funcdata:List[FunctionData],
                       dwarf_locals:pd.DataFrame, data_folder:Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ## Locals table

    # combine into single df
    debug_locals = pd.concat([fd.locals_df for fd in debug_funcdata])
    stripped_locals = pd.concat([fd.locals_df for fd in stripped_funcdata])

    # TEMP: save off raw dataframes so I can figure out why nothing lined up...lol
    dwarf_locals.to_csv(data_folder/'_raw_dwarf_locals.csv', index=False)
    debug_locals.to_csv(data_folder/'_raw_debug_locals.csv', index=False)
    stripped_locals.to_csv(data_folder/'_raw_stripped_locals.csv', index=False)

    print(f'Combining stripped/debug local vars...')
    return build_var_table_by_signatures(debug_locals, stripped_locals, dwarf_locals)

def build_var_table_by_signatures(debug_vars:pd.DataFrame, stripped_vars:pd.DataFrame,
        dwarf_vars:pd.DataFrame,
        drop_extra_debug_vars:bool=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    drop_extra_debug_vars: Drop any debug variables that do NOT align with DWARF variables
                           by name (within the same function)
    '''

    # drop empty and duplicate signatures before merge
    num_empty_debug = len(debug_vars[debug_vars.Signature==''])
    num_empty_stripped = len(stripped_vars[stripped_vars.Signature==''])

    print(f'Dropped {num_empty_debug:,} debug vars with empty signatures (no refs)')
    print(f'Dropped {num_empty_stripped:,} stripped vars with empty signatures (no refs)')

    debug_df = debug_vars.loc[debug_vars.Signature!='', :]
    stripped_df = stripped_vars.loc[stripped_vars.Signature!='', :]

    dcount = debug_df.groupby(['FunctionStart','Signature']).count()
    scount = stripped_df.groupby(['FunctionStart','Signature']).count()
    num_dup_debug = dcount[dcount.TypeCategory>1].TypeCategory.sum()
    num_dup_stripped = scount[scount.TypeCategory>1].TypeCategory.sum()

    print(f'Dropped {num_dup_debug:,} duplicate debug vars')
    print(f'Dropped {num_dup_stripped:,} duplicate stripped vars')

    debug_df = drop_duplicate_vars(debug_df)
    stripped_df = drop_duplicate_vars(stripped_df)

    # REDUCE DEBUG TO ONLY TRUE DWARF VARS
    if drop_extra_debug_vars:
        DWARF_IDX_COLS = ['FunctionStart', 'Name']  # no var signature for dwarf, use var name
        tmp = debug_df.merge(dwarf_vars, how='left',on=DWARF_IDX_COLS,suffixes=['_Debug','_DWARF'])
        tmp_good = tmp[~tmp.Type_DWARF.isna()].set_index(DWARF_IDX_COLS)
        debug_dwarf = debug_df.set_index(DWARF_IDX_COLS).loc[tmp_good.index].reset_index()

        num_extra_debug_vars = len(debug_df) - len(debug_dwarf)
        print(f'Dropped {num_extra_debug_vars:,} debug vars that did not align with true DWARF vars ({len(debug_df):,} down to {len(debug_dwarf):,})')
    else:
        num_extra_debug_vars = -1   # indicates we did not drop (or count) extra debug vars
        debug_dwarf = debug_df  # don't drop anything, just merge (below)

    # merge based on (FunctionStart, Signature) instead of Loc
    df = debug_dwarf.merge(stripped_df, how='outer',
                        on=['FunctionStart', 'Signature'],
                        suffixes=['_Debug', '_Strip'])

    # reduce to only good matches

    # NOTE: using TypeCategory instead of Name here specifically so it works for return types
    df_good = df[(~df.TypeCategory_Strip.isna()) & (~df.TypeCategory_Debug.isna())]

    # reset the index to avoid strange behavior going forward
    # (setting .loc[:'newcol'] = pd.Series() doesn't work as you'd expect when indices skip around)
    df_good = df_good.reset_index(drop=True)

    yield_pcnt = len(df_good)/len(stripped_df)*100 if not stripped_df.empty else 0.0
    print(f'{len(df_good):,} of {len(stripped_df):,} (reduced) stripped vars align with debug var by signature ({yield_pcnt:.2f}% yield)')
    print(f'{len(df_good)/len(stripped_vars)*100 if not stripped_vars.empty else 0.0:.2f}% yield overall (before removing empty/dup signatures)')

    stats_df = pd.DataFrame({
        # Raw: input to function before processing
        'NumRawDebugVars': [len(debug_vars)],
        'NumRawStrippedVars': [len(stripped_vars)],
        # Empty: no references to these variables -> empty signatures
        'NumEmptyDebug': [num_empty_debug],
        'NumEmptyStripped': [num_empty_stripped],
        # Duplicate: variables with duplicate signatures (referenced in all the same instructions)
        'NumDupDebug': [num_dup_debug],
        'NumDupStripped': [num_dup_stripped],
        # Extra: debug vars that don't align with DWARF vars (e.g. not explicit source variables)
        'NumExtraDebugVars': [num_extra_debug_vars],
        # Good: remaining stripped vars that aligned with remaining debug vars
        'NumGoodVars': [len(df_good)]
    })

    return df_good, stats_df

def combine_fb_tables_into_rundata(run:Run, bin_list:List[FlatLayoutBinary], csv_name:str):
    '''
    Read each of the pandas tables (in file csv_name) from the list of flat binary folders
    and combine them into a single run-level data frame, writing it to the run data folder
    '''
    # if we pd.concat() with an empty dataframe it messes up the column data types
    df_list = (pd.read_csv(fb.data_folder/csv_name) for fb in bin_list)
    combined_df = pd.concat(df for df in df_list if not df.empty)
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

        with PrintRuntime():
            extract_data_tables(fb)
            # temp_member_expression_logic(fb)

    # combine into unified files
    flat_bins = outputs['flatten_binaries'].values()

    bins_df = pd.DataFrame([(fb.id, fb.binary_file.name) for fb in flat_bins], columns=['BinaryId', 'Name'])
    bins_df.to_csv(run.data_folder/'binaries.csv', index=False)

    combine_fb_tables_into_rundata(run, flat_bins, 'locals.csv')
    combine_fb_tables_into_rundata(run, flat_bins, 'functions.csv')
    combine_fb_tables_into_rundata(run, flat_bins, 'function_params.csv')
    combine_fb_tables_into_rundata(run, flat_bins, 'locals.stats.csv')
    combine_fb_tables_into_rundata(run, flat_bins, 'function_params.stats.csv')

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
            # mname = slib_debug[sid].fields_by_offset[offset].name if sid in slib_debug else ''
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

def do_install_cc_wrapper(run:Run, params:Dict[str,Any], outputs:Dict[str,Any]):
    # full path to cc_wrapper
    cc_wrapper_path = Path(subprocess.check_output(['which', 'cc_wrapper']).decode('utf-8').strip())
    cxx_wrapper_path = Path(subprocess.check_output(['which', 'cc_wrapper']).decode('utf-8').strip())

    # find full path to target compiler
    c_compiler_path = Path(subprocess.check_output(['which', run.config.c_options.compiler_path]).decode('utf-8').strip())
    cpp_compiler_path = Path(subprocess.check_output(['which', run.config.cpp_options.compiler_path]).decode('utf-8').strip())

    cc_link = Path('/wrapper_bin')/c_compiler_path.name
    cxx_link = Path('/wrapper_bin')/cpp_compiler_path.name

    # create symlink to cc_wrapper named <target_compiler> in /wrapper_bin
    subprocess.run(['ln', '-s', cc_wrapper_path, cc_link])
    subprocess.run(['ln', '-s', cxx_wrapper_path, cxx_link])
    subprocess.run(['hash', '-r'], shell=True)  # apparently bash caches program locations... https://unix.stackexchange.com/a/91176

    # allow cc_wrapper to find full path to target compiler for actual invocation later
    # NOTE: we have a unique container name (from docker run --name) for each run. I don't think
    # the files (not mounted) will persist...CHECK THIS
    # - if so, I can write the full path to a file at ~/compiler_full_path
    with open(Path.home()/'cc_path.txt', 'w') as f:
        f.write(str(c_compiler_path))
    with open(Path.home()/'cxx_path.txt', 'w') as f:
        f.write(str(cpp_compiler_path))

    # DEBUGGING
    print(subprocess.check_output(['ls', '-al', '/wrapper_bin']))

def install_cc_wrapper() -> RunStep:
    return RunStep('install_cc_wrapper', do_install_cc_wrapper, run_in_docker=True)

def get_dtlabels_tempfolder_for_build(run:Run):
    '''Compute a deterministic temp folder name derived from hashing the build folder path'''
    buildfolder_hash = hashlib.md5(str(run.build.build_folder).encode('utf-8')).hexdigest()
    return run.exp_root/f'dtlabels_{buildfolder_hash}'

def do_dump_dt_labels(run:Run, params:Dict[str,Any], outputs:Dict[str,Any]):
    from wildebeest.defaultbuildalgorithm import build, configure

    # ---- save original compiler options
    orig_compiler_path = run.config.c_options.compiler_path
    orig_compiler_flags = [f for f in run.config.c_options.compiler_flags]

    run.config.c_options.compiler_flags.extend([
        '-Xclang', '-load', '-Xclang', '/clang-dtlabels/build/libdtlabels.so',
        '-Xclang', '-add-plugin', '-Xclang', 'dtlabels',
        # -fsyntax-only doesn't work in practice - build system will fail because
        # unable to link the missing .o files!
    ])

    run.config.c_options.compiler_path = '/llvm-build/bin/clang'    # force our build of clang

    configure(run, params, outputs)
    build(run, params, outputs)

    # ---- put the original options back
    run.config.c_options.compiler_flags = orig_compiler_flags
    run.config.c_options.compiler_path = orig_compiler_path

    # should have .dtlabels files scattered throughout source folder now

    # ---- delete and remake a fresh build folder
    # (we're about to actually build the project, need a clean starting point)
    shutil.rmtree(run.build.build_folder)
    run.build.build_folder.mkdir(parents=True, exist_ok=True)

def dump_dt_labels() -> RunStep:
    return RunStep('dump_dt_labels', do_dump_dt_labels, run_in_docker=True)

def do_process_dt_labels(run:Run, params:Dict[str,Any], outputs:Dict[str,Any]):
    # ---- move dtlabels files to rundata folder
    # (have to do this AFTER rundata folder gets reset)
    dtlabels_folder = run.data_folder/'dtlabels'
    dtlabels_folder.mkdir(parents=True, exist_ok=True)

    dtlabels_files = run.build.project_root.glob('**/*.dtlabels')
    for f in dtlabels_files:
        # insert a hash into filename to avoid filename collisions (main.c? lol)
        hashval = hashlib.md5(str(f).encode('utf-8')).hexdigest()[:5]   # take portion of md5
        newfilename = f.with_suffix(f'.{hashval}.dtlabels').name
        newfile = dtlabels_folder/newfilename
        # moving each file, so we shouldn't have any leftovers in source tree
        f.rename(newfile)

    # import IPython; IPython.embed()

def process_dt_labels() -> RunStep:
    return RunStep('process_dt_labels', do_process_dt_labels)

class BasicDatasetExp(Experiment):
    def __init__(self,
        exp_folder:Path=None,
        runconfigs:List[RunConfig]=None,
        projectlist:List[ProjectRecipe]=[],
        params={}) -> None:
        '''
        exp_folder: The experiment folder
        projectlist: List of project recipes
        clang_dir: Root directory of Clang for extracting funcprotos
        funcproto_so_dir: Folder containing the Clang funcproto plugin shared object
        '''

        # experiment runs
        gcc_config = RunConfig('gcc')
        gcc_config.c_options.compiler_path = 'gcc'
        gcc_config.cpp_options.compiler_path = 'g++'
        # gcc_config.env_vars['WDB_CC'] = 'gcc'
        # gcc_config.env_vars['WDB_CXX'] = 'g++'

        # clang_config = RunConfig('clang')
        # clang_config.c_options.compiler_path = 'clang'
        # clang_config.cpp_options.compiler_path = 'clang++'

        # runconfigs = [gcc_config, clang_config]
        runconfigs = [gcc_config]

        # configure each runconfig with experiment-specific settings
        for rc in runconfigs:
            # append our flags
            rc.linker_flags.extend(['-fuse-ld=lld'])

            # enable debug info
            rc.c_options.enable_debug_info()
            rc.cpp_options.enable_debug_info()

            # I don't think we need this for C?
            # rc.c_options.compiler_flags.extend(['-Xlinker', '--no-export-dynamic'])
            rc.cpp_options.compiler_flags.extend(['-Xlinker', '--no-export-dynamic'])

            # everything is -O0 for now
            rc.env_vars['OPT_LEVEL'] = '-O0'

            # rc.c_options.compiler_flags.append('-O0')
            # rc.cpp_options.compiler_flags.append('-O0')
            # rc.c_options.compiler_flags.append('-O1')
            # rc.cpp_options.compiler_flags.append('-O1')
            # rc.c_options.compiler_flags.append('-g3')
            # rc.cpp_options.compiler_flags.append('-g3')

        exp_params = {
            'exp_docker_cmds': [
                # install ourselves into docker :)
                'RUN pip install --upgrade pip',
                'RUN --mount=type=ssh pip install git+ssh://git@github.com/lasserre/datatype-recovery-experiments.git',
                'RUN apt update && apt install -y gcc g++ clang',
                'ENV WRAPPER_BIN="/wrapper_bin"',
                'ENV PATH="${WRAPPER_BIN}:${PATH}"',
                'RUN mkdir -p ${WRAPPER_BIN} && chmod 777 ${WRAPPER_BIN}'
            ],
            'GHIDRA_INSTALL': Path.home()/'software'/'ghidra_10.3_DEV',
        }

        decompile_script = astlib.decompile_all_script()

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
                ghidra_import('strip_binaries', decompile_script),
                ghidra_import('debug_binaries', decompile_script),

                # TODO: look at debug binaries to see if we can use the member offsets
                # from this (check against DWARF debug info...maybe check against
                # dtlabels too if that makes sense?)

                # -----------------------------
                # TODO: look at variable aliasing in Ghidra debug_binaries (see OneNote notes)
                # >> how bad is this? (in my tiny data set compared to their 62%?)
                # -----------------------------

                # NOTE: much as I hate to say it, if the Ghidra/debug version decompilation
                # approach gets us what we need for pulling in member offsets and the
                # MEMORY ADDRESSES (and from there...AST nodes) they correspond to, we may
                # NOT need to do the dt-labels step, at least for this purpose.
                # - Good news is if we need to pull out other random info from clang, we have
                # all the plumbing right here ready to go

                # process_dt_labels(),

                extract_debuginfo_labels(),

                # calculate_similarity_metric(),
            ],
            postprocess_steps = [
            ])

        super().__init__('basic-dataset', algorithm, runconfigs,
            projectlist, exp_folder=exp_folder, params=exp_params)
