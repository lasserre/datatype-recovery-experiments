from pathlib import Path
import hashlib
from io import StringIO
import itertools
import json
import pandas as pd
from rich.console import Console
import shutil
import sys
from typing import List, Tuple

from wildebeest import Experiment, RunConfig, ProjectRecipe
from wildebeest import DockerBuildAlgorithm, DefaultBuildAlgorithm
from wildebeest.postprocessing import find_binaries, flatten_binaries, strip_binaries, find_instrumentation_files
from wildebeest.postprocessing import ghidra_import
from wildebeest.preprocessing.ghidra import start_ghidra_server, create_ghidra_repo
from wildebeest import *
from wildebeest.run import Run

from wildebeest.postprocessing.flatlayoutbinary import FlatLayoutBinary

import astlib
from varlib.location import Location
from varlib.datatype import *
from dwarflib import *

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
    '''
    # pull in debug info for this binary
    ddi = DwarfDebugInfo.fromElf(debug_binary_file)

    # NOTE: I can't think of a particularly good reason for doing this one way over
    # another, so arbitrarily deciding to use Ghidra addresses for consistency
    # (if nothing else, I will be looking at Ghidra much more often than DWARF data)

    locals_dfs = []
    func_names = []
    func_starts = []
    params_df_list = []

    for dwarf_addr, fdie in ddi.funcdies_by_addr.items():
        if fdie.artificial:
            print(f'Skipping artificial function {fdie.name} (intrinsic?)')
            continue
        if fdie.inline:
            print(f'Skipping inlined function {fdie.name}')
            continue

        ### Locals
        locals = ddi.get_function_locals(fdie)
        locations = [l.location_varlib for l in locals]

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

        locals_dfs.append(df)

        ### Functions
        func_names.append(fdie.name)
        func_starts.append(dwarf_to_ghidra_addr(dwarf_addr))

        ### Function prototype
        params = list(ddi.get_function_params(fdie))
        param_locs = [p.location_varlib for p in params]

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
        self.func_df:pd.DataFrame = None    # only 1 row expected, but ready to pd.concat()
        self.params_df:pd.DataFrame = None
        self.locals_df:pd.DataFrame = None
        self.globals_accessed_df:pd.DataFrame = None

def extract_funcdata_from_ast(ast:astlib.ASTNode) -> FunctionData:

    fdecl = ast.inner[-1]
    fbody = fdecl.inner[-1]

    # prototype
    params = fdecl.inner[:-1]
    return_type = fdecl.return_dtype

    # locals
    local_decls = itertools.takewhile(lambda node: node.kind == 'DeclStmt', fbody.inner)
    local_vars = [decl_stmt.inner[0] for decl_stmt in local_decls]

    fd = FunctionData()
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
    df = pd.DataFrame({
        'FunctionStart': pd.array([fdecl.address] * len(params), dtype=pd.UInt64Dtype()),
        'Name': [p.name for p in params],
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
        'IsReturnType': pd.array([True], dtype=pd.BooleanDtype()),
        'Type': [return_type.dtype_varlib],
        'LocType': [return_type.location.loc_type if return_type.location else None],
        'LocRegName': [return_type.location.reg_name if return_type.location else None],
        'LocOffset': pd.array([return_type.location.offset if return_type.location else None],
                              dtype=pd.Int64Dtype()),
    })], ignore_index=True)

    df['TypeCategory'] = [t.category for t in df.Type]

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
        'Type': [v.dtype.dtype_varlib for v in local_vars],
        # 'Location': [v.location for v in local_vars]
        'LocType': [v.location.loc_type if v.location else None for v in local_vars],
        'LocRegName': [v.location.reg_name if v.location else None for v in local_vars],
        'LocOffset': pd.array([v.location.offset if v.location else None for v in local_vars],
                              dtype=pd.Int64Dtype()),
    })

    df['TypeCategory'] = [t.category for t in df.Type]

    return df

def collect_passing_asts(fb:FlatLayoutBinary):
    '''Collect the debug and stripped ASTs that did not have failures'''
    debug_asts = fb.data['debug_asts']
    stripped_asts = fb.data['stripped_asts']

    # collect functions for this binary, partitioned into debug/stripped sets
    stripped_funcs = set(stripped_asts.glob('*.json'))
    debug_funcs = set(debug_asts.glob('*.json'))

    # separate the functions that had errors (log files)
    stripped_fails = separate_ast_fails(stripped_funcs)
    debug_fails = separate_ast_fails(debug_funcs)
    print(f'# stripped fails = {len(stripped_fails)}')
    print(f'# debug fails = {len(debug_fails)}')
    return (debug_funcs, stripped_funcs)

def extract_funcdata_from_ast_set(ast_funcs:Set[Path]) -> List[FunctionData]:
    '''Extracts FunctionData content from each of the provided ASTs'''
    fdatas:List[FunctionData] = []
    num_funcs = len(ast_funcs)
    for i, ast_json in enumerate(sorted(ast_funcs)):
        if (i+1) % 500 == 0:
            print(f'{i+1}/{num_funcs} ({(i+1)/num_funcs*100:.0f}%)...')
        ast, slib = astlib.json_to_ast(ast_json)
        fdatas.append(extract_funcdata_from_ast(ast))
    return fdatas

def extract_data_tables(fb:FlatLayoutBinary):
    '''
    Build our master table of local variables
    '''
    # Function | Binary |
    # ... DWARF local var | Stripped AST local var | Debug AST local var | Stripped Function AST

    debug_funcs, stripped_funcs = collect_passing_asts(fb)

    # extract data from ASTs/DWARF debug symbols
    print(f'Extracting data from {len(debug_funcs):,} debug ASTs for binary {fb.debug_binary_file.name}...')
    debug_funcdata = extract_funcdata_from_ast_set(debug_funcs)
    print(f'Extracting data from {len(stripped_funcs):,} stripped ASTs for binary {fb.binary_file.name}...')
    stripped_funcdata = extract_funcdata_from_ast_set(stripped_funcs)

    # NOTE when we need more DWARF data, extract it all at once while we have
    # the file with DWARF debug info opened

    print(f'Extracting DWARF data for binary {fb.debug_binary_file.name}...')
    dwarf_tables = build_dwarf_data_tables(fb.debug_binary_file)

    ### Locals
    locals_df = build_locals_table(debug_funcdata, stripped_funcdata, dwarf_tables.locals_df)
    locals_df['BinaryId'] = fb.id
    locals_df.to_csv(fb.data_folder/'locals.csv', index=False)

    ### Functions
    funcs_df = build_funcs_table(debug_funcdata, stripped_funcdata, dwarf_tables.funcs_df)
    funcs_df['BinaryId'] = fb.id
    funcs_df.to_csv(fb.data_folder/'functions.csv', index=False)

    ### Function Parameters (prototype)
    params_df = build_params_table(debug_funcdata, stripped_funcdata, dwarf_tables.params_df)
    params_df['BinaryId'] = fb.id
    params_df.to_csv(fb.data_folder/'function_params.csv', index=False)

def build_params_table(debug_funcdata:List[FunctionData], strip_funcdata:List[FunctionData],
                      dwarf_df:pd.DataFrame):
    debug_df = pd.concat(fd.params_df for fd in debug_funcdata)
    strip_df = pd.concat(fd.params_df for fd in strip_funcdata)

    # Because DWARF data doesn't have a Loc for return types, we need to join return types
    # differently than parameters. Thus:
    # ------------------------------
    # 1. Merge Debug/Strip as normal
    # 2. Divide the RESULTING df into params/return types (df_params, df_rtypes)
    # 3. Merge df_params and dwarf_params as normal
    # 4. Merge df_rtypes and dwarf_rtypes on FunctionStart/IsReturnType only
    #    (probably want to delete the DWARF Loc columns first since they are
    #     always None...just take w/e df_rtypes has for Loc)
    # 5. Concat the two merged sets (params and return types) into a single df

    df = debug_df.merge(strip_df, how='outer',
                        on=['FunctionStart','LocType','LocRegName','LocOffset','IsReturnType'],
                        suffixes=['_Debug','_Strip'])

    # dwarf doesn't have any location for return type, so we need to split up
    # the dataframe into params and return types and do it separately
    dwarf_rtypes = dwarf_df.loc[dwarf_df.IsReturnType,:]
    dwarf_params = dwarf_df.loc[~dwarf_df.IsReturnType,:]

    # divide strip/debug-joined data into params/return types
    df_rtypes = df.loc[df.IsReturnType,:]
    df_params = df.loc[~df.IsReturnType,:]

    # 3. merge debug/strip params and dwarf params as normal
    df_params = df_params.merge(dwarf_params, how='outer',
                    on=['FunctionStart','LocType','LocRegName','LocOffset'],
                    suffixes=[None, '_DWARF'])

    # 4. merge debug/strip rtypes and dwarf rtypes on FunctionStart/IsReturnType only
    # (after deleting dwarf loc columns)
    dwarf_rtypes = dwarf_rtypes.drop(columns=['LocType','LocRegName','LocOffset'])

    df_rtypes = df_rtypes.merge(dwarf_rtypes, how='outer',
                                on=['FunctionStart'],
                                suffixes=[None, '_DWARF'])

    # 5. concat the two merged sets
    df = pd.concat([df_params, df_rtypes], ignore_index=True)

    rename_cols = {
        'Name': 'Name_DWARF',
        'Type': 'Type_DWARF',
        'TypeCategory': 'TypeCategory_DWARF',
    }
    df.rename(columns=rename_cols, inplace=True)

    # import IPython; IPython.embed()
    return df


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
    })

    strip_df = pd.DataFrame({
        'FunctionStart': [f.address for f in strip_funcdata],
        'FunctionName': [f.name for f in strip_funcdata],
    })

    df = debug_df.merge(strip_df, on='FunctionStart', how='outer', suffixes=['_Debug','_Strip'])
    df = df.merge(dwarf_funcs, on='FunctionStart', how='outer', suffixes=[None, '_DWARF'])
    df.rename(columns={'FunctionName': 'FunctionName_DWARF'}, inplace=True)

    return df

def build_locals_table(debug_funcdata:List[FunctionData], stripped_funcdata:List[FunctionData],
                       dwarf_locals:pd.DataFrame):
    ## Locals table

    # combine into single df
    debug_locals = pd.concat([fd.locals_df for fd in debug_funcdata])
    stripped_locals = pd.concat([fd.locals_df for fd in stripped_funcdata])

    # merge debug/stripped
    df = debug_locals.merge(stripped_locals, how='outer',
                        on=['FunctionStart','LocType','LocRegName','LocOffset'],
                        suffixes=['_Debug','_Strip'])

    # merge both with dwarf
    df = df.merge(dwarf_locals, how='outer',
             on=['FunctionStart', 'LocType', 'LocRegName', 'LocOffset'],
             suffixes=[None, '_DWARF'])

    # suffixes doesn't catch these since _Debug and _Strip have already been added
    # to the first two dataframes
    df.rename(columns={
        'Name': 'Name_DWARF',
        'Type': 'Type_DWARF',
        'TypeCategory': 'TypeCategory_DWARF',
        }, inplace=True)

    df['TrueDebugVar'] = (~df.TypeCategory_Debug.isna()) & (~df.TypeCategory_DWARF.isna())
    df['TrueStripVar'] = (~df.TypeCategory_Strip.isna()) & (~df.TypeCategory_DWARF.isna())

    df['Size_DWARF'] = df.Type_DWARF.apply(lambda x: x.size if pd.notna(x) else -1)
    df['Size_Debug'] = df.Type_Debug.apply(lambda x: x.size if pd.notna(x) else -1)
    df['Size_Strip'] = df.Type_Strip.apply(lambda x: x.size if pd.notna(x) else -1)

    return df

    # TODO: collect all the functions up into a pandas dataframe
    # show a rich table with:
    # - total # of functions (raw)
    # - # of uniquely-named functions (we want to avoid duplicates)
        # - ok, multiple "main" functions are ok...
        # - are any other duplicates ok?
        # - put in logic to allow "ok duplicates"??
    # - compute "% yield" based on this

    # TODO: add in function parameters...maybe a separate table?
    # TODO: move this into characterize_dataset functions
    # TODO: start printing stats in a rich table:
    # - totals (# binaries, # functions, # vars)
    # - totals vars breakout: (# vars by category, storage class (LocType))

    ######
    # maybe move these to the end/outside?
    # - assemble tables (save them off)
    # - run analysis to produce result tables/charts/etc
    num_dwarf_locals = len(df[~df.TypeCategory_DWARF.isna()])
    num_debug_locals = len(df[~df.TypeCategory_Debug.isna()])
    num_strip_locals = len(df[~df.TypeCategory_Strip.isna()])

    num_true_debug_locals = len(df[df.TrueDebugVar])
    num_true_debug_locals = len(df[(~df.TypeCategory_Debug.isna()) & (~df.TypeCategory_DWARF.isna())])
    num_true_strip_locals = len(df[(~df.TypeCategory_Strip.isna()) & (~df.TypeCategory_DWARF.isna())])

    num_extra_strip_locals = num_strip_locals - num_true_strip_locals
    num_extra_debug_locals = num_debug_locals - num_true_debug_locals

    # example plot
    import seaborn as sns
    sns.set_style()

    # ax = df.groupby('TypeCategory_DWARF').count().Name_DWARF.plot.bar(zorder=3)

    # do separate group bys so we count ALL of the vars for each source (DWARF, debug, etc)
    # independently - we don't want a multi-column group by
    gb1 = df.groupby('TypeCategory_DWARF').count().Name_DWARF
    gb2 = df.groupby('TypeCategory_Debug').count().Name_Debug
    gb3 = df.groupby('TypeCategory_Strip').count().Name_Strip

    categories_df = pd.DataFrame({
        'DWARF': gb1,
        'Debug': gb2,
        'Strip': gb3
    })

    cats_pcnt_df = pd.DataFrame({
        'DWARF': gb1/num_dwarf_locals,
        'Debug': gb2/num_debug_locals,
        'Strip': gb3/num_strip_locals
    })

    ax = categories_df.plot.bar(zorder=3)

    ax.set_title(f'Local var categories ({fb.binary_file.name})')
    ax.set_ylabel('# Locals')
    for c in ax.containers:
        ax.bar_label(c, fmt=lambda x: f'{int(x):,}')

    ax.figure.savefig('plot.png', bbox_inches='tight')

    # TODO: take out this IPython.embed(), create a Jupyter notebook to drive plots
    # TODO: develop a set of plots/tables for dataset characterization (use astera for now)
    # TODO: although the model will "mean nothing" using only astera as a dataset...
    # go ahead and BUILD A DUMMY MODEL in PyG so I can see what's required to get
    # it actually set up end-to-end
    # - develop what my dataset output looks like for the model
    # - start figuring out actual model input format
    # - figure out GNN (pyg) api and how to do this...
    # - test it out and see what performance I get...
    # - then go back and develop an actual training set and do a proper experiment
    import IPython; IPython.embed()

    return df

def combine_fb_tables_into_rundata(run:Run, bin_list:List[FlatLayoutBinary], csv_name:str):
    '''
    Read each of the pandas tables (in file csv_name) from the list of flat binary folders
    and combine them into a single run-level data frame, writing it to the run data folder
    '''
    combined_df = pd.concat(pd.read_csv(fb.data_folder/csv_name) for fb in bin_list)
    combined_df.to_csv(run.data_folder/csv_name, index=False)

def do_extract_debuginfo_labels(run:Run, params:Dict[str,Any], outputs:Dict[str,Any]):
    console = Console()

    locals_dfs = []
    for bin_id, fb in outputs['flatten_binaries'].items():
        fb:FlatLayoutBinary
        console.rule(f'Processing binary: [bold]{fb.binary_file.name}')
        extract_data_tables(fb)
        # temp_member_expression_logic(fb)

        # console.log('[bold yellow]WARNING [normal] temporarily skipping other binaries...')
        # break

    # combine into unified files
    flat_bins = outputs['flatten_binaries'].values()

    bins_df = pd.DataFrame([(fb.id, fb.binary_file.name) for fb in flat_bins], columns=['BinaryId', 'Name'])
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
            # rc.c_options.compiler_flags.append('-g3')
            # rc.cpp_options.compiler_flags.append('-g3')

        exp_params = {
            'exp_docker_cmds': [
                # install ourselves into docker :)
                'RUN pip install --upgrade pip',
                'RUN --mount=type=ssh pip install git+ssh://git@github.com/lasserre/datatype-recovery-experiments.git',
                'RUN apt update && apt install -y gcc g++ clang'
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

                # - dtlabels: we should know a **superset** of member accesses within
                # a function (not an exact set - some code could be removed)
                # ...also, maybe not a superset...loop unrolling will produce >1
                # member access for the unrolled iterations

                # TODO: also add a step somewhere in here (optionally) to dump the
                # AST graph for each function (using latest updates in astlib)
                # - we shouldn't have to have this long term...we can graph it on the fly...
                # but it might be nice to have a bunch sitting here I can click through to find
                # interesting example functions!
                # NOTE: implement this as a function called graph_asts(folder) or plot_asts(folder)
                # where it accepts a folder location (and possibly recursively does this?)
                #   e.g. glob('**/*.json')
                # and if I need to do it on the fly later after I remove it, I can quickly
                # get the whole folder of data plotted/graphed

                # NOTE: much as I hate to say it, if the Ghidra/debug version decompilation
                # approach gets us what we need for pulling in member offsets and the
                # MEMORY ADDRESSES (and from there...AST nodes) they correspond to, we may
                # NOT need to do the dt-labels step, at least for this purpose.
                # - Good news is if we need to pull out other random info from clang, we have
                # all the plumbing right here ready to go

                # process_dt_labels(),

                extract_debuginfo_labels(),
                # TODO: combine AST data with true data type info...
                # 1) analysis questions (compare variable recovery, etc)
                # 2) compile GNN dataset...

                # calculate_similarity_metric(),
            ],
            postprocess_steps = [
            ])

        super().__init__('basic-dataset', algorithm, runconfigs,
            projectlist, exp_folder=exp_folder, params=exp_params)
