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

    FIRST: try modifying the set itself...

    Returns a tuple of (success, fail) where pass is the set of AST paths that
    exported successfully and fail is the set of AST paths that failed to export
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

def build_dwarf_locals_table(debug_binary_file:Path) -> pd.DataFrame:
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

    for dwarf_addr, fdie in ddi.funcdies_by_addr.items():
        if fdie.artificial:
            print(f'Skipping artificial function {fdie.name} (intrinsic?)')
            continue

        print(fdie.name)

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
        # import IPython; IPython.embed()

    return pd.concat(locals_dfs)

def build_ast_locals_table(ast:astlib.ASTNode):
    '''
    Build the local variables table for the given AST

    Ghidra Function Addr | Var Name? | Location | Type | Type Category
    '''
    fdecl = ast.inner[-1]
    fbody = fdecl.inner[-1]
    if fbody.kind != 'CompoundStmt':
        # no function body -> no locals
        return pd.DataFrame()

    print(f'Bulding AST locals for function: {fdecl.name}')

    local_decls = itertools.takewhile(lambda node: node.kind == 'DeclStmt', fbody.inner)
    local_vars = [decl_stmt.inner[0] for decl_stmt in local_decls]
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


def build_localvars_table(fb:FlatLayoutBinary):
    '''
    Build our master table of local variables
    '''
    # Function | Binary |
    # ... DWARF local var | Stripped AST local var | Debug AST local var | Stripped Function AST

    dwarf_locals = build_dwarf_locals_table(fb.debug_binary_file)

    debug_asts = fb.data['debug_asts']
    stripped_asts = fb.data['stripped_asts']

    # collect functions for this binary, partitioned into debug/stripped sets
    stripped_funcs = set(stripped_asts.glob('*.json'))
    debug_funcs = set(debug_asts.glob('*.json'))

    # import IPython; IPython.embed()

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

    # TODO: collect all the functions up into a pandas dataframe
    # show a rich table with:
    # - total # of functions (raw)
    # - # of uniquely-named functions (we want to avoid duplicates)
        # - ok, multiple "main" functions are ok...
        # - are any other duplicates ok?
        # - put in logic to allow "ok duplicates"??
    # - compute "% yield" based on this

    locals_dfs = []

    for ast_json_debug in sorted(debug_funcs):
        ast_debug, slib_debug = astlib.json_to_ast(ast_json_debug)
        locals_dfs.append(build_ast_locals_table(ast_debug))

    debug_df = pd.concat(locals_dfs)

    locals_dfs = []
    for ast_json in sorted(stripped_funcs):
        ast, slib = astlib.json_to_ast(ast_json)
        locals_dfs.append(build_ast_locals_table(ast))

    stripped_df = pd.concat(locals_dfs)

    df = debug_df.merge(stripped_df, how='outer',
                        on=['FunctionStart','LocType','LocRegName','LocOffset'],
                        suffixes=['_Debug','_Strip'])

    # df = debug_df.merge(stripped_df, how='outer',
    #                     on=['FunctionStart','Location'],
    #                     suffixes=['Debug','Strip'])

    # NOTE: USE PARENTHESES FOR FILTERING!!!
    # df[(df.FunctionStart==0x1e7af0) & (df.NameDebug.isna())]

    # TODO: convert DWARF variables to this format, join with AST vars (AST suffix = '', DWARF = 'NameDWARF')
    # TODO: start printing stats in a rich table:
    # - totals (# binaries, # functions, # vars)
    # - totals vars breakout: (# vars by category, storage class (LocType))

    # TODO: add in function parameters...maybe a separate table?
    # TODO: once I get all this working, then add coreutils to my dataset and compute
    # descriptive stats/charts showing the composition of this dataset
    # (to help understand complexity of published prior work)
    # TODO: build the first/basic model using the chosen subset of the data (true vars?)

    # TODO: join dwarf vars in with df...

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

    df['BinaryId'] = fb.id

    ######
    # maybe move these to the end/outside?
    # - assemble tables (save them off)
    # - run analysis to produce result tables/charts/etc
    num_dwarf_locals = len(df[~df.TypeCategory_DWARF.isna()])
    num_debug_locals = len(df[~df.TypeCategory_Debug.isna()])
    num_strip_locals = len(df[~df.TypeCategory_Strip.isna()])

    df['TrueDebugVar'] = (~df.TypeCategory_Debug.isna()) & (~df.TypeCategory_DWARF.isna())
    df['TrueStripVar'] = (~df.TypeCategory_Strip.isna()) & (~df.TypeCategory_DWARF.isna())

    df['Size_DWARF'] = df.Type_DWARF.apply(lambda x: x.size if pd.notna(x) else -1)
    df['Size_Debug'] = df.Type_Debug.apply(lambda x: x.size if pd.notna(x) else -1)
    df['Size_Strip'] = df.Type_Strip.apply(lambda x: x.size if pd.notna(x) else -1)

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

    # TODO: rerun and test str(DataType)
    # TODO: convert Type columns to their string representation and save to CSV
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


def do_extract_debuginfo_labels(run:Run, params:Dict[str,Any], outputs:Dict[str,Any]):
    console = Console()

    locals_dfs = []
    for bin_id, fb in outputs['flatten_binaries'].items():
        fb:FlatLayoutBinary
        print(f'processing binary: {fb.binary_file.name}')

        loc_df = build_localvars_table(fb)
        locals_dfs.append(loc_df)

        # temp_member_expression_logic(fb)

    loc_df = pd.concat(locals_dfs)

    import IPython; IPython.embed()

    # how well did Ghidra recover variables?
    # -> variable recovered when stripped variable exists at the same location as a
    #    true variable
    #    -> what percentage of the true variables did Ghidra recover?

    # TODO: replace this with ~loc_df.NameDWARF.isna() to have "real" truth vars
    true_vars = loc_df[~loc_df.Name_Debug.isna()]    # non-null debug name indicates true variable here
    recov_true_vars = true_vars[~true_vars.Name_Strip.isna()]  # true vars that ALSO have a stripped variable here

    # we can calculate overall stats, or use groupby first to compute over interesting
    # subsets (like per binary...)
    recov_by_typecat = recov_true_vars.groupby('TypeCategory_Debug').count()/true_vars.groupby('TypeCategory_Debug').count()

    import IPython; IPython.embed()

    return

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
