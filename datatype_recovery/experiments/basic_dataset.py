from pathlib import Path
import hashlib
from io import StringIO
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
from .dwarflib import *

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

def do_extract_debuginfo_labels(run:Run, params:Dict[str,Any], outputs:Dict[str,Any]):
    console = Console()

    for bin_id, fb in outputs['flatten_binaries'].items():
        fb:FlatLayoutBinary
        debug_asts = fb.data['debug_asts']
        stripped_asts = fb.data['stripped_asts']
        print(f'processing binary: {fb.binary_file.name}')

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
            if ast_json_debug.stem != 'r_batch_add':
                continue

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
                matches = [l for l in intliterals if l.value == offset]
                sname = slib_debug[sid].name if sid in slib_debug else ''
                if isinstance(slib_debug[sid], astlib.UnionDef):
                    print(f'Skipping UNION type {sname}')
                    continue
                # mname = slib_debug[sid].fields_by_offset[offset].name if sid in slib_debug else ''
                mname = node.name
                print(f'{len(matches)} matches found for member access @ 0x{instr_addr:x} ({sname}.{mname})')
                if len(matches) > 1:
                    for x in matches:
                        print(f'{x.kind}: value={x.value} parent={x.parent}')
                elif not matches:
                    print(f'NO MATCH FOUND!')

                for m in matches:
                    m.IS_MEMBER_ACCESS = True

            render_ast(ast_debug, ast_json_debug.stem, ast_json_debug.parent, 'svg', 'MemberExpr')
            render_ast(ast, ast_json.stem, ast_json.parent, 'svg', 'MemberExpr')

            import IPython; IPython.embed()

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
