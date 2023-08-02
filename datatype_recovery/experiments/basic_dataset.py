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
    def highlight_kind(node, attrs):
        if node.kind == highlight_kind:
            attrs.font_color = highlight_color
    ast.render(format=format, ast_name=ast_name, outfolder=outfolder, format_node=highlight_kind)

def separate_ast_fails(ast_paths:set) -> set: #-> Tuple[set, set]:
    '''Checks for AST export failures (based on .log file presence) among the
    given set of AST export Paths, and partitions the set into failures and no failures.

    FIRST: try modifying the set itself...

    Returns a tuple of (success, fail) where pass is the set of AST paths that
    exported successfully and fail is the set of AST paths that failed to export
    '''
    fails = set([x for x in ast_paths if x.with_suffix('log').exists()])
    ast_paths -= fails
    return fails

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

        stripped_by_addr = {int(x.stem[4:], 16): x for x in stripped_funcs}

        # separate the functions that had errors (log files)
        print(f'# stripped funcs before = {len(stripped_funcs)}')
        stripped_fails = separate_ast_fails(stripped_funcs)
        print(f'# stripped funcs after = {len(stripped_funcs)}')
        print(f'# stripped fails = {len(stripped_fails)}')

        # stripped_fails = set([x for x in stripped_funcs if x.with_suffix('log').exists()])
        debug_fails = set([x for x in debug_funcs if x.with_suffix('log').exists()])
        debug_funcs -= debug_fails
        # stripped_funcs -= stripped_fails

        for ast_json_debug in debug_funcs:
            ast_debug, slib_debug = astlib.json_to_ast(ast_json_debug)

            # top-level ast node is TranslationUnitDecl
            # ghidra_addr = int(ast.inner[-1].address, 16)   # currently string, but changing to be int

            funcdbg_addr_gh = ast_debug.inner[-1].address
            funcdbg_addr_df = ghidra_to_dwarf_addr(ghidra_addr)

            # todo: replace w/ above
            ghidra_addr = ast_debug.inner[-1].address
            dwarf_addr = ghidra_to_dwarf_addr(ghidra_addr)

            if funcdbg_addr_df not in ddi.funcdies_by_addr:
                # this is ok in general - startup functions like _DT_INIT and _DT_FINI don't have debug info
                console.print(f'No debug info for function @ 0x{funcdbg_addr_df:x} (ghidra name = {ast_json_debug.stem}, ghidra addr = 0x{funcdbg_addr_gh:x})',
                            style='red')
                continue

            fdie = ddi.funcdies_by_addr[dwarf_addr]

            print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            print(f'AST for function {ast_debug.inner[-1].name}')
            print(f'Function {fdie.name} at {dwarf_addr:x} (ghidra address = {funcdbg_addr_gh:x})')


            # TODO: start with an AST function -> read in JSON using astlib
            # 2) in DWARF data, map variable addresses -> variable DIEs

            locals = ddi.get_function_locals(fdie)
            params = list(ddi.get_function_params(fdie))

            # TODO: this is probably about where I start diverging from the dtlabels/DWARF lines
            # approach - I'll go ahead and cut off that code here to get out of the way of what I'm
            # trying to do now. The lastest version of that code is in a branch dwarf_line_info or something

            ast_json = stripped_by_addr[funcdbg_addr_gh]
            ast, slib = astlib.json_to_ast(ast_json)

            render_ast(ast_debug, ast_json_debug.stem, ast_json_debug.parent, 'svg', 'MemberExpr')
            render_ast(ast, ast_json.stem, ast_json.parent, 'svg', 'MemberExpr')

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
                dump_dt_labels()
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
                process_dt_labels(),
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
