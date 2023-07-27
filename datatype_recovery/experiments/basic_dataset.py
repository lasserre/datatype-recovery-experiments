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

def do_extract_debuginfo_labels(run:Run, params:Dict[str,Any], outputs:Dict[str,Any]):
    console = Console()

    for bin_id, fb in outputs['flatten_binaries'].items():
        fb:FlatLayoutBinary
        ast_dumps = fb.data_folder/'ast_dumps'

        # pull in debug info for this binary
        with open(fb.debug_binary_file, 'rb') as f:
            ef = ELFFile(f)
            if ef.structs.e_type != 'ET_DYN':
                raise Exception(f'Need to handle potential non-PIE e_type "{ef.structs.e_type}" for {fb.debug_binary_file}')
            dwarf = ef.get_dwarf_info()
            init_pyelftools_from_dwarf(dwarf)

        ddi = DwarfDebugInfo(dwarf)

        # exclude the functions that had errors (log files)
        for ast_json in ast_dumps.glob('*.json'):

            # HACK TEMP TEMP TEMP
            # if ast_json.stem != 'FUN_00101b19':
            #     continue

            export_failed = (ast_dumps/f'{ast_json.stem}.log').exists()
            if export_failed:
                print(f'{fb.debug_binary_file.stem} AST export failed for function {ast_json.stem}')
                continue

            # option 1: pull address from filename (if we ever have a symbol that breaks it...idk if want that?)
            # faddr = int(ast_json.stem.split('_')[1],16)

            with open(ast_json) as f:
                data = json.load(f)

            ast, struct_lib = astlib.dict_to_ast(data)
            ast.render(format='svg', ast_name=ast_json.stem, outfolder=ast_json.parent)
                #format_node=highlight_binop,

            # top-level ast node is TranslationUnitDecl
            # ghidra_addr = int(ast.inner[-1].address, 16)   # currently string, but changing to be int
            ghidra_addr = ast.inner[-1].address
            dwarf_addr = ghidra_to_dwarf_addr(ghidra_addr)

            if dwarf_addr not in ddi.funcdies_by_addr:
                # this is ok in general - startup functions like _DT_INIT and _DT_FINI don't have debug info
                console.print(f'No debug info for function @ 0x{dwarf_addr:x} (ghidra name = {ast_json.stem}, ghidra addr = 0x{ghidra_addr:x})',
                            style='red')
                continue

            fdie = ddi.funcdies_by_addr[dwarf_addr]

            # fdie = ddi.get_function_dies()[0]
            # dwarf_addr = fdie.low_pc
            # ghidra_addr = fdie.low_pc + 0x100000

            print(f'Function {fdie.name} at {dwarf_addr:x} (ghidra address = {ghidra_addr:x})')


            # TODO: start with an AST function -> read in JSON using astlib
            # 2) in DWARF data, map variable addresses -> variable DIEs

            locals = ddi.get_function_locals(fdie)
            params = list(ddi.get_function_params(fdie))

            # TODO: this is probably about where I start diverging from the dtlabels/DWARF lines
            # approach - I'll go ahead and cut off that code here to get out of the way of what I'm
            # trying to do now. The lastest version of that code is in a branch dwarf_line_info or something

            # ----------------------
            # OLD...maybe we still want to try to find close matches??? not sure, let's see how it looks
            # ----------------------
            # TODO: build an AST visitor to collect nodes within address range (low, high)
            # -> find these nodes
            # -> print the matching nodes (individually and in tree structures)
            # -> print the AST graph (jupyter) and highlight these matching nodes
            #    (use AST address?)
            # ----------------------

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
                #
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
