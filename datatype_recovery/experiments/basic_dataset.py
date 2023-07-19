from pathlib import Path
import hashlib
from io import StringIO
import json
import pandas as pd
from rich.console import Console
import shutil
import sys
from typing import List

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

            # ---------------------------------------------------
            # LABEL MEMBER EXPRESSIONS
            # ---------------------------------------------------
            #
            # TODO: use this lookup with the SOURCE AST file/line/col triples
            # (NOTE: files are full/canonical path strings) to map each
            # SOURCE member expression -> address
            # - then, go through AST nodes and find all nodes at the same address
            # of the source member expression.
            # - are there tons of spurious matches?
            # - can we reduce this to find only legit matches?

            # ddi._build_lineinfo_lookup()
            # ddi.lineinfo_lookup

            for dtl in (run.data_folder/'dtlabels').iterdir():
                df = pd.read_csv(dtl)
                # list of Tuple(filename, line #, col #) for each member expression
                member_expr_lineinfos = [(x,*[int(z) for z in y.split(':')]) for x, y in zip(df['Filename'], df['Loc'])]

            # TEST lookups...
            ddi._build_lineinfo_lookup()

            # TODO: reduce to relevant FILE
            # TODO: sort by line/col
            # TODO: find match (<= line that is closest to line)
            # -> so put in a DF, filter <= line and then take the max line val
            # (if multiple max lines, take MIN column)
            # TODO: then find the very next LINE (not same line bigger column)
            # and use these two as min/max addresses
            # TODO: walk the AST and print all nodes that match this address range
            # --> any way we can use this??

            # I could jump straight to trying LLVM metadata approach
            # (attache metadata to member expressions I already have, dump these out
            # at the end once they are bound to machine addresses)
            # but I have my doubts as to if the metadata will survive correctly
            # that long
            # NOTE: it very well could, this is what it was designed to do...you
            # just never know with compiler optimizations...
            k = list(ddi.lineinfo_lookup.keys())[1]
            k2 = list(ddi.lineinfo_lookup.keys())[2]

            closest_addr_dwarf = ddi.lineinfo_lookup[k]
            closest_addr = dwarf_to_ghidra_addr(closest_addr_dwarf)

            # map file: {line: [cols]}
            byfile = {}
            for k, v in ddi.lineinfo_lookup.items():
                if k[0] not in byfile:
                    byfile[k[0]] = {}
                if k[1] not in byfile[k[0]]:
                    byfile[k[0]][k[1]] = []
                byfile[k[0]][k[1]].append(k[2])

            # TODO: try bounding with closest and next line?
            for mem_expr_li in member_expr_lineinfos:
                filename, line, col = mem_expr_li
                lcdict = byfile[filename]

                # find previous location
                low_line = max([k for k in lcdict.keys() if k <= line])
                low_col = max([c for c in lcdict[low_line] if c <= col])

                higher_cols = [c for c in lcdict[low_line] if c > col]
                if higher_cols:
                    high_line = low_line    # same line
                    high_col = min(higher_cols)
                else:
                    # take first col of next line
                    high_line = min([k for k in lcdict.keys() if k > line])
                    high_col = lcdict[high_line][0]

                low_addr = dwarf_to_ghidra_addr(ddi.lineinfo_lookup[(filename, low_line, low_col)])
                high_addr = dwarf_to_ghidra_addr(ddi.lineinfo_lookup[(filename, high_line, high_col)])

                print(f'Member expr loc: {line}:{col}')
                print(f'LOW BOUND: {low_line}:{low_col} (0x{low_addr:x})')
                print(f'HIGH BOUND: {high_line}:{high_col} (0x{high_addr:x})')
                print(f'Address Delta = {high_addr - low_addr}')
                print('-------------------')

                # TODO: build an AST visitor to collect nodes within address range (low, high)
                # -> find these nodes
                # -> print the matching nodes (individually and in tree structures)
                # -> print the AST graph (jupyter) and highlight these matching nodes
                #    (use AST address?)

            import IPython; IPython.embed()

def extract_debuginfo_labels() -> RunStep:
    return RunStep('extract_debuginfo_labels', do_extract_debuginfo_labels)

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
            # rc.c_options.enable_debug_info()
            # rc.cpp_options.enable_debug_info()
            rc.c_options.compiler_flags.append('-g3')
            rc.cpp_options.compiler_flags.append('-g3')

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
                # TODO: pull true data types/categories from unstripped binary

                ghidra_import('strip_binaries', decompile_script),

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
