from pathlib import Path
from io import StringIO
import json
from rich.console import Console
from typing import List
import sys

from wildebeest import Experiment, RunConfig, ProjectRecipe
from wildebeest import DockerBuildAlgorithm, DefaultBuildAlgorithm
from wildebeest.postprocessing import find_binaries, flatten_binaries, strip_binaries
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

            import IPython; IPython.embed()

def extract_debuginfo_labels() -> RunStep:
    return RunStep('extract_debuginfo_labels', do_extract_debuginfo_labels)

def do_dump_source_ast(run:Run, params:Dict[str,Any], outputs:Dict[str,Any]):
    from wildebeest.defaultbuildalgorithm import build

    orig_compiler_path = run.config.c_options.compiler_path
    orig_compiler_flags = [f for f in run.config.c_options.compiler_flags]

    # clang -Xclang -ast-dump=json -fsyntax-only C_FILE > out.json
    run.config.c_options.compiler_flags.extend(['-Xclang', '-ast-dump=json', '-fsyntax-only'])

    # run.config.c_options.compiler_path = '/home/cls0027/software/llvm-features-12.0.1/bin/clang'
    run.config.c_options.compiler_path = 'clang'    # force clang

    stdout_capture = StringIO()
    try:
        sys.stdout = stdout_capture
        build(run, params, outputs)     # run the build driver...
        sys.stdout = sys.__stdout__     # Reset stdout to its original value
    except Exception as e:
        print("An exception occurred:", str(e))
        sys.stdout = sys.__stdout__

    # Get the captured stdout as a string
    captured_output = stdout_capture.getvalue()
    dump_ast_file = run.build.build_folder/'dump_ast_output.txt'
    # dump_ast_file.mkdir(exist_ok=True, parents=True)
    # run.data_folder.mkdir(exist_ok=True, parents=True)
    with open(dump_ast_file, 'w') as f:
        f.write(captured_output)

    # put the original options back
    run.config.c_options.compiler_flags = orig_compiler_flags
    run.config.c_options.compiler_path = orig_compiler_path

    return dump_ast_file

def dump_source_ast() -> RunStep:
    return RunStep('dump_source_ast', do_dump_source_ast, run_in_docker=True)

def do_process_source_ast_dump(run:Run, params:Dict[str,Any], outputs:Dict[str,Any]):
    print('READY!')
    # import IPython; IPython.embed()

def process_source_ast_dump() -> RunStep:
    return RunStep('process_source_ast_dump', do_process_source_ast_dump)

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
            pre_build_steps=[
                dump_source_ast()
            ],
            post_build_steps = [
                find_binaries(),
                flatten_binaries(),
                strip_binaries(),
                # TODO: pull true data types/categories from unstripped binary

                ghidra_import('strip_binaries', decompile_script),

                process_source_ast_dump(),
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
