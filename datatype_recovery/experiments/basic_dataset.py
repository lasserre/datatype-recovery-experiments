from pathlib import Path
from typing import List

from wildebeest import Experiment, RunConfig, ProjectRecipe
from wildebeest import DockerBuildAlgorithm, DefaultBuildAlgorithm
from wildebeest.postprocessing import find_binaries, flatten_binaries, strip_binaries
from wildebeest.postprocessing import ghidra_import
from wildebeest.preprocessing.ghidra import start_ghidra_server, create_ghidra_repo

import astlib

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


        # tmux new-window "<ghidra_install>/server/ghidraSvr console"
        # ---------------------------------------------
        # NOTE ghidra server should be configured with desired cmd-line options in
        # its server.conf:
        # ----------
        # wrapper.java.maxmemory = 16 + (32 * FileCount/10000) + (2 * ClientCount)
        # wrapper.java.maxmemory=XX   // in MB
        # ghidra.repositories.dir=/home/cls0027/ghidra_server_projects
        # <parameters>: -anonymous <ghidra.repositories.dir> OR
        #               -a0 -e0 -u <ghidra.repositories.dir>
        # ---------------------------------------------

        decompile_script = astlib.decompile_all_script()

        # algorithm = DefaultBuildAlgorithm(
        algorithm = DockerBuildAlgorithm(
            preprocess_steps=[
                start_ghidra_server(),
                create_ghidra_repo(),
            ],
            post_build_steps = [
                find_binaries(),
                flatten_binaries(),
                # find_instrumentation_files(['funcprotos']),
                # gen_truthprotos(),
                strip_binaries(),
                # TODO: pull true data types/categories from unstripped binary

                ghidra_import('strip_binaries', decompile_script),

                # TODO: combine AST data with true data type info...
                # 1) analysis questions (compare variable recovery, etc)
                # 2) compile GNN dataset...

                # ghidra_import('strip_binaries',
                #     ghidra_extract_script,
                #     get_extract_args),
                # calculate_similarity_metric(),
            ],
            postprocess_steps = [
            ])

        # TODO: add other .linker-objects steps (reuse) - can I build and get .linker-objects files generated?
        # TODO: now test if I can SWITCH COMPILERS! (gcc or clang) and everything still work with llvm-features linker :D

        super().__init__('basic-dataset', algorithm, runconfigs,
            projectlist, exp_folder=exp_folder, params=exp_params)
