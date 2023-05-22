from pathlib import Path
from typing import List

from wildebeest import Experiment, RunConfig, ProjectRecipe
from wildebeest import DockerBuildAlgorithm, DefaultBuildAlgorithm

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
        clang_config = RunConfig('clang')
        clang_config.c_options.compiler_path = 'clang'
        clang_config.cpp_options.compiler_path = 'clang++'

        runconfigs = [gcc_config, clang_config]

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
                'RUN apt update && apt install -y gcc g++'
            ]
        }

        # algorithm = DefaultBuildAlgorithm(
        algorithm = DockerBuildAlgorithm(
            preprocess_steps = [
            ],
            post_build_steps = [
            ],
            postprocess_steps = [
            ])

        # TODO: add other .linker-objects steps (reuse) - can I build and get .linker-objects files generated?
        # TODO: now test if I can SWITCH COMPILERS! (gcc or clang) and everything still work with llvm-features linker :D

        super().__init__('basic-dataset', algorithm, runconfigs,
            projectlist, exp_folder=exp_folder, params=exp_params)
