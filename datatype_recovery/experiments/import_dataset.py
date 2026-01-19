from pathlib import Path
from typing import List, Tuple

from wildebeest import Experiment, RunConfig, ProjectRecipe, RunStep
from wildebeest.experimentalgorithm import ExperimentAlgorithm
from wildebeest.defaultbuildalgorithm import reset_data_folder
from wildebeest.postprocessing import find_binaries, flatten_binaries, strip_binaries, find_binaries_in_path
from wildebeest.postprocessing import ghidra_import
from wildebeest.preprocessing.ghidra import start_ghidra_server, create_ghidra_repo

import astlib
from ghidralib import export_asts

from .basic_dataset import extract_debuginfo_labels

class ImportDatasetExp(Experiment):
    def __init__(self,
        exp_folder:Path=None,
        runconfigs:List[RunConfig]=None,
        projectlist:List[ProjectRecipe]=[],
        bin_folder:str=None,
        strip_exe:str=None,
        params={}) -> None:
        '''
        exp_folder: The experiment folder
        projectlist: List of project recipes - UNUSED
        bin_folder: Folder containing debug binaries to be imported
        '''
        if bin_folder:
            bin_folder = Path(bin_folder)
            if not bin_folder.exists():
                raise Exception(f'bin_folder {bin_folder} does not exist')

            # 1 run per binary to execute in parallel
            binaries = find_binaries_in_path(bin_folder, no_cmake=False)
            runconfigs = [RunConfig(f'{i+1}.{x.name}', new_params={'binary': x.resolve()}) for i, x in enumerate(binaries)]
        else:
            runconfigs = [RunConfig()]     # just in case I need this

        projectlist = [ProjectRecipe('IMPORT', git_remote='')]  # wdb wants one to exist

        exp_params = {
            'GHIDRA_INSTALL': Path.home()/'software'/'ghidra_10.3_DEV',
        }

        if strip_exe:
            for rc in runconfigs:
                rc.strip_executable = strip_exe

        algorithm = ExperimentAlgorithm(
            preprocess_steps=[
                start_ghidra_server(),
                create_ghidra_repo(),
            ],
            steps=[
                # reset_data resets the data folder if it exists, so if we want to
                # clean and rerun postprocessing, this is the spot to run from
                RunStep('reset_data', reset_data_folder),

                find_binaries(import_binaries=True),
                flatten_binaries(),
                strip_binaries(run_in_docker=False),
                ghidra_import(debug=False, prescript=astlib.set_analysis_options_script()),
                ghidra_import(debug=True, prescript=astlib.set_analysis_options_script()),
                export_asts(debug=False),
                export_asts(debug=True),
                extract_debuginfo_labels(),
            ],
        )

        super().__init__('import-dataset', algorithm, runconfigs,
            projectlist, exp_folder=exp_folder, params=exp_params)
