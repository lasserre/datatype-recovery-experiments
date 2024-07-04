# DRAGON RYDER
# -------------
# DRAGON incremental RetYping DrivER

import pandas as pd
from pathlib import Path
from rich.console import Console
from tqdm import tqdm
from typing import List, Tuple

from datatype_recovery.models.eval import make_predictions_on_dataset
from datatype_recovery.models.dataset import load_dataset_from_path

from astlib import read_json_str
from varlib.datatype import DataType

import pyhidra
pyhidra.start()

from ghidralib.projects import *
from ghidralib.ghidraretyper import GhidraRetyper

class DragonRyder:
    def __init__(self, dragon_model_path:Path, repo_name:str,
                device:str='cpu',
                resume:bool=False, numrefs_thresh:int=5,
                rollback_delete:bool=False,
                ghidra_server:str='localhost', ghidra_port:int=13100,
                confidence_strategy:str='refs',
                binary_list:List[str]=None) -> None:
        self.dragon_model_path = dragon_model_path
        self.repo_name = repo_name
        self.device = device
        self.resume = resume
        self.numrefs_thresh = numrefs_thresh
        self.rollback_delete = rollback_delete
        self.ghidra_server = ghidra_server
        self.ghidra_port = ghidra_port
        self.confidence_strategy = confidence_strategy
        self.binary_list = binary_list if binary_list else []

        self._shared_proj = None
        self.console = Console()
        self.bin_files:List[DomainFile] = []     # generate this from binary_list or all files in the repo

        # self.dataset = load_dataset_from_path(dataset_path)
        # self._check_dataset()   # ensure dataset is valid for dragon-ryder (not balanced, etc.)

    @property
    def ryder_folder(self) -> Path:
        # for simplicity/sanity, create a NAME.dragon-ryder folder to store all the intermediate data
        # - this will help easily isolate independent runs, help debugging, etc. especially bc I'm going fast
        # assumes cwd
        return Path(f'{self.repo_name}-{self.dragon_model_path.name}.dragon-ryder')

    @property
    def binary_paths(self) -> Path:
        '''Ghidra file paths for selected binaries in the repo'''
        return self.ryder_folder/'binary_paths.csv'

    @property
    def dragon_initial_preds(self) -> Path:
        return self.ryder_folder/'dragon_initial_preds.csv'

    @property
    def high_confidence_vars(self) -> Path:
        return self.ryder_folder/'high_conf_vars.csv'

    @property
    def low_confidence_vars(self) -> Path:
        # everything but high confidence vars...
        return self.ryder_folder/'low_conf_vars.csv'

    @property
    def retyped_vars(self) -> Path:
        return self.ryder_folder/'retyped_vars.csv'

    @property
    def status_file(self) -> Path:
        # right now, simply write last completed step # (1-6)
        return self.ryder_folder/'status.csv'

    @staticmethod
    def get_last_completed_step(ryder_folder:Path) -> Tuple[int, str]:
        status_file = ryder_folder/'status.csv'
        if not status_file.exists():
            return -1, ''
        with open(status_file, 'r') as f:
            last_completed_step, comment = f.readline().strip().split(',')
        return last_completed_step, comment

    @staticmethod
    def report_status(ryder_folder:Path):
        last_completed_step, comment = DragonRyder.get_last_completed_step(ryder_folder)
        if last_completed_step == -1:
            print(f'No progress made - status.csv does not exist')
        else:
            print(f'Last completed step: {last_completed_step} - {comment}')
        return 0

    def _check_dataset(self):
        if self.dataset.balance_dataset:
            raise Exception(f'Dataset was built with --balance, which is invalid for dragon-ryder')

    def _update_status_file(self, completed_step:int, text:str):
        with open(self.status_file, 'w') as f:
            f.write(f'{completed_step},{text}')

    def make_initial_predictions(self) -> pd.DataFrame:
        if self.dragon_initial_preds.exists():
            print(f'Initial predictions file already exists - moving on to next step')
            model_pred = pd.read_csv(self.dragon_initial_preds)
            return model_pred.set_index('Index')

        model_pred = make_predictions_on_dataset(self.dragon_model_path, self.device, self.dataset)

        # join with Name_Strip so we can identify the variable for retyping by name
        vdf = self.dataset.read_vars_csv()[['BinaryId','FunctionStart','Signature','Vartype',
                                            'Name_Strip']]
        model_pred = model_pred.merge(vdf, how='left', on=['BinaryId','FunctionStart','Signature','Vartype'])

        print(f'Saving model predictions to {self.dragon_initial_preds}')
        # save index since we will refer to it to partition high/low confidence
        model_pred.to_csv(self.dragon_initial_preds, index_label='Index')

        return model_pred

    def collect_high_confidence_preds(self, init_preds:pd.DataFrame) -> List[int]:
        # NOTE: the generic form of this is to track variables we have
        # retyped (so we don't retype >1x) and continue iterating until all
        # new vars have been accounted for
        #
        # -> save our retyping decisions in a special file for later use:

        if self.high_confidence_vars.exists():
            print(f'High confidence vars file already exists - moving on to next step')
            with open(self.high_confidence_vars, 'r') as f:
                hc_idx = [int(x) for x in f.readline().strip().split(',')]
            return hc_idx

        print(f'Taking all variables with {self.numrefs_thresh} or more references as high confidence')
        high_conf = init_preds.loc[init_preds.NumRefs >= self.numrefs_thresh]

        # just write index as a single csv
        with open(self.high_confidence_vars, 'w') as f:
            f.write(",".join(str(x) for x in high_conf.index.to_list()))

        return high_conf.index.to_list()

    def verify_ghidra_revision(self, bdf:pd.DataFrame, expected_ghidra_revision:int):
        '''
        Verifies each binary in bdf is at the expected Ghidra revision

        Expected columns in bdf:
            - OrigBinaryId
            - ExpName
            - RunName
        '''
        failure = False

        for exp_name, exp_bins in bdf.groupby('ExpName'):
            for bid, bin_df in exp_bins.groupby('OrigBinaryId'):
                run_name = bin_df.iloc[0].RunName
                bin_file = locate_ghidra_binary(self.proj, run_name, bid, debug_binary=False)

                if bin_file.version != expected_ghidra_revision:
                    print(f'{bin_file.name} in {exp_name} @ version {bin_file.version} does not match expected version {expected_ghidra_revision}')

                    if bin_file.version > expected_ghidra_revision and self.rollback_delete:
                        print(f'Rolling back {bin_file.name} from version {bin_file.version} to version {expected_ghidra_revision}...')
                        for v in range(bin_file.version, expected_ghidra_revision, -1):
                            bin_file.delete(v)
                    else:
                        failure = True

        if failure:
            raise Exception(f'Some files did not match expected version')

    def _retype_variable(self, retyper, var_highsym, new_type:DataType):

        if new_type.leaf_type.category != 'BUILTIN':
            # print(f'Skipping non-builtin leaf type: {new_type}')
            return
        elif 'ARR' in new_type.type_sequence_str:
            # skip arrays for now - we don't have an array length
            # NOTE: we could arbitrarily choose a length of 1 and see what that does?
            return

        # TODO: only apply BUILTIN types for now...
        # if dt.category == 'BUILTIN':

        # TODO - handle UNION, STRUCT, ENUM, FUNC leaf types
        # NOTE: STRUCT leaf type options
        # - a) don't apply these
        # - b) apply a dummy struct def (idk if this is a good idea...)
        # - c) go ahead and apply the struct type we plan on using for member recovery (char*)
        # UNION,

        retyper.set_localvar_type(var_highsym, new_type)

    def apply_predictions_to_ghidra(self, preds:pd.DataFrame, expected_ghidra_revision:int=None, checkin_msg:str='') -> pd.DataFrame:
        '''
        Applies our predicted data types (Pred column in preds) to the
        variables in each Ghidra binary

        preds: Data type predictions to apply
        expected_ghidra_revision: Expected Ghidra revision (e.g. 1 for initial state) of each database. If
                                  this is specified but does not match the apply will fail (for now - later we can roll back)
        '''
        if self.retyped_vars.exists():
            print(f'Retyped vars file already exists - moving on to next step')
            return pd.read_csv(self.retyped_vars)

        bt = self.dataset.full_binaries_table[['BinaryId','OrigBinaryId','ExpName','RunName']]
        preds = preds.merge(bt, on='BinaryId', how='left')
        preds['PredType'] = preds.PredJson.apply(lambda x: DataType.from_json(x))

        if expected_ghidra_revision is not None:
            self.verify_ghidra_revision(preds, expected_ghidra_revision)

        for exp_name, exp_preds in preds.groupby('ExpName'):
            for bid, bin_preds in exp_preds.groupby('OrigBinaryId'):
                run_name = bin_preds.iloc[0].RunName
                bin_file = locate_ghidra_binary(self.proj, run_name, bid, debug_binary=False)
                with GhidraCheckoutProgram(self.proj, bin_file) as co:
                    retyper = GhidraRetyper(co.program, None)

                    for func_addr, func_preds in tqdm(bin_preds.groupby('FunctionStart'), desc=f'Retyping {bin_file.name}...'):
                        func_syms = retyper.get_function_symbols(func_addr)     # decompiles function, only do 1x
                        [self._retype_variable(retyper, func_syms[x[0]], x[1]) for x in zip(func_preds.Name_Strip, func_preds.PredType)]

                    # ------------------------------------------------------------------
                    # TODO: if name not in syms then try matching by location?
                    # - convert Ghidra HighSymbol storage to Location
                    # - collect Location_Strip from original vdf (before this function)
                    # ------------------------------------------------------------------

                    self.proj.save(co.program)
                    self.proj.close(co.program)
                    co.checkin_msg = checkin_msg

        # save preds to retyped_vars (update/add to this file in general...)
        # TODO: - do we want to only save vars we actually retyped?
        preds.to_csv(self.retyped_vars, index=False)
        return preds

    def _load_bin_files(self):
        if self.binary_paths.exists():
            print(f'Loading saved binary paths')
            with open(self.binary_paths, 'r') as f:
                binary_paths = [p.strip() for p in f.readlines()]
            self.bin_files = [self.proj.projectData.getFile(p) for p in binary_paths]
            return

        if self.binary_list:
            print(f'Locating selected binaries from repo {self.repo_name}')
            repo_file_paths = {}  # map de-numbered name -> DomainFile paths
            for f in get_all_files_in_project(self.proj, no_debug=True):
                # remove initial binary number (e.g. 4.binary_name -> binary_name)
                denumbered_name = str(f.name)[str(f.name).find('.')+1:]
                repo_file_paths[denumbered_name] = f.pathname
            binary_paths = [repo_file_paths[binary] for binary in self.binary_list]
        else:
            print(f'Populating list of binaries')
            binary_paths = [f.pathname for f in get_all_files_in_project(self.proj, no_debug=True)]

        self.bin_files = [self.proj.projectData.getFile(p) for p in binary_paths]

        # save binary paths to csv
        with open(self.binary_paths, 'w') as f:
            f.write('\n'.join(binary_paths))

    @property
    def proj(self) -> GhidraProject:
        '''Handle to the GhidraProject'''
        return self._shared_proj.shared_gp if self._shared_proj else None

    def __enter__(self):
        self.console.print(f'Starting pyhidra...')
        self._shared_proj = OpenSharedGhidraProject(self.ghidra_server, self.repo_name, self.ghidra_port)
        self._shared_proj.__enter__()
        return self

    def __exit__(self, etype, value, traceback):
        self._shared_proj.__exit__(etype, value, traceback)
        self._shared_proj = None

    def run(self):
        print(f'{"Resuming" if self.resume else "Running"} dragon-ryder on {self.repo_name} using DRAGON model {self.dragon_model_path}')

        if not self.resume:
            if self.ryder_folder.exists():
                self.console.print(f'[yellow]Warning: {self.ryder_folder} folder already exists. Use --resume to continue if unfinished')
                return 1

        if not self.ryder_folder.exists():
            self.ryder_folder.mkdir()

        self._load_bin_files()
        for f in self.bin_files:
            print(f)

        # TODO: NEW ALGORITHM --------------------------------------
        # for each binary... (retyping binary X of Y (%))
        # A) Gen 1 (go function by function) -- arbitrary order right now, maybe we care later?

        ##########################
        # TODO: --> PICK UP HERE
        ##########################
        # 1. decompile function, extract AST
        # from ghidralib.decompiler import get_decompiler_interface
        # from ghidralib.export_ast import decompile_all

        from ghidralib.decompiler import get_decompiler_interface

        for i, bin_file in enumerate(self.bin_files):
            self.console.rule(f'Processing binary {bin_file.name} ({i+1} of {len(self.bin_files)})')
            with GhidraCheckoutProgram(self.proj, bin_file) as co:
                ifc = get_decompiler_interface(co.program)

                fm = co.program.getFunctionManager()
                nonthunks = [x for x in fm.getFunctions(True) if not x.isThunk()]
                total_funcs = len(nonthunks)

                for func in tqdm(nonthunks):
                    timeout_sec = 240
                    res = ifc.decompileFunction(func, timeout_sec, None)

                    # --------------------
                    # TODO - eventually, this + read_json_str() call needs to get
                    # wrapped into a helper function so the magic "BEGIN AST" string
                    # is only in one place
                    # --------------------
                    error_msg, ast_json = res.errorMessage.split('#$#$# BEGIN AST #@#@#')

                    if not res.decompileCompleted:
                        print('Decompilation failed:')
                        print(error_msg)
                        # failed_decompilations.append(address)
                        continue

                    ast = read_json_str(ast_json, sdb=None)

                    # TODO: build var graphs...

                    import IPython; IPython.embed()

                    break

                # NOTE: all this will need to go in a function (and probably call other functions)
                # ...we need to keep the decompiler interface/state alive for the duration so we
                # can reuse all the "open handles"

                # TODO: modify GhidraRetyper to accept an "already-live" decompiler interface
                # (instead of manually creating its own internally)

                # TODO - save/close/checkin the binary
                # self.proj.save(co.program)
                # self.proj.close(co.program)
                # co.checkin_msg = checkin_msg

            self.console.print(f'[bold red]TEMP: bailing after first binary')
            break

        # 2. build var graphs, convert to pytorch Data objects (here we actually don't "need" the y data...not training so no loss!
        #    ...and we'll wait to "eval" until the very end, doing that in pandas)
        # 3. predictions using DRAGON
        # 4. determine high confidence predictions
        # 5. retype high confidence vars (KEEP TRACK/SAVE THESE IN PANDAS/CSV)
        #    -> continue to next function...
        #
        # B) Gen 2 (go function by function)
        # 6. decompile function, extract AST (this is rerunning decompiler)
        # 7. build var graphs, convert to pytorch Data objects
        # 8. predictions using DRAGON (for remaining variables...or all vars if we care to see if our pred's changed?)
        # 9. retype remaining vars (UPDATE/ADD THESE TO PANDAS/CSV)
        #
        # 10. save final/full CSV output predictions
        #
        # --> separate script: eval this CSV by:
        # a) extracting all debug ASTs into a pandas DF/CSV
        # b) aligning vars by signature...
        # c) compute accuracy/metrics

        return 0

        # 1. Make initial predictions on dataset (save these somewhere - pd/csv)
        console.rule(f'Step 1/6: make initial model predictions')
        init_preds = self.make_initial_predictions()

        # 2. Collect high confidence predictions (via strategy)
        console.rule(f'Step 2/6: collect high confidence predictions')
        hc_idx = self.collect_high_confidence_preds(init_preds)

        high_conf = init_preds.loc[hc_idx, :]
        # low_conf = init_preds.loc[~init_preds.index.isin(hc_idx),:]

        # 3. Apply high confidence predictions to each binary
        console.rule(f'Step 3/6: apply high confidence predictions to Ghidra')
        self.apply_predictions_to_ghidra(high_conf, expected_ghidra_revision=1,
                checkin_msg='dragon-ryder: high confidence')

        # TODO: implement...
        # 4. Update/rebuild the dataset from this latest state
        #   - re-export all function ASTs
        #   - re-build the dataset (via dragon build)
        #       >> this means we need to KNOW how it was initially built (run same cmd-line params...)

        #       NOTE: we DO know this I think (via dataset/raw/input_params.json)

        #   - LATER: we don't HAVE to rebuild var graphs for variables we have already predicted...
        #     ...for now, just redo everything for simplicity
        # 5. Make predictions on remaining variables (using this updated dataset)
        #   OPTIONALLY: make predictions on all updated variables...did anything change?
        # 6. Final predictions are JOIN of initial high-confidence predictions + gen 2 predictions

        # NOTE: to rollback:
        # - check out version 1
        # - "save as" the binary file as a new name (and check in)
        # - this new file is the "rolled back" copy we can use...
        return 0

