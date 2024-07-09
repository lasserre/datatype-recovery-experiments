# DRAGON RYDER
# -------------
# DRAGON incremental RetYping DrivER

import pandas as pd
from pathlib import Path
from rich.console import Console
from tqdm import tqdm
from typing import List, Tuple

from datatype_recovery.models.eval import make_predictions_on_dataset
from datatype_recovery.models.dataset.encoding import *
from datatype_recovery.models.homomodels import DragonModel, VarPrediction

from astlib import read_json_str
from varlib.datatype import DataType
from ghidralib.decompiler import AstDecompiler

import pyhidra
pyhidra.start()

import typing
if typing.TYPE_CHECKING:
    import ghidra
    from ghidra.ghidra_builtins import *

from ghidra.program.model.pcode import HighSymbol

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
        self.bin_files:List[DomainFile] = []        # generate this from binary_list or all files in the repo
        self.dragon_model:DragonModel = None        # loaded model will go here

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

    def filter_high_confidence_pred(self, var_pred:VarPrediction) -> bool:
        '''
        Function suitable for use with filter() that returns True if this item
        is considered a high confidence prediction (per our confidence strategy)
        '''
        if self.confidence_strategy == 'refs':
            return var_pred.num_refs >= self.numrefs_thresh
        else:
            raise Exception(f'Unhandled confidence strategy {self.confidence_strategy}')

    def verify_ghidra_revision(self, domain_file:DomainFile, expected_revision:int):
        '''
        Verifies this Ghidra file is at the expected revision.

        If self.rollback_delete is true, files beyond expected_revision will be rolled back
        by deleting revisions. Otherwise, an exception will be thrown.

        An exception will be thrown in any case if the revision is < the expected revision
        '''
        if domain_file.version != expected_revision:
            msg = f'{domain_file.name} @ version {domain_file.version} does not match expected version {expected_revision}'

            if domain_file.version > expected_revision and self.rollback_delete:
                print(msg)
                print(f'Rolling back {domain_file.name} from version {domain_file.version} to version {expected_revision}...')
                for v in range(domain_file.version, expected_revision, -1):
                    domain_file.delete(v)
            else:
                # version < expected_revision or rollback_delete is false
                raise Exception(msg)

    def _retype_variable(self, retyper:GhidraRetyper, var_highsym:HighSymbol, new_type:DataType) -> bool:
        '''
        Retypes the given variable if possible based on the desired new_type.

        Returns true if the variable was retyped, false otherwise.
        '''

        if new_type.leaf_type.category != 'BUILTIN':
            # print(f'Skipping non-builtin leaf type: {new_type}')
            return False
        elif 'ARR' in new_type.type_sequence_str:
            # skip arrays for now - we don't have an array length
            # NOTE: we could arbitrarily choose a length of 1 and see what that does?
            return False
        elif isinstance(new_type, BuiltinType) and new_type.is_void:
            # NOTE - we cannot retype a local or param as void, this only is valid for
            # return types
            # TODO: we could convert this to void* ?
            return False

        # TODO - handle UNION, STRUCT, ENUM, FUNC leaf types
        # NOTE: STRUCT leaf type options
        # - a) don't apply these
        # - b) apply a dummy struct def (idk if this is a good idea...)
        # - c) go ahead and apply the struct type we plan on using for member recovery (char*)
        # UNION,

        try:
            retyper.set_funcvar_type(var_highsym, new_type)
        except Exception as e:
            self.console.print(f'[yellow]{e}')
            return False

        return True

    def _load_bin_files(self):
        if self.binary_paths.exists():
            print(f'Loading saved binary paths')
            if self.binary_list:
                self.console.print(f'[yellow]Warning: ignoring -b option and restoring saved binary paths as-is')
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

    @staticmethod
    def binary_id(bin_file:DomainFile) -> int:
        # we can use these directly since we stay within a single Ghidra repo for evaluations
        # (no need to recompute global bid across repos)
        return int(bin_file.name.split('.')[0])

    def _run_generation(self, bin_file:DomainFile,
                        skip_var_signatures:pd.DataFrame=None,
                        filter_preds_to_retype:Callable=None,
                        expected_revision:int=None,
                        checkin_msg:str='',
                        generation_console_msg:str='') -> pd.DataFrame:
        '''
        Generic algorithm to run one generation of decompilation, variable prediction, and retyping

        bin_file: The program to decompile
        skip_signatures: An optional table of variables that should be skipped for both prediction and retyping
                         (typically because we've already retyped them). The table must include the varid
                         columns FunctionStart and Signature.
        filter_preds_to_retype: An optional callable that will be passed to filter() after predictions
                                are made in order to filter predictions we will actually attempt to retype
                                in Ghidra for this generation
        expected_revision: The expected Ghidra revision (if any). Any rollback actions will be taken based on
                           this version and the configured rollback mode.
        checkin_msg: The Ghidra check-in message to use for this generation.
        generation_console_msg: Optional message to print on the console at the start of this generation
        '''
        svs = skip_var_signatures   # alias for shorter pandas line
        bid = DragonRyder.binary_id(bin_file)

        if expected_revision is not None:
            self.verify_ghidra_revision(bin_file, expected_revision=expected_revision)

        with GhidraCheckoutProgram(self.proj, bin_file) as co:
            with AstDecompiler(co.program, bid, timeout_sec=240) as decompiler:
                retyper = GhidraRetyper(co.program, sdb=None)
                retyped_rows = []   # record for each high confidence variable

                # -----------------------------------
                LIMIT_FUNCS = 50
                self.console.print(f'[bold orange1]TEMP - only running on first {LIMIT_FUNCS:,} functions')
                nonthunks = decompiler.nonthunk_functions[:LIMIT_FUNCS]
                # -----------------------------------

                if generation_console_msg:
                    self.console.print(f'[blue]{generation_console_msg}')

                for func in tqdm(nonthunks):
                    ast = decompiler.decompile_ast(func)
                    if ast is None:
                        self.console.print('[bold orange1]Decompilation failed:')
                        self.console.print(f'[orange1]{decompiler.last_error_msg}')
                        continue
                    fdecl = ast.get_fdecl()

                    skip_signatures = svs[svs.FunctionStart==fdecl.address].Signature.to_list() if svs else None
                    var_preds = self.dragon_model.predict_func_types(ast, self.device, bid,
                                                                    skip_unique_vars=True,
                                                                    skip_signatures=skip_signatures)

                    for p in filter(filter_preds_to_retype, var_preds):
                        success = self._retype_variable(retyper, decompiler.local_sym_dict[p.vardecl.name], p.pred_dt)
                        retyped_rows.append([*p.varid, p.vardecl.name, p.vardecl.location, p.pred_dt, p.pred_dt.to_dict(), success])

                co.checkin_msg = checkin_msg

                return pd.DataFrame.from_records(retyped_rows, columns=[
                    'BinaryId','FunctionStart','Signature','Vartype','Name','Location','Pred','PredJson','Retyped'
                ])

    def _gen1_high_confidence(self, bin_file:DomainFile, idx:int, total:int) -> pd.DataFrame:
        '''
        Performs generation 1 (predict/retype high confidence variable types)
        and returns a table containing the high confidence predictions (all predictions,
        including ones we were not able to actually retype in Ghidra)
        '''
        # if self.high_confidence_vars.exists():
        #     self.console.print(f'[bold red]TEMP - skipping gen 1 since {self.high_confidence_vars.name} exists')
        #     # TODO - return saved DF...?
        # TODO: check if this Binary ID has already been retyped in high_conf.csv (if so, skip to next...)

        self.console.rule(f'[bold blue]GEN 1: High Confidence[/] processing binary {bin_file.name} ({idx+1} of {total})')

        return self._run_generation(bin_file,
                                    filter_preds_to_retype=self.filter_high_confidence_pred,
                                    expected_revision=1,
                                    checkin_msg='dragon-ryder: high confidence',
                                    generation_console_msg=f'[{bin_file.name}]: initial predictions/retypings')

    def _gen2_remaining_vars(self, bin_file:DomainFile, hc_vars:pd.DataFrame, idx:int, total:int) -> pd.DataFrame:
        '''
        Performs generation 2 (predict remaining variable types)
        and returns a table containing the remaining predictions
        '''
        self.console.rule(f'[bold green]GEN 2: Remaining Vars[/] processing binary {bin_file.name} ({idx+1} of {total})')
        bid = DragonRyder.binary_id(bin_file)
        binary_hc = hc_vars.loc[hc_vars.BinaryId==bid,:]
        return self._run_generation(bin_file,
                                    skip_var_signatures=binary_hc,
                                    expected_revision=2,
                                    checkin_msg='dragon-ryder: gen2',
                                    generation_console_msg=f'[{bin_file.name}]: remaining variable predictions/retyping')

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

        self.dragon_model = DragonModel.load_model(self.dragon_model_path, self.device, eval=True)

        #### run gen 1
        # TODO - to save progress, have _gen1_high...() check for file, read it, check bid exists
        # - if bid exists, filter hc down to bid and return it (don't do anything)
        # - if not, then we continue running from here...
        bin_hcs = [self._gen1_high_confidence(bin_file, i, len(self.bin_files)) for i, bin_file in enumerate(self.bin_files)]
        hc = pd.concat(bin_hcs).reset_index(drop=True)
        hc.to_csv(self.high_confidence_vars, index=False)

        #### run gen 2
        bin_rdfs = [self._gen2_remaining_vars(bin_file, hc, i, len(self.bin_files)) for i, bin_file in enumerate(self.bin_files)]
        rdf = pd.concat([hc, *bin_rdfs]).reset_index(drop=True)
        rdf.to_csv(self.retyped_vars, index=False)

        import IPython; IPython.embed()

        # -----------------------------------
        # TODO: I can validate this also by re-decompiling everything in Ghidra
        # and verifying that each variable marked Retyped=True has a type that matches
        # the type from decompiled Ghidra
        # -----------------------------------
        # NOTE: OOOOOOOOOOO!!
        # --> to do this, I basically need an export_variables() or export_var_types() function
        #
        #           export_var_df(func=func)
        #                   -> if func=None, then do it for all functions?
        #                   -> or specify the list of functions funcs=[f] vs. funcs=nonthunks vs. funcs=None does all...
        #
        #     that 1) decompiles a function (or all functions) and returns the ast, (TODO 1) decompile_ast())
        #          2) gives me all the func variables (we already have this as fdecl.local_vars/fdecl.params)
        #          3) computes signatures (TODO 2) FindAllRefs/compute_sig...maybe these)
        #          4) saves data in table format (TODO 3) [varid columns], Name, Type, etc... )
        #
        # TODO - then refactor code above this to USE THESE FUNCTIONS (this run() function should read
        #        very easy/be very simple to follow)
        #
        # >>>>> this same function is what I will reuse basically AS-IS for
        # BUILDING THE TRUTH DATA FROM DEBUG ASTS! (which I need for eval script)
        # -----------------------------------------

        # 10. save final/full CSV output predictions
        #
        # --> TODO: separate script: eval this CSV by:
        # a) extracting all debug ASTs into a pandas DF/CSV
        # b) aligning vars by signature...
        # c) compute accuracy/metrics

        return 0

