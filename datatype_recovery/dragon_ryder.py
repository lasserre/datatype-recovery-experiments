# DRAGON RYDER
# -------------
# DRAGON incremental RetYping DrivER

import pyhidra
pyhidra.start()

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
from ghidralib.export_vars import *

import typing
if typing.TYPE_CHECKING:
    import ghidra
    from ghidra.ghidra_builtins import *

from ghidra.program.model.pcode import HighSymbol

from ghidralib.projects import *
from ghidralib.ghidraretyper import GhidraRetyper

def load_bin_files(proj:GhidraProject, bin_paths_csv:Path, console:Console, binary_list:List[str]=None) -> List[DomainFile]:
    '''
    Loads the stripped binary files for this project and returns them as a list of DomainFiles.

    - binary_list optionally specifies a list of binaries to be loaded instead of finding all binaries in the project
    - bin_paths_csv will be used to save the binary paths used. If it already exists, it will be used
        and ignore binary_list

    proj: The GhidraProject to pull from
    bin_paths_csv: File path to save binary paths to/load saved paths from
    binary_list: A list of specific binary names (no id or suffix) to use instead of all binaries in the project
    '''
    bin_files = []

    if bin_paths_csv.exists():
        print(f'Loading saved binary paths')
        if binary_list:
            console.print(f'[yellow]Warning: ignoring -b option and restoring saved binary paths as-is')
        with open(bin_paths_csv, 'r') as f:
            binary_paths = [p.strip() for p in f.readlines()]
        return [proj.projectData.getFile(p) for p in binary_paths]

    if binary_list:
        print(f'Locating selected binaries from repo {proj.projectData.repository.name}')
        bin_files = locate_binaries_from_project(proj, binary_list, strip_only=True)
    else:
        print(f'Populating list of binaries')
        bin_files = get_all_files_in_project(proj, strip_only=True)

    binary_paths = [f.pathname for f in bin_files]

    # save binary paths to csv
    with open(bin_paths_csv, 'w') as f:
        f.write('\n'.join(binary_paths))

    return bin_files

class DragonRyder:
    def __init__(self, dragon_model_path:Path, repo_name:str,
                device:str='cpu',
                resume:bool=False, numrefs_thresh:int=5,
                rollback_delete:bool=False,
                ghidra_server:str='localhost', ghidra_port:int=13100,
                confidence_strategy:str='refs',
                binary_list:List[str]=None,
                limit_funcs:int=-1) -> None:
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
        self.limit_funcs = limit_funcs      # max # funcs per binary (for testing)

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
    def high_confidence_vars(self) -> Path:
        return self.ryder_folder/'high_conf_vars.csv'

    @property
    def predictions_csv(self) -> Path:
        return self.ryder_folder/'predictions.csv'

    def _check_dataset(self):
        if self.dataset.balance_dataset:
            raise Exception(f'Dataset was built with --balance, which is invalid for dragon-ryder')

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
        bid = binary_id(bin_file.name)

        if expected_revision is not None:
            self.verify_ghidra_revision(bin_file, expected_revision=expected_revision)

        with GhidraCheckoutProgram(self.proj, bin_file) as co:
            with AstDecompiler(co.program, bid, timeout_sec=240) as decompiler:
                retyper = GhidraRetyper(co.program, sdb=None)
                retyped_rows = []   # record for each high confidence variable
                nonthunks = decompiler.nonthunk_functions

                if self.limit_funcs > 0:
                    self.console.print(f'[bold orange1] only running on first {self.limit_funcs:,} functions')
                    nonthunks = nonthunks[:self.limit_funcs]

                if generation_console_msg:
                    self.console.print(f'[blue]{generation_console_msg}')

                for func in tqdm(nonthunks):
                    ast = decompiler.decompile_ast(func)
                    if ast is None:
                        self.console.print('[bold orange1]Decompilation failed:')
                        self.console.print(f'[orange1]{decompiler.last_error_msg}')
                        continue
                    fdecl = ast.get_fdecl()

                    skip_signatures = None if svs is None else svs[svs.FunctionStart==fdecl.address].Signature.to_list()
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
        bid = binary_id(bin_file.name)
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

        self.bin_files = load_bin_files(self.proj, self.binary_paths, self.console, self.binary_list)
        for f in self.bin_files:
            print(f)

        self.dragon_model = DragonModel.load_model(self.dragon_model_path, self.device, eval=True)

        #### run gen 1
        # TODO - to save progress, have _gen1_high...() check for file, read it, check bid exists
        # - if bid exists, filter hc down to bid and return it (don't do anything)
        # - if not, then we continue running from here...
        bin_gen1 = [self._gen1_high_confidence(bin_file, i, len(self.bin_files)) for i, bin_file in enumerate(self.bin_files)]
        gen1 = pd.concat(bin_gen1).reset_index(drop=True)
        gen1.to_csv(self.high_confidence_vars, index=False)

        #### run gen 2
        bin_gen2 = [self._gen2_remaining_vars(bin_file, gen1, i, len(self.bin_files)) for i, bin_file in enumerate(self.bin_files)]
        rdf = pd.concat([gen1, *bin_gen2]).reset_index(drop=True)
        rdf.to_csv(self.predictions_csv, index=False)

        # -----------------------------------
        # TODO: I can validate this also by re-decompiling everything in Ghidra
        # and verifying that each variable marked Retyped=True has a type that matches
        # the type from decompiled Ghidra
        # -----------------------------------

        return 0

