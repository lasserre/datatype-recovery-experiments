# DRAGON RYDER
# -------------
# DRAGON incremental RetYping DrivER

import pyhidra
pyhidra.start()

import json
import pandas as pd
from pathlib import Path
from rich.console import Console
from tqdm import tqdm
from typing import List, Tuple

from datatype_recovery.models.dataset.encoding import *
from datatype_recovery.models.homomodels import DragonModel, VarPrediction

from astlib import read_json_str
from varlib.datatype import DataType
from ghidralib.decompiler import AstDecompiler
from ghidralib.projects import *
from ghidralib.export_vars import *

import typing
if typing.TYPE_CHECKING:
    import ghidra
    from ghidra.ghidra_builtins import *

from ghidra.program.model.pcode import HighSymbol

from ghidralib.projects import *
from ghidralib.ghidraretyper import GhidraRetyper

def load_bin_files_from_repo(repo_name:str, bin_paths_csv:Path, console:Console, binary_list:List[str]=None,
                            host:str='localhost', port:int=13100) -> Tuple[List[DomainFile], List[DomainFile]]:
    with OpenSharedGhidraProject(host, repo_name, port) as proj:
        return load_bin_files(proj, bin_paths_csv, console, binary_list)

def load_bin_files(proj:GhidraProject, bin_paths_csv:Path, console:Console,
                    binary_list:List[str]=None) -> Tuple[List[DomainFile], List[DomainFile]]:
    '''
    Loads the stripped and debug binary files for this project and returns them as a pair
    of (strip_filelist, debug_filelist).

    - binary_list optionally specifies a list of binaries to be loaded instead of finding all binaries in the project
    - bin_paths_csv will be used to save the binary paths used. If it already exists, it will be used
        and ignore binary_list

    proj: The GhidraProject to pull from
    bin_paths_csv: File path to save binary paths to/load saved paths from
    binary_list: A list of specific binary names (no id or suffix) to use instead of all binaries in the project
    '''
    strip_bins = []

    if bin_paths_csv.exists():
        print(f'Loading saved binary paths')
        if binary_list:
            console.print(f'[yellow]Warning: ignoring -b option and restoring saved binary paths as-is')
        with open(bin_paths_csv, 'r') as f:
            binary_paths = [p.strip() for p in f.readlines()]
        strip_bins = [proj.projectData.getFile(p) for p in binary_paths]
        debug_bins = [get_debug_binary(b) for b in strip_bins]
        return (strip_bins, debug_bins)

    if binary_list:
        print(f'Locating selected binaries from repo {proj.projectData.repository.name}')
        strip_bins = locate_binaries_from_project(proj, binary_list, strip_only=True)
    else:
        print(f'Populating list of binaries')
        strip_bins = get_all_files_in_project(proj, strip_only=True)

    strip_paths = [f.pathname for f in strip_bins]

    # save stripped binary paths to csv
    with open(bin_paths_csv, 'w') as f:
        f.write('\n'.join(strip_paths))

    debug_bins = [get_debug_binary(b) for b in strip_bins]
    return (strip_bins, debug_bins)

def replace_arr_with_ptr(dt:DataType) -> DataType:
    '''
    Replace each occurrence of arrays in this data type with pointers (for retyping)
    '''
    if isinstance(dt, ArrayType):
        return PointerType(replace_arr_with_ptr(dt.element_type), pointer_size=8)
    elif isinstance(dt, PointerType):
        return PointerType(replace_arr_with_ptr(dt.pointed_to), dt.pointer_size)
    return dt

def replace_leaf_type(dt:DataType, new_leaftype:DataType) -> DataType:
    '''
    Return a data type that replaces dt's leaf type with new_leaftype
    '''
    if isinstance(dt, PointerType):
        return PointerType(replace_leaf_type(dt.pointed_to, new_leaftype), pointer_size=dt.pointer_size)
    elif isinstance(dt, ArrayType):
        return ArrayType(replace_leaf_type(dt.element_type, new_leaftype), dt.num_elements)
    # this is the leaf type
    return new_leaftype

class DragonRyder:
    def __init__(self, dragon_model_path:Path, repo_name:str,
                device:str='cpu',
                resume:bool=False, numrefs_thresh:int=5,
                rollback_delete:bool=False,
                ghidra_server:str='localhost', ghidra_port:int=13100,
                confidence_strategy:str='refs',
                binary_list:List[str]=None,
                limit_funcs:int=-1,
                confidence:float=0.9,
                influence:int=10,
                medium_conf:float=0.65,
                ryder_folder:Path=None) -> None:
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
        self.confidence = confidence
        self.influence = influence
        self.medium_conf = medium_conf
        self._ryder_folder = ryder_folder

        self._shared_proj = None
        self.console = Console()
        self.bin_files:List[DomainFile] = []        # generate this from binary_list or all files in the repo
        self.dragon_model:DragonModel = None        # loaded model will go here
        self._placeholder_sid = -1                  # sid for placeholder struct

    @property
    def ryder_folder(self) -> Path:
        if self._ryder_folder:
            return self._ryder_folder
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

    def filter_high_confidence_pred(self, var_pred:VarPrediction) -> bool:
        '''
        Function suitable for use with filter() that returns True if this item
        is considered a high confidence prediction (per our confidence strategy)
        '''
        if self.confidence_strategy == 'refs':
            return var_pred.num_refs >= self.numrefs_thresh
        elif self.confidence_strategy == 'conf':
            return var_pred.confidence >= self.confidence
        elif self.confidence_strategy == 'conf_inf':
            high_conf = var_pred.confidence >= self.confidence
            med_conf_high_influence = (var_pred.confidence >= self.medium_conf) and (var_pred.influence >= self.influence)
            return high_conf or med_conf_high_influence
        elif self.confidence_strategy == 'inf_conf':
            # 1st: divide into 4 impact/influenceability quadrants
            # 2nd: then use confidence within these quadrants

            high_impact = var_pred.influence >= self.influence
            high_inflbl = var_pred.num_other_vars >= 4      # FIXME: make this a param

            # LOW IMPACT/LOW INFLUENCEABILITY - predict vars above high confidence (or wait)
            # LOW IMPACT/HIGH INFLUENCEABILITY - wait for gen 2 no matter what
            # HIGH IMPACT/LOW INFLUENCEABILITY - predict vars above MED confidence
            # HIGH IMPACT/HIGH INFLUENCEABILITY - predict vars above high confidence

            if not high_impact and not high_inflbl:
                return var_pred.confidence >= self.confidence
            elif not high_impact and high_inflbl:
                return False
            elif high_impact and not high_inflbl:
                return var_pred.confidence >= self.medium_conf
            else:
                return var_pred.confidence >= self.confidence

        else:
            raise Exception(f'Unhandled confidence strategy {self.confidence_strategy}')

    def _retype_struct_with_placeholder(self, retyper:GhidraRetyper, dt:DataType) -> DataType:
        '''
        Replace the leaf type with the placeholder structure that has an arbitrary definition with a single char* member
        '''
        placeholder_struct = StructType(retyper.sdb, self._placeholder_sid)
        return replace_leaf_type(dt, placeholder_struct)

    def _retype_variable(self, retyper:GhidraRetyper, var_highsym:HighSymbol, new_type:DataType) -> bool:
        '''
        Retypes the given variable if possible based on the desired new_type.

        Returns true if the variable was retyped, false otherwise.
        '''
        if isinstance(new_type.leaf_type, StructType):
            # 1. use placeholder structure
            # new_type = self._retype_struct_with_placeholder(retyper, new_type)

            # 2. simply DO NOT retype Struct leaf types
            return False

            # 3. don't retype STRUCT, otherwise replace STRUCT leaf type with void
            # (e.g. STRUCT -> don't retype, STRUCT* -> void*, STRUCT** -> void**)
            # if isinstance(new_type, StructType):
            #     return False    # don't retype STRUCT
            # new_type = replace_leaf_type(new_type, BuiltinType.create_void_type())
        elif isinstance(new_type, BuiltinType) and new_type.is_void:
            # NOTE - we cannot retype a local or param as void, this only is valid for
            # return types
            # --> should we convert this to void* ?
            return False
        elif not isinstance(new_type.leaf_type, BuiltinType):
            # skip remaining non-primitive types
            return False

        # convert arrays in the pointer hierarchy
        if 'ARR' in new_type.type_sequence_str:
            new_type = replace_arr_with_ptr(new_type)   # replace arrays with pointers

        # TODO - handle UNION, ENUM, FUNC leaf types?

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
        rid = run_id(bin_file.parent.name)

        if expected_revision is not None:
            verify_ghidra_revision(bin_file, expected_revision, self.rollback_delete)

        with GhidraCheckoutProgram(self.proj, bin_file) as co:
            with AstDecompiler(co.program, bid, timeout_sec=240) as decompiler:
                sdb = StructDatabase()
                placeholder_def = StructDefinition('PLACEHOLDER',
                    StructLayout({0: StructField(PointerType(BuiltinType.from_standard_name('char'), 8), 'dummy_field')})
                )
                self._placeholder_sid = sdb.map_struct_type('', placeholder_def, is_union=False)

                retyper = GhidraRetyper(co.program, sdb=sdb)
                retyper.define_all_reference_types()

                retyped_rows = []   # record for each high confidence variable
                nonthunks = decompiler.nonthunk_functions

                if self.limit_funcs:
                    self.console.print(f'[bold orange1] only running on first {self.limit_funcs:,} functions')
                    nonthunks = nonthunks[:self.limit_funcs]

                for func in tqdm(nonthunks, desc=generation_console_msg):
                    ast = decompiler.decompile_ast(func)
                    num_callers = len(func.getCallingFunctions(None))

                    if ast is None:
                        self.console.print('[bold orange1]Decompilation failed:')
                        self.console.print(f'[orange1]{decompiler.last_error_msg}')
                        continue
                    fdecl = ast.fdecl

                    skip_signatures = None if svs is None else svs[svs.FunctionStart==fdecl.address].Signature.to_list()
                    var_preds = self.dragon_model.predict_func_types(ast, self.device, bid,
                                                                    skip_unique_vars=True,
                                                                    skip_signatures=skip_signatures,
                                                                    num_callers=num_callers)

                    for p in filter(filter_preds_to_retype, var_preds):
                        success = self._retype_variable(retyper, decompiler.local_sym_dict[p.vardecl.name], p.pred_dt)
                        retyped_rows.append([*p.to_record(), success, rid])

                co.checkin_msg = checkin_msg

                return pd.DataFrame.from_records(retyped_rows, columns=[*VarPrediction.record_columns(), 'Retyped', 'RunId'])

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

        self.console.rule(f'[bold blue]GEN 1: High Confidence[/] processing binary {bin_file.name} ({idx+1} of {total})',
                            align='left')

        df = self._run_generation(bin_file,
                                    filter_preds_to_retype=self.filter_high_confidence_pred,
                                    expected_revision=1,
                                    checkin_msg='dragon-ryder: high confidence',
                                    generation_console_msg=f'[{bin_file.name}]: high confidence vars')
        df['Gen'] = 1
        return df

    def _gen2_remaining_vars(self, bin_file:DomainFile, hc_vars:pd.DataFrame, idx:int, total:int) -> pd.DataFrame:
        '''
        Performs generation 2 (predict remaining variable types)
        and returns a table containing the remaining predictions
        '''
        self.console.rule(f'[bold green]GEN 2: Remaining Vars[/] processing binary {bin_file.name} ({idx+1} of {total})',
                            align='left')
        bid = binary_id(bin_file.name)
        binary_hc = hc_vars.loc[hc_vars.BinaryId==bid,:]
        df = self._run_generation(bin_file,
                                    skip_var_signatures=binary_hc,
                                    expected_revision=2,
                                    checkin_msg='dragon-ryder: gen2',
                                    generation_console_msg=f'[{bin_file.name}]: remaining vars')
        df['Gen'] = 2
        return df

    def run(self):
        print(f'{"Resuming" if self.resume else "Running"} dragon-ryder on {self.repo_name} using DRAGON model {self.dragon_model_path}')

        if not self.resume:
            if self.ryder_folder.exists():
                self.console.print(f'[yellow]Warning: {self.ryder_folder} folder already exists. Use --resume to continue if unfinished')
                return 1

        if not self.ryder_folder.exists():
            self.ryder_folder.mkdir()

        # we only need the stripped binaries here
        self.bin_files, _ = load_bin_files(self.proj, self.binary_paths, self.console, self.binary_list)
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

