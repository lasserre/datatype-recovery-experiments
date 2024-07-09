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

    def collect_high_confidence_preds(self, var_preds:List[VarPrediction]) -> List[VarPrediction]:
        if self.confidence_strategy == 'refs':
            return [p for p in var_preds if p.num_refs >= self.numrefs_thresh]
        else:
            raise Exception(f'Unhandled confidence strategy {self.confidence_strategy}')

    # def collect_high_confidence_preds(self, init_preds:pd.DataFrame) -> List[int]:
    #     # NOTE: the generic form of this is to track variables we have
    #     # retyped (so we don't retype >1x) and continue iterating until all
    #     # new vars have been accounted for
    #     #
    #     # -> save our retyping decisions in a special file for later use:

    #     if self.high_confidence_vars.exists():
    #         print(f'High confidence vars file already exists - moving on to next step')
    #         with open(self.high_confidence_vars, 'r') as f:
    #             hc_idx = [int(x) for x in f.readline().strip().split(',')]
    #         return hc_idx

    #     print(f'Taking all variables with {self.numrefs_thresh} or more references as high confidence')
    #     high_conf = init_preds.loc[init_preds.NumRefs >= self.numrefs_thresh]

    #     # just write index as a single csv
    #     with open(self.high_confidence_vars, 'w') as f:
    #         f.write(",".join(str(x) for x in high_conf.index.to_list()))

    #     return high_conf.index.to_list()

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
        # A) Gen 1 (go function by function) -- arbitrary order right now, maybe we care later?

        from ghidralib.decompiler import get_decompiler_interface

        model_load = torch.load(self.dragon_model_path)

        model = DragonModel(model_load.num_hops, model_load.hidden_channels, model_load.num_heads,
                            model_load.num_shared_linear_layers)
        model.load_state_dict(model_load.state_dict())
        model.to(self.device)
        model.eval()

        self.console.print(f'[bold green]Gen 1: Retype high-confidence predictions (strategy={self.confidence_strategy})')

        for i, bin_file in enumerate(self.bin_files):
            if self.high_confidence_vars.exists():
                self.console.print(f'[bold red]TEMP - skipping gen 1 since {self.high_confidence_vars.name} exists')
                break

            self.console.rule(f'Processing binary {bin_file.name} ({i+1} of {len(self.bin_files)})')

            # TODO: check if this Binary ID has already been retyped in high_conf.csv (if so, skip to next...)

            self.verify_ghidra_revision(bin_file, expected_revision=1)

            # we can use these directly since we stay within a single Ghidra repo for evaluations
            # (no need to recompute global bid across repos)
            bid = int(bin_file.name.split('.')[0])

            with GhidraCheckoutProgram(self.proj, bin_file) as co:
                ifc = get_decompiler_interface(co.program)
                fm = co.program.getFunctionManager()

                hc_retyped_vars = []   # record for each retyped variable

                nonthunks = [x for x in fm.getFunctions(True) if not x.isThunk()]

                # -----------------------------------
                LIMIT_FUNCS = 50
                self.console.print(f'[bold orange1]TEMP - only running on first {LIMIT_FUNCS:,} functions')
                nonthunks = nonthunks[:LIMIT_FUNCS]
                # -----------------------------------

                total_funcs = len(nonthunks)

                retyper = GhidraRetyper(co.program, sdb=None)

                self.console.print(f'[blue][{bin_file.name}]: initial predictions/retypings')

                for func in tqdm(nonthunks):
                    timeout_sec = 240

                    # 1. decompile function, extract AST
                    res = ifc.decompileFunction(func, timeout_sec, None)

                    # --------------------
                    # TODO - eventually, this + read_json_str() call needs to get
                    # wrapped into a helper function so the magic "BEGIN AST" string
                    # is only in one place
                    # --------------------
                    error_msg, ast_json = res.errorMessage.split('#$#$# BEGIN AST #@#@#')

                    if not res.decompileCompleted():
                        self.console.print('[bold orange]Decompilation failed:')
                        self.console.print(f'[orange]{error_msg}')
                        # failed_decompilations.append(address)
                        continue

                    name_to_sym = dict(res.highFunction.localSymbolMap.nameToSymbolMap)
                    ast = read_json_str(ast_json, sdb=None)
                    fdecl = ast.get_fdecl()

                    empty_locs = [v for v in fdecl.local_vars + fdecl.params if v.location.loc_type == '']
                    for v in empty_locs:
                        sym_storage = name_to_sym[v.name].storage
                        if not sym_storage.hashStorage and not sym_storage.uniqueStorage:
                            self.console.print(f'[bold red]Found empty AST location that is not unique or hash storage {v.name}')
                            import IPython; IPython.embed()

                    # 2-3. build vargraphs, predict each function variable
                    var_preds = model.predict_func_types(ast, self.device, bid, skip_unique_vars=True)

                    # 4. determine high confidence predictions
                    hc_preds = self.collect_high_confidence_preds(var_preds)

                    # 5. retype high confidence predictions (RETAIN THESE/SAVE IN PANDAS/CSV)
                    # local/param symbols
                    for p in hc_preds:
                        success = self._retype_variable(retyper, name_to_sym[p.vardecl.name], p.pred_dt)
                        hc_retyped_vars.append(tuple([
                            *p.varid, p.vardecl.name, p.vardecl.location,
                            p.pred_dt, p.pred_dt.to_dict(), success
                        ]))

                co.checkin_msg = 'dragon-ryder: high confidence'

                # save all HC preds here, even if they haven't been retyped in Ghidra (Retyped column will be False)
                binary_hc_df = pd.DataFrame.from_records(hc_retyped_vars, columns=[
                    'BinaryId','FunctionStart','Signature','Vartype','Name','Location','Pred','PredJson','Retyped'
                ])

                self.console.print(f'[bold orange1]TEMP - overwriting high_conf_vars.csv (not combining across binaries)')
                binary_hc_df.to_csv(self.high_confidence_vars, index=False)

                # import IPython; IPython.embed()

                # TODO: combine with existing master retyped_df
                # TODO: write entire updated rdf to csv (overwrite)

            self.console.print(f'[bold red]TEMP: bailing after first binary')
            break


        # B) Gen 2 (go function by function)
        self.console.print(f'[bold green]Gen 2: Retype remaining variables')

        # TODO: alot of this was copy/paste - refactor into a nicer function and reuse it...

        hc = pd.read_csv(self.high_confidence_vars)

        for i, bin_file in enumerate(self.bin_files):
            self.console.rule(f'Processing binary {bin_file.name} ({i+1} of {len(self.bin_files)})')

            self.verify_ghidra_revision(bin_file, expected_revision=2)

            # we can use these directly since we stay within a single Ghidra repo for evaluations
            # (no need to recompute global bid across repos)
            bid = int(bin_file.name.split('.')[0])

            with GhidraCheckoutProgram(self.proj, bin_file) as co:
                ifc = get_decompiler_interface(co.program)
                fm = co.program.getFunctionManager()

                gen2_retyped_vars = []   # record for each retyped variable

                nonthunks = [x for x in fm.getFunctions(True) if not x.isThunk()]

                # -----------------------------------
                LIMIT_FUNCS = 50
                self.console.print(f'[bold orange1]TEMP - only running on first {LIMIT_FUNCS:,} functions')
                nonthunks = nonthunks[:LIMIT_FUNCS]
                # -----------------------------------

                total_funcs = len(nonthunks)

                retyper = GhidraRetyper(co.program, sdb=None)

                self.console.print(f'[blue][{bin_file.name}]: initial predictions/retypings')

                for func in tqdm(nonthunks):
                    timeout_sec = 240

                    # 1. decompile function, extract AST
                    res = ifc.decompileFunction(func, timeout_sec, None)

                    # --------------------
                    # TODO - eventually, this + read_json_str() call needs to get
                    # wrapped into a helper function so the magic "BEGIN AST" string
                    # is only in one place
                    # --------------------
                    error_msg, ast_json = res.errorMessage.split('#$#$# BEGIN AST #@#@#')

                    if not res.decompileCompleted():
                        self.console.print('[bold orange]Decompilation failed:')
                        self.console.print(f'[orange]{error_msg}')
                        # failed_decompilations.append(address)
                        continue

                    name_to_sym = dict(res.highFunction.localSymbolMap.nameToSymbolMap)
                    ast = read_json_str(ast_json, sdb=None)
                    fdecl = ast.get_fdecl()

                    # --------------------------------------------------------
                    # TODO: scroll back up and find chunks of this code to pull out into
                    # reusable functions (don't copy/paste so much!!! lol)
                    # --------------------------------------------------------

                    # TODO: pass in list of excluded var signatures to retyping function:
                    exclude_sigs = hc[hc.FunctionStart==fdecl.address].Signature.to_list()

                    var_preds = model.predict_func_types(ast, self.device, bid,
                                                        skip_unique_vars=True,
                                                        skip_signatures=exclude_sigs)

                    for p in var_preds:
                        success = self._retype_variable(retyper, name_to_sym[p.vardecl.name], p.pred_dt)
                        gen2_retyped_vars.append(tuple([
                            *p.varid, p.vardecl.name, p.vardecl.location,
                            p.pred_dt, p.pred_dt.to_dict(), success
                        ]))

                co.checkin_msg = 'dragon-ryder: gen2'

                # save all gen2 preds here, even if they haven't been retyped in Ghidra (Retyped column will be False)
                gen2df = pd.DataFrame.from_records(gen2_retyped_vars, columns=[
                    'BinaryId','FunctionStart','Signature','Vartype','Name','Location','Pred','PredJson','Retyped'
                ])

                self.console.print(f'[bold orange1]TEMP - combining hc and gen2 dfs (not combining across binaries)')
                rdf = pd.concat((hc, gen2df)).reset_index(drop=True)

                self.console.print(f'[bold orange1]TEMP - overwriting retyped_vars (not combining across binaries)')
                rdf.to_csv(self.retyped_vars)

                # NOTE: do we want to process gen1/gen2 FOR A SINGLE BINARY COMPLETELY before moving on
                # to another binary? (I don't have to do gen1 for all binaries first...)

            self.console.print(f'[bold red]TEMP: bailing after first binary')
            break

        import IPython; IPython.embed()

        # -----------------------------------
        # TODO: I can validate this also by re-decompiling everything in Ghidra
        # and verifying that each variable marked Retyped=True has a type that matches
        # the type from decompiled Ghidra
        # -----------------------------------
        # NOTE: OOOOOOOOOOO!!
        # --> to do this, I basically need an export_variables() or export_var_types() function
        #     that 1) decompiles a function (or all functions) and returns the ast, (decompile_ast())
        #          2) gives me all the func variables (we already have this as fdecl.local_vars/fdecl.params)
        #          3) computes signatures (FindAllRefs/compute_sig)
        #          4) saves data in table format ([varid columns], Name, Type, etc...)
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

