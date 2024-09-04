from itertools import chain
import json
import multiprocessing
import os
from pathlib import Path
import pandas as pd
from rich.console import Console
import shutil
import subprocess
import tempfile
from typing import List, Generator, Callable
from tqdm import tqdm

import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.data.data import BaseData

from wildebeest.utils import pretty_memsize_str
import astlib
from varlib.datatype import DataType
from .variablegraphbuilder import VariableGraphBuilder
from .encoding import TypeSequence, TypeEncoder
from .typeseqprojections import DatasetBalanceProjection

def convert_funcvars_to_data_gb(funcs_df:pd.DataFrame, max_hops:int, include_comp:bool, structural_only:bool, node_typeseq_len:int) -> Callable:
    '''
    Pandas group by function that converts function variables (locals or params) into
    PyG Data objects. This function should be applied to a groupby object via groupby.pipe()

    gb_bin_and_func: Pandas groupby object that has grouped the desired dataframe
                     by (BinaryId, FunctionStart)
    '''
    def do_convert_funcvars_to_data(gb_bin_and_func) -> Generator[Data,None,None]:
        # groupby[0]: tuple of grouped-by column values
        # groupby[1]: data frame for this subset of the data
        for gb_vals, df in gb_bin_and_func:
            # we can reuse the AST object here within a function
            bid, addr = gb_vals
            ast_file = funcs_df[(funcs_df.FunctionStart==addr)&(funcs_df.BinaryId==bid)].iloc[0].AstJson_Strip
            ast = astlib.read_json(ast_file)

            for i in range(len(df)):
                name_strip = df.iloc[i].Name_Strip
                builder = VariableGraphBuilder(name_strip, ast, max_hops, sdb=None, node_kind_only=structural_only)

                try:
                    data = builder.build(bid)
                except:
                    # keeping this here so if something goes wrong on a large dataset I can see what binary/function/variable
                    # had the issue! (from varid)
                    print(f'Failed to build variable graph for variable {name_strip} (bid={bid},func={addr:x})', flush=True)
                    raise

                if data is None:
                    continue    # no references to this var

                # Debug holds ground truth prediction
                data.y = TypeEncoder.encode(DataType.from_json(df.iloc[i].TypeJson_Debug))
                yield data

                # yield Data(x=node_list, edge_index=edge_index, y=y, varid=varid, edge_attr=edge_attr)

    return do_convert_funcvars_to_data

def convert_funcvars_to_data(df:pd.DataFrame, funcs_df:pd.DataFrame, max_hops:int, include_comp:bool,
                            structural_only:bool, node_typeseq_len:int) -> Generator[Data,None,None]:
    '''
    Converts function variables (locals or params) into PyG Data objects
    given either the locals or params data frame
    '''
    return df.groupby(['BinaryId','FunctionStart']).pipe(convert_funcvars_to_data_gb(funcs_df, max_hops, include_comp, structural_only, node_typeseq_len))

class TypeSequenceDataset(Dataset):
    def __init__(self, root:str, input_params:dict=None, transform=None, pre_transform=None, pre_filter=None):
        '''
        TypeSequenceDataset is the data type prediction dataset for simple type sequences (i.e. no structure layout prediction).
        Data types are predicted as a sequence of types such as: ['int16_t'], ['STRUCT'], or ['PTR','float'].

        input_params: This is a dictionary of inputs specifying where the raw data comes from
            name:               A descriptive name for this particular dataset
            experiment_runs:    List of paths to individual experiment RUN FOLDERS which should be included
                                ->  This is not simply a path to the overall experiment because runs typically correspond
                                    to varying parameters (e.g. -O0 vs. -O1) which would also typically correspond
                                    to separate datasets.
                                ->  To combine corresponding runs (e.g. -O0) across experiments (e.g. exp1 and exp2),
                                    the individual run folders would be listed (e.g. exp1/run1, exp2/run1)
                                ->  If all runs of an experiment are desired, they can be listed explicitly
                                    (exp1/run1, exp1/run2, exp1/run3)
            copy_data:          True if raw .csv files should be copied to the raw_dir folder.
                                If False, data will be pulled from the original file locations
            drop_component:     If True, drop COMP entries from the dataset
            structural_only:    If True, generate node features for structural-only model
            node_typeseq_len:   Type sequence length of node features for Dragon model
            balance_dataset:    True if we should balance the dataset (using hardcoded type projection for now)
            keep_all:           Colon-separated list of CSV PROJECTED type sequences which should not be downsampled or used to compute
                                the dataset balance (e.g. use this for a rare 10-sample class you don't want to lose any samples of and
                                don't want to cause the whole dataset to shrink too much by balancing to its level)
            dedup_funcs:        Eliminate duplicate functions from the dataset
        '''
        self.input_params = input_params
        self.root = root
        self._cached_batch:List[Data] = None    # cached list of Data objects
        self._cached_batchidx:int = None        # batch index for the cached batch
        self._dataset_len:int = None            # amazingly, len() gets called ALL the time...cache this value

        if self.input_params and self.input_params_path.exists():
            print(f'Warning: input_params dict supplied but saved .json file also found')
            print(f'input_params will be IGNORED in favor of saved .json file ({self.input_params_path})')

        if self.input_params_path.exists():
            self._load_params()
        elif self.input_params:
            self._save_params()
        else:
            raise Exception(f'No input_params specified and no saved input_params JSON found at {self.input_params_path}')

        super().__init__(root, transform, pre_transform, pre_filter)

    def _load_params(self):
        with open(self.input_params_path, 'r') as f:
            self.input_params = json.load(f)

    def _save_params(self):
        self.input_params_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.input_params_path, 'w') as f:
            json.dump(self.input_params, f, indent=2)

    @property
    def input_params_path(self) -> Path:
        return Path(self.raw_dir)/'input_params.json'

    @property
    def exp_runs_filename(self):
        return 'experiment_runs.csv'

    @property
    def exp_runs_path(self) -> Path:
        return Path(self.raw_dir)/self.exp_runs_filename

    @property
    def funcs_path(self) -> Path:
        return Path(self.raw_dir)/'functions.csv'

    @property
    def binaries_path(self) -> Path:
        return Path(self.raw_dir)/'binaries.csv'

    @property
    def variables_path(self) -> Path:
        return Path(self.raw_dir)/'variables.csv'

    @property
    def unfiltered_variables_path(self) -> Path:
        '''
        Holds all variables before filtering
        (dropping component vars, balancing, etc)
        '''
        return Path(self.raw_dir)/'unfiltered_variables.csv'

    @property
    def raw_file_names(self):
        # return ['locals.csv', 'functions.csv', 'function_params.csv']
        # if we have this one, we finished "downloading" (if desired)
        return [self.exp_runs_filename]

    @property
    def process_finished_filename(self):
        return 'processing_finished'

    @property
    def processed_file_names(self):
        # Documentation says:
        #   "The name of the files in the self.processed_dir folder that must be present in
        #   order to skip processing."
        # ----
        # So just try using this sentinel file to indicate we are done processing since
        # the actual files will be dynamic based on content of dataset (batching Data objects up into files)

        # NOTE: are we saving locals separate from params?
        # --> right now just combine all variables into a single dataset, but later
        # this could be something to try (if nothing else, just to see if one performs better)

        return [self.process_finished_filename]

    @property
    def copy_data(self) -> bool:
        return self.input_params['copy_data'] if 'copy_data' in self.input_params else False

    @property
    def drop_component(self) -> bool:
        return self.input_params['drop_component'] if 'drop_component' in self.input_params else False

    @property
    def include_component(self) -> bool:
        # This is the form typically used and I found myself converting all over the place
        return bool(not self.drop_component)

    @property
    def balance_dataset(self) -> bool:
        return self.input_params['balance_dataset'] if 'balance_dataset' in self.input_params else False

    @property
    def keep_all(self) -> List[str]:
        return self.input_params['keep_all'].split(':') if 'keep_all' in self.input_params else []

    @property
    def dedup_funcs(self) -> bool:
        return self.input_params['dedup_funcs'] if 'dedup_funcs' in self.input_params else False

    @property
    def structural_only(self) -> bool:
        return self.input_params['structural_only'] if 'structural_only' in self.input_params else True

    @property
    def node_typeseq_len(self) -> int:
        return self.input_params['node_typeseq_len'] if 'node_typeseq_len' in self.input_params else 3

    @property
    def max_hops(self) -> int:
        return self.input_params['max_hops'] if 'max_hops' in self.input_params else 3

    def _generate_exp_runs_df(self) -> pd.DataFrame:
        run_data = []
        raw_dir = Path(self.raw_dir)

        for rid, run_folder in enumerate(self.input_params['experiment_runs']):
            rf = Path(run_folder)
            run_data.append((rid, rf,
                raw_dir/f'run{rid}_binaries.csv' if self.copy_data else rf/'binaries.csv',
                raw_dir/f'run{rid}_functions.csv' if self.copy_data else rf/'functions.csv',
                raw_dir/f'run{rid}_function_params.csv' if self.copy_data else rf/'function_params.csv',
                raw_dir/f'run{rid}_locals.csv' if self.copy_data else rf/'locals.csv'))

        return pd.DataFrame.from_records(run_data, columns=['RunGid','RunFolder','BinariesCsv','FuncsCsv','ParamsCsv','LocalsCsv'])

    def _copy_raw_csvs(self, exp_runs_df:pd.DataFrame):
        '''
        Copy raw csv files locally
        '''
        if not self.copy_data:
            return

        for i in range(len(exp_runs_df)):
            rf = exp_runs_df.iloc[i].RunFolder
            shutil.copy(rf/'binaries.csv', exp_runs_df.iloc[i].BinariesCsv)
            shutil.copy(rf/'functions.csv', exp_runs_df.iloc[i].FuncsCsv)
            if (rf/'function_params.csv').exists():
                shutil.copy(rf/'function_params.csv', exp_runs_df.iloc[i].ParamsCsv)
            if (rf/'locals.csv').exists():
                shutil.copy(rf/'locals.csv', exp_runs_df.iloc[i].LocalsCsv)

    def _generate_global_binids(self, exp_runs:pd.DataFrame):
        base_gid = 1000     # start here

        bindf_list = []

        for i in range(len(exp_runs)):
            bin_df = pd.read_csv(exp_runs.iloc[i].BinariesCsv)
            # CLS: I didn't have the full path to the debug binary saved in the table, so hacking
            # this in here now so I don't have to regenerate everything
            bin_df['FolderName'] = bin_df.apply(lambda x: f'{x.BinaryId}.{x.Name[:-3] if x.Name.endswith(".so") else x.Name}', axis=1)
            bin_df['RunGid'] = exp_runs.iloc[i].RunGid
            bin_df['OrigBinaryId'] = bin_df['BinaryId']
            bin_df['BinaryId'] = bin_df.BinaryId + base_gid

            bindf_list.append(bin_df)

            # update base_gid for the next exp_run entry
            next_base_gid = base_gid + int(len(bin_df)/1000)*1000 + 1000
            base_gid = next_base_gid

        return pd.concat(bindf_list).reset_index(drop=True)

    def _convert_run_binids(self, bin_df:pd.DataFrame, exp_run_col:str):
        '''
        Read the csv file specified in exp_run_col of the exp_runs table, and
        convert the BinaryId column to use the global binary id for this dataset
        (as mapped in bin_df)

        bin_df: The combined binaries df with columns
                    BinaryId - the new global binary id
                    OrigBinaryId - the original (local) binary id
        '''
        # maps (RunGid, OrigBinaryId) -> BinaryId
        bid_lookup = bin_df.set_index(['RunGid','OrigBinaryId'])[['BinaryId']].to_dict()['BinaryId']

        def do_convert_run_binids(rungid_groupby):
            df_list = []
            print(f'Combining {exp_run_col}...')
            for rungid, exp_run in tqdm(rungid_groupby, total=len(rungid_groupby)):
                assert len(exp_run) == 1, f'Expected 1 exp run in group by, found {len(exp_run)}'

                # filter down to only the run of interest
                csvfile = Path(exp_run.iloc[0][exp_run_col])

                if csvfile.stat().st_size < 5:
                    df_list.append(pd.DataFrame())   # this is an empty file - pandas throws an exception if we call read_csv()
                else:
                    run_df = pd.read_csv(csvfile)
                    run_df['BinaryId'] = run_df.BinaryId.apply(lambda bid: bid_lookup[(rungid, bid)])
                    df_list.append(run_df)

            # combine since we grabbed every occurrence of this df type
            return pd.concat(df_list).reset_index(drop=True)

        return do_convert_run_binids

    @staticmethod
    def build_hash_df(bin_df:pd.DataFrame) -> pd.DataFrame:
        '''
        Build the function hash dataframe for each binary in this dataset
        by running the TyGR angr_func_hash script
        '''
        console = Console()
        script_dir = (Path(__file__)/'../../../scripts').resolve()
        # script_dir/'angr_func_hash.py'

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            input_bins = tmp/'bins'
            out_folder = tmp/'output'

            input_bins.mkdir()

            for i in range(len(bin_df)):
                # NOTE: only compute hashes on the DEBUG build
                # (binaries are identical, func addrs are too, DEBUG gives us func names)
                debug_bin = bin_df.iloc[i].DebugBinary
                bid = bin_df.iloc[i].BinaryId
                if not Path(debug_bin).exists():
                    # CLS: working around issue with .so names not being generated properly
                    console.print(f'[yellow]Skipping binary {debug_bin} (does not exist at this path)...')
                    continue

                # prepend binary id to ensure unique names
                shutil.copy2(debug_bin, input_bins/f'{bid}.{Path(debug_bin).name}')
                # os.symlink(debug_bin, input_bins/f'{bid}.{Path(debug_bin).name}')

            # run angr_func_hash on each debug binary
            ncores = multiprocessing.cpu_count()
            print(f'Using {ncores} cores')
            p = subprocess.run([script_dir/'angr_func_hash.py', input_bins, '--results', out_folder, f'-j{ncores}'])
            if p.returncode != 0:
                # NOTE: not sure we want to fail here, but right now idk what this might imply so I
                # want to catch it if/when it happens
                raise Exception(f'angr_func_hash failed with return code {p.returncode}')

            # bin_name is contained (over and over) inside each JSON file - we can just grab all of them
            # and collect binary name from there
            df_list = []
            for json_file in out_folder.glob('**/*.json'):
                with open(json_file, 'r') as f:
                    func_hashes = json.load(f)

                if not func_hashes:
                    console.print(f'[yellow]No hashes computed for {json_file} (timed out?)')
                    continue

                # binary id the is same for everything in this file
                first_key = list(func_hashes.keys())[0]
                filename = Path(func_hashes[first_key]['bin_name'][0]).name

                # we named each file "<bid>.<name>" above - recover bid from filename
                bid = int(filename.split('.')[0])

                df_list.append(pd.DataFrame(
                    [(bid, entry['name'], fhash) for fhash, entry in func_hashes.items()],
                    columns=['BinaryId','FunctionName_Debug','Hash']    # FunctionName_Debug just to match funcs_df for merging
                ))

            return pd.concat(df_list)

    @staticmethod
    def dedup_by_function(bin_df:pd.DataFrame, funcs_df:pd.DataFrame,
                        params_df:pd.DataFrame, locals_df:pd.DataFrame):
        '''
        Deduplicate the data by using the TyGR script to compute function hashes
        and eliminating duplicate functions from the dataset (and their associated locals/params)
        '''
        console = Console()
        hash_df = TypeSequenceDataset.build_hash_df(bin_df)

        # drop duplicate functions from each table before merging
        # -------------------------------------------------------
        # NOTE: since we are merging on BinaryId/FunctionName_Debug we need to drop duplicate function names
        # occuring in the same binary from BOTH lists (I think these are static functions compiled in separate
        # translation units)
        orig_count = len(funcs_df)
        orig_hash = len(hash_df)

        funcs_df = funcs_df.drop_duplicates(subset=['BinaryId','FunctionName_Debug'], keep=False)
        hash_df = hash_df.drop_duplicates(subset=['BinaryId','FunctionName_Debug'], keep=False)
        console.print(f'Removing {orig_hash-len(hash_df):,} functions from hashed set with duplicate names (in the same binary)')
        console.print(f'Removing {orig_count-len(funcs_df):,} functions from funcs set with duplicate names (in the same binary)')

        # eliminate duplicates by hash
        # ----------------------------
        orig_count = len(funcs_df)  # reset this for stats below

        funcs_df = funcs_df.merge(hash_df, how='left', on=['BinaryId','FunctionName_Debug'])
        funcs_df = funcs_df.dropna(subset='Hash')
        num_nans = orig_count - len(funcs_df)
        funcs_df = funcs_df.drop_duplicates(subset='Hash')
        num_dups = orig_count - num_nans - len(funcs_df)

        print(f'Started with {orig_count:,} functions (unique names per binary)')
        print(f'Dropped {num_nans:,} functions with no hash ({num_nans/orig_count*100:.2f}%)')
        console.print(f'[bold orange1]Dropped {num_dups:,} duplicate functions by hash ({num_dups/orig_count*100:.2f}%)')
        console.print(f'[bold blue]Retained {len(funcs_df):,} functions ({len(funcs_df)/orig_count*100:.2f}%)')

        orig_locals = len(locals_df)
        orig_params = len(params_df)

        # drop variables associated with duplicate functions
        # (we have to match by both BinaryId/FunctionStart, not just FunctionStart)
        ff = funcs_df.set_index(['BinaryId','FunctionStart'])
        pp = params_df.set_index(['BinaryId','FunctionStart'])
        ll = locals_df.set_index(['BinaryId','FunctionStart'])

        params_df = pp[pp.index.isin(ff.index)].reset_index()   # don't drop=True - we want to keep BinaryId/FunctionStart
        locals_df = ll[ll.index.isin(ff.index)].reset_index()

        dropped_locals = orig_locals - len(locals_df)
        dropped_params = orig_params - len(params_df)
        console.print(f'[bold blue]Retained {len(locals_df):,} locals ({len(locals_df)/orig_locals*100:.2f}%) - dropped {dropped_locals:,}')
        console.print(f'[bold blue]Retained {len(params_df):,} params ({len(params_df)/orig_params*100:.2f}%) - dropped {dropped_params:,}')

        funcs_df = funcs_df.reset_index(drop=True)
        params_df = params_df.reset_index(drop=True)
        locals_df = locals_df.reset_index(drop=True)

        return funcs_df, params_df, locals_df

    def download(self):
        # generate a dataframe/csv file mapping runs/run gids to their associated files

        # NOTE FOR LATER: if we ever do need to download (e.g. a tar.gz from google drive), we
        # could use gdown: https://github.com/wkentaro/gdown

        console = Console()
        print(f'Downloading data...')

        df = self._generate_exp_runs_df()

        # NOTE - I don't think this makes sense anymore since I'm going to be
        # combining everything into a unified df (unless I later run into memory limitations
        # and have to keep things separated...)
        # if self.copy_data:
        #     self._copy_raw_csvs(df)

        # generate global binids, unified csvs for dataset
        bin_df = self._generate_global_binids(df)

        rungid_gb = df.groupby('RunGid')

        funcs_df = rungid_gb.pipe(self._convert_run_binids(bin_df, 'FuncsCsv'))
        params_df = rungid_gb.pipe(self._convert_run_binids(bin_df, 'ParamsCsv'))
        locals_df = rungid_gb.pipe(self._convert_run_binids(bin_df, 'LocalsCsv'))

        if self.dedup_funcs:
            funcs_df, params_df, locals_df = TypeSequenceDataset.dedup_by_function(bin_df, funcs_df, params_df, locals_df)

        # combine all variables into one table
        params_df['Vartype'] = 'p'
        locals_df['Vartype'] = 'l'
        locals_df['IsReturnType_Debug'] = False     # fill these in so they aren't NaN
        locals_df['IsReturnType_Strip'] = False
        vars_df = pd.concat([locals_df, params_df])

        print(f'Binaries table memory usage: {pretty_memsize_str(bin_df.memory_usage(deep=True).sum())}')
        print(f'Funcs table memory usage: {pretty_memsize_str(funcs_df.memory_usage(deep=True).sum())}')
        print(f'Variables table memory usage: {pretty_memsize_str(vars_df.memory_usage(deep=True).sum())}')

        # write combined tables to local csvs
        bin_df.to_csv(self.binaries_path, index=False)
        funcs_df.to_csv(self.funcs_path, index=False)
        vars_df.to_csv(self.unfiltered_variables_path, index=False)

        vars_df = self._filter_vars_df(vars_df)

        if self.balance_dataset:
            print(f'Balancing dataset...')
            vars_df = self._balance_dataset(vars_df)

        print(f'Final leaf category balance:')
        print(vars_df.groupby('LeafCategory').count().FunctionStart.sort_values())
        print(f'Final pointer levels balance:')
        print(vars_df.groupby('PtrLevels').count().FunctionStart.sort_values())
        console.print(f'[bold blue]Final dataset size: {len(vars_df):,} variables across {len(funcs_df):,} functions')

        vars_df.to_csv(self.variables_path, index=False)

        # now write the exp_runs file to indicate download is complete
        df.to_csv(self.exp_runs_path, index=False)

    def _balance_dataset(self, vars_df:pd.DataFrame, raw:bool=False) -> pd.DataFrame:

        # Set aside ALL our "unicorn" samples and then balancing down the remainder
        # Unicorns: minorities for PtrL3, PtrL2, and LeafCategory (enum/func/union)
        self.input_params['keep_all'] = 'ENUM:UNION:FUNC'
        print(f'NOTE: ignoring keep-all parameter and using hardcoded keep_all value of {self.keep_all}')

        keep_p1s = vars_df.PtrL1.isin(['A','P'])     # this includes ALL non-leaf PtrL2/PtrL3 (plus other variants where PtrL1 is non-leaf)
        keep_leaf_cats = vars_df.LeafCategory.isin(self.keep_all)
        keep_unicorns = keep_leaf_cats | keep_p1s if self.keep_all else keep_p1s

        keep_df = vars_df.loc[keep_unicorns,:]      # keep all of these

        if self.keep_all:
            print(f'Keeping all {len(vars_df[keep_leaf_cats]):,} vars from the set {self.keep_all}')
            print(f'Keeping all {len(vars_df[keep_p1s]):,} vars from the set of non-leaf PtrL1\'s')
            print(f'=> keeping all {len(keep_df):,} "unicorns"')

        print(f'Remaining leaf categories (non-unicorns):')
        print(vars_df.loc[~keep_unicorns,:].groupby('LeafCategory').count().FunctionStart.sort_values())

        # sample_n = vars_df.loc[~keep_unicorns,:].groupby('LeafCategory').count().FunctionStart.min()

        # NOTE: balance the leaf builtins down to the level of struct pointers...when I balanced it
        # down to the LEAF structs this cuts WAY too much
        num_struct_ptrs = len(vars_df[(vars_df.PtrL1!='L')&(vars_df.LeafCategory=='STRUCT')])
        sample_n = num_struct_ptrs

        leaf_structs = vars_df.loc[(~keep_unicorns)&(vars_df.LeafCategory=='STRUCT'),:]
        leaf_builtins = vars_df.loc[(~keep_unicorns)&(vars_df.LeafCategory=='BUILTIN'),:]

        print(f'There are {num_struct_ptrs:,} STRUCT "pointers" (non-leaf vars) - balance BUILTINs down to this level')
        print(f'Keeping all {len(leaf_structs):,} remaining leaf STRUCTs')

        if sample_n < len(leaf_builtins):
            print(f'Sample {sample_n:,} of the {len(leaf_builtins):,} remaining leaf BUILTINs')
            sampled_df = leaf_builtins.groupby('LeafCategory').sample(n=sample_n, random_state=33)
        else:
            print(f'Keeping all {len(leaf_builtins):,} of the leaf BULTINS (since we can\'t sample {sample_n:,} of them)')
            sampled_df = leaf_builtins

        balanced_df = pd.concat([keep_df, leaf_structs, sampled_df]).reset_index()

        percent_dropped = 100-len(balanced_df)/len(vars_df)*100
        print(f'Dropped {percent_dropped:.2f}% of the original dataset (from {len(vars_df):,} down to {len(balanced_df):,})')

        return balanced_df

    def _filter_vars_df(self, vars_df:pd.DataFrame) -> pd.DataFrame:
        # NOTE: skipping all return types for now - later if we want to predict
        # these (as opposed to letting Ghidra type propagate) then we need to
        # treat them special in VariableGraphBuilder by joining all ReturnStmt nodes
        # instead of looking for references (as the return expression is not a named variable)
        # rtypes = params_df.loc[params_df.IsReturnType_Debug,:]

        print(f'Dropped {len(vars_df.loc[vars_df.IsReturnType_Debug,:]):,} return types')
        vars_df = vars_df.loc[~vars_df.IsReturnType_Debug,:]

        if self.drop_component:
            # params don't have COMP entries, so just drop it from locals
            num_comp_vars = len(vars_df[vars_df.TypeSeq_Debug=='COMP'])
            print(f'Dropped {num_comp_vars:,} COMP local variables')

            vars_df = vars_df.loc[vars_df.TypeSeq_Debug!='COMP',:]

        # TEMP - drop all FUNC vars (PTR->FUNC is only valid form of function pointer)
        num_func_vars = len(vars_df[vars_df.TypeSeq_Debug=='FUNC'])
        if num_func_vars > 0:
            print(f'TEMP - dropping {num_func_vars:,} FUNC vars (until these get corrected upstream)')
            vars_df = vars_df.loc[vars_df.TypeSeq_Debug!='FUNC',:]

        return vars_df


    @property
    def batchsize(self) -> int:
        return 10000  # FIXME: assuming fixed batchsize for now

    def _get_varfile_for_idx(self, data_idx:int) -> Path:
        batch_idx = int(data_idx/self.batchsize)
        return Path(self.processed_dir)/f'vars-{self.batchsize}_{batch_idx}.pt'

    def _save_batch(self, current_batch, i):
        torch.save(current_batch, self._get_varfile_for_idx(i))

    def _save_data_in_batches(self, gen_data_objs:Generator, expected_total_vars:int):
        current_batch = []
        from tqdm import tqdm
        for i, data in enumerate(tqdm(gen_data_objs, total=expected_total_vars)):
            current_batch.append(data)
            if i % self.batchsize == (self.batchsize-1):
                self._save_batch(current_batch, i)
                current_batch = []      # reset batch

        # save leftovers
        if current_batch:
            self._save_batch(current_batch, i)

    def process(self):
        # convert data from csv files into pyg Data objects and save to .pt files
        funcs_df = pd.read_csv(self.funcs_path)
        vars_df = pd.read_csv(self.variables_path)

        include_comp = bool(not self.drop_component)

        print(f'Generating var graphs and saving in batches...')
        self._save_data_in_batches(
            convert_funcvars_to_data(vars_df, funcs_df, max_hops=self.max_hops,
                                    include_comp=include_comp,
                                    structural_only=self.structural_only,
                                    node_typeseq_len=self.node_typeseq_len),
            expected_total_vars=len(vars_df)
        )

        # write processing_finished file to self.processed_dir to indicate we are done processing
        with open(Path(self.processed_dir)/self.process_finished_filename, 'w') as f:
            f.write('')

    def get(self, idx: int) -> BaseData:
        batch_idx = int(idx/self.batchsize)
        list_idx = idx % self.batchsize

        if self._cached_batchidx != batch_idx:
            vfile = self._get_varfile_for_idx(idx)
            self._cached_batch = torch.load(vfile)
            self._cached_batchidx = batch_idx

        return self._cached_batch[list_idx]

    def len(self) -> int:
        if self._dataset_len is None:
            self._dataset_len = self._calc_len()
        return self._dataset_len

    def _calc_len(self) -> int:
        var_filenames = [x.stem for x in Path(self.processed_dir).glob('vars*.pt')] # remove .pt
        largest_batch_idx_stem, largest_batch_idx = sorted([(x, int(x.split('_')[1])) for x in var_filenames],
                                    key=lambda x: x[1])[-1]
        largest_batch_idx_file = (Path(self.processed_dir)/largest_batch_idx_stem).with_suffix('.pt')
        data_list = torch.load(largest_batch_idx_file)

        return largest_batch_idx*self.batchsize + len(data_list)

    @property
    def num_classes(self) -> int:
        data = torch.load(self._get_varfile_for_idx(0))[0]
        return data.y.size(-1)  # all y vectors should be the same size