import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Callable, Any, List, Generator

import torch
from torch_geometric.data import InMemoryDataset, HeteroData
from torch_geometric.data.data import BaseData

import astlib
from varlib.datatype import DataType
from wildebeest.utils import pretty_memsize_str
from .inmemtypesequencedataset import extract_expname, extract_runname
from .variablegraphbuilder import VariableHeteroGraphBuilder
from .encoding import TypeEncoder, HeteroNodeEncoder

def convert_funcvars_to_heterodata_gb(funcs_df:pd.DataFrame, max_hops:int) -> Callable:
    '''
    Pandas group by function that converts function variables (locals or params) into
    PyG Data objects. This function should be applied to a groupby object via groupby.pipe()

    gb_bin_and_func: Pandas groupby object that has grouped the desired dataframe
                     by (BinaryId, FunctionStart)
    '''
    def do_convert_funcvars_to_data(gb_bin_and_func) -> Generator[HeteroData,None,None]:
        # groupby[0]: tuple of grouped-by column values
        # groupby[1]: data frame for this subset of the data
        for gb_vals, df in gb_bin_and_func:
            # we can reuse the AST object here within a function
            bid, addr = gb_vals
            ast_file = funcs_df[(funcs_df.FunctionStart==addr)&(funcs_df.BinaryId==bid)].iloc[0].AstJson_Strip
            ast = astlib.read_json(ast_file)

            for i in range(len(df)):
                name_strip = df.iloc[i].Name_Strip
                builder = VariableHeteroGraphBuilder(name_strip, ast, max_hops, sdb=None)

                try:
                    data = builder.build(bid)
                except:
                    # keeping this here so if something goes wrong on a large dataset I can see what binary/function/variable
                    # had the issue! (from varid)
                    print(f'Failed to build variable graph for variable {name_strip} (bid={bid},func={addr:x})', flush=True)
                    raise

                # Debug holds ground truth prediction
                data.y = TypeEncoder.encode(DataType.from_json(df.iloc[i].TypeJson_Debug))
                yield data

                # yield Data(x=node_list, edge_index=edge_index, y=y, varid=varid, edge_attr=edge_attr)

    return do_convert_funcvars_to_data

def convert_funcvars_to_heterodata(df:pd.DataFrame, funcs_df:pd.DataFrame, max_hops:int) -> Generator[HeteroData,None,None]:
    '''
    Converts function variables (locals or params) into PyG Data objects
    given either the locals or params data frame
    '''
    return df.groupby(['BinaryId','FunctionStart']).pipe(convert_funcvars_to_heterodata_gb(funcs_df, max_hops))

class SimpleTypeDataset(InMemoryDataset):
    def __init__(self, root:str, input_params:dict=None, transform=None, pre_transform=None, pre_filter=None):
        '''
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
            max_hops:           Max hops for variable graphs during generation
        '''
        self.input_params = input_params
        self.root = root
        self._cached_batch:List[HeteroData] = None    # cached list of Data objects
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

        self.hetero_data_list = torch.load(self.processed_paths[0])

        # NOTE: something isn't working with collate function
        # self.load(self.processed_paths[0])

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

    def read_exp_runs_csv(self) -> pd.DataFrame:
        return pd.read_csv(self.src_dataset.exp_runs_path)

    def read_vars_csv(self) -> pd.DataFrame:
        return pd.read_csv(self.variables_path)

    def read_binaries_csv(self) -> pd.DataFrame:
        return pd.read_csv(self.binaries_path)

    def read_funcs_csv(self) -> pd.DataFrame:
        return pd.read_csv(self.funcs_path)

    @property
    def full_binaries_table(self) -> pd.DataFrame:
        '''
        Join the binaries csv and exp_runs csv for a full binaries table
        '''
        df = self.read_binaries_csv().merge(
                self.read_exp_runs_csv(),
                on='RunGid',
                how='left')

        df['ExpName'] = df.RunFolder.apply(extract_expname)
        df['RunName'] = df.RunFolder.apply(extract_runname)

        return df

    @property
    def raw_file_names(self):
        # return ['locals.csv', 'functions.csv', 'function_params.csv']
        # if we have this one, we finished "downloading" (if desired)
        return [self.exp_runs_filename]

    @property
    def processed_file_names(self):
        return ['heterodata.pt']

    @property
    def max_hops(self) -> int:
        return self.input_params['max_hops'] if 'max_hops' in self.input_params else 3

    @property
    def limit(self) -> int:
        return self.input_params['limit'] if 'limit' in self.input_params else None

    def _generate_exp_runs_df(self) -> pd.DataFrame:
        run_data = []
        raw_dir = Path(self.raw_dir)

        for rid, run_folder in enumerate(self.input_params['experiment_runs']):
            rf = Path(run_folder)
            run_data.append((rid, rf,
                                rf/'binaries.csv',
                                rf/'functions.csv',
                                rf/'function_params.csv',
                                rf/'locals.csv'))

        return pd.DataFrame.from_records(run_data, columns=['RunGid','RunFolder','BinariesCsv','FuncsCsv','ParamsCsv','LocalsCsv'])

    def _generate_global_binids(self, exp_runs:pd.DataFrame):
        base_gid = 1000     # start here

        bindf_list = []

        for i in range(len(exp_runs)):
            bin_df = pd.read_csv(exp_runs.iloc[i].BinariesCsv)
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
        def do_convert_run_binids(rungid_groupby):
            df_list = []
            print(f'Combining {exp_run_col}...')
            for rungid, exp_run in tqdm(rungid_groupby, total=len(rungid_groupby)):
                assert len(exp_run) == 1, f'Expected 1 exp run in group by, found {len(exp_run)}'

                # filter down to only the run of interest
                runbin = bin_df.loc[bin_df.RunGid==rungid, :]
                run_df = pd.read_csv(exp_run.iloc[0][exp_run_col])
                run_df['BinaryId'] = run_df.BinaryId.apply(
                    lambda bid: runbin[runbin.OrigBinaryId==bid].BinaryId.iloc[0]
                )
                df_list.append(run_df)

            # combine since we grabbed every occurrence of this df type
            return pd.concat(df_list).reset_index(drop=True)

        return do_convert_run_binids

    def download(self):
        # generate a dataframe/csv file mapping runs/run gids to their associated files

        # NOTE FOR LATER: if we ever do need to download (e.g. a tar.gz from google drive), we
        # could use gdown: https://github.com/wkentaro/gdown

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
        vars_df.to_csv(self.variables_path, index=False)

        # now write the exp_runs file to indicate download is complete
        df.to_csv(self.exp_runs_path, index=False)

    def _filter_vars_df(self, vars_df:pd.DataFrame) -> pd.DataFrame:
        # NOTE: skipping all return types for now - later if we want to predict
        # these (as opposed to letting Ghidra type propagate) then we need to
        # treat them special in VariableGraphBuilder by joining all ReturnStmt nodes
        # instead of looking for references (as the return expression is not a named variable)
        # rtypes = params_df.loc[params_df.IsReturnType_Debug,:]

        # DROP return types
        print(f'Dropped {len(vars_df.loc[vars_df.IsReturnType_Debug,:]):,} return types')
        vars_df = vars_df.loc[~vars_df.IsReturnType_Debug,:]

        # DROP COMP vars
        # params don't have COMP entries, so just drop it from locals
        num_comp_vars = len(vars_df[vars_df.TypeSeq_Debug=='COMP'])
        print(f'Dropped {num_comp_vars:,} COMP local variables')
        vars_df = vars_df.loc[vars_df.TypeSeq_Debug!='COMP',:]

        # TEMP - drop all FUNC vars (PTR->FUNC is only valid form of function pointer)
        num_func_vars = len(vars_df[vars_df.TypeSeq_Debug=='FUNC'])
        if num_func_vars > 0:
            print(f'TEMP - dropping {num_func_vars:,} FUNC vars (until these get corrected upstream)')
            vars_df = vars_df.loc[vars_df.TypeSeq_Debug!='FUNC',:]

        if self.limit:
            print(f'Keeping only the first {self.limit:,} variables')
            vars_df = vars_df[:self.limit]

        return vars_df

    def _save_data_in_batches(self, gen_data_objs:Generator, expected_total_vars:int):
        current_batch = []
        for i, data in enumerate(tqdm(gen_data_objs, total=expected_total_vars)):
            current_batch.append(data)
            if i % self.batchsize == (self.batchsize-1):
                self._save_batch(current_batch, i)
                current_batch = []      # reset batch

        # save leftovers
        if current_batch:
            self._save_batch(current_batch, i)

    def generate_data_from_vars(self) -> List[HeteroData]:
        # convert data from csv files into pyg Data objects and save to .pt file
        funcs_df = pd.read_csv(self.funcs_path)
        vars_df = pd.read_csv(self.variables_path)

        print(f'Generating var graphs and saving in batches...')
        data_list = [x for x in tqdm(convert_funcvars_to_heterodata(vars_df, funcs_df, self.max_hops), total=len(vars_df))]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        return data_list

    # TEMP: only implementing these since we can't use self.save/load
    def get(self, idx: int) -> BaseData:
        return self.hetero_data_list[idx]

    def len(self) -> int:
        return len(self.hetero_data_list)

    def process(self):
        torch.save(self.generate_data_from_vars(), self.processed_paths[0])

        # NOTE: something isn't working with collate function
        # self.save(self.generate_data_from_vars(), self.processed_paths[0])
        # For PyG<2.4:
        # torch.save(self.collate(data_list), self.processed_paths[0])
