from itertools import chain
import json
from pathlib import Path
import pandas as pd
import shutil
from typing import List, Generator, Callable

import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.data.data import BaseData

import astlib
from datatype_recovery.models.dataset import VariableGraphBuilder
from datatype_recovery.models.dataset.encoding import encode_typeseq

def convert_funcvars_to_data_gb(funcs_df:pd.DataFrame, rungid:int, vartype:str, max_hops:int) -> Callable:
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
            ast, slib = astlib.json_to_ast(ast_file)

            for i in range(len(df)):
                name_strip = df.iloc[i].Name_Strip

                # Debug holds ground truth prediction
                type_seq = df.iloc[i].TypeSeq_Debug.split(',')  # list of str

                # ------------------------
                # FIXME: only using first type sequence element for bare bones model
                type_seq = type_seq[:1]
                # ------------------------

                builder = VariableGraphBuilder(name_strip, ast, slib)
                node_list, edge_index = builder.build_variable_graph(max_hops=max_hops)
                y = encode_typeseq(type_seq)

                varid = (rungid, bid, addr, df.iloc[i].Signature, vartype)   #  save enough metadata to "get back to" full truth data
                yield Data(x=node_list, edge_index=edge_index, y=y, varid=varid)

    return do_convert_funcvars_to_data

def convert_funcvars_to_data(df:pd.DataFrame, funcs_df:pd.DataFrame, rungid:int, vartype:str, max_hops:int) -> Generator[Data,None,None]:
    '''
    Converts function variables (locals or params) into PyG Data objects
    given either the locals or params data frame
    '''
    return df.groupby(['BinaryId','FunctionStart']).pipe(convert_funcvars_to_data_gb(funcs_df, rungid, vartype, max_hops))

class TypeSequenceDataset(Dataset):
    def __init__(self, root:str, input_params:dict=None, max_hops:int=3, transform=None, pre_transform=None, pre_filter=None):
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
        '''
        self.max_hops = max_hops
        self.input_params = input_params
        self.root = root
        self._cached_batch:List[Data] = None    # cached list of Data objects
        self._cached_batchidx:int = None        # batch index for the cached batch

        if self.input_params and self.input_params_path.exists():
            print(f'Warning: input_params dict supplied but saved .json file also found')
            print(f'input_params will be IGNORED in favor of saved .json file ({self.input_params_path})')

        if self.input_params_path.exists():
            with open(self.input_params_path, 'r') as f:
                self.input_params = json.load(f)
        elif self.input_params:
            self.input_params_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.input_params_path, 'w') as f:
                json.dump(self.input_params, f, indent=2)
        else:
            raise Exception(f'No input_params specified and no saved input_params JSON found at {self.input_params_path}')

        super().__init__(root, transform, pre_transform, pre_filter)

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

    def download(self):
        # generate a dataframe/csv file mapping runs/run gids to their associated files

        # NOTE FOR LATER: if we ever do need to download (e.g. a tar.gz from google drive), we
        # could use gdown: https://github.com/wkentaro/gdown

        print(f'Downloading data...')
        copy_data = self.input_params['copy_data']
        raw_dir = Path(self.raw_dir)

        run_data = []
        for rid, run_folder in enumerate(self.input_params['experiment_runs']):
            rf = Path(run_folder)
            run_data.append((rid, rf,
                raw_dir/f'run{rid}_binaries.csv' if copy_data else rf/'binaries.csv',
                raw_dir/f'run{rid}_functions.csv' if copy_data else rf/'functions.csv',
                raw_dir/f'run{rid}_function_params.csv' if copy_data else rf/'function_params.csv',
                raw_dir/f'run{rid}_locals.csv' if copy_data else rf/'locals.csv'))

        df = pd.DataFrame.from_records(run_data, columns=['RunGid','RunFolder','BinariesCsv','FuncsCsv','ParamsCsv','LocalsCsv'])

        if copy_data:
            # copy files locally
            for i in range(len(df)):
                rf = df.iloc[i].RunFolder
                shutil.copy(rf/'binaries.csv', df.iloc[i].BinariesCsv)
                shutil.copy(rf/'functions.csv', df.iloc[i].FuncsCsv)
                if (rf/'function_params.csv').exists():
                    shutil.copy(rf/'function_params.csv', df.iloc[i].ParamsCsv)
                if (rf/'locals.csv').exists():
                    shutil.copy(rf/'locals.csv', df.iloc[i].LocalsCsv)

        df.to_csv(self.exp_runs_path, index=False)

    @property
    def batchsize(self) -> int:
        return 1000  # FIXME: assuming fixed batchsize for now

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
        # go through experiment_runs.csv to get files instead of hardcoded locals.csv, etc
        runs_df = pd.read_csv(self.exp_runs_path)
        locals_gen_list = []
        params_gen_list = []
        expected_total_vars = 0

        for i in range(len(runs_df)):
            rungid = runs_df.iloc[i].RunGid
            locals_df = pd.read_csv(runs_df.iloc[i].LocalsCsv)
            funcs_df = pd.read_csv(runs_df.iloc[i].FuncsCsv)
            params_df = pd.read_csv(runs_df.iloc[i].ParamsCsv)

            # FIXME: skipping all return types for now - later if we want to predict
            # these (as opposed to letting Ghidra type propagate) then we need to
            # treat them special in VariableGraphBuilder by joining all ReturnStmt nodes
            # instead of looking for references (as the return expression is not a named variable)
            # rtypes = params_df.loc[params_df.IsReturnType_Debug,:]

            no_rtypes = params_df.loc[~params_df.IsReturnType_Debug,:]

            # l: local, p: param (later rt: return type)
            locals_gen_list.append(convert_funcvars_to_data(locals_df, funcs_df, rungid, vartype='l', max_hops=self.max_hops))
            params_gen_list.append(convert_funcvars_to_data(no_rtypes, funcs_df, rungid, vartype='p', max_hops=self.max_hops))

            expected_total_vars += len(no_rtypes) + len(locals_df)

        self._save_data_in_batches(chain(*locals_gen_list, *params_gen_list), expected_total_vars)

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