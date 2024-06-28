import pandas as pd
from pathlib import Path
import subprocess
import torch
from torch_geometric.data import InMemoryDataset
from tqdm.auto import trange

from .typesequencedataset import TypeSequenceDataset

# NOTE: these probably belong elsewhere (in experiment.py?) but in a hurry...

def extract_expname(run_folder:str):
    run_folder = Path(run_folder)
    exp_folder = [x for x in run_folder.parts if '.exp' in x]
    return Path(exp_folder[0]).stem

def extract_runname(run_folder:str):
    run_folder = Path(run_folder)
    return run_folder.name

class InMemTypeSequenceDataset(InMemoryDataset):
    '''
    For datasets that fit, use an in-memory dataset for HUGE performance boost!!
    '''
    def __init__(self, dataset:TypeSequenceDataset):
        self.src_dataset = dataset

        super().__init__(dataset.root,
            transform=dataset.transform,
            pre_transform=dataset.pre_transform,
            pre_filter=dataset.pre_filter)

        self.load(self.processed_paths[0])
        # For PyG<2.4:
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # we need these to be here before we can copy to our version
        return self.src_dataset.raw_file_names

    @property
    def variables_path(self):
        return self.src_dataset.variables_path

    @property
    def unfiltered_variables_path(self) -> Path:
        return self.src_dataset.unfiltered_variables_path

    @property
    def binaries_path(self) -> Path:
        return self.src_dataset.binaries_path

    @property
    def funcs_path(self) -> Path:
        return self.src_dataset.funcs_path

    @property
    def drop_component(self) -> bool:
        '''True if this dataset drops all COMP variables'''
        return self.src_dataset.drop_component

    @property
    def include_component(self) -> bool:
        return self.src_dataset.include_component

    @property
    def balance_dataset(self) -> bool:
        return self.src_dataset.balance_dataset

    @property
    def structural_only(self) -> bool:
        return self.src_dataset.structural_only

    @property
    def node_typeseq_len(self) -> int:
        return self.src_dataset.node_typeseq_len

    @property
    def max_hops(self) -> int:
        return self.src_dataset.max_hops

    @property
    def input_params(self) -> dict:
        return self.src_dataset.input_params

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

    def _balance_dataset(self, vars_df:pd.DataFrame, raw:bool=False) -> pd.DataFrame:
        # pass through to src_dataset
        return self.src_dataset._balance_dataset(vars_df, raw)

    def _filter_vars_df(self, vars_df:pd.DataFrame) -> pd.DataFrame:
        # pass through to src_dataset
        return self.src_dataset._filter_vars_df(vars_df)

    @property
    def processed_file_names(self):
        return ['IN_MEMORY_COPY.pt']

    def download(self):
        # Download to `self.raw_dir`
        print(f'Looks like the raw source dataset is missing for: {self.src_dataset}')

    def process(self):
        # Read data into huge `Data` list.
        for fname in self.src_dataset.processed_file_names:
            if not (Path(self.root)/'processed'/fname).exists():
                raise Exception(f'Looks like the raw source dataset is missing for: {self.src_dataset}')

        folder_size_str = subprocess.check_output(f'du -ch {self.src_dataset.root} | tail -1',
            shell=True).decode('utf-8').split()[0]
        print(f'Loading dataset into memory of size {folder_size_str}')

        # data_list = list(self.src_dataset)      # we assume it fits in memory...lol
        ds_len = len(self.src_dataset)
        data_list = [self.src_dataset[i] for i in trange(ds_len)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
        # For PyG<2.4:
        # torch.save(self.collate(data_list), self.processed_paths[0])