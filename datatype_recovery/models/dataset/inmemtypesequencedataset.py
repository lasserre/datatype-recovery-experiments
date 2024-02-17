import pandas as pd
from pathlib import Path
import subprocess
import torch
from torch_geometric.data import InMemoryDataset
from tqdm.auto import trange

from .typesequencedataset import TypeSequenceDataset

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
    def drop_component(self) -> bool:
        '''True if this dataset drops all COMP variables'''
        return self.src_dataset.drop_component

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