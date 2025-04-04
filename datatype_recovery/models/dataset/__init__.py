from .split import split_train_test
from .variablegraphbuilder import VariableGraphBuilder, VariableGraphViewer
from .typesequencedataset import TypeSequenceDataset
from .inmemtypesequencedataset import InMemTypeSequenceDataset
from .simpletypedataset import SimpleTypeDataset


import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from typing import Tuple

from pathlib import Path

def load_dataset_from_path(dataset_path:Path):
    if not dataset_path.exists():
        raise Exception(f'Dataset {dataset_path} does not exist!')

    in_mem_file = dataset_path/'processed'/'IN_MEMORY_COPY.pt'
    hetero_file = dataset_path/'processed'/'heterodata.pt'

    if hetero_file.exists():
        return SimpleTypeDataset(dataset_path)

    ds = TypeSequenceDataset(dataset_path)
    if in_mem_file.exists():
        return InMemTypeSequenceDataset(ds)
    else:
        from rich.console import Console
        console = Console()
        console.print(f'[bold yellow] Warning: no in-memory dataset exists! Using file-based TypeSequenceDataset')
    return ds

def max_typesequence_len_in_dataset(dataset_path:Path) -> int:
    '''Calculate the max true type sequence length in the dataset'''
    print(f'Calculating max true sequence length in dataset {dataset_path.name}')
    dataset_datafmt = load_dataset_from_path(dataset_path)
    myloader = DataLoader(dataset_datafmt, batch_size=1)
    # no transforms, so our data.y is (N, 22)
    return max([data.y.shape[0] for data in tqdm(myloader, total=len(myloader))])
