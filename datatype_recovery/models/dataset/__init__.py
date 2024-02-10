from .variablegraphbuilder import VariableGraphBuilder, VariableGraphViewer
from .typesequencedataset import TypeSequenceDataset
from .inmemtypesequencedataset import InMemTypeSequenceDataset

from pathlib import Path

def load_dataset_from_path(dataset_path:Path):
    if not dataset_path.exists():
        raise Exception(f'Dataset {dataset_path} does not exist!')

    in_mem_file = dataset_path/'processed'/'IN_MEMORY_COPY.pt'

    ds = TypeSequenceDataset(dataset_path)
    if in_mem_file.exists():
        return InMemTypeSequenceDataset(ds)
    return ds
