from itertools import chain
from pathlib import Path
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from datatype_recovery.models.dataset.encoding import *


def make_predictions_on_dataset(model_path:Path, device:str, dataset, max_true_seq_len:int) -> pd.DataFrame:
    '''
    Evaluates the model on the given dataset and returns a DataFrame containing the varid columns
    of each variable in the dataset along with its predicted type sequence (raw and corrected)

    NOTE: this is similar but separate from EvalContext.eval(); that implementation is designed for
    use during training and computes loss and accuracy metrics. This version simply makes predictions
    using the model and allows the metrics of interest to be computed later on from the resulting
    DataFrame
    '''
    model = torch.load(model_path)
    print(model)

    # take the max of model seq length and max seq length of dataset so we
    # calculate accuracy correctly (without truncating something)
    max_len = max(model.max_seq_len, max_true_seq_len)

    # prepare the data loaders
    batch_size = 512
    dataset.transform = T.Compose([ToBatchTensors(), ToFixedLengthTypeSeq(max_len, dataset.include_component)])

    # split the dataset into the part divisible by batch size and the leftovers
    # we can chain these together for performance - our metrics simply iterate
    # through all elements in the batch
    batched_total = len(dataset)-(len(dataset)%batch_size)
    batch_loader = DataLoader(dataset[:batched_total], batch_size=batch_size)
    leftovers_loader = DataLoader(dataset[batched_total:], batch_size=1)

    print(f'Running eval...')

    model.to(device)
    model.eval()

    model_outputs = []

    tseq = TypeSequence(dataset.include_component)

    for data in tqdm(chain(batch_loader, leftovers_loader), total=len(batch_loader)+len(leftovers_loader)):
        data.to(device)
        edge_attr = data.edge_attr if model.uses_edge_features else None
        out = model(data.x, data.edge_index, data.batch, edge_attr=edge_attr)

        for i, pred in enumerate(out):
            raw = tseq.decode(pred, drop_empty_elems=True)
            corrected = tseq.decode(pred, force_valid_seq=True)

            binid, funcstart, sig, vartype = data.varid[i]

            model_outputs.append((
                binid, funcstart, sig, vartype,
                ','.join(raw),
                ','.join(corrected),
            ))

    return pd.DataFrame.from_records(model_outputs, columns=['BinaryId','FunctionStart','Signature','Vartype','RawPred','Pred'])

# --------------------------------------------------------------------------------
# NOTE: I don't think I need these anymore, but I'm afraid to delete them yet lol
# Keeping them here in case I need to revive them in the near future...
# --------------------------------------------------------------------------------

# def eval_model_on_dataset(model_path:Path, device:str, dataset_path:Path) -> float:
#     '''
#     Evaluates the model on the given dataset and returns the accuracy of the corrected
#     model output against the dataset labels
#     '''
#     dataset = load_dataset_from_path(dataset_path)
#     include_comp = not dataset.drop_component
#     max_true_seq_len = max_typesequence_len_in_dataset(dataset_path)
#     return eval_model_on_subset(model_path, device, dataset, max_true_seq_len, include_comp)

# def eval_model_on_subset(model_path:Path, device:str, dataset, max_true_seq_len:int, include_comp:bool) -> float:
#     '''
#     Evaluates the model on the given subset and returns the accuracy of the corrected
#     model output against the dataset labels
#     '''
#     model = torch.load(model_path)
#     print(model)

#     # take the max of model seq length and max seq length of dataset so we
#     # calculate accuracy correctly (without truncating something)
#     max_len = max(model.max_seq_len, max_true_seq_len)

#     # prepare the data loaders
#     batch_size = 64
#     dataset.transform = T.Compose([ToBatchTensors(), ToFixedLengthTypeSeq(max_len)])

#     # split the dataset into the part divisible by batch size and the leftovers
#     # we can chain these together for performance - our metrics simply iterate
#     # through all elements in the batch
#     batched_total = len(dataset)-(len(dataset)%batch_size)
#     batch_loader = DataLoader(dataset[:batched_total], batch_size=batch_size)
#     leftovers_loader = DataLoader(dataset[batched_total:], batch_size=1)

#     print(f'Running eval...')

#     model.to(device)
#     model.eval()
#     num_correct = 0

#     for data in tqdm(chain(batch_loader, leftovers_loader), total=len(batch_loader)+len(leftovers_loader)):
#         data.to(device)
#         out = model(data.x, data.edge_index, data.batch)
#         num_correct += acc_heuristic_numcorrect(data.y, out, include_comp)

#     accuracy = num_correct/len(dataset)
#     print(f'Accuracy = {accuracy*100:,.2f}%')

#     return accuracy