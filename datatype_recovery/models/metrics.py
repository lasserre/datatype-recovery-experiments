import torch
from .dataset.encoding import decode_typeseq

def probabilities_to_indices(data:torch.tensor, max_seq_len:int) -> torch.tensor:
    '''
    Converts a batch of type sequence prediction probabilities (output of softmax)
    into a tensor of predicted indices corresponding to the max probability
    for each type sequence

    Returns tensor of shape (batch_size, max_seq_len)
    '''
    batch_size = int(data.shape[0]/max_seq_len)
    return data.argmax(dim=1).view((batch_size, max_seq_len))

def acc_raw_numcorrect(truth:torch.tensor, pred:torch.tensor):
    '''
    For each type sequence in the corresponding truth and pred tensors,
    compute accuracy as a 1 if fully correct or 0 otherwise

    Input tensors are expected to be logits for predicted tensors and truth probability
    labels (1 or 0), but should work for any combination. This works because argmax()
    will both identify the correct label probability of 1 and also identify the correct
    (max) model logit and convert both into the same "dimension" of "max index" before comparing.
    '''
    # expecting input shapes of ([batch_size,] num_classes, num_typeseq_elem)
    pred_idxs = pred.argmax(dim=pred.dim()-2)
    truth_idxs = truth.argmax(dim=truth.dim()-2)
    return (pred_idxs == truth_idxs).all(dim=truth.dim()-2).to(float)

def accuracy_weighted(truth, pred):
    '''
    For each type sequence row vector, compute its accuracy as a weighted
    average for each element within the type sequence considered independently
    '''
    typeseq_len = truth.shape[-1]
    pred_idxs = pred.argmax(dim=pred.dim()-2)
    truth_idxs = truth.argmax(dim=truth.dim()-2)
    return ((truth_idxs == pred_idxs).sum(dim=truth.dim()-2)/typeseq_len).to(float)

def acc_heuristic_numcorrect(truth:torch.tensor, pred:torch.tensor):
    # expecting input shapes of ([batch_size,] num_classes, num_typeseq_elem)
    corrected_acc = 0
    for i, y in enumerate(truth):
        y_class = ','.join(decode_typeseq(y, drop_empty_elems=True))
        pred_class = ','.join(decode_typeseq(pred[i], force_valid_seq=True))
        corrected_acc += int(pred_class == y_class)
    return corrected_acc
