import torch

def probabilities_to_indices(data:torch.tensor, max_seq_len:int) -> torch.tensor:
    '''
    Converts a batch of type sequence prediction probabilities (output of softmax)
    into a tensor of predicted indices corresponding to the max probability
    for each type sequence

    Returns tensor of shape (batch_size, max_seq_len)
    '''
    batch_size = int(data.shape[0]/max_seq_len)
    return data.argmax(dim=1).view((batch_size, max_seq_len))

def accuracy_complete(y1_indices, y2_indices):
    '''
    For each type sequence row vector, compute its accuracy as a 1 if
    fully correct or 0 otherwise.
    '''
    return (y1_indices == y2_indices).all(dim=1).to(float)

def accuracy_weighted(y1_indices, y2_indices):
    '''
    For each type sequence row vector, compute its accuracy as a weighted
    average for each element within the type sequence considered independently
    '''
    typeseq_len = y1_indices.shape[1]
    return ((y1_indices == y2_indices).sum(dim=1)/typeseq_len).to(float)
