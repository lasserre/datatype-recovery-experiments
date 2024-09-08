import torch
from typing import Tuple

def split_train_test(dataset_size:int, test_split:float=0.1, batch_size:int=1) -> Tuple[list, list]:
    '''
    Splits the dataset into training and test subsets, and returns a tuple of
    (train_set, test_set) sets of indices
    '''
    num_test = int(dataset_size*test_split/batch_size) * batch_size
    num_train = int((dataset_size-num_test)/batch_size) * batch_size

    print(f'Dataset size is {dataset_size:,}')
    print(f'Test set has {num_test:,} samples ({num_test/dataset_size*100:.1f}%)')
    print(f'Train set has {num_train:,} samples ({num_train/dataset_size*100:.1f}%)')

    unused = dataset_size - num_test - num_train
    print(f'{unused:,} samples unused due to batch alignment (batch_size = {batch_size})')

    train_set = set([int(x) for x in torch.randperm(dataset_size)[:num_train]])
    test_set = set(list(set(range(dataset_size)) - train_set)[:num_test])

    # verify expected sizes
    assert len(test_set) == num_test
    assert len(train_set) == num_train
    # test + train should not intersect
    assert len(test_set.intersection(train_set)) == 0, f'Train and test indices overlap'
    # union of test + train should be full dataset - unused
    assert len(test_set.union(train_set)) == (dataset_size - unused), f'Train + test indices do not equal full (utilized) dataset size ({dataset_size-unused:,})'

    return (list(train_set), list(test_set))