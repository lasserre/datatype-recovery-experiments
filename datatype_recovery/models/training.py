from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from torch.nn import functional as F
from pathlib import Path
import wandb
from wandb import AlertLevel
from tqdm.auto import trange
import rich

from .metrics import *
from .dataset.encoding import ToFixedLengthTypeSeq
from .dataset import load_dataset_from_path

class TrainContext:
    def __init__(self, model, device, optimizer, criterion, max_seq_len:int) -> None:
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.max_seq_len = max_seq_len

    def __enter__(self):
        cuda_dev = torch.cuda.current_device()
        cuda_devname = torch.cuda.get_device_name(cuda_dev)
        print(f'---------------')
        print(f'Current device: {self.device}')
        print(f'Current CUDA device: {cuda_dev} ({cuda_devname})')

        # move model to device
        self.model = self.model.to(self.device)

        return self

    def __exit__(self, etype, value, traceback):
        pass

    def train_one_epoch(self, train_loader:DataLoader):
        self.model.train()

        for data in train_loader:
            data.to(self.device)
            out = self.model(data.x, data.edge_index, data.batch)
            loss = self.criterion(out, data.y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def eval(self, loader:DataLoader):
        return EvalContext(self.model, self.device, self.criterion, self.max_seq_len).eval(loader)

class EvalContext:
    def __init__(self, model, device, criterion, max_seq_len:int) -> None:
        self.model = model
        self.device = device
        self.criterion = criterion
        self.max_seq_len = max_seq_len

    def eval(self, loader:DataLoader):
        '''
        Evaluates the model on all the data from the DataLoader and return
        the following metrics as a tuple:

        acc_binary - Binary accuracy, predictions only correct if fully accurate
        acc_weighted - Weighted accuracy of all components
        avg_loss - Average loss across the loader dataset
        '''
        self.model.eval()

        # WARNING: accuracy computed based on max length - need to calculate
        # full accuracy (accounting for longer true type sequences)

        num_correct = 0
        total_loss = 0
        weighted_correct = 0.0

        for data in loader:
            # make model prediction
            data.to(self.device)
            out = self.model(data.x, data.edge_index, data.batch)
            loss = self.criterion(out, data.y)
            y_indices = probabilities_to_indices(data.y, self.max_seq_len)
            pred_indices = probabilities_to_indices(F.softmax(out, dim=1), self.max_seq_len)

            # compute loss and metrics
            num_correct += int(accuracy_complete(pred_indices, y_indices).sum())
            weighted_correct += accuracy_weighted(pred_indices, y_indices).sum()
            total_loss += loss.item()

        acc_complete = num_correct/len(loader.dataset)
        acc_weighted = float(weighted_correct/len(loader.dataset))
        avg_loss = total_loss/len(loader.dataset)

        return acc_complete, acc_weighted, avg_loss

def partition_dataset(dataset, max_seq_len:int, train_split:float, batch_size:int, data_limit:int=None,
                      shuffle_data_each_epoch:bool=True):

    transform = ToFixedLengthTypeSeq(max_seq_len)
    dataset.transform = transform   # apply transform here so we can remove it if desired

    console = rich.console.Console()

    console.print(f'[yellow]Warning: only computing accuracy on fixed-length size of {max_seq_len}')
    console.print(f'[yellow]TODO: compute accuracy based on raw type sequence')

    # OVERFIT_SIZE = 4096

    if data_limit is not None:
        dataset = dataset[:data_limit]    # TEMP: overfit on tiny subset
        console.rule(f'Limiting training dataset to the first {data_limit:,} samples')

    # divide into train/test sets - aligning to batch size
    train_size = int(len(dataset)*train_split/batch_size) * batch_size
    test_size = int((len(dataset) - train_size)/batch_size) * batch_size

    train_indices = [int(x) for x in torch.randperm(len(dataset))[:train_size]]
    test_indices = set(range(1024)) - set(train_indices)
    test_indices = list(test_indices)[:test_size]   # align to batch size

    train_set = Subset(dataset, train_indices)
    test_set = Subset(dataset, range(len(train_set), len(train_set)+test_size))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_data_each_epoch, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle_data_each_epoch, pin_memory=False)

    total_usable = len(train_set)+len(test_set)
    non_batch_aligned = len(dataset)-total_usable
    print()
    print(f'Train set: {len(train_set):,} samples ({len(train_set)/len(dataset)*100:.2f}%)')
    print(f'Test set: {len(test_set):,} samples ({len(test_set)/len(dataset)*100:.2f}%)')
    print(f'Batch size: {batch_size}')
    print(f'Total usable dataset size (batch-aligned): {total_usable:,}')
    print(f'Loss due to batch alignment: {non_batch_aligned:,} ({non_batch_aligned/len(dataset)*100:.2f}%)')

    return train_loader, test_loader

def train_model(model_path:Path, dataset_path:Path, run_name:str, train_split:float, batch_size:int, num_epochs:int,
                learn_rate:float=0.001, data_limit:int=None, cuda_dev_idx:int=0, seed:int=33):

    torch.manual_seed(seed)   # deterministic hopefully? lol

    model = torch.load(model_path)
    dataset = load_dataset_from_path(dataset_path)
    dataset_name = Path(dataset.root).name

    train_loader, test_loader = partition_dataset(dataset, model.max_seq_len, train_split, batch_size, data_limit)

    train_metrics_file = Path(f'{run_name}.train_metrics.csv')

    if cuda_dev_idx >= torch.cuda.device_count():
        dev_count = torch.cuda.device_count()
        raise Exception(f'CUDA device idx {cuda_dev_idx} is out of the range of the {dev_count} CUDA devices')

    device = f'cuda:{cuda_dev_idx}' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {device}')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    wandb.login()
    wandb.init(
        # set the wandb project where this run will be logged
        project="StructuralModel",
        name=run_name,

        # track hyperparameters and run metadata
        config={
            "learning_rate": learn_rate,
            "architecture": "GATConv",
            'max_hops': model.num_hops,
            'max_seq_len': model.max_seq_len,
            'hidden_channels': model.hidden_channels,
            "dataset": dataset_name,
            'dataset_size': data_limit if data_limit is not None else len(dataset),
            'train_split': train_split,
            "epochs": num_epochs,
            'batch_size': batch_size,
        }
    )

    print(f'Training for {num_epochs} epochs')

    notify_acc_levels = [0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 2]     # the 2 is to stop notifications :)
    curr_acc_idx = 0

    with TrainContext(model, device, optimizer, criterion, model.max_seq_len) as ctx:

        # write header
        with open(train_metrics_file, 'w') as f:
            f.write(f'train_loss,train_acc_bin,train_acc_weight,test_loss,test_acc_bin,test_acc_weight\n')

        print(f'Computing initial accuracy/loss...')
        train_acc_bin, train_acc_weight, train_loss = ctx.eval(train_loader)
        test_acc_bin, test_acc_weight, test_loss = ctx.eval(test_loader)
        print(f'Train loss = {train_loss:.4f}, train accuracy = {train_acc_bin*100:,.2f}%')
        print(f'Test loss = {test_loss:.4f}, test accuracy = {test_acc_bin*100:,.2f}%')

        for epoch in trange(num_epochs):
            ctx.train_one_epoch(train_loader)
            train_acc_bin, train_acc_weight, train_loss = ctx.eval(train_loader)
            test_acc_bin, test_acc_weight, test_loss = ctx.eval(test_loader)
            wandb.log({
                'train/loss': train_loss,
                'train/acc': train_acc_bin,
                'train/acc_weighted': train_acc_weight,
                'test/loss': test_loss,
                'test/acc': test_acc_bin,
                'test/acc_weighted': test_acc_weight,
            })
            torch.save(model, model_path)

            with open(train_metrics_file, 'a') as f:
                f.write(f'{train_loss},{train_acc_bin},{train_acc_weight},{test_loss},{test_acc_bin},{test_acc_weight}\n')

            if test_acc_bin >= notify_acc_levels[curr_acc_idx]:
                wandb.alert(
                    title='Test accuracy',
                    text=f'Reached test accuracy of {test_acc_bin*100:.2f}%',
                    level=AlertLevel.INFO,
                )
                curr_acc_idx += 1

        train_acc_bin, train_acc_weight, train_loss = ctx.eval(train_loader)
        test_acc_bin, test_acc_weight, test_loss = ctx.eval(test_loader)
        print(f'Train loss = {train_loss:.4f}, train accuracy = {train_acc_bin*100:,.2f}%')
        print(f'Test loss = {test_loss:.4f}, test accuracy = {test_acc_bin*100:,.2f}%')
        wandb.finish()