from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch.utils.data import Subset
from torch.nn import functional as F
from pathlib import Path
import wandb
from wandb import AlertLevel
from tqdm.auto import trange
from tqdm import tqdm
import rich
from typing import List, Tuple

from .metrics import *
from .dataset.encoding import ToFixedLengthTypeSeq, ToBatchTensors
from .dataset import load_dataset_from_path, max_typesequence_len_in_dataset

class TrainContext:
    def __init__(self, model, device, optimizer, criterion) -> None:
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion

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
            edge_attr = data.edge_attr if self.model.uses_edge_features else None
            out = self.model(data.x, data.edge_index, data.batch, edge_attr=edge_attr)
            loss = self.criterion(out, data.y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def eval(self, loader:DataLoader, eval_metrics:List[EvalMetric], use_tqdm:bool=False):
        evalctx = EvalContext(self.model, self.device, eval_metrics)
        return evalctx.eval(loader, use_tqdm)

class EvalContext:
    def __init__(self, model, device, eval_metrics:List[EvalMetric]) -> None:
        self.model = model
        self.device = device
        self.eval_metrics = eval_metrics

    def eval(self, loader:DataLoader, use_tqdm:bool=False):
        '''
        Evaluates the model on all the data from the DataLoader and compute the
        eval_metrics
        '''
        self.model.eval()

        for m in self.eval_metrics:
            m.reset_state()

        get_data = tqdm(loader, total=len(loader)) if use_tqdm else loader

        for data in get_data:
            # make model prediction
            data.to(self.device)
            edge_attr = data.edge_attr if self.model.uses_edge_features else None
            out = self.model(data.x, data.edge_index, data.batch, edge_attr=edge_attr)

            # compute loss and metrics
            for m in self.eval_metrics:
                m.compute_for_batch(data.y, out)

        for m in self.eval_metrics:
            m._save_result(len(loader.dataset))

def partition_dataset(dataset, train_split:float, batch_size:int, data_limit:int=None,
                      shuffle_data_each_epoch:bool=True):

    console = rich.console.Console()

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

class DragonModelLoss:
    def __init__(self) -> None:
        # NOTE: I think this is overkill...i.e. I think I could reuse the same loss
        # instance for all items that use that kind of loss...but just to be safe
        # I'm keeping things separate
        self.p1_criterion = torch.nn.CrossEntropyLoss()
        self.p2_criterion = torch.nn.CrossEntropyLoss()
        self.p3_criterion = torch.nn.CrossEntropyLoss()

        self.cat_criterion = torch.nn.CrossEntropyLoss()
        self.sign_criterion = torch.nn.BCEWithLogitsLoss()
        self.float_criterion = torch.nn.BCEWithLogitsLoss()
        self.size_criterion = torch.nn.CrossEntropyLoss()

    def __call__(self, out:Tuple[torch.Tensor], data_y:torch.Tensor):
        # model output tuple
        # PTR LEVELS: [L1 ptr_type (3)][L2 ptr_type (3)][L3 ptr_type (3)]
        # LEAF TYPE: [category (5)][sign (1)][float (1)][size (6)]

        p1_out, p2_out, p3_out, cat_out, sign_out, float_out, size_out = out

        p1_loss = self.p1_criterion(p1_out, data_y[:,:3])
        p2_loss = self.p2_criterion(p2_out, data_y[:,3:6])
        p3_loss = self.p3_criterion(p3_out, data_y[:,6:9])
        cat_loss = self.cat_criterion(cat_out, data_y[:,9:14])
        sign_loss = self.sign_criterion(sign_out, data_y[:,14])
        float_loss = self.float_criterion(float_out, data_y[:,15])
        size_loss = self.size_criterion(size_out, data_y[:,16:])
        return p1_loss + p2_loss + p3_loss + \
                cat_loss + sign_loss + float_loss + size_loss


def train_model(model_path:Path, dataset_path:Path, run_name:str, train_split:float, batch_size:int, num_epochs:int,
                learn_rate:float=0.001, data_limit:int=None, cuda_dev_idx:int=0, seed:int=33, save_every:int=50):

    torch.manual_seed(seed)   # deterministic hopefully? lol

    if not run_name:
        run_name = model_path.stem

    model = torch.load(model_path)
    dataset = load_dataset_from_path(dataset_path)
    dataset_name = Path(dataset.root).name

    train_loader, test_loader = partition_dataset(dataset, train_split, batch_size, data_limit)

    train_metrics_file = Path(f'{run_name}.train_metrics.csv')

    if cuda_dev_idx >= torch.cuda.device_count():
        dev_count = torch.cuda.device_count()
        raise Exception(f'CUDA device idx {cuda_dev_idx} is out of the range of the {dev_count} CUDA devices')

    device = f'cuda:{cuda_dev_idx}' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {device}')

    criterion = DragonModelLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    wandb.login()
    wandb.init(
        # set the wandb project where this run will be logged
        project="DRAGON",
        name=run_name,

        # track hyperparameters and run metadata
        config={
            "learning_rate": learn_rate,
            "architecture": "GATConv",
            'max_hops': model.num_hops,
            'hidden_channels': model.hidden_channels,
            "dataset": dataset_name,
            'dataset_size': data_limit if data_limit is not None else len(dataset),
            'train_split': train_split,
            "epochs": num_epochs,
            'batch_size': batch_size,
        }
    )

    print(f'Training for {num_epochs} epochs')

    train_metrics = [
        AccuracyMetric('Train Acc'),
        AccuracyMetric('Train Acc Raw', raw_predictions=True),
        LossMetric('Train Loss', DragonModelLoss())
    ]

    test_metrics = [
        AccuracyMetric('Test Acc', notify=[0.65, 0.7, 0.8, 0.9, 0.95]),
        AccuracyMetric('Test Acc Raw', raw_predictions=True),
        LossMetric('Test Loss', DragonModelLoss())
    ]

    with TrainContext(model, device, optimizer, criterion) as ctx:

        # write header
        with open(train_metrics_file, 'w') as f:
            f.write(f"{','.join(m.name for m in chain(train_metrics, test_metrics))}\n")

        print(f'Computing initial accuracy/loss...')
        ctx.eval(train_loader, train_metrics, use_tqdm=True)
        ctx.eval(test_loader, test_metrics, use_tqdm=True)
        print(','.join([str(m) for m in train_metrics]))
        print(','.join([str(m) for m in test_metrics]))

        for epoch in trange(num_epochs):
            ctx.train_one_epoch(train_loader)
            ctx.eval(train_loader, train_metrics)
            ctx.eval(test_loader, test_metrics)

            wandb.log({m.name: m.value for m in chain(train_metrics, test_metrics)})
            torch.save(model, model_path)

            if save_every > 0 and (epoch+1) % save_every == 0:
                torch.save(model, f'{model_path.stem}_ep{epoch+1}.pt')      # save these in current directory

            with open(train_metrics_file, 'a') as f:
                f.write(f"{','.join(m.value for m in chain(train_metrics, test_metrics))}\n")

            for m in chain(train_metrics, test_metrics):
                m:EvalMetric
                if m.should_notify:
                    wandb.alert(
                        title=f'{m.name}',
                        text=f'{m.name} reached next threshold: {m}',
                        level=AlertLevel.INFO,
                    )

        print(f'Computing final accuracy/loss...')
        ctx.eval(train_loader, train_metrics, use_tqdm=True)
        ctx.eval(test_loader, test_metrics, use_tqdm=True)
        print(','.join([str(m) for m in train_metrics]))
        print(','.join([str(m) for m in test_metrics]))
        wandb.finish()