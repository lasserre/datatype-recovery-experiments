from torch_geometric.loader import DataLoader
from torch.nn import functional as F

from .metrics import *

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
