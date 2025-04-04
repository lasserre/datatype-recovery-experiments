import torch
from .dataset.encoding import *

class EvalMetric:
    '''
    Encapsulates logic to compute an eval metric so we can mix and match
    different metrics without changing the internals of EvalContext
    '''
    def __init__(self, name:str, is_percentage:bool=False, notify_levels:List[float]=None,
                print_in_summary:bool=False):
        '''
        name: The name of the metric, which will be supplied to wandb for logging
        '''
        self.name = name
        self.is_percentage = is_percentage
        self.notify_levels = notify_levels if notify_levels else []
        self.print_in_summary = print_in_summary
        self._value = 0.0
        self._next_notify_value = self.notify_levels[0] if self.notify_levels else None

    def __str__(self) -> str:
        val_str = f'{self.value*100:,.2f}%' if self.is_percentage else f'{self.value:,.4f}'
        return f'{self.name} = {val_str}'

    @property
    def value(self) -> float:
        '''Computed value of the metric'''
        return self._value

    @property
    def should_notify(self) -> bool:
        '''True if the current metric value has just exceeded a notification level'''
        if self._next_notify_value is not None:
            if self.value >= self._next_notify_value:
                self._next_notify_value = self.notify_levels.pop(0) if self.notify_levels else None
                return True
        return False

    def _save_result(self, dataset_size:int):
        self._value = self.result(dataset_size)

    def reset_state(self):
        raise Exception(f'reset_state not implemented by {self.__class__.__name__}')

    def compute_for_batch(self, batch_y:torch.Tensor, batch_out:torch.Tensor) -> None:
        '''
        Computes the metric on this batch from the data. The metric should store any state
        needed to return the final metric from
        '''
        raise Exception(f'compute_for_batch not implemented by {self.__class__.__name__}')

    def result(self, dataset_size:int) -> float:
        '''
        Returns the final result of the metric accumulated over the entire dataset.

        dataset_size: The size of the entire dataset for convenience (e.g. for average value)
        '''
        raise Exception(f'result not implemented by {self.__class__.__name__}')

# I just need an easy way to read hyperparameter values for our loss metric
# and fit it into the logging like other eval metrics. Hacking this in for now,
# later would make more sense to call this something other than an eval metric
class HyperparamMetric(EvalMetric):
    def __init__(self, name:str, get_value:Callable[[int], float], print_in_summary:bool=False):
        super().__init__(name, print_in_summary=print_in_summary)
        self._get_value = get_value

    def reset_state(self):
        pass

    def compute_for_batch(self, batch_y: torch.Tensor, batch_out: torch.Tensor) -> None:
        pass

    def result(self, dataset_size: int) -> float:
        return self._get_value(dataset_size)

class LossMetric(EvalMetric):
    def __init__(self, name:str, criterion, print_in_summary:bool=False):
        super().__init__(name, print_in_summary=print_in_summary)
        self.criterion = criterion
        self.reset_state()
        self.confidence_loss = hasattr(criterion, '_last_Lc')
        self.has_eval_mode = hasattr(criterion, 'eval_mode')

    def reset_state(self):
        self.total_loss = 0.0
        self.total_conf_loss = 0.0
        self.batch_size = None

    def compute_for_batch(self, batch_y:torch.Tensor, batch_out:Tuple[torch.Tensor]) -> None:
        orig_eval_mode = None
        if self.has_eval_mode:
            orig_eval_mode = self.criterion.eval_mode
            self.criterion.eval_mode = True

        self.batch_size = batch_y.shape[0]  # save batch size
        self.total_loss += self.criterion(batch_out, batch_y).item()
        if self.confidence_loss:
            self.total_conf_loss += self.criterion._last_Lc

        if self.has_eval_mode:
            self.criterion.eval_mode = orig_eval_mode

    def result(self, dataset_size: int) -> float:
        # individual loss criteria are bean reduced using 'mean', so don't divide our total
        # by every sample - divide by # batches (since we added each batch avg)
        num_batches = dataset_size/self.batch_size
        return self.total_loss/num_batches     # avg loss

class AccuracyMetric(EvalMetric):
    def __init__(self, name:str, raw_predictions:bool=False, specific_output:str=None,
                notify:List[float]=None, print_in_summary:bool=False,
                thresholds:LeafTypeThresholds=None):
        '''
        name: Name of the metric visible in wandb
        raw_preds: Use raw prediction if true, constrain to valid data types if false
        specific_output: Name of a specific output (e.g. PtrL1) to log accuracy for instead of overall type
        notify: List of accuracy values at which to send a notification
        print_in_summary: Set this if the training function should print this metric in the before/after summaries
        thresholds: Override default logit threshold of 0 for binary classifiers
        '''
        super().__init__(name, is_percentage=True, notify_levels=notify, print_in_summary=print_in_summary)
        self.reset_state()
        self.raw_preds = raw_predictions
        self.specific_output = specific_output
        self.binary_thresholds = thresholds if thresholds else LeafTypeThresholds()

    def reset_state(self):
        self.num_correct = 0

    def compute_individual_output_acc(self, out, data_y, output_name:str):
        # NOTE: it is important here to NOT use TypeEncoder.decode() since I
        # want full visibility into each task-specific-output's performance.
        # TypeEncoder will create a valid type and do so by preferring some
        # outputs over others, which we don't want here

        p1_out = out[:,:3]
        p2_out = out[:,3:6]
        p3_out = out[:,6:9]
        cat_out = out[:,9:14]
        sign_out = out[:,14].unsqueeze(1).item()
        float_out = out[:,15].unsqueeze(1).item()
        bool_out = out[:,16].unsqueeze(1).item()
        size_out = out[:,17:]

        p1_y = data_y[:,:3]
        p2_y = data_y[:,3:6]
        p3_y = data_y[:,6:9]
        cat_y = data_y[:,9:14]
        sign_y = data_y[:,14].unsqueeze(1).item()
        float_y = data_y[:,15].unsqueeze(1).item()
        bool_y = data_y[:,16].unsqueeze(1).item()
        size_y = data_y[:,17:]

        if output_name == 'PtrL1':
            return p1_out.argmax().item() == p1_y.argmax().item()
        if output_name == 'PtrL2':
            return p2_out.argmax().item() == p2_y.argmax().item()
        if output_name == 'PtrL3':
            return p3_out.argmax().item() == p3_y.argmax().item()
        if output_name == 'LeafCategory':
            return cat_out.argmax().item() == cat_y.argmax().item()
        if output_name == 'LeafSigned':
            is_signed = sign_out > self.binary_thresholds.signed_threshold
            return is_signed == bool(sign_y)
        if output_name == 'LeafFloating':
            is_float = float_out > self.binary_thresholds.floating_threshold
            return is_float == bool(float_y)
        if output_name == 'LeafBool':
            is_bool = bool_out > self.binary_thresholds.bool_threshold
            return is_bool == bool(bool_y)
        if output_name == 'LeafSize':
            return size_out.argmax().item() == size_y.argmax().item()

        raise Exception(f'Unrecognized task-specific output "{output_name}"')

    def compute_for_batch(self, batch_y:torch.Tensor, batch_out:Tuple[torch.Tensor]) -> None:
        if len(batch_out) == 2:     # (pred_tuple, confidence)
            batch_out = batch_out[0]    # we only care about pred_tuple

        # convert tuple of outputs into a single tensor
        batch_out = torch.cat(batch_out, dim=1)

        if self.specific_output:
            # task-specific accuracy
            for i, y in enumerate(batch_y):
                sample_acc = self.compute_individual_output_acc(batch_out[None,i], y[None,:], self.specific_output)
                self.num_correct += int(sample_acc)
        else:
            # overall recovered type accuracy
            for i, y in enumerate(batch_y):
                y_seq = TypeEncoder.decode(y[None,:]).type_sequence_str
                if self.raw_preds:
                    pred_seq = TypeEncoder.decode_raw_typeseq(batch_out[None,i], self.binary_thresholds)
                else:
                    pred_seq = TypeEncoder.decode(batch_out[None,i], self.binary_thresholds).type_sequence_str
                self.num_correct += int(pred_seq == y_seq)

    def result(self, dataset_size: int) -> float:
        return self.num_correct/dataset_size

# -------------------------------------------------------------------------------------
# OLD STUFF
# -------------------------------------------------------------------------------------

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

def acc_heuristic_numcorrect(truth:torch.tensor, pred:torch.tensor, include_comp:bool):
    # expecting input shapes of ([batch_size,] num_classes, num_typeseq_elem)
    corrected_acc = 0
    tseq = TypeSequence(include_comp)
    for i, y in enumerate(truth):
        y_class = ','.join(tseq.decode(y, drop_empty_elems=True))
        pred_class = ','.join(tseq.decode(pred[i], force_valid_seq=True))
        corrected_acc += int(pred_class == y_class)
    return corrected_acc
