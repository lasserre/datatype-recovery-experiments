import pandas as pd
from rich.console import Console
from sklearn.metrics import precision_recall_fscore_support
from typing import List

def drop_duplicates(df:pd.DataFrame) -> pd.DataFrame:
    idx = ['BinaryId','FunctionStart','Signature','Vartype']
    num_dups = df.groupby(idx).count()

    # keep all unique rows (with < 2 entries for that index)
    return df.set_index(idx).loc[num_dups[num_dups.Name<2].index, :].reset_index()

def align_variables(strip_df:pd.DataFrame, debug_df:pd.DataFrame) -> pd.DataFrame:
    '''
    Aligns a debug variable with each of the stripped variables if possible, and returns
    a merged data frame with the aligned variables. All duplicates are dropped from
    both sides before merging, as we can't deal with duplicate signatures
    '''
    # drop duplicates first (we may have retyped some of these, but we can't evaluate their
    # accuracy based on our signature alignment method)
    strip_unique = drop_duplicates(strip_df)
    debug_unique = drop_duplicates(debug_df)

    mdf_all = strip_unique.merge(debug_unique, how='inner', on=['BinaryId','FunctionStart','Signature','Vartype'], suffixes=['Strip','Debug'])

    with pd.option_context("mode.copy_on_write", True):
        mdf_all['TypeSeq'] = mdf_all.Type.apply(lambda dt: dt.type_sequence_str)
        mdf_all['PredSeq'] = mdf_all.Pred.apply(lambda dt: dt.type_sequence_str)

    return mdf_all

class PandasEvalMetrics:
    # NOTE: there is another EvalMetric class, but that is for pytorch during training...

    def __init__(self, mdf:pd.DataFrame, truth_col:str, pred_col:str, name:str=None, scaleby_col:str=None, f1_avg:str='weighted') -> None:
        '''
        Compute evaluation metrics on this merged/aligned data frame
        '''
        self.truth_col = truth_col
        self.pred_col = pred_col
        self.mdf = mdf
        self.name = name if name else f'{pred_col} metric'
        self.scaleby_col = scaleby_col
        self.f1_avg = f1_avg

        self.accuracy = 0.0
        self.scaled_accuracy = 0.0
        self.num_scaled_occurrences = 0
        self.f1 = 0.0
        self.precision = 0.0
        self.recall = 0.0
        # ...more?

        self._compute_metrics()

    def _compute_metrics(self):
        '''
        Compute metrics
        '''
        df = self.mdf

        self.accuracy = (df[self.truth_col] == df[self.pred_col]).sum()/len(df)

        self.precision, self.recall, self.f1, _ = precision_recall_fscore_support(
                y_true=df[self.truth_col],
                y_pred=df[self.pred_col],
                average=self.f1_avg,
                zero_division=0,
                labels=df[self.truth_col].unique()      # we only consider truth classes
            )

        if self.scaleby_col:
            correct = df[self.truth_col] == df[self.pred_col]
            num_occurrences = df[self.scaleby_col]
            scaled = correct * num_occurrences
            self.scaled_accuracy = scaled.sum()/num_occurrences.sum()
            self.num_scaled_occurrences = num_occurrences.sum()

    def to_dataframe(self) -> pd.DataFrame:
        '''
        Return the metrics as a DataFrame
        (which can be combined with other metrics instances for comparison via pd.concat)
        '''
        return pd.DataFrame({
            'Name': [self.name],
            'Accuracy': [self.scaled_accuracy if self.scaleby_col else self.accuracy],
            'F1': [-1 if self.scaleby_col else self.f1],                # TODO - implement scaled versions
            'Precision': [-1 if self.scaleby_col else self.precision],  # TODO - implement scaled versions
            'Recall': [-1 if self.scaleby_col else self.recall],        # TODO - implement scaled versions
            'ScaledBy': [self.scaleby_col],
            'NumVars': [self.num_scaled_occurrences if self.scaleby_col else self.dataset_size],
        }).set_index('Name')

    @property
    def dataset_size(self) -> int:
        '''Number of samples over which metrics are computed'''
        return len(self.mdf)

    def print_summary(self, console:Console=None):
        if console is None:
            console = Console()

        console.print(f'[green]{self.name}[/] - Summary')
        console.print(f'{self.pred_col} vs. {self.truth_col} (dataset size = {len(self.mdf):,})')
        console.print(f'Accuracy: {self.accuracy*100:.2f}%')

        if self.scaleby_col:
            console.print(f'Scaled Accuracy ({self.scaleby_col}): {self.scaled_accuracy*100:.2f}%')

def project_types(df:pd.DataFrame, col_names:List[str], project_type:callable):
    '''
    Project the data types in the given columns using the project_type function,
    saving them in new columns in the dataframe named <col>Proj for each column
    '''
    for col in col_names:
        df[f'{col}Proj'] = df[col].apply(project_type)

def compute_metrics(df:pd.DataFrame, truth_col:str, pred_col:str, projected_types:bool=False,
                    name:str=None, scaleby_col:str=None, groupby:str=None) -> pd.DataFrame:
    truth_col = f'{truth_col}Proj' if projected_types else truth_col
    pred_col = f'{pred_col}Proj' if projected_types else pred_col
    if groupby is None:
        return PandasEvalMetrics(df, truth_col, pred_col, name, scaleby_col).to_dataframe()
    else:
        return df.groupby(groupby).apply(lambda x: PandasEvalMetrics(x, truth_col, pred_col, name, scaleby_col).to_dataframe())

def compute_dragon_metrics(df:pd.DataFrame, projected_types:bool=False, name:str=None,
                           scaleby_col:str=None, groupby:str=None) -> pd.DataFrame:
    '''
    Return PandasEvalMetrics for this dataframe using the appropriate column names for DRAGON
    '''
    return compute_metrics(df, 'TypeSeq', 'PredSeq', projected_types, name, scaleby_col, groupby)
