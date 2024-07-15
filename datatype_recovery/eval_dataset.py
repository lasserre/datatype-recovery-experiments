import pandas as pd

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

    mdf_all = strip_unique.merge(debug_unique, how='left', on=['BinaryId','FunctionStart','Signature','Vartype'], suffixes=['Strip','Debug'])
    mdf_all['TypeSeq'] = mdf_all.Type.apply(lambda dt: dt.type_sequence_str)
    mdf_all['PredSeq'] = mdf_all.Pred.apply(lambda dt: dt.type_sequence_str)

    # keep only aligned variables
    return mdf_all.loc[~mdf_all.NameDebug.isna(), :]

class PandasEvalMetrics:
    # NOTE: there is another EvalMetric class, but that is for pytorch during training...

    def __init__(self, mdf:pd.DataFrame, truth_col:str, pred_col:str) -> None:
        '''
        Compute evaluation metrics on this merged/aligned data frame
        '''
        self.truth_col = truth_col
        self.pred_col = pred_col

        # TODO: project_types() would happen first...

        self.accuracy = 0.0
        self.f1 = 0.0
        self.precision = 0.0
        self.recall = 0.0
        # ...more?

        self._compute_metrics()

    def _compute_metrics(self):
        '''
        Compute metrics
        '''
        pass
