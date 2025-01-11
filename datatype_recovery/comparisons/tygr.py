from collections import defaultdict
from pathlib import Path
import pandas as pd

from ..eval_dataset import *
from ..experiments.eval_simple_types import DragonEval

def read_tygr_preds(tygr_folder:Path, first_only:bool=False):
    tygr_csvs = list(tygr_folder.glob('*.preds.csv'))
    # coreutils binaries named "true" and "false" make the Binary column
    # intepreted as a bool instead of string...enforce string interpretation
    dtypes = defaultdict(lambda: str, Binary="str")
    if first_only:
        if len(tygr_csvs) > 1:
            print(f'{len(tygr_csvs)} found, only using first one ({tygr_csvs[0]})')
        return pd.read_csv(tygr_csvs[0], dtype=dtypes)
    return [pd.read_csv(x, dtype=dtypes) for x in tygr_csvs]

def reduce_tygr_preds(tygr_preds:pd.DataFrame) -> pd.DataFrame:
    '''
    Reduce the TYGR predictions to one per (debug info) variable by taking
    the most frequently predicted type across all the nodes corresponding to a
    variable
    '''
    agg_columns = {
        'Type': 'first',                        # Type should all be the same (truth)
        'PredType': lambda pt: pt.mode()[0]     # take the most commonly predicted type (or first of these if there are ties)
    }

    if 'TypeProj' in tygr_preds:
        agg_columns['TypeProj'] = 'first'
        agg_columns['PredTypeProj'] = lambda pt: pt.mode()[0]

    return tygr_preds.groupby(['Binary','FunctionName','VarName']).agg(agg_columns).reset_index()

def project_tygr_type(t:str) -> str:
    '''
    Project a TYGR type to the corresponding DRAGON-compatible
    type
    '''
    # 1. struct_XXX -> struct
    return 'struct' if t.startswith('struct_') else t

def project_dragon_type(typeseq:str) -> str:
    '''
    Project a DRAGON type sequence (e.g. PTR,char) to the
    corresponding TYGR-compatible type sequence

    (e.g. PTR,uchar -> PTR,char and ARR,int -> ARR,void)
    '''
    # 1. uchar -> char
    x = typeseq.replace('uchar','char')

    # 2. FUNC -> void
    x = x.replace('FUNC','void')

    # 3. Arrays don't preserve element type, so ARR -> ARR,void
    if 'ARR' in x:
        comp_list = []
        for comp in x.split(','):
            if comp == 'ARR':
                # array can only hold void, so we terminate the type sequence
                comp_list.append('ARR')
                comp_list.append('void')
                break
            else:
                comp_list.append(comp)
        x = ','.join(comp_list)

    # 4. Only 1 pointer level supported, anything with 2 pointers goes to void** (PTR,PTR,void)
    if 'PTR,PTR' in x:
        # ARR has already been convered to ARR[void], so
        # we can safely convert this to PTR,PTR,void (we can't have ARR,PTR,PTR)
        x = 'PTR,PTR,void'

    return x

def project_dragon_types(df:pd.DataFrame, dragon_truth_col:str='TypeSeq', dragon_pred_col:str='PredSeq'):
    '''Project the TypeSeq/PredSeq columns to TYGR-compatible types'''
    project_types(df, [dragon_truth_col, dragon_pred_col], project_dragon_type)

def project_tygr_types(df:pd.DataFrame, tygr_truth_col:str='Type', tygr_pred_col:str='PredType'):
    '''Project the Type/PredType columns to DRAGON-compatible types'''
    project_types(df, [tygr_truth_col, tygr_pred_col], project_tygr_type)

def compute_tygr_metrics(df:pd.DataFrame, projected_types:bool=False, name:str=None, groupby:str=None) -> PandasEvalMetrics:
    '''
    Return PandasEvalMetrics for this dataframe using the appropriate column names for TYGR
    '''
    return compute_metrics(df, 'Type', 'PredType', projected_types, name, groupby=groupby)

class TygrEval:
    '''
    Convenience class to wrap the analysis we perform for TYGR evals
    and keep the related data together
    '''
    def __init__(self, eval_folder:Path):
        self.eval_folder = eval_folder
        self.nested_tygr_folders = list(self.eval_folder.glob('*.tygr'))

        print(f'Processing TYGR data from {eval_folder.name}...')

        self.df = self._read_raw_data()                 # 1. Read all & concat raw data
        project_tygr_types(self.df)                     # 2. Project TYGR types (adds columns)
        self.df_reduced = reduce_tygr_preds(self.df)    # 3. Reduce (now we have tygr_all and tygr_reduced)

    def __repr__(self):
        repr_str = f'TygrEval: {self.eval_folder}'
        if self.nested_tygr_folders:
            repr_str += f' ({len(self.nested_tygr_folders)} nested)'
        return repr_str

    def _read_raw_data(self) -> pd.DataFrame:
        '''
        Read the single predictions .csv file, or combine all csv files if we
        have nested .tygr folders
        '''
        if self.nested_tygr_folders:
            return pd.concat([read_tygr_preds(f, first_only=True) for f in self.nested_tygr_folders])
        else:
            return read_tygr_preds(self.eval_folder)

    @property
    def avg_nodes_per_var(self) -> float:
        return self.df.groupby(['Binary','FunctionName','VarName']).count().NodeId.mean()

def compare_dragon_tygr_final(dragon:pd.DataFrame, tygr:pd.DataFrame, proj_types:bool=True, scale_dragon:bool=False, groupby:str=None) -> pd.DataFrame:
    scaleby = 'NumRefs' if scale_dragon else None

    df = pd.concat([
        compute_dragon_metrics(dragon, proj_types, 'DRAGON', scaleby,  groupby).to_dataframe(),
        compute_tygr_metrics(tygr, proj_types, 'TYGR', groupby).to_dataframe()
    ])

    if groupby:
        df = df.reset_index('Name').pivot(columns='Name')
        df.columns.names = (None, None)     # removes 'Name' from legend

    return df

def compare_dragon_tygr_all(dragon, tygr_red, tygr) -> pd.DataFrame:

    final_df = compare_dragon_tygr_final(dragon, tygr_red)

    return pd.concat([
        compute_dragon_metrics(dragon, name='DRAGON').to_dataframe(),
        compute_dragon_metrics(dragon, name='DRAGON (S)', scaleby_col='NumRefs').to_dataframe(),
        compute_dragon_metrics(dragon, name='DRAGON (S/P)', projected_types=True, scaleby_col='NumRefs').to_dataframe(),
        compute_tygr_metrics(tygr, name='TYGR').to_dataframe(),
        compute_tygr_metrics(tygr_red, name='TYGR (R)').to_dataframe(),
        final_df
    ])

class DragonEvalVsTygr(DragonEval):
    '''
    Wrap DragonEval with TYGR-specific aspects (project types)
    '''
    def __init__(self, eval_folder):
        super().__init__(eval_folder)
        project_dragon_types(self.df)        # 2. Project to TYGR-compatible data types
