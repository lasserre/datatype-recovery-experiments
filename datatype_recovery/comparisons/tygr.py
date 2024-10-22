import pandas as pd

from ..eval_dataset import project_types

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