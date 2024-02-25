from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
from datatype_recovery.models.dataset import load_dataset_from_path

def plot_prefilter_variables(var_df:pd.DataFrame):
    '''
    Plot the kinds of variables before filtering
    (<Component>, Local/Param, Return Type) since we will be potentially filtering
    out the component and return types
    '''
    ax = (var_df.groupby('Label').count()/len(var_df)*100).BinaryId.plot(kind='bar', rot=0)
    ax.set_title('Variables (before filtering)')
    ax.set_ylabel('Percentage (%)')
    ax.set_xlabel('')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f%%', label_type='center', color='white')
    plt.show()

def prefilter(var_df:pd.DataFrame, drop_comp:bool, drop_return_types:bool) -> pd.DataFrame:
    '''
    Compute the initial variable balance, plot the balance, and drop the
    desired variables returning the resulting data frame
    '''
    # show initial balance here
    var_df.loc[var_df.TypeSeq_Debug=='COMP','Label'] = '<Component>'
    var_df.loc[var_df.IsReturnType_Debug, 'Label'] = 'Return Type'
    var_df.Label = var_df.Label.fillna('Local/Param')

    plot_prefilter_variables(var_df)

    filter_df = var_df
    if drop_comp:
        filter_df = filter_df.loc[filter_df.TypeSeq_Debug!='COMP']
    if drop_return_types:
        filter_df = filter_df.loc[~filter_df.IsReturnType_Debug, :]
    return filter_df

def plot_locals_vs_params(var_df:pd.DataFrame, title_suffix:str=''):
    ax = (var_df.groupby('Vartype').count()/len(var_df)*100).BinaryId.rename({'l': 'Local', 'p': 'Param'}).plot(kind='bar', rot=0)
    ax.set_title(f'Variable Kinds{title_suffix}')
    ax.set_ylabel('Percentage (%)')
    ax.set_xlabel('')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f%%', label_type='edge', color='black')
    ax.grid(False)
    sns.despine()
    plt.show()

def compute_datatype_class_balance(var_df:pd.DataFrame) -> pd.DataFrame:
    '''
    Compute the datatype class balance of the dataset and return a DataFrame
    where each data type class is a row, and each column corresponds to a type sequence
    length (only one column/row combo has a value, the others are NaN).

    This layout allows plotting the balance and coloring by type sequence length
    '''
    dt_classes = var_df.groupby('TypeSeq_Debug').count()[['BinaryId']]
    dt_classes['TypeSeqLen'] = [len(x.split(',')) for x in dt_classes.index]

    order_idx = sorted(dt_classes.index, key=lambda x: f'{len(x.split(","))}{x}')
    bal_df = dt_classes.pivot(columns='TypeSeqLen', values='BinaryId').loc[order_idx]/len(var_df)*100
    return bal_df

def plot_typeseq_len(bal_df:pd.DataFrame):
    ax = bal_df.sum().plot(kind='bar', rot=0)
    ax.set_title('Type Sequence Length')
    ax.set_ylabel('Percentage (%)')
    ax.set_xlabel('')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f%%', label_type='edge', color='black')
    ax.grid(False)
    sns.despine()
    plt.show()

def plot_full_dataset_balance(bal_df:pd.DataFrame, title_suffix:str=''):
    ax = bal_df.plot(kind='bar', stacked=True, figsize=(18, 9))
    ax.set_title(f'Data Type Balance{title_suffix}')
    ax.set_ylabel('Percentage (%)')
    ax.set_xlabel('')
    # ax.set_ylim([0, 0.025])
    for container in ax.containers:
        ax.bar_label(container, label_type='edge', fmt=lambda x: f'{x:.2f}%' if x > 5 else '')
    ax.grid(False)
    sns.despine()
    plt.show()

def plot_dataset_balance(dataset_path:Path, drop_comp:bool=None, drop_return_types:bool=True):
    '''
    Plot the balance of the dataset

    drop_comp: If unspecified, the value from the dataset will be used
    '''
    dataset = load_dataset_from_path(dataset_path)
    var_df = dataset.read_vars_csv()

    if drop_comp is None:
        drop_comp = dataset.drop_component

    var_df = prefilter(var_df, drop_comp, drop_return_types)
    plot_locals_vs_params(var_df, title_suffix=' (after filtering)')

    bal_df = compute_datatype_class_balance(var_df)
    plot_typeseq_len(bal_df)
    plot_full_dataset_balance(bal_df, f' ({len(var_df):,} total vars)')

    # TODO:
    # plt.savefig('type_seq_len.png')

    return var_df, bal_df
