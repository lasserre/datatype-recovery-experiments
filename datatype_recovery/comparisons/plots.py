import matplotlib.pyplot as plt
import pandas as pd

def annotate_tygr(ax, name_column:pd.Series):
    for i, name in enumerate(name_column.tolist()):
        if 'TYGR' in name:
            ax.patches[i].set_hatch('X')

def bar_labels_percent(ax, precision:int=2, custom_values:list=None):
    for i, container in enumerate(ax.containers):
        values = custom_values if custom_values else [x.get_height()*100 for x in container]
        ax.bar_label(container, labels=[f'{{:.{precision}f}}%'.format(v) for v in values])

def bar_labels(ax, comma_sep:bool=True, custom_values:list=None, precision:int=0):
    for i, container in enumerate(ax.containers):
        values = custom_values if custom_values else [x.get_height() for x in container]
        comma_fmt = ',' if comma_sep else ''
        ax.bar_label(container, labels=[f'{{:{comma_fmt}.{precision}f}}'.format(v) for v in values])

def plot_benchmark_comparison(df, title:str='Benchmark') -> pd.DataFrame:
    '''
    Plot the entire benchmark (# vars, accuracy)
    '''
    if 'Accuracy' in df:
        ax = df['Accuracy'].plot(kind='bar',
            title=title,
            ylim=[0.4, 1],
            ylabel='Accuracy',
            rot=0,
        )
        bar_labels_percent(ax, precision=1)
        annotate_tygr(ax, df.index)
        plt.show()

    if 'NumVars' in df:
        ax = (df['NumVars']/1000).plot(kind='bar',
            title=title,
            ylabel='Number of Samples (k)',
            rot=0,     # this might go away w/ style
        )
        # bar_labels(ax)
        annotate_tygr(ax, df.index)
        plt.show()

    return df