import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def annotate_tygr(ax, name_column:pd.Series):
    for i, name in enumerate(name_column.tolist()):
        if 'TYGR' in name:
            ax.patches[i].set_hatch('X')

def bar_labels(ax, comma_sep:bool=True, custom_values:list=None, precision:int=0, make_percent:bool=False):
    for i, container in enumerate(ax.containers):
        values = custom_values if custom_values else [x.get_height()*100 if make_percent else x.get_height() for x in container]
        comma_fmt = ',' if comma_sep else ''
        pcnt = '%' if make_percent else ''
        ax.bar_label(container, labels=[f'{{:{comma_fmt}.{precision}f}}{pcnt}'.format(v) for v in values])

def plot_and_savefig(df, save_file:Path=None, labels:bool=True, label_precision:int=1, label_percent:bool=False,
                    title_kwargs:dict=None, legend_kwargs:dict=None, **kwargs):
    '''
    Plot the entire benchmark (# vars, accuracy)
    '''
    plot_kwargs = {
        # these will be defaults if not specified
        'xlabel': '',
        'width': 0.7,
        'kind': 'bar',
        'rot': 0
    }
    plot_kwargs.update(kwargs)
    ax = df.plot(**plot_kwargs)

    if title_kwargs:
        plt.title(kwargs['title'], figure=ax.get_figure(), **title_kwargs)

    if legend_kwargs:
        ax.legend(**legend_kwargs)

    if labels:
        bar_labels(ax, precision=label_precision, make_percent=label_percent)

    if save_file:
        save_file.parent.mkdir(parents=True, exist_ok=True)
        ax.get_figure().savefig(save_file)
    plt.show()

    return ax
