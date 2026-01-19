import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict

from typing import Tuple

# this is for my jupyter notebooks
def set_stylesheet_and_get_output_paths(stylesheet:str, root_path:Path=None) -> Tuple[Path, Path]:
    '''
    Sets the stylesheet and returns the appropriate (figures, tables) output
    folders corresponding with the stylesheet
    '''
    plt.style.use(stylesheet)
    name = Path(stylesheet).with_suffix('').name
    if not root_path:
        root_path = Path.cwd()
    figs = root_path/f'figures_{name}'
    tables = root_path/f'tables_{name}'
    if not tables.exists():
        tables.mkdir(parents=True)
    if not figs.exists():
        figs.mkdir(parents=True)
    return (figs, tables)

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

def annotate_text(ax:plt.Axes, text:str, xy:tuple, fontsize:int, flip_left:bool=False,
                xpad_points:int=5, ypad_points:int=0,
                xycoords:str='data', **kwargs) -> plt.Annotation:
    '''
    ax: Axes to draw on
    text: Text to render
    xy: Coordinates passed to ax.annotate
    fontsize: Font size of text
    flip_left: Move the text to where the end is just to the left of the xy point (including padding)
    xpad_points: Horizontal padding of text relative to xy in points
    ypad_points: Vertical padding of text relative to xy in points
    xycoords: Coordinate system for xy, passed to ax.annotate
    '''
    horiz_align = 'right' if flip_left else 'left'
    xcoord = -xpad_points if flip_left else xpad_points
    ycoord = ypad_points
    return ax.annotate(text, xy=xy, xycoords=xycoords,
            xytext=(xcoord, ycoord), textcoords='offset points',
            horizontalalignment=horiz_align,
            fontsize=fontsize,
            **kwargs)

def plot_and_savefig(df, save_file:Path=None, labels:bool=True, label_precision:int=1, label_percent:bool=False,
                    title_kwargs:dict=None, legend_kwargs:dict=None, savefig_kwargs:dict=None, **kwargs):
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

    savefig_kwargs = savefig_kwargs if savefig_kwargs else {}

    if title_kwargs:
        plt.title(kwargs['title'], figure=ax.get_figure(), **title_kwargs)

    if legend_kwargs:
        ax.legend(**legend_kwargs)

    if labels:
        bar_labels(ax, precision=label_precision, make_percent=label_percent)

    if save_file:
        save_file.parent.mkdir(parents=True, exist_ok=True)
        ax.get_figure().savefig(save_file, **savefig_kwargs)
    plt.show()

    return ax

def format_column_headers(df, use_title_case:bool=True):
    '''Bold headers'''
    if use_title_case:
        df.columns = (df.columns.to_series().apply(lambda r: f"\\textbf{{{r.replace('_',' ').title()}}}"))
    else:
        df.columns = (df.columns.to_series().apply(lambda r: "\\textbf}".format(r.replace("_", " "))))

def write_latex(df:pd.DataFrame, outfile:Path, colfmt:str='c', fixed_colfmt:str=None, **kwargs):
    latex_kwargs = {
        # these will be defaults if not specified
        'index': False,
        'escape': False
    }
    latex_kwargs.update(kwargs)

    num_cols = len(df.columns) + 1 if latex_kwargs['index'] else len(df.columns)
    column_format = colfmt*num_cols if fixed_colfmt is None else fixed_colfmt

    df.to_latex(outfile,
        column_format=column_format,
        **latex_kwargs
    )

def get_column_format(column, bold_max:bool=True, precision:int=2):
    def do_format_column(x):
        display_val = f'{{:,.{precision}f}}'.format(x) if isinstance(x, float) else '{:,}'.format(x)
        if bold_max:
            return f'\\bfseries {display_val}' if x == column.max() else display_val
        return display_val
    return do_format_column

def format_columns(df, bold_max:bool=True, precision:int=2, column_overrides:Dict[str,dict]=None):
    '''
    Format all columns with the defaults, except any columns specified in
    column_overrides (mapping col_name -> prop_dict)
    '''
    default_opts = {
        'bold_max': bold_max,
        'precision': precision
    }
    col_opts = {colname: default_opts.copy() for colname in df.columns}
    if column_overrides:
        col_opts.update(column_overrides)
    return {colname: get_column_format(df[colname], **opts) for colname, opts in col_opts.items()}
