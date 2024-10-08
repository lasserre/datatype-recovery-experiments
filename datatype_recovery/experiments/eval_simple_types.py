import argparse
import pandas as pd
from pathlib import Path
from rich.console import Console
import os
import pickle
from tqdm import tqdm
from typing import List

from datatype_recovery.models.homomodels import DragonModel, VarPrediction
from datatype_recovery.dragon_ryder_cmdline import *
from datatype_recovery.eval_dataset import *
from varlib.datatype import DataType
from astlib import binary_id
from wildebeest.utils import print_runtime

# eval_simple_types --dragon=<model> --dragon-ryder=<model> --tygr=<model>

def create_eval_folder(name:str, resume:bool) -> Path:
    eval_folder = Path(name)
    if eval_folder.exists() and not resume:
        raise Exception(f'Eval folder {eval_folder} already exists')
    eval_folder.mkdir(exist_ok=True)
    return eval_folder

def export_truth_types(args:argparse.Namespace, console:Console, debug_csv:Path, bin_paths_csv:Path) -> pd.DataFrame:
    from ghidralib.export_vars import export_debug_vars
    from ghidralib.projects import OpenSharedGhidraProject
    from datatype_recovery.dragon_ryder import load_bin_files

    if debug_csv.exists():
        console.print(f'[yellow]debug vars already exported - skipping this step and reusing them')
        debug_df = pd.read_csv(debug_csv)
        # load Type from json so it's not just a string
        debug_df['Type'] = debug_df.TypeJson.apply(DataType.from_json)
    else:
        console.rule(f'Exporting debug variables (truth)')
        with OpenSharedGhidraProject(args.host, args.ghidra_repo, args.port) as proj:
            _, debug_bins = load_bin_files(proj, bin_paths_csv, console, args.binaries)
            if args.limit:
                console.print(f'[bold orange1] only exporting first {args.limit:,} debug functions')
            debug_df = export_debug_vars(proj, debug_bins, args.limit)
            debug_df.to_csv(debug_csv, index=False)
    return debug_df

def run_dragon(dragon_model:Path, args:argparse.Namespace, out_csv:Path, proj, strip_bins:List, console:Console):
    '''
    Runs dragon with the provided arguments and returns a path to
    the output predictions csv
    '''
    from ghidralib.projects import verify_ghidra_revision, GhidraCheckoutProgram
    from ghidralib.decompiler import AstDecompiler

    model = DragonModel.load_model(dragon_model, args.device, eval=True)

    table_rows = []

    for bin_file in strip_bins:
        bid = binary_id(bin_file.name)
        verify_ghidra_revision(bin_file, expected_revision=1, rollback_delete=True)

        with GhidraCheckoutProgram(proj, bin_file) as co:
            with AstDecompiler(co.program, bid, timeout_sec=240) as decompiler:
                nonthunks = co.decompiler.nonthunk_functions[:args.limit]
                for func in tqdm(nonthunks, desc=bin_file.name):
                    ast = decompiler.decompile_ast(func)
                    num_callers = len(func.getCallingFunctions(None))
                    if ast is None:
                        console.print('[bold orange1]Decompilation failed:')
                        console.print(f'[orange1]{decompiler.last_error_msg}')
                        continue
                    var_preds = model.predict_func_types(ast, args.device, bid, skip_unique_vars=True, num_callers=num_callers)
                    table_rows.extend([
                        p.to_record() for p in var_preds
                    ])

    pd.DataFrame.from_records(table_rows, columns=VarPrediction.record_columns()).to_csv(out_csv, index=False)

def run_dragon_ryder(model_path:Path, ryder_folder:Path, args:argparse.Namespace, console:Console) -> Path:
    '''
    Runs dragon-ryder with the provided arguments and returns a path to
    the output predictions csv
    '''
    from ..dragon_ryder import DragonRyder
    dragon_ryder = DragonRyder(model_path, '', ryder_folder=ryder_folder)

    if dragon_ryder.predictions_csv.exists():
        console.print(f'[yellow]dragon-ryder predictions already exist - skipping this step and reusing these')
    else:
        with init_dragon_ryder_from_args(args, model_path, ryder_folder) as dragon_ryder:
            console.rule(f'Running dragon-ryder with strategy {args.strategy}')
            with print_runtime('Dragon-ryder'):
                rcode = dragon_ryder.run()
            if rcode != 0:
                raise Exception(f'Dragon-ryder failed with return code {rcode}')

    return dragon_ryder.predictions_csv

def print_args(args, console:Console):
    for name in args.__dict__:
        console.print(f'{name}: {args.__dict__[name]}')

def save_or_load_args(saved_args:Path, args:argparse.Namespace, console:Console) -> argparse.Namespace:
    '''
    If saved_args exists, it is loaded and returned. Otherwise, args (parsed by this cmd-line)
    is saved to saved_args and returned. A summary of the selected args object is printed
    '''
    if saved_args.exists():
        with open(saved_args, 'rb') as f:
            args = pickle.load(f)
        console.rule(f'[bold yellow]Loaded saved arguments (ignoring current cmd-line):', align='left', style='blue')
    else:
        console.rule(f'Arguments', align='left', style='blue')
        # save arguments in case we resume
        with open(saved_args, 'wb') as f:
            pickle.dump(args, f, protocol=5)

    print_args(args, console)
    console.rule('', style='blue')

    return args

def load_models_from_arg(model_arg) -> List[Path]:
    '''
    Determine if the argument points to a model file or folder of models (*.pt), and
    return a list of Paths to all the specified model files
    '''
    if Path(model_arg).is_dir():
        return list(Path(model_arg).glob('*.pt'))
    return [Path(model_arg)]

def eval_dragon_model(model_path:Path, dragon_results:Path, proj, args, strip_bins, console, debug_df:pd.DataFrame) -> PandasEvalMetrics:

    dragon_preds_csv = dragon_results/f'{model_path.stem}.preds.csv'
    dragon_aligned_csv = dragon_results/f'{model_path.stem}.aligned.csv'

    if not dragon_preds_csv.exists():
        console.rule(f'Running dragon model {model_path.name}')
        with print_runtime('Dragon'):
            run_dragon(model_path, args, dragon_preds_csv, proj, strip_bins, console)

    if not dragon_aligned_csv.exists():
        dragon_df = pd.read_csv(dragon_preds_csv)
        dragon_df['Pred'] = dragon_df.PredJson.apply(DataType.from_json)
        dragon_mdf = align_variables(dragon_df, debug_df)   # TODO - save as CSV?
        dragon_mdf.to_csv(dragon_aligned_csv, index=False)
    else:
        dragon_mdf = pd.read_csv(dragon_aligned_csv)
        dragon_mdf['Pred'] = dragon_mdf.PredJson.apply(DataType.from_json)

    return PandasEvalMetrics(dragon_mdf, truth_col='TypeSeq', pred_col='PredSeq', name=f'DRAGON {model_path.stem}')

def eval_dragon_models(dragon_models:List[Path], dragon_results:Path, args, console, bin_paths_csv:Path,
                       debug_df:pd.DataFrame) -> List[PandasEvalMetrics]:
    from datatype_recovery.dragon_ryder import load_bin_files
    from ghidralib.projects import OpenSharedGhidraProject

    metrics = []

    if dragon_models:
        if not dragon_results.exists():
            dragon_results.mkdir()

        with OpenSharedGhidraProject(args.host, args.ghidra_repo, args.port) as proj:
            strip_bins, _ = load_bin_files(proj, bin_paths_csv, console, args.binaries)

            for model_path in dragon_models:
                metrics.append(eval_dragon_model(model_path, dragon_results, proj, args, strip_bins, console, debug_df))

    return metrics

def eval_dragonryder_model(model_path:Path, dragon_ryder_results:Path, args, console, debug_df:pd.DataFrame) -> PandasEvalMetrics:
    ryder_folder = dragon_ryder_results/f'{model_path.stem}.dragon-ryder'
    ryder_preds_csv = ryder_folder/'predictions.csv'
    ryder_aligned_csv = dragon_ryder_results/f'{model_path.stem}.aligned.csv'

    if not ryder_preds_csv.exists():
        ryder_preds_csv = run_dragon_ryder(model_path, args, console)

    if not ryder_aligned_csv.exists():
        ryder_df = pd.read_csv(ryder_preds_csv)
        ryder_df['Pred'] = ryder_df.PredJson.apply(DataType.from_json)
        ryder_mdf = align_variables(ryder_df, debug_df)
        ryder_mdf.to_csv(ryder_aligned_csv, index=False)
        # TODO - project types?
    else:
        ryder_mdf = pd.read_csv(ryder_aligned_csv)
        ryder_mdf['Pred'] = ryder_mdf.PredJson.apply(DataType.from_json)

    return PandasEvalMetrics(ryder_mdf, truth_col='TypeSeq', pred_col='PredSeq', name=f'DRAGON-RYDER {model_path.stem}')

def eval_dragonryder_models(dragon_ryder_models:List[Path], dragon_ryder_results:Path, args,
                            console, debug_df:pd.DataFrame) -> List[PandasEvalMetrics]:

    metrics = []

    if dragon_ryder_models:
        if not dragon_ryder_results.exists():
            dragon_ryder_results.mkdir()
        for model_path in dragon_ryder_models:
            metrics.append(eval_dragonryder_model(model_path, dragon_ryder_results, args, console, debug_df))

    return metrics

def main():
    p = argparse.ArgumentParser(description='Evaluate simple type prediction models')
    p.add_argument('name', type=str, help='Name for this experiment folder')
    p.add_argument('--dragon', type=Path, help='Path to DRAGON model (or a folder of DRAGON models) to evaluate')
    p.add_argument('--dragon-ryder', type=Path, help='Path to DRAGON model (or a folder of DRAGON models) to use for DRAGON-RYDER evaluation')
    # TODO - tygr model

    add_binary_opts(p)
    add_ghidra_opts(p)
    add_model_opts(p)
    add_dragon_ryder_opts(p)

    args = p.parse_args()
    console = Console()

    # create eval folder
    eval_folder = create_eval_folder(args.name, args.resume).absolute()
    os.chdir(eval_folder)

    saved_args = eval_folder/'args.pickle'
    args = save_or_load_args(saved_args, args, console)

    dragon_models = []
    dragon_ryder_models = []

    if args.dragon:
        dragon_models = load_models_from_arg(args.dragon)
    if args.dragon_ryder:
        dragon_ryder_models = load_models_from_arg(args.dragon_ryder)

    debug_csv = eval_folder/'debug_vars.csv'
    bin_paths_csv = eval_folder/'binary_paths.csv'

    dragon_results = eval_folder/'dragon'               # each dragon model's results go in here
    dragon_ryder_results = eval_folder/'dragon_ryder'   # same for dragon-ryder

    dragon_metrics = []
    ryder_metrics = []

    # ------------- start pyhidra
    console.print(f'Starting pyhidra...')
    import pyhidra
    pyhidra.start()

    debug_df = export_truth_types(args, console, debug_csv, bin_paths_csv)
    dragon_metrics = eval_dragon_models(dragon_models, dragon_results, args, console, bin_paths_csv, debug_df)
    ryder_metrics = eval_dragonryder_models(dragon_ryder_models, dragon_ryder_results, args, console, debug_df)

    for m in dragon_metrics:
        m.print_summary(console)
    for m in ryder_metrics:
        m.print_summary(console)

    # import IPython; IPython.embed()

    # TODO - project types?
    # 1. compute INDEPENDENT metrics and combine (for comparison)
    # 2. join predictions on ONLY SHARED variables and compute SHARED metrics
    #
    # // both versions print their metrics (possibly write metrics.csv?)
    # // notebook can read both predictions.csv and generate nice plots

    return 0

if __name__ == '__main__':
    exit(main())
