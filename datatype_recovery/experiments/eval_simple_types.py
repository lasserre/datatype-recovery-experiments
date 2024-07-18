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

def export_truth_types(args:argparse.Namespace, console:Console, debug_csv:Path, debug_bins:List) -> pd.DataFrame:
    from ghidralib.export_vars import export_debug_vars
    from ghidralib.projects import OpenSharedGhidraProject

    console.rule(f'Exporting debug variables (truth)')

    with OpenSharedGhidraProject(args.host, args.ghidra_repo, args.port) as proj:
        if debug_csv.exists():
            console.print(f'[yellow]debug vars already exported - skipping this step and reusing them')
            debug_df = pd.read_csv(debug_csv)
            # load Type from json so it's not just a string
            debug_df['Type'] = debug_df.TypeJson.apply(DataType.from_json)
        else:
            if args.limit:
                console.print(f'[bold orange1] only exporting first {args.limit:,} debug functions')
            debug_df = export_debug_vars(proj, debug_bins, args.limit)
            debug_df.to_csv(debug_csv, index=False)
        return debug_df

def run_dragon(args:argparse.Namespace, out_csv:Path, proj, strip_bins:List, console:Console):
    '''
    Runs dragon with the provided arguments and returns a path to
    the output predictions csv
    '''
    from ghidralib.projects import verify_ghidra_revision, GhidraCheckoutProgram
    from ghidralib.decompiler import AstDecompiler

    model = DragonModel.load_model(args.dragon, args.device, eval=True)

    table_rows = []

    for bin_file in strip_bins:
        bid = binary_id(bin_file.name)
        verify_ghidra_revision(bin_file, expected_revision=1, rollback_delete=True)

        with GhidraCheckoutProgram(proj, bin_file) as co:
            with AstDecompiler(co.program, bid, timeout_sec=240) as decompiler:
                nonthunks = co.decompiler.nonthunk_functions[:args.limit]
                for func in tqdm(nonthunks, desc=bin_file.name):
                    ast = decompiler.decompile_ast(func)
                    if ast is None:
                        console.print('[bold orange1]Decompilation failed:')
                        console.print(f'[orange1]{decompiler.last_error_msg}')
                        continue
                    var_preds = model.predict_func_types(ast, args.device, bid, skip_unique_vars=True)
                    table_rows.extend([
                        [*p.varid, p.vardecl.name, p.vardecl.location, p.pred_dt, p.pred_dt.to_json()] for p in var_preds
                    ])

    pd.DataFrame.from_records(table_rows, columns=[
        'BinaryId','FunctionStart','Signature','Vartype','Name','Location','Pred','PredJson'
    ]).to_csv(out_csv, index=False)

def run_dragon_ryder(args:argparse.Namespace, console:Console) -> Path:
    '''
    Runs dragon-ryder with the provided arguments and returns a path to
    the output predictions csv
    '''
    from datatype_recovery.dragon_ryder import DragonRyder
    console.rule(f'Running dragon-ryder')

    with DragonRyder(args.dragon_ryder, args.ghidra_repo, args.device,
                    args.resume, args.nrefs, args.rollback_delete,
                    args.host, args.port, args.strategy,
                    args.binaries, args.limit) as dragon_ryder:

        if dragon_ryder.predictions_csv.exists():
            console.print(f'[yellow]dragon-ryder predictions already exist - skipping this step and reusing these')
        else:
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
            console.rule(f'[bold yellow]Loaded saved arguments:', align='left', style='blue')
        else:
            console.rule(f'Arguments', align='left', style='blue')
            # save arguments in case we resume
            with open(saved_args, 'wb') as f:
                pickle.dump(args, f, protocol=5)

        print_args(args, console)
        console.rule('', style='blue')

        return args

def main():
    p = argparse.ArgumentParser(description='Evaluate simple type prediction models')
    p.add_argument('name', type=str, help='Name for this experiment folder')
    p.add_argument('--dragon', type=Path, help='Path to DRAGON model to evaluate')
    p.add_argument('--dragon-ryder', type=Path, help='Path to DRAGON model to use for DRAGON-RYDER evaluation')
    # TODO - tygr model

    add_binary_opts(p)
    add_ghidra_opts(p)
    add_model_opts(p)
    add_dragon_ryder_opts(p)

    args = p.parse_args()
    console = Console()

    # NOTE: just get it working right now - later I can refactor to be more generic...
    # - function to add_simpletype_model(model_name, eval_model_callback, etc...)

    # create eval folder
    eval_folder = create_eval_folder(args.name, args.resume).absolute()
    os.chdir(eval_folder)

    saved_args = eval_folder/'args.pickle'
    args = save_or_load_args(saved_args, args, console)

    # import IPython; IPython.embed()

    debug_csv = eval_folder/'debug_vars.csv'
    bin_paths_csv = eval_folder/'binary_paths.csv'
    dragon_preds_csv = eval_folder/'dragon_predictions.csv'
    dragon_aligned_csv = eval_folder/'dragon_aligned.csv'
    ryder_aligned_csv = eval_folder/'ryder_aligned.csv'
    ryder_preds_csv = None
    dragon_metrics = None
    ryder_metrics = None

    # ------------- start pyhidra
    console.print(f'Starting pyhidra...')
    import pyhidra
    pyhidra.start()

    from datatype_recovery.dragon_ryder import load_bin_files
    from ghidralib.projects import OpenSharedGhidraProject

    # TODO: accept/check for an already-exported debug CSV (just accept it on cmd-line -> driver will provide it)

    with OpenSharedGhidraProject(args.host, args.ghidra_repo, args.port) as proj:
        strip_bins, debug_bins = load_bin_files(proj, bin_paths_csv, console, args.binaries)
        debug_df = export_truth_types(args, console, debug_csv, debug_bins)

        if args.dragon:
            console.rule(f'Running dragon')
            if not dragon_preds_csv.exists():
                with print_runtime('Dragon'):
                    run_dragon(args, dragon_preds_csv, proj, strip_bins, console)

            dragon_df = pd.read_csv(dragon_preds_csv)
            dragon_df['Pred'] = dragon_df.PredJson.apply(DataType.from_json)
            dragon_mdf = align_variables(dragon_df, debug_df)   # TODO - save as CSV?
            dragon_mdf.to_csv(dragon_aligned_csv, index=False)
            dragon_metrics = PandasEvalMetrics(dragon_mdf, truth_col='TypeSeq', pred_col='PredSeq')

    if args.dragon_ryder:
        ryder_preds_csv = run_dragon_ryder(args, console)
        ryder_df = pd.read_csv(ryder_preds_csv)
        ryder_df['Pred'] = ryder_df.PredJson.apply(DataType.from_json)
        ryder_mdf = align_variables(ryder_df, debug_df)
        ryder_mdf.to_csv(ryder_aligned_csv, index=False)
        # TODO - project types?
        # TODO - calculate metrics...(entire thing, subset, etc..)
        ryder_metrics = PandasEvalMetrics(ryder_mdf, truth_col='TypeSeq', pred_col='PredSeq')

    if dragon_metrics:
        dragon_metrics.print_summary('DRAGON', console)
    if ryder_metrics:
        ryder_metrics.print_summary('DRAGON-RYDER', console)

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
