import argparse
from collections import defaultdict
import pandas as pd
from pathlib import Path
from rich.console import Console
import os
import pickle
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from typing import List, Tuple

from datatype_recovery.models.homomodels import DragonModel, VarPrediction
from datatype_recovery.models.dataset import load_dataset_from_path
from datatype_recovery.dragon_ryder_cmdline import *
from datatype_recovery.eval_dataset import *
from varlib.datatype import DataType
from varlib.location import Location
from astlib import binary_id, run_id
from wildebeest.utils import print_runtime

# eval_simple_types --dragon=<model> --dragon-ryder=<model> --tygr=<model>

def create_eval_folder(name:str, resume:bool) -> Path:
    eval_folder = Path(name)
    if eval_folder.exists() and not resume:
        raise Exception(f'Eval folder {eval_folder} already exists')
    eval_folder.mkdir(exist_ok=True)
    return eval_folder

def read_dragon_preds(eval_folder:Path, first_only:bool=False, dragon_ryder:bool=False) -> List[Tuple[Path,pd.DataFrame]]:
    '''
    Read each of the aligned dragon prediction files (*.aligned.csv) in
    this eval folder and return them as a list of DataFrames

    eval_folder: Eval folder to read from
    first_only: Only read and return the first csv if there are multiple
    dragon_ryder: Read dragon-ryder predictions instead of dragon predictions
    '''
    foldername = 'dragon_ryder' if dragon_ryder else 'dragon'
    dragon_pred_csvs = list((eval_folder/foldername).glob('*.aligned.csv'))
    # coreutils binaries named "true" and "false" make the Binary column
    # intepreted as a bool instead of string...enforce string interpretation
    dtypes = defaultdict(lambda: str, Binary="str")
    if first_only:
        if len(dragon_pred_csvs) > 1:
            print(f'{len(dragon_pred_csvs)} model csvs found, only using the first one ({dragon_pred_csvs[0]})')
        return (dragon_pred_csvs[0], pd.read_csv(dragon_pred_csvs[0], dtype=dtypes))
    return [(x, pd.read_csv(x, dtype=dtypes)) for x in dragon_pred_csvs]

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
            debug_df = export_debug_vars(proj, debug_bins, args.limit, skip_unique_vars=True)
            debug_df.to_csv(debug_csv, index=False)
    return debug_df

def remap_bid_from_debug_df(df:pd.DataFrame, debug_df:pd.DataFrame) -> pd.DataFrame:
    '''
    Remap the BinaryId column in df using the unique (arbitrary) mapping from debug_df.
    Values are mapped using the OrigBinaryId/RunId columns and the resulting dataframe
    has BinaryId/OrigBinaryId/RunId entries which match debug_df
    '''
    # map (OrigBinaryId, RunId) -> BinaryId
    bid_lookup = dict(zip(zip(debug_df['OrigBinaryId'], debug_df['RunId']), debug_df['BinaryId']))
    df2 = df.rename({'BinaryId': 'OrigBinaryId'}, axis=1)
    df2['BinaryId'] = df2.apply(lambda x: (x.OrigBinaryId, x.RunId), axis=1).map(bid_lookup)
    return df2

def run_dragon_offline(dragon_model:Path, dataset_path:Path, device:str, out_csv:Path, console:Console, var_df:pd.DataFrame,
                        use_test_split:bool=False):
    model = DragonModel.load_model(dragon_model, device, eval=True)
    ds = load_dataset_from_path(dataset_path)

    if use_test_split:
        console.print(f'[yellow]Using test split from dataset {dataset_path}')
        ds = ds.test_split      # only eval on test split

    loader = DataLoader(ds, batch_size=1024)     # arbitrarily chosen batch size
    preds = model.predict_loader_types(loader, show_progress=True)

    preds_df = pd.DataFrame([p.to_record() for p in preds], columns=VarPrediction.record_columns())

    # fill in stripped varname, varloc from var_df using varid
    varid_cols = ['BinaryId','FunctionStart','Signature','Vartype']
    preds_df = preds_df.drop('Name',axis=1).drop('Location',axis=1)
    preds_df = preds_df.merge(var_df[[*varid_cols, 'Name_Strip','Location_Strip']], how='left', on=varid_cols)
    preds_df = preds_df.rename({'Name_Strip': 'Name', 'Location_Strip': 'Location'}, axis=1)

    preds_df.to_csv(out_csv, index=False)

def run_dragon(dragon_model:Path, args:argparse.Namespace, out_csv:Path, proj, strip_bins:List, console:Console, debug_df:pd.DataFrame=None):
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
        rid = run_id(bin_file.parent.name)
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
                        [rid, *p.to_record()] for p in var_preds
                    ])

    df = pd.DataFrame.from_records(table_rows, columns=['RunId', *VarPrediction.record_columns()])

    if debug_df is not None:
        df = remap_bid_from_debug_df(df, debug_df)

    df.to_csv(out_csv, index=False)

def run_dragon_ryder(model_path:Path, ryder_folder:Path, args:argparse.Namespace, console:Console, debug_df:pd.DataFrame=None) -> Path:
    '''
    Runs dragon-ryder with the provided arguments and returns a path to
    the output predictions csv
    '''
    from datatype_recovery.dragon_ryder import DragonRyder
    # from ..dragon_ryder import DragonRyder
    dragon_ryder = DragonRyder(model_path, '', ryder_folder=ryder_folder)

    if dragon_ryder.predictions_csv.exists():
        console.print(f'[yellow]dragon-ryder predictions already exist - skipping this step and reusing these')
    else:
        with init_dragon_ryder_from_args(args, model_path, ryder_folder) as dragon_ryder:
            console.rule(f'Running dragon-ryder with strategy {args.strategy} (conf={dragon_ryder.confidence})')
            with print_runtime('Dragon-ryder'):
                rcode = dragon_ryder.run()
            if rcode != 0:
                raise Exception(f'Dragon-ryder failed with return code {rcode}')

    if debug_df is not None:
        df = pd.read_csv(dragon_ryder.predictions_csv)
        df = remap_bid_from_debug_df(df, debug_df)
        df.to_csv(dragon_ryder.predictions_csv, index=False)    # update csv file

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

def get_dragon_preds_csvpath(dragon_results:Path, model_path:Path) -> Path:
    return dragon_results/f'{model_path.stem}.preds.csv'
def get_dragon_aligned_csvpath(dragon_results:Path, model_path:Path) -> Path:
    return dragon_results/f'{model_path.stem}.aligned.csv'

def align_preds(preds_csv:Path, debug_df:pd.DataFrame, aligned_csv:Path) -> pd.DataFrame:
    '''
    Align the predictions in preds_csv, saving them in aligned_csv and returning
    the result as a DataFrame
    '''
    if not aligned_csv.exists():
        df = pd.read_csv(preds_csv)
        df['Pred'] = df.PredJson.apply(DataType.from_json)
        mdf = align_variables(df, debug_df)
        mdf.to_csv(aligned_csv, index=False)
    else:
        mdf = pd.read_csv(aligned_csv)
        mdf['Pred'] = mdf.PredJson.apply(DataType.from_json)
        mdf['Type'] = mdf.TypeJson.apply(DataType.from_json)
    return mdf

def eval_dragon_model(model_path:Path, dragon_results:Path, proj, args, strip_bins, console, debug_df:pd.DataFrame) -> PandasEvalMetrics:

    dragon_preds_csv = get_dragon_preds_csvpath(dragon_results, model_path)
    dragon_aligned_csv = get_dragon_aligned_csvpath(dragon_results, model_path)

    if not dragon_preds_csv.exists():
        console.rule(f'Running dragon model {model_path.name}')
        with print_runtime('Dragon'):
            run_dragon(model_path, args, dragon_preds_csv, proj, strip_bins, console, debug_df)

    dragon_mdf = align_preds(dragon_preds_csv, debug_df, dragon_aligned_csv)

    return PandasEvalMetrics(dragon_mdf, truth_col='TypeSeq', pred_col='PredSeq', name=f'DRAGON {model_path.stem}')

def eval_dragon_model_offline(model_path:Path, dragon_results:Path, args, console, var_df:pd.DataFrame) -> PandasEvalMetrics:
    dragon_preds_csv = get_dragon_preds_csvpath(dragon_results, model_path)
    dragon_aligned_csv = get_dragon_aligned_csvpath(dragon_results, model_path)

    if not dragon_preds_csv.exists():
        console.rule(f'Running dragon model {model_path.name} (offline)')
        with print_runtime('Dragon'):
            run_dragon_offline(model_path, args.dataset, args.device, dragon_preds_csv, console, var_df, args.test_split)

    # drop Name_Strip, Location_Strip for debug_df
    debug_df = var_df.drop('Name_Strip',axis=1).drop('Location_Strip',axis=1)

    dragon_mdf = align_preds(dragon_preds_csv, debug_df, dragon_aligned_csv)
    return PandasEvalMetrics(dragon_mdf, truth_col='TypeSeq', pred_col='PredSeq', name=f'DRAGON {model_path.stem}')

def eval_dragon_models_offline(dragon_models:List[Path], dragon_results:Path, args, console, var_df:pd.DataFrame):
    if dragon_models and not dragon_results.exists():
        dragon_results.mkdir()
    return [eval_dragon_model_offline(model_path, dragon_results, args, console, var_df) for model_path in dragon_models]

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

    if not dragon_ryder_results.exists():
        dragon_ryder_results.mkdir()

    if not ryder_preds_csv.exists():
        ryder_preds_csv = run_dragon_ryder(model_path, ryder_folder, args, console, debug_df)

    if not ryder_aligned_csv.exists():
        ryder_df = pd.read_csv(ryder_preds_csv)
        ryder_df['Pred'] = ryder_df.PredJson.apply(DataType.from_json)
        ryder_mdf = align_variables(ryder_df, debug_df)
        ryder_mdf.to_csv(ryder_aligned_csv, index=False)
    else:
        ryder_mdf = pd.read_csv(ryder_aligned_csv)
        ryder_mdf['Pred'] = ryder_mdf.PredJson.apply(DataType.from_json)

    return PandasEvalMetrics(ryder_mdf, truth_col='TypeSeq', pred_col='PredSeq', name=f'DRAGON-RYDER {model_path.stem}')

def eval_dragonryder_models(dragon_ryder_models:List[Path], dragon_ryder_results:Path, args,
                            console, debug_df:pd.DataFrame) -> List[PandasEvalMetrics]:

    metrics = []

    if dragon_ryder_models:
        for model_path in dragon_ryder_models:
            if args.sweep_confidence:
                start, stop, step = [int(float(x)*100) for x in args.sweep_confidence]
                for conf_int in range(start, stop, step):
                    conf = conf_int/100     # convert back to float in range [0, 1]
                    res_folder = dragon_ryder_results.with_name(f'{dragon_ryder_results.name}_conf{conf_int}')
                    args.confidence = conf      # overwrite default confidence for this run
                    metrics.append(eval_dragonryder_model(model_path, res_folder, args, console, debug_df))
            else:
                metrics.append(eval_dragonryder_model(model_path, dragon_ryder_results, args, console, debug_df))

    return metrics

def convert_vardf_to_debugdf(dataset_folder:Path) -> pd.DataFrame:
    '''
    Read in the variables.csv file associated with a dataset and convert the
    debug columns into the names/types expected by align_variables (as would be
    present when we run the eval online)
    '''
    var_csv = dataset_folder/'raw/variables.csv'
    bin_csv = dataset_folder/'raw/binaries.csv'

    var_df = pd.read_csv(var_csv).rename({
        'Name_Debug': 'Name',
        'Type_Debug': 'Type',
        'TypeJson_Debug': 'TypeJson',
    }, axis=1)

    def reconstruct_location(x:pd.Series, suffix:str='') -> Location:
        return Location(
            x[f'LocType{suffix}'],
            x[f'LocRegName{suffix}'] if isinstance(x[f'LocRegName{suffix}'], str) else '',
            int(x[f'LocOffset{suffix}']))

    var_df['Location'] = var_df.apply(lambda x: reconstruct_location(x,'_Debug'), axis=1)
    var_df['Location_Strip'] = var_df.apply(lambda x: reconstruct_location(x,'_Strip'), axis=1)
    var_df['Type'] = var_df.TypeJson.apply(DataType.from_json)  # load Type from json so it's not just a string

    if 'Binary' not in var_df:
        # Binary gets put in when we use --func-list, so we only need to do this lookup
        # if we don't already have this column filled in
        bnames = pd.read_csv(bin_csv)[['BinaryId','Name']].rename({'Name': 'Binary'},axis=1)
        var_df = var_df.merge(bnames, how='left', on='BinaryId')

    return var_df[['BinaryId','Binary','FunctionStart','Signature','Vartype',
                   'Name','Location','Type','TypeJson',
                   'Name_Strip','Location_Strip']]      # include Name_Strip/Location_Strip to label stripped vars

def main():
    p = argparse.ArgumentParser(description='Evaluate simple type prediction models',
                    epilog='Either --ghidra_repo or --dataset should be used to eval online or offline respectively')
    p.add_argument('name', type=str, help='Name for this experiment folder')
    p.add_argument('--dragon', type=Path, help='Path to DRAGON model (or a folder of DRAGON models) to evaluate')
    p.add_argument('--dragon-ryder', type=Path, help='Path to DRAGON model (or a folder of DRAGON models) to use for DRAGON-RYDER evaluation')
    p.add_argument('--sweep_confidence', nargs=3, help='Array of [START, STOP, STEP] (e.g. 0.5 1 0.1) where START is inclusive, STOP is not')

    add_binary_opts(p)
    add_ghidra_opts(p)
    p.add_argument('--dataset', type=Path, help='Path to offline dataset to use for model evaluation')
    p.add_argument('--test-split', action='store_true', help='Eval on the test split of the specified dataset (assumes --dataset)')
    add_model_opts(p)
    add_dragon_ryder_opts(p)

    args = p.parse_args()
    console = Console()

    # CLS: I'm only doing this manual args checking so I can reuse the --ghidra_repo
    # arguments elsewhere without --dataset (via add_ghidra_opts)
    if not args.ghidra_repo and not args.dataset:
        console.print(f'[red]One of --ghidra_repo or --dataset must be specified')
        return 1
    if args.ghidra_repo and args.dataset:
        console.print(f'[red]Either --ghidra_repo or --dataset must be specified, not both')
        return 1

    # create eval folder
    eval_folder = create_eval_folder(args.name, args.resume).absolute()

    if args.dragon:
        args.dragon = Path(args.dragon).absolute()
    if args.dragon_ryder:
        args.dragon_ryder = Path(args.dragon_ryder).absolute()
    if args.dataset:
        args.dataset = Path(args.dataset).absolute()

    saved_args = eval_folder/'args.pickle'
    args = save_or_load_args(saved_args, args, console)

    dragon_models = []
    dragon_ryder_models = []

    if args.dragon:
        dragon_models = load_models_from_arg(args.dragon)
    if args.dragon_ryder:
        dragon_ryder_models = load_models_from_arg(args.dragon_ryder)

    os.chdir(eval_folder)

    debug_csv = eval_folder/'debug_vars.csv'
    bin_paths_csv = eval_folder/'binary_paths.csv'

    dragon_results = eval_folder/'dragon'               # each dragon model's results go in here
    dragon_ryder_results = eval_folder/'dragon_ryder'   # same for dragon-ryder

    dragon_metrics = []
    ryder_metrics = []

    # ------------- start pyhidra
    if args.ghidra_repo:
        console.print(f'Starting pyhidra...')
        import pyhidra
        pyhidra.start()
        debug_df = export_truth_types(args, console, debug_csv, bin_paths_csv)
        dragon_metrics = eval_dragon_models(dragon_models, dragon_results, args, console, bin_paths_csv, debug_df)
        ryder_metrics = eval_dragonryder_models(dragon_ryder_models, dragon_ryder_results, args, console, debug_df)
    else:
        var_df = convert_vardf_to_debugdf(args.dataset)
        dragon_metrics = eval_dragon_models_offline(dragon_models, dragon_results, args, console, var_df)

    for m in dragon_metrics:
        m.print_summary(console)
    for m in ryder_metrics:
        m.print_summary(console)

    return 0

if __name__ == '__main__':
    exit(main())
