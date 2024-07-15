import argparse
import pandas as pd
from pathlib import Path
from rich.console import Console
import os
from typing import List

from datatype_recovery.dragon_ryder_cmdline import *
from datatype_recovery.eval_dataset import align_variables

# eval_simple_types --dragon=<model> --dragon-ryder=<model> --tygr=<model>

def create_eval_folder(name:str) -> Path:
    eval_folder = Path(name)
    if eval_folder.exists():
        raise Exception(f'Eval folder {eval_folder} already exists')
    eval_folder.mkdir()
    return eval_folder

def main():
    p = argparse.ArgumentParser(description='Evaluate simple type prediction models')
    p.add_argument('--dragon', type=Path, help='Path to DRAGON model to evaluate')
    p.add_argument('--dragon-ryder', type=Path, help='Path to DRAGON model to use for DRAGON-RYDER evaluation')
    # TODO - tygr model
    p.add_argument('--name', type=str, help='Name for this experiment folder')

    add_binary_opts(p)
    add_ghidra_opts(p)
    add_model_opts(p)
    add_dragon_ryder_opts(p)

    args = p.parse_args()
    console = Console()

    # NOTE: just get it working right now - later I can refactor to be more generic...
    # - function to add_simpletype_model(model_name, eval_model_callback, etc...)

    if args.dragon:
        console.print(f'Evaluating DRAGON model: {args.dragon}')
    if args.dragon_ryder:
        console.print(f'Evaluating DRAGON-RYDER with dragon model: {args.dragon_ryder}')

    # create eval folder
    eval_folder = create_eval_folder(args.name).absolute()
    os.chdir(eval_folder)

    debug_csv = eval_folder/'debug_vars.csv'
    bin_paths_csv = eval_folder/'binary_paths.csv'

    # ------------- start pyhidra
    import pyhidra
    pyhidra.start()
    from ghidralib.export_vars import export_debug_vars
    from ghidralib.projects import OpenSharedGhidraProject, DomainFile
    from dragon_ryder import DragonRyder, load_bin_files

    # TODO: accept/check for an already-exported debug CSV (just accept it on cmd-line -> driver will provide it)
    # TODO: implement overall experiment:

    bin_files:List[DomainFile] = []  # load these below from ghidra project

    console.rule(f'Exporting debug variables (truth)')
    with OpenSharedGhidraProject(args.host, args.ghidra_repo, args.port) as proj:
        bin_files = load_bin_files(proj, bin_paths_csv, console, args.binaries)

        if debug_csv.exists():
            console.print(f'[yellow]debug vars already exported - skipping this step and reusing them')
            debug_df = pd.read_csv(debug_csv)
        else:
            if args.limit > 0:
                console.print(f'[bold orange1] only exporting first {args.limit:,} debug functions')
            debug_df = export_debug_vars(proj, bin_files, args.limit if args.limit > 0 else None)
            debug_df.to_csv(debug_csv, index=False)

    if args.dragon_ryder:
        console.rule(f'Running dragon-ryder')
        with DragonRyder(args.dragon_ryder, args.ghidra_repo, args.device,
                                args.resume, args.nrefs, args.rollback_delete,
                                args.host, args.port, args.strategy,
                                args.binaries, args.limit) as dragon_ryder:

                if dragon_ryder.predictions_csv.exists():
                    console.print(f'[yellow]dragon-ryder predictions already exist - skipping this step and reusing these')
                else:
                    rcode = dragon_ryder.run()
                    if rcode != 0:
                        console.print(f'[red]Dragon-ryder failed with return code {rcode}')
                        return rcode

    import IPython; IPython.embed()

    # TODO: dragon could be similar to this...
    # def make_initial_predictions(self) -> pd.DataFrame:
    #     if self.dragon_initial_preds.exists():
    #         print(f'Initial predictions file already exists - moving on to next step')
    #         model_pred = pd.read_csv(self.dragon_initial_preds)
    #         return model_pred.set_index('Index')

    #     model_pred = make_predictions_on_dataset(self.dragon_model_path, self.device, self.dataset)

    #     # join with Name_Strip so we can identify the variable for retyping by name
    #     vdf = self.dataset.read_vars_csv()[['BinaryId','FunctionStart','Signature','Vartype',
    #                                         'Name_Strip']]
    #     model_pred = model_pred.merge(vdf, how='left', on=['BinaryId','FunctionStart','Signature','Vartype'])

    #     print(f'Saving model predictions to {self.dragon_initial_preds}')
    #     # save index since we will refer to it to partition high/low confidence
    #     model_pred.to_csv(self.dragon_initial_preds, index_label='Index')

    #     return model_pred


    ########## eval() -> CSV
    # mdf = align_variables(rdf, vdf)
    # TODO - project types?


    # TODO - USE LIMIT to run end-to-end QUICKLY
    # TODO - check for output of each step and skip it (with a warning) if we ran with --resume
    #
    # export_debug_vars <repo> <binary_list> debug_vars.csv
    #
    # dragon eval XYZ --truth=debug_vars.csv
    #       {XYZ}.dragon/predictions.csv
    #
    # dragon-ryder run XYZ
    #       // generates {XYZ}.dragon-ryder/predictions.csv
    #
    # eval dragon/predictions.csv debug_vars.csv --project-types=TYGR   // <- align, save merged CSV
    # eval dragon-ryder/predictions.csv debug_vars.csv --project-types=TYGR
    #
    # 1. compute INDEPENDENT metrics and combine (for comparison)
    # 2. join predictions on ONLY SHARED variables and compute SHARED metrics
    #
    # // both versions print their metrics (possibly write metrics.csv?)
    # // notebook can read both predictions.csv and generate nice plots

    return 0

if __name__ == '__main__':
    exit(main())
