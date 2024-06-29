# DRAGON RYDER
# -------------
# DRAGON incremental RetYping DrivER

import argparse
import pandas as pd
from pathlib import Path
from rich.console import Console
from typing import List, Tuple

from datatype_recovery.models.eval import make_predictions_on_dataset
from datatype_recovery.models.dataset import load_dataset_from_path

class DragonRyder:
    def __init__(self, dragon_model_path:Path, dataset_path:Path, device:str='cpu',
                resume:bool=False, numrefs_thresh:int=5) -> None:
        self.dragon_model_path = dragon_model_path
        self.dataset_path = dataset_path
        self.device = device
        self.resume = resume
        self.numrefs_thresh = numrefs_thresh
        self.dataset = load_dataset_from_path(dataset_path)

        self._check_dataset()   # ensure dataset is valid for dragon-ryder (not balanced, etc.)

    @property
    def ryder_folder(self) -> Path:
        # for simplicity/sanity, create a NAME.dragon-ryder folder to store all the intermediate data
        # - this will help easily isolate independent runs, help debugging, etc. especially bc I'm going fast
        # assumes cwd
        return Path(f'{self.dataset_path.name}-{self.dragon_model_path.name}.dragon-ryder')

    @property
    def dragon_initial_preds(self) -> Path:
        return self.ryder_folder/'dragon_initial_preds.csv'

    @property
    def high_confidence_vars(self) -> Path:
        return self.ryder_folder/'high_conf_vars.csv'

    @property
    def low_confidence_vars(self) -> Path:
        # everything but high confidence vars...
        return self.ryder_folder/'low_conf_vars.csv'

    @property
    def retyped_vars(self) -> Path:
        return self.ryder_folder/'retyped_vars.csv'

    @property
    def status_file(self) -> Path:
        # right now, simply write last completed step # (1-6)
        return self.ryder_folder/'status.csv'

    @staticmethod
    def get_last_completed_step(ryder_folder:Path) -> Tuple[int, str]:
        status_file = ryder_folder/'status.csv'
        if not status_file.exists():
            return -1, ''
        with open(status_file, 'r') as f:
            last_completed_step, comment = f.readline().strip().split(',')
        return last_completed_step, comment

    @staticmethod
    def report_status(ryder_folder:Path):
        last_completed_step, comment = DragonRyder.get_last_completed_step(ryder_folder)
        if last_completed_step == -1:
            print(f'No progress made - status.csv does not exist')
        else:
            print(f'Last completed step: {last_completed_step} - {comment}')
        return 0

    def _check_dataset(self):
        if self.dataset.balance_dataset:
            raise Exception(f'Dataset was built with --balance, which is invalid for dragon-ryder')

    def _update_status_file(self, completed_step:int, text:str):
        with open(self.status_file, 'w') as f:
            f.write(f'{completed_step},{text}')

    def make_initial_predictions(self) -> pd.DataFrame:
        if self.dragon_initial_preds.exists():
            print(f'Initial predictions file already exists - moving on to next step')
            model_pred = pd.read_csv(self.dragon_initial_preds)
            return model_pred.set_index('Index')

        model_pred = make_predictions_on_dataset(self.dragon_model_path, self.device, self.dataset)
        print(f'Saving model predictions to {self.dragon_initial_preds}')

        # save index since we will refer to it to partition high/low confidence
        model_pred.to_csv(self.dragon_initial_preds, index_label='Index')

        return model_pred

    def collect_high_confidence_preds(self, init_preds:pd.DataFrame) -> List[int]:
        # NOTE: the generic form of this is to track variables we have
        # retyped (so we don't retype >1x) and continue iterating until all
        # new vars have been accounted for
        #
        # -> save our retyping decisions in a special file for later use:

        if self.high_confidence_vars.exists():
            print(f'High confidence vars file already exists - moving on to next step')
            with open(self.high_confidence_vars, 'r') as f:
                hc_idx = [int(x) for x in f.readline().strip().split(',')]
            return hc_idx

        print(f'Taking all variables with {self.numrefs_thresh} or more references as high confidence')
        high_conf = init_preds.loc[init_preds.NumRefs >= self.numrefs_thresh]

        # just write index as a single csv
        with open(self.high_confidence_vars, 'w') as f:
            f.write(",".join(str(x) for x in high_conf.index.to_list()))

        return high_conf.index.to_list()

    def apply_predictions_to_ghidra(self, preds:pd.DataFrame, expected_ghidra_revision:int=None):
        '''
        Applies our predicted data types (Pred column in preds) to the
        variables in each Ghidra binary

        preds: Data type predictions to apply
        expected_ghidra_revision: Expected Ghidra revision (e.g. 1 for initial state) of each database. If
                                  this is specified but does not match the apply will fail (for now - later we can roll back)
        '''
        # TODO: group by binaryid
        bt = self.dataset.full_binaries_table

        # TODO: check expected revision (if specified)
        #   - create (new?) rollback version for each binary -- or otherwise delete history...

        # TODO: use OpenSharedGhidraProject, GhidraCheckout, GhidraRetyper...

        # TODO: save preds to retyped_vars (update/add to this file in general...)

        import IPython; IPython.embed()



    def run(self):
        console = Console()
        print(f'{"Resuming" if self.resume else "Running"} dragon-ryder on {self.dataset_path} using DRAGON model {self.dragon_model_path}')

        if not self.resume:
            if self.ryder_folder.exists():
                console.print(f'[yellow]Warning: {self.ryder_folder} folder already exists. Use --resume to continue if unfinished')
                return 1

        if not self.ryder_folder.exists():
            self.ryder_folder.mkdir()

        # 1. Make initial predictions on dataset (save these somewhere - pd/csv)
        console.rule(f'Step 1/6: make initial model predictions')
        init_preds = self.make_initial_predictions()

        # 2. Collect high confidence predictions (via strategy)
        console.rule(f'Step 2/6: collect high confidence predictions')
        hc_idx = self.collect_high_confidence_preds(init_preds)

        high_conf = init_preds.loc[hc_idx, :]
        # low_conf = init_preds.loc[~init_preds.index.isin(hc_idx),:]

        # 3. Apply high confidence predictions to each binary
        console.rule(f'Step 3/6: apply high confidence predictions to Ghidra')
        self.apply_predictions_to_ghidra(high_conf, expected_ghidra_revision=1)

        # TODO: implement...
        # 4. Update/rebuild the dataset from this latest state
        #   - re-export all function ASTs
        #   - re-build the dataset (via dragon build)
        #       >> this means we need to KNOW how it was initially built (run same cmd-line params...)

        #       NOTE: we DO know this I think (via dataset/raw/input_params.json)

        #   - LATER: we don't HAVE to rebuild var graphs for variables we have already predicted...
        #     ...for now, just redo everything for simplicity
        # 5. Make predictions on remaining variables (using this updated dataset)
        #   OPTIONALLY: make predictions on all updated variables...did anything change?
        # 6. Final predictions are JOIN of initial high-confidence predictions + gen 2 predictions

        # NOTE: to rollback:
        # - check out version 1
        # - "save as" the binary file as a new name (and check in)
        # - this new file is the "rolled back" copy we can use...
        return 0

def main():
    p = argparse.ArgumentParser(description='DRAGON incremental RetYping DrivER - recovers variable types using DRAGON and incremental retyping')

    subparsers = p.add_subparsers(dest='subcmd')

    # run: run dragon-ryder
    run_p = subparsers.add_parser('run', help='Run dragon-ryder')
    run_p.add_argument('dragon_model', type=Path, help='The trained DRAGON model to use')
    run_p.add_argument('dataset', type=Path, help='The dataset on which to run dragon ryder (and recover basic variable types)')
    run_p.add_argument('--device', type=str, help='Pytorch device string on which to run the DRAGON model', default='cpu')
    run_p.add_argument('--resume', action='store_true', help='Continue after last completed step')
    run_p.add_argument('--nrefs', type=int, default=5, help='Number of references to use for high confidence variables')
    # TODO: add strategy selection? --strategy=truth|refs|confidence

    # status: show where we are
    status_p = subparsers.add_parser('status', help='Show the status of a dragon-ryder run')
    status_p.add_argument('folder', type=Path, help='The .dragon-ryder eval folder for which to report status')

    args = p.parse_args()

    if args.subcmd == 'run':
        return DragonRyder(args.dragon_model, args.dataset, args.device, args.resume, args.nrefs).run()
    elif args.subcmd == 'status':
        return DragonRyder.report_status(args.folder)

if __name__ == '__main__':
    exit(main())
