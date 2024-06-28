# DRAGON RYDER
# -------------
# DRAGON incremental RetYping DrivER

import argparse
import pandas as pd
from pathlib import Path
from rich.console import Console

from .models.eval import make_predictions_on_dataset
from .models.dataset import load_dataset_from_path

class DragonRyder:
    def __init__(self, dragon_model_path:Path, dataset_path:Path, device:str='cpu') -> None:
        self.dragon_model_path = dragon_model_path
        self.dataset_path = dataset_path
        self.device = device
        self.dataset = load_dataset_from_path(dataset_path)

        self._check_dataset()   # ensure dataset is valid for dragon-ryder (not balanced, etc.)

    @property
    def ryder_folder(self) -> Path:
        # assumes cwd
        return Path(f'{self.dataset_path.name}-{self.dragon_model_path.name}.dragon-ryder')

    @property
    def dragon_initial_preds(self) -> Path:
        return self.ryder_folder/'dragon_initial_preds.csv'

    @property
    def retyped_vars(self) -> Path:
        return self.ryder_folder/'retyped_vars.csv'

    def _check_dataset(self):
        if self.dataset.balance_dataset:
            raise Exception(f'Dataset was built with --balance, which is invalid for dragon-ryder')

    def make_initial_predictions(self):
        # make_predictions_on_dataset()
        print(f'Making initial model predictions')
        model_pred = make_predictions_on_dataset(self.dragon_model_path, self.device, self.dataset)
        print(f'Saving model predictions to {self.dragon_initial_preds}')
        model_pred.to_csv(self.dragon_initial_preds, index=False)

    def collect_high_confidence_preds(self, model_pred:pd.DataFrame):
        # NOTE: the generic form of this is to track variables we have
        # retyped (so we don't retype >1x) and continue iterating until all
        # new vars have been accounted for
        #
        # -> save our retyping decisions in a special file for later use:
        import IPython; IPython.embed()

        if self.retyped_vars.exists():
            raise Exception(f'{self.retyped_vars} file already exists!')

        # TODO: write next generation to retyped_vars, then apply retyping to Ghidra

    def run(self):
        console = Console()
        print(f'Running dragon-ryder on {self.dataset_path} using DRAGON model {self.dragon_model_path}')

        # for simplicity/sanity, create a NAME.dragon-ryder folder to store all the intermediate data
        # - this will help easily isolate independent runs, help debugging, etc. especially bc I'm going fast
        if self.ryder_folder.exists():
            console.print(f'[yellow]Warning: {self.ryder_folder} folder already exists')
            return 1

        # 1. Make initial predictions on dataset (save these somewhere - pd/csv)
        self.make_initial_predictions()

        # 2. Collect high confidence predictions (via strategy)
        self.collect_high_confidence_preds()

        # TODO: implement...
        # 3. Apply high confidence predictions to each binary
        #   - we can get to Ghidra repos from dataset/raw/input_params.csv:experiment_runs (.exp folder stem is the repo name)
        #   - create (new?) rollback version for each binary -- or otherwise delete history...
        # 4. Update/rebuild the dataset from this latest state
        #   - re-export all function ASTs
        #   - re-build the dataset (via dragon build)
        #       >> this means we need to KNOW how it was initially built (run same cmd-line params...)

        #       NOTE: we DO know this I think (via dataset/raw/input_params.json)

        #   - LATER: we don't HAVE to rebuild var graphs for variables we have already predicted...
        #     ...for now, just redo everything for simplicity
        # 5. Make predictions on remaining variables (using this updated dataset)
        # 6. OPTIONALLY: make predictions on all updated variables...did anything change?
        # 7. Final predictions are JOIN of initial high-confidence predictions + gen 2 predictions

        # NOTE: to rollback:
        # - check out version 1
        # - "save as" the binary file as a new name (and check in)
        # - this new file is the "rolled back" copy we can use...
        return 0

def main():
    p = argparse.ArgumentParser(description='DRAGON incremental RetYping DrivER - recovers variable types using DRAGON and incremental retyping')
    p.add_argument('dataset', type=Path, help='The dataset on which to run dragon ryder (and recover basic variable types)')
    p.add_argument('dragon_model', type=Path, help='The trained DRAGON model to use')
    p.add_argument('--device', type=str, help='Pytorch device string on which to run the DRAGON model', default='cpu')
    # TODO: add strategy selection? --strategy=truth|refs|confidence
    args = p.parse_args()
    # p.add_argument('--exp', type=Path, default=Path().cwd(), help='The experiment folder')

    return DragonRyder(args.dragon_model, args.dataset, args.device).run()

if __name__ == '__main__':
    exit(main())
