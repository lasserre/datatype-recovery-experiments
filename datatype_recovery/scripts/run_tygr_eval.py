#!/usr/bin/env python3
import argparse
from pathlib import Path
# import pandas as pd
import subprocess
from tqdm.auto import tqdm
from rich.console import Console

def main(args):
    '''
    This script automates the TYGR eval for benchmarks with multiple binaries
    (specifically coreutils)
    '''
    benchmark_binaries = Path(args.benchmark_binaries)
    output_folder = Path(args.output_folder)
    tygr_model = Path(args.tygr_model)

    console = Console()

    output_folder.mkdir(exist_ok=True)

    binaries = [x for x in benchmark_binaries.iterdir() if not x.is_dir()]
    output_csvs = []

    # we assume we are already in the TYGR folder
    for bin_file in (pbar := tqdm(binaries, desc='Running TYGR on binaries')):
        pbar.set_description(f'Processing {bin_file.name}')
        bin_output_folder = output_folder/f'{bin_file.name}.tygr'

        if bin_output_folder.exists():
            console.print(f'[yellow]Output folder for {bin_file.name} already exists - skipping')
            continue

        bin_output_folder.mkdir()

        with open(bin_output_folder/'log.txt', 'w') as f:
            p = subprocess.run(f'./TYGR datagen {bin_file} {bin_output_folder} --eval {tygr_model}',
                                stdout=f, stderr=subprocess.STDOUT,
                                shell=True)
            if p.returncode != 0:
                console.print(f'[bold red]TYGR failed for binary {bin_file.name} with error code {p.returncode}')

        csv_file = list(bin_output_folder.glob('*.csv'))
        if csv_file:
            output_csvs.append(csv_file)
        else:
            console.print(f'[yellow]No CSV file found for binary {bin_file.name}')

    # combined_df = pd.concat([pd.read_csv(x) for x in output_csvs])
    # combined_df.to_csv(output_folder/'combined_preds.csv', index=False)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('benchmark_binaries', help='Path to folder containing binaries to evaluate')
    p.add_argument('output_folder', help='Output folder')
    p.add_argument('tygr_model', help='TYGR model file to use')
    args = p.parse_args()
    exit(main(args))
