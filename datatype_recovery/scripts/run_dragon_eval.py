#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess

def main(args):
    cmd = f'eval_simple_types {args.eval_folder} ' \
            f'--ghidra_repo benchmark_{args.benchmark_name} ' \
            f'--dragon {args.model_folder} ' \
            '--strategy conf ' \
            '--confidence 0.75 ' \
            '--resume --rollback-delete'
            # f'--dragon-ryder {args.model_folder} ' \

    p = subprocess.run(cmd, shell=True)
    if p.returncode != 0:
        print(f'eval exited with returncode {p.returncode}')

    return p.returncode

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Run the DRAGON eval on one of the benchmarks')
    p.add_argument('eval_folder', type=Path, help='Path to output folder')
    p.add_argument('benchmark_name', help='Name of the benchmark to run (e.g. nginx, coreutils, etc)')
    p.add_argument('model_folder', type=Path, help='Path to DRAGON model (or a folder of DRAGON models) to evaluate')
    args = p.parse_args()
    exit(main(args))
