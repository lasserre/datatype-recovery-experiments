# DRAGON RYDER
# -------------
# DRAGON incremental RetYping DrivER

import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description='DRAGON incremental RetYping DrivER - recovers variable types using DRAGON and incremental retyping')
    p.add_argument('dataset', type=Path, help='The dataset on which to run dragon ryder (and recover basic variable types)')
    p.add_argument('dragon_model', type=Path, help='The trained DRAGON model to use')
    # TODO: add strategy selection? --strategy=truth|refs|confidence
    p.parse_args()
    # p.add_argument('--exp', type=Path, default=Path().cwd(), help='The experiment folder')

    print(f'Hello from DRAGON RYDER')

    # TODO: implement...

if __name__ == '__main__':
    exit(main())
