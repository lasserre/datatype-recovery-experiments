import os
from pathlib import Path
import subprocess
import sys
from typing import List

from wildebeest.utils import env

_opt_arg_list = [
    '-O',
    '-O0',
    '-O1',
    '-O2',
    '-O3',
    '-Os',
    '-Ofast',
    '-Oz',
    '-Og',
]

def optflag_in_string(s:str) -> bool:
    global _opt_arg_list
    return any([x for x in _opt_arg_list if x in s])

def filter_optimization_args(argv:List[str]) -> List[str]:
    return [x for x in argv if not optflag_in_string(x)]

def main():
    '''
    From GCC docs: https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html

    "If you use multiple -O options, with or without level numbers, the last such option is the one that is effective."

    Instead of trying to ensure we are LAST, we just eat everything and replace with the one we want :)
    '''
    is_cxx = 'cxx' in sys.argv[0]
    opt_level = os.environ['OPT_LEVEL'] if 'OPT_LEVEL' in os.environ else '-O0'
    FLAGS_VAR = 'CXXFLAGS' if is_cxx else 'CFLAGS'
    # print(f'Using optimization level {opt_level}')

    cc_path_filename = 'cxx_path.txt' if is_cxx else 'cc_path.txt'

    with open(Path.home()/cc_path_filename, 'r') as f:
        compiler = f.readlines()[0].strip()

    print(f'Using compiler at: {compiler}')

    filtered_flags = ''
    if FLAGS_VAR in os.environ:
        print(f'Original {FLAGS_VAR}: {os.environ[FLAGS_VAR]}', file=sys.stderr)
        filtered_flags = ' '.join(filter_optimization_args(os.environ[FLAGS_VAR].split()))
        print(f'Filtered {FLAGS_VAR}: {filtered_flags}', file=sys.stderr)

    compiler_args = filter_optimization_args(sys.argv[1:])
    compiler_args.append(opt_level)

    if any(['@' in x for x in [*compiler_args, *filtered_flags]]):
        raise Exception(f'Found @ arguments: {[*compiler_args, "CFLAGS...", *filtered_flags]}')

    envdict = {FLAGS_VAR: filtered_flags} if filtered_flags else {}

    with env(envdict):
        return subprocess.run([compiler, *compiler_args], shell=True).returncode
