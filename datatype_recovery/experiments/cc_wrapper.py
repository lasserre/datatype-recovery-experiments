import os
from pathlib import Path
import subprocess
import sys
from typing import List

from wildebeest.utils import env

_opt_arg_list = [
    # '-O', # this is also a valid linker argument, so let's not get in the way of that
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
    global _opt_arg_list
    return [x for x in argv if x not in _opt_arg_list]

def main():
    '''
    From GCC docs: https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html

    "If you use multiple -O options, with or without level numbers, the last such option is the one that is effective."

    Instead of trying to ensure we are LAST, we just eat everything and replace with the one we want :)
    '''
    # we ASSUME we are called via the symlink - our symlink name will match
    # the name of our target compiler
    symlink_path = Path(sys.arvg[0])
    opt_level = os.environ['OPT_LEVEL'] if 'OPT_LEVEL' in os.environ else '-O0'

    # handle cc vs cxx compiler
    with open(Path.home()/'cxx_path.txt', 'r') as f:
        cxx_compiler = Path(f.readlines()[0].strip())
    with open(Path.home()/'cc_path.txt', 'r') as f:
        c_compiler = Path(f.readlines()[0].strip())

    is_cxx = symlink_path.name == cxx_compiler.name
    compiler = cxx_compiler if is_cxx else c_compiler
    FLAGS_VAR = 'CXXFLAGS' if is_cxx else 'CFLAGS'

    filtered_flags = ''
    if FLAGS_VAR in os.environ:
        filtered_flags = ' '.join(filter_optimization_args(os.environ[FLAGS_VAR].split()))

    compiler_args = filter_optimization_args(sys.argv[1:])
    # put the desired optimization level at the front
    # (for C++, -lstdc++ MUST go last!! https://stackoverflow.com/a/6045967)
    compiler_args.insert(0, opt_level)

    # NOTE: this did not work because some paths have '@' symbol in it
    # -> if we want to prevent options getting past us via "gcc @file" then
    # we can come back and re-address, but I haven't actually seen this be an
    # issue yet
    # if any(['@' in x for x in [*compiler_args, *filtered_flags]]):
    #     raise Exception(f'Found @ arguments: {[*compiler_args, "CFLAGS...", *filtered_flags]}')

    envdict = {FLAGS_VAR: filtered_flags} if filtered_flags else {}

    # print(f'Using optimization level {opt_level}')
    # print(f'Using compiler at: {compiler}')
    # print(f'Original {FLAGS_VAR}: {os.environ[FLAGS_VAR]}', file=sys.stderr)
    # print(f'Filtered {FLAGS_VAR}: {filtered_flags}', file=sys.stderr)
    # print(f'Called with: {sys.argv}', file=sys.stderr)
    # print(f'Filtered to: {compiler_args}', file=sys.stderr)

    with env(envdict):
        print(f'sys.arv was: {" ".join(sys.argv)}', flush=True)
        print(f'sys.arv was: {" ".join(sys.argv)}', file=sys.stderr, flush=True)
        print(f'CALLING COMPILER: {" ".join([compiler, *compiler_args])}', flush=True)
        print(f'CALLING COMPILER: {" ".join([compiler, *compiler_args])}', file=sys.stderr, flush=True)
        return subprocess.run(' '.join([compiler, *compiler_args]), shell=True).returncode
