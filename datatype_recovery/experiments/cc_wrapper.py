import os
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

def filter_optimization_args(argv:List[str]) -> List[str]:
    global _opt_arg_list
    return [x for x in argv if x not in _opt_arg_list]

def main():
    '''
    From GCC docs: https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html

    "If you use multiple -O options, with or without level numbers, the last such option is the one that is effective."

    Instead of trying to ensure we are LAST, we just eat everything and replace with the one we want :)
    '''
    # sys.argv
    # with env({'DOCKER_BUILDKIT': '1'}):
    is_cxx = 'cxx' in sys.argv[0]

    opt_level = os.environ['OPT_LEVEL'] if 'OPT_LEVEL' in os.environ else '-O0'
    FLAGS_VAR = 'CXXFLAGS' if is_cxx else 'CFLAGS'
    # print(f'Using optimization level {opt_level}')

    if is_cxx:
        compiler = os.environ['WDB_CXX'] if 'WDB_CXX' in os.environ else 'g++'
    else:
        compiler = os.environ['WDB_CC'] if 'WDB_CC' in os.environ else 'gcc'

    filtered_flags = ''
    if FLAGS_VAR in os.environ:
        print(f'Original {FLAGS_VAR}: {os.environ[FLAGS_VAR]}', file=sys.stderr)
        filtered_flags = ' '.join(filter_optimization_args(os.environ[FLAGS_VAR].split()))
        print(f'Filtered {FLAGS_VAR}: {filtered_flags}', file=sys.stderr)

    compiler_args = filter_optimization_args(sys.argv[1:])
    compiler_args.append(opt_level)

    envdict = {FLAGS_VAR: filtered_flags} if filtered_flags else {}

    with env(envdict):
        return subprocess.run([compiler, *compiler_args], shell=True).returncode
    # print(' '.join([compiler, *compiler_args]))
