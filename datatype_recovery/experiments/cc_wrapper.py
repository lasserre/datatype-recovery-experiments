import os
import subprocess
import sys

from wildebeest.utils import env

def main():
    # sys.argv
    # with env({'DOCKER_BUILDKIT': '1'}):

    c_compiler = os.environ['WDB_CC'] if 'WDB_CC' in os.environ else ''
    cxx_compiler = os.environ['WDB_CXX'] if 'WDB_CXX' in os.environ else ''

    print(sys.argv)
    print(sys.argv[0])
    # subprocess.run([''])