#!/usr/bin/env python3
################################################################
# Taken from the TYGR project with minor modification to fix
# bug in get_proj() function
# https://github.com/sefcom/TYGR/tree/master
#
# this script works in my python 3 environment with:
#
# pip install angr==9.0.7491 protobuf==3.17.1
#
# (per the environment.yml in TYGR)
################################################################

import argparse
import json
import hashlib
import multiprocessing
import signal
import os
import logging

from pathlib import Path

import angr
from tqdm import tqdm

OPTIMIZATIONS = ['O0', 'O1', 'O2', 'O3']
BLACKLIST = []

logging.disable(logging.CRITICAL)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help='Input directory containing the optimization folders')
    parser.add_argument("--results", help="Results directory", default="results")
    parser.add_argument("-j", type=int, help="Cores to use", default=1)
    return parser.parse_args()


REG = 1
IMM = 2
MEM = 3

def is_addr(proj: angr.Project, addr: int) -> bool:
    return any(x.min_addr <= addr <= x.max_addr for x in proj.loader.all_objects)


def get_function_hash(proj: angr.Project, func: angr.knowledge_plugins.Function) -> str:
    blocks = sorted(func.blocks, key=lambda x: x.addr)
    all_insns = [y for x in blocks for y in x.disassembly.insns]
    out_insns = []
    for idx, insn in enumerate(all_insns):
        insn_str = "\t".join(str(insn).split("\t")[1:])
        for operand in insn.operands:
            # THIS ONLY WORKS FOR X86/64
            if operand.type == IMM:
                if is_addr(proj, operand.imm):
                    insn_str = insn_str.replace(hex(operand.imm), "ADDR")
            elif operand.type == MEM:
                base = insn.reg_name(operand.mem.base)
                if base == proj.arch.register_names[proj.arch.ip_offset]: # RIP-relative
                    start = insn_str.find(base)
                    insn_str = insn_str[:start] + "RIP RELATIVE"
        out_insns.append(insn_str)

    disassembly = '\n'.join(out_insns)
    h = hashlib.new("sha256")
    h.update(disassembly.encode())
    return h.hexdigest(), disassembly


def get_function_hashes(proj: angr.Project, res_dir: Path, opt_level: str):
    out_file = res_dir / f"{opt_level}_func_hashes.json"
    if not out_file.exists():
        hashes = {}
    else:
        hashes = json.loads(out_file.read_text())
        if any(x['bin_name'] == proj.filename for x in hashes.values()):
            return
    for function in proj.kb.functions.values():
        if function.is_simprocedure or function.is_plt:
            continue

        if function.name in BLACKLIST:
            continue

        function_hash, disassembly = get_function_hash(proj, function)
        hashes[function_hash] = {
            'len': len(disassembly),
            'name': function.name,
            'bin_name': [proj.filename]
        }

    with out_file.open('w') as f:
        json.dump(hashes, f, indent=4)


def get_proj(bin_file: Path) -> angr.Project:
    try:
        proj = angr.Project(str(bin_file.absolute()), load_options={'auto_load_libs': False})
        proj.analyses.CFGFast(show_progressbar=True)
    except:
        proj = None
    return proj

def handler(signum, frame):
    print("Timout occured")
    raise TimeoutError

def get_handler(bin_file:Path):
    def handler(signum, frame):
        print(f"Timeout occured for binary {bin_file.name}")
        raise TimeoutError
    return handler

class ReturnCode:
    SUCCESS = 0
    TIMEOUT = 1
    FILE_EXISTS = 2
    OTHER = 3

def get_duplicates_for_file(bin_file: Path) -> int:

    TIMEOUT_MIN = 7    # longer timeout to try and get everything included for tydamin

    signal.signal(signal.SIGALRM, get_handler(bin_file))
    res_dir = RESULTS_DIR / bin_file.name
    res_dir.mkdir(parents=True, exist_ok=True)

    parts = bin_file.parts
    opt = parts[-2]
    out_file = res_dir / f"{opt}_func_hashes.json"
    if out_file.exists():
        return ReturnCode.FILE_EXISTS

    try:
        signal.alarm(60*TIMEOUT_MIN)
        proj = get_proj(bin_file)
        if proj is None:
            raise AttributeError
        get_function_hashes(proj, res_dir, opt)
        return ReturnCode.SUCCESS
    except (angr.errors.AngrCFGError, AttributeError, TimeoutError) as e:
        signal.alarm(0)     # cancel any pending alarms (https://manpages.debian.org/bookworm/manpages-dev/alarm.2.en.html)

        rcode = ReturnCode.OTHER
        if isinstance(e, TimeoutError):
            rcode = ReturnCode.TIMEOUT

        if not out_file.exists():
            with out_file.open("w") as f:
                f.write("{}")
        with (res_dir / "failed.txt").open("a") as f:
            f.write(f"Failed to analyze {bin_file}\n")

        return rcode

if __name__ == '__main__':
    args = get_args()
    RESULTS_DIR = Path(args.results)

    if not RESULTS_DIR.exists():
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_files = []
    for path, _, files in os.walk(args.dir):
        for name in files:
            file = Path(path) / name
            if file.is_file():
                all_files.append(file)
    count = len(all_files)
    with multiprocessing.Pool(args.j) as pool:
        # original TYGR code:
        # -------------------
        # pool.map(get_duplicates_for_file, all_files)
        # for _ in tqdm(pool.imap_unordered(get_duplicates_for_file, all_files), "Analyzing Files...", total=len(all_files)):
        #     count -= 1
            # print(f"{count} files remaining...")

        rcodes = [rc for rc in tqdm(pool.map(get_duplicates_for_file, all_files), 'Analyzing Files...', total=len(all_files))]

        num_success = len([x for x in rcodes if x == ReturnCode.SUCCESS])
        num_fexists = len([x for x in rcodes if x == ReturnCode.FILE_EXISTS])
        num_timeout = len([x for x in rcodes if x == ReturnCode.TIMEOUT])
        num_other = len([x for x in rcodes if x == ReturnCode.OTHER])
        total = len(rcodes)

        from rich.console import Console

        console = Console()

        console.print(f'[green]{num_success:,} binaries succeeded ({num_success/total*100:.2f}%)')
        console.print(f'[orange1]{num_timeout:,} binaries timed out ({num_timeout/total*100:.2f}%)')
        console.print(f'[red]{num_fexists:,} binaries skipped since outfile already exists ({num_fexists/total*100:.2f}%)')
        console.print(f'[yellow]{num_other:,} binaries failed for some other reason ({num_other/total*100:.2f}%)')

    exit(0)
