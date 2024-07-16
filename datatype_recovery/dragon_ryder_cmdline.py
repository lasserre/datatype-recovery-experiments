import argparse
from pathlib import Path

# NOTE: making this its own file so I can wait to import pyhidra/dragon_ryder
# after we parse cmd-line args (and not make that unneccessarily slow)

def add_ghidra_opts(parser:argparse.ArgumentParser):
    parser.add_argument('ghidra_repo', type=str, help='Name of the Ghidra repository holding the binaries to retype')
    parser.add_argument('--host', type=str, default='localhost', help='Ghidra server address/hostname')
    parser.add_argument('-p', '--port', type=int, default=13100, help='Ghidra server port')

def add_binary_opts(parser:argparse.ArgumentParser):
    parser.add_argument('-b', '--binaries', nargs='*',
                                help='Specific binaries to evaluate (instead of all non-debug binaries in the repository)')
    parser.add_argument('-l', '--limit', type=int, default=None, help='Max # funcs per binary (for testing)')

def add_model_opts(parser:argparse.ArgumentParser):
    parser.add_argument('--device', type=str, help='Pytorch device string on which to run the DRAGON model', default='cpu')

def add_dragon_ryder_opts(parser:argparse.ArgumentParser):
    parser.add_argument('--resume', action='store_true', help='Continue after last completed step')
    parser.add_argument('--nrefs', type=int, default=5, help='Number of references to use for high confidence variables')
    parser.add_argument('--rollback-delete', action='store_true', help='Rollback any Ghidra programs with a version > 1 by deleting revisions')
    parser.add_argument('--strategy', default='refs', nargs='?',
                        help='Strategy for identifying high confidence predictions',
                        choices=['truth', 'refs'])

def main():
    p = argparse.ArgumentParser(description='DRAGON incremental RetYping DrivER - recovers variable types using DRAGON and incremental retyping')

    subparsers = p.add_subparsers(dest='subcmd')

    # run: run dragon-ryder
    run_p = subparsers.add_parser('run', help='Run dragon-ryder')
    run_p.add_argument('dragon_model', type=Path, help='The trained DRAGON model to use')
    add_binary_opts(run_p)
    add_ghidra_opts(run_p)
    add_model_opts(run_p)
    # run_p.add_argument('dataset', type=Path, help='The dataset on which to run dragon ryder (and recover basic variable types)')
    add_dragon_ryder_opts(run_p)

    # status: show where we are
    status_p = subparsers.add_parser('status', help='Show the status of a dragon-ryder run')
    status_p.add_argument('folder', type=Path, help='The .dragon-ryder eval folder for which to report status')

    args = p.parse_args()

    from datatype_recovery.dragon_ryder import DragonRyder

    if args.subcmd == 'run':
        with DragonRyder(args.dragon_model,
                            args.ghidra_repo,
                            args.device,
                            args.resume,
                            args.nrefs,
                            args.rollback_delete,
                            args.host, args.port,
                            args.strategy,
                            args.binaries,
                            args.limit) as dragon_ryder:
            return dragon_ryder.run()
    elif args.subcmd == 'status':
        return DragonRyder.report_status(args.folder)

if __name__ == '__main__':
    exit(main())
