import argparse
from pathlib import Path

# NOTE: making this its own file so I can wait to import pyhidra/dragon_ryder
# after we parse cmd-line args (and not make that unneccessarily slow)

def main():
    p = argparse.ArgumentParser(description='DRAGON incremental RetYping DrivER - recovers variable types using DRAGON and incremental retyping')

    subparsers = p.add_subparsers(dest='subcmd')

    # -------------------------------------------------------------------------------
    # NOTE: updated version needs to NOT require a dataset (since we will extract the
    # INITIAL ASTS from Ghidra! we won't already have a dataset)
    # OPTIONS:
    # - Accept a Ghidra repo - process every binary within that repo
    # - Accept a Ghidra repo + binary (or list of binaries)
    # - Accept multiple Ghidra repos (low priority - this )
    #
    # what do we care about for TyGR comparison?
    # -> to use TyDA-min and/or my own builds, I will have to run their script to process
    #    each binary, then combine
    # -> I was going to add the ability to wdb (just for evaluating) to create an "experiment"
    #    from an arbitrary folder of binaries so I could control the specific set of
    #    binaries used in an evaluation
    #    NOTE: even if I want to use binaries I have built, this should be easy enough
    #          to either 1) point to a wdb experiment I already have or 2) create a
    #          new Ghidra repo for this particular eval
    #
    # ...point being - I think it should be FINE to just accept a SINGLE REPO
    # (either all binaries within it or a list/subset/single one)
    # -------------------------------------------------------------------------------
    # TODO: pick up here - accept a single Ghidra repo, run on every **NON-DEBUG BINARY**
    # within that repo
    # -> test with astera for now
    # -> later add a script that pulls in a folder of debug binaries (copy, strip, import both to ghidra, dragon-ryder)
    # (we will maintain the convention that .debug versions are for reference/truth)
    # -------------------------------------------------------------------------------

    # run: run dragon-ryder
    run_p = subparsers.add_parser('run', help='Run dragon-ryder')
    run_p.add_argument('dragon_model', type=Path, help='The trained DRAGON model to use')
    run_p.add_argument('ghidra_repo', type=str, help='Name of the Ghidra repository holding the non-debug binaries to retype')
    run_p.add_argument('-b', '--binaries', nargs='*', help='Specific binaries to evaluate (instead of all non-debug binaries in the repository)')
    run_p.add_argument('--host', type=str, default='localhost', help='Ghidra server address/hostname')
    run_p.add_argument('-p', '--port', type=int, default=13100, help='Ghidra server port')
    # run_p.add_argument('dataset', type=Path, help='The dataset on which to run dragon ryder (and recover basic variable types)')
    run_p.add_argument('--device', type=str, help='Pytorch device string on which to run the DRAGON model', default='cpu')
    run_p.add_argument('--resume', action='store_true', help='Continue after last completed step')
    run_p.add_argument('--nrefs', type=int, default=5, help='Number of references to use for high confidence variables')
    run_p.add_argument('--rollback-delete', action='store_true', help='Rollback any Ghidra programs with a version > 1 by deleting revisions')
    run_p.add_argument('--strategy', default='refs', nargs='?',
                        help='Strategy for identifying high confidence predictions',
                        choices=['truth', 'refs'])
    run_p.add_argument('-l', '--limit', type=int, default=-1, help='Max # funcs per binary (for testing)')

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
