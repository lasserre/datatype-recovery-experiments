import argparse
import argcomplete
import pandas as pd
from pathlib import Path
import torch

from datatype_recovery.models.model_repo import get_registered_models
from datatype_recovery.models.dataset import TypeSequenceDataset, InMemTypeSequenceDataset, SimpleTypeDataset
from datatype_recovery.models.training import train_model
from datatype_recovery.models.dataset import load_dataset_from_path
from datatype_recovery.models.dataset.balance import plot_dataset_balance

def cmd_create(args):
    registered_models = get_registered_models()
    if args.model_type not in registered_models:
        print(f'Model type {args.model_type} not recognized')
        return 1

    create_model = registered_models[args.model_type]
    model_kwargs = {k: v for k,  v in [p.split('=') for p in args.params]} if args.params else {}
    try:
        new_model = create_model(**model_kwargs)
    except Exception as e:
        print(e)
        return 1
    print(new_model)
    torch.save(new_model, args.model_path)
    print(f'Saved model to {args.model_path}')
    return 0

def cmd_build(args):
    if args.from_file:
        with open(Path(args.exp_runfolders[0]), 'r') as f:
            exp_runs = [l.strip() for l in f.readlines() if l]
    else:
        exp_runs = args.exp_runfolders

    if args.from_exps:
        # exp_runs currently holds an experiment list - collect all run folders from this
        exp_list = exp_runs
        exp_runs = []
        excludes = {x.split(':')[0]: x.split(':')[1] for x in args.exclude} if args.exclude else {}

        for exp in exp_list:
            rundata_folder = Path(exp)/'rundata'
            run_folders = [str(x.absolute()) for x in rundata_folder.iterdir() if x.is_dir() and x.name.startswith('run')]
            if Path(exp).name in excludes:
                exp_excludes = excludes[Path(exp).name]
                exp_runs.extend([r for r in run_folders if Path(r).name not in exp_excludes])
            else:
                exp_runs.extend(run_folders)

        print(f'Found {len(exp_runs):,} runs across {len(exp_list)} experiments')
        print(f'Experiments: {[Path(x).name for x in exp_list]}')

    params = {
        'experiment_runs': exp_runs,
        'max_hops': args.max_hops,
        'limit': args.limit,
    }

    if args.hetero:
        print('Building HeteroData simple type dataset')
        ds = SimpleTypeDataset(args.dataset_folder, params)
    else:
        params['copy_data'] = bool(args.copy_data)
        params['drop_component'] = bool(args.drop_comp)
        params['node_typeseq_len'] = args.node_typeseq_len
        params['structural_only'] = bool(args.structural)
        params['balance_dataset'] = bool(args.balance)
        params['keep_all'] = args.keep_all

        ds = TypeSequenceDataset(args.dataset_folder, params)

        if args.inmem:
            print('Converting to in-memory dataset')
            inmem = InMemTypeSequenceDataset(ds)

def cmd_train(args):
    train_model(Path(args.model_path), Path(args.dataset_path), args.name, args.train_split, args.batch_size,
                args.num_epochs, args.lr, args.data_limit, args.cuda_dev, args.seed, args.save_every)

def cmd_show_model(args):
    model = torch.load(args.model_path)
    print(model)

def cmd_show_ds(args):
    if args.plot:
        plot_dataset_balance(Path(args.dataset_folder))
    else:
        ds = load_dataset_from_path(Path(args.dataset_folder))
        ds.input_params['keep_all'] = args.keep_all

        vars_df = pd.read_csv(ds.unfiltered_variables_path)
        vars_df = ds._filter_vars_df(vars_df)
        bal_df = ds._balance_dataset(vars_df, raw=args.raw)
    return 0

def cmd_ls_models(args):
    for model_type in get_registered_models().keys():
        print(model_type)
    return 0

def main():

    p = argparse.ArgumentParser(description='Create, train, and eval DRAGON models and datasets')
    # p.add_argument('--exp', type=Path, default=Path().cwd(), help='The experiment folder')

    subparsers = p.add_subparsers(dest='subcmd')

    # --- create: creates a DRAGON model
    #   --> instantiate untrained model with the params, save to file we can train later
    create_p = subparsers.add_parser('create', help='Create a DRAGON model instance with specified params')
    create_p.add_argument('model_path', type=str, help='The path where the new model will be saved')
    create_p.add_argument('model_type', type=str, help='Type of model to create')
    create_p.add_argument('-p', '--params', nargs='+', help='Additional model-specific params to pass as-is to the model constructor')

    # --- build: build datasets from wildebeest runs
    build_p = subparsers.add_parser('build', help='Build a DRAGON dataset from wildebeest runs')
    build_p.add_argument('dataset_folder', type=str, help='Folder where the dataset should be stored')
    build_p.add_argument('max_hops', type=int, help='Number of hops to traverse when building variable reference graphs')
    build_p.add_argument('exp_runfolders', nargs='+', help='Run folders from which to pull the data for this dataset')
    build_p.add_argument('--from-exps', action='store_true', help='Interpret exp_runfolders as a list of experiments, each of which will have all runs included (this can be from a file or cmd line)')
    build_p.add_argument('--from-file', action='store_true', help='Read the exp_runfolders from each nonempty line of a file (path given instead of exp_runfolders)')
    build_p.add_argument('--exclude', nargs='+', help='Runs to exclude in the form <exp_foldername>:run<number>')
    build_p.add_argument('--drop-comp', action='store_true', help='Do not include COMP entries in this dataset')
    build_p.add_argument('--inmem', action='store_true', help='Use an in-memory datast')
    build_p.add_argument('--copy-data', action='store_true', help='Copy data to local dataset folder')
    build_p.add_argument('--structural', action='store_true', help='Generate node features for structural-only model')
    build_p.add_argument('--node_typeseq_len', type=int, help='Type sequence length for node data type features', default=3)
    build_p.add_argument('--balance', action='store_true', help='Balance the dataset (will greatly reduce in size also)')
    build_p.add_argument('--keep-all', type=str, help='Colon-separated list of leaf categories which must all be kept and will not influence the balance', default='')
    build_p.add_argument('--hetero', action='store_true', help='Build a HeteroData dataset (default is homogenous Data dataset)')
    build_p.add_argument('--limit', type=int, default=None, help='Hard limit on number of variables in dataset')
    #   --> --convert: convert existing to inmem

    # --- show_ds: Show the dataset balance
    showds_p = subparsers.add_parser('show_ds', help='Show information about DRAGON datasets')
    showds_p.add_argument('dataset_folder', type=str, help='Dataset folder')
    showds_p.add_argument('--raw', action='store_true', help="Show the raw dataset balance")
    showds_p.add_argument('--plot', action='store_true', help='Plot dataset balance using matplotlib')
    showds_p.add_argument('--keep-all', type=str, help='Colon-separated list of CSV PROJECTED type sequences which must all be kept and will not influence the balance', default='')

    # --- show_model: Show a DRAGON model
    showmodel_p = subparsers.add_parser('show_model', help='Show information about DRAGON models')
    showmodel_p.add_argument('model_path', type=str, help='Model .pt file')

    # --- rebalance: Rebalance an existing dataset
    # rebalance_p = subparsers.add_parser('rebalance', help='Rebalance an existing dataset')
    # show_p.add_argument('dataset_folder', type=str, help='Dataset folder')
    # show_p.add_argument('--raw', action='store_true', help="Show the raw dataset balance")
    # show_p.add_argument('--keep-all', type=str, help='Colon-separated list of CSV PROJECTED type sequences which must all be kept and will not influence the balance', default='')

    # --- train: train an existing DRAGON model
    train_p = subparsers.add_parser('train', help='Train a DRAGON model located at the given path')
    train_p.add_argument('model_path', type=str, help='The path to the model file to be trained')
    train_p.add_argument('dataset_path', type=str, help='Path to the training dataset folder')
    train_p.add_argument('--name', type=str, help='Run name for wandb', default='')
    train_p.add_argument('--num_epochs', type=int, help='Number of epochs to run', default=500)
    train_p.add_argument('--train_split', type=float, help='Percent of dataset to use for the train set (remainder is test set)', default=0.7)
    train_p.add_argument('--batch_size', type=int, help='Number of epochs to run', default=64)
    train_p.add_argument('--lr', type=float, help='Learning rate for training', default=0.001)
    train_p.add_argument('--data_limit', type=int, help='Limit training data to first N samples', default=None)
    train_p.add_argument('--cuda-dev', type=int, help='CUDA device index (0 to N-1)', default=0)
    train_p.add_argument('--seed', type=int, help='Random seed', default=33)
    train_p.add_argument('--save-every', type=int, help='Save model snapshot every N epochs', default=50)
    #   --> DEFINITELY create a training_run folder (model does not have to be here, just stats/params/output)

    # --- eval: evaluate a DRAGON model
    eval_p = subparsers.add_parser('eval', help='Eval a DRAGON model on the given dataset')
    #   --> point to dataset to eval on
    #   --> produce stats DF (we can plot offline or now)

    # TODO: add ls and info commands as appropriate to show dataset/model contents, print model summary etc

    # --- ls: list things
    ls_p = subparsers.add_parser('ls', help='List information about requested content')
    ls_p.add_argument('object', help='The object to list', choices=['models'])

    argcomplete.autocomplete(p)
    args = p.parse_args()

    # --- dragon create
    if args.subcmd == 'create':
        return cmd_create(args)
    # --- dragon build
    elif args.subcmd == 'build':
        return cmd_build(args)
    # --- dragon train
    elif args.subcmd == 'train':
        return cmd_train(args)
    # --- dragon show
    elif args.subcmd == 'show_model':
        return cmd_show_model(args)
    elif args.subcmd == 'show_ds':
        return cmd_show_ds(args)
    # --- dragon rebalance
    # elif args.subcmd == 'rebalance':
    #     return cmd_rebalance(args)
    # --- dragon ls
    elif args.subcmd == 'ls':
        if args.object == 'models':
            return cmd_ls_models(args)
        raise Exception(f'Unhandled ls type {args.subcmd}')

if __name__ == '__main__':
    exit(main())
