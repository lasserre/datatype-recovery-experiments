import argparse
import argcomplete
from pathlib import Path
import torch

from datatype_recovery.models.model_repo import get_registered_models
from datatype_recovery.models.dataset import TypeSequenceDataset, InMemTypeSequenceDataset
from datatype_recovery.models.training import train_model

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
    params = {
        'experiment_runs': args.exp_runfolders,
        'copy_data': bool(args.copy_data),
        'drop_component': bool(args.drop_comp),
        'max_hops': args.max_hops,
        'node_typeseq_len': args.node_typeseq_len,
        'structural_only': bool(args.structural)
    }

    ds = TypeSequenceDataset(args.dataset_folder, params)

    if args.inmem:
        print('Converting to in-memory dataset')
        inmem = InMemTypeSequenceDataset(ds)

def cmd_train(args):
    train_model(Path(args.model_path), Path(args.dataset_path), args.run_name, args.train_split, args.batch_size,
                args.num_epochs, args.lr, args.data_limit, args.cuda_dev, args.seed)

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
    build_p.add_argument('--drop-comp', action='store_true', help='Do not include COMP entries in this dataset')
    build_p.add_argument('--inmem', action='store_true', help='Use an in-memory datast')
    build_p.add_argument('--copy-data', action='store_true', help='Copy data to local dataset folder')
    build_p.add_argument('--structural', action='store_true', help='Generate node features for structural-only model')
    build_p.add_argument('--node_typeseq_len', type=int, help='Type sequence length for node data type features', default=3)
    #   --> --convert: convert existing to inmem

    # --- train: train an existing DRAGON model
    train_p = subparsers.add_parser('train', help='Train a DRAGON model located at the given path')
    train_p.add_argument('model_path', type=str, help='The path to the model file to be trained')
    train_p.add_argument('dataset_path', type=str, help='Path to the training dataset folder')
    train_p.add_argument('run_name', type=str, help='Run name for wandb')
    train_p.add_argument('--num_epochs', type=int, help='Number of epochs to run', default=500)
    train_p.add_argument('--train_split', type=float, help='Percent of dataset to use for the train set (remainder is test set)', default=0.7)
    train_p.add_argument('--batch_size', type=int, help='Number of epochs to run', default=64)
    train_p.add_argument('--lr', type=float, help='Learning rate for training', default=0.001)
    train_p.add_argument('--data_limit', type=int, help='Limit training data to first N samples', default=None)
    train_p.add_argument('--cuda-dev', type=int, help='CUDA device index (0 to N-1)', default=0)
    train_p.add_argument('--seed', type=int, help='Random seed', default=33)
    #   --> DEFINITELY create a training_run folder (model does not have to be here, just stats/params/output)

    # --- eval: evaluate a DRAGON model
    eval_p = subparsers.add_parser('eval', help='Eval a DRAGON model on the given dataset')
    #   --> point to dataset to eval on
    #   --> produce stats DF (we can plot offline or now)

    # TODO: info: compute/plot the dataset BALANCE

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
    # --- dragon ls
    elif args.subcmd == 'ls':
        if args.object == 'models':
            return cmd_ls_models(args)
        raise Exception(f'Unhandled ls type {args.subcmd}')

if __name__ == '__main__':
    exit(main())
