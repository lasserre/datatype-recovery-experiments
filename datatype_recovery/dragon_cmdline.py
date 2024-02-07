import argparse
from pathlib import Path
import torch

from .models.model_repo import get_registered_models

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
    # create_p.add_argument('exp_folder', type=Path, default=None, help='The experiment folder', nargs='?')
    # create_p.add_argument('-l', '--project-list', type=str, help='The name of the project list to use for this experiment')
    # create_p.add_argument('-r', '--recipe', type=str, help='The name of the recipe to use for this experiment (overrides -l)')

    # --- build: build datasets from wildebeest runs
    build_p = subparsers.add_parser('build', help='Build a DRAGON dataset from wildebeest runs')
    #   --> drop_component or no
    #   --> inmem or no
    #   --> --convert: convert existing to inmem

    # --- train: train an existing DRAGON model
    train_p = subparsers.add_parser('train', help='Train a DRAGON model located at the given path')
    train_p.add_argument('model_path', type=str, help='The path to the model file to be trained')
    #   --> models are referred to by .pt filename
    #   --> run_name is a req'd arg?
    #   --> SAVE PARAMS TO DF/CSV (need to be able to go back and look at a folder)
    #   --> DEFINITELY create a training_run folder (model does not have to be here, just stats/params/output)

    # --- eval: evaluate a DRAGON model
    eval_p = subparsers.add_parser('eval', help='Eval a DRAGON model on the given dataset')
    #   --> point to dataset to eval on
    #   --> produce stats DF (we can plot offline or now)

    # TODO: add ls and info commands as appropriate to show dataset/model contents, print model summary etc

    # --- ls: list things
    ls_p = subparsers.add_parser('ls', help='List information about requested content')
    ls_p.add_argument('object', help='The object to list', choices=['models'])

    args = p.parse_args()

    # --- dragon create
    if args.subcmd == 'create':
        return cmd_create(args)
    # --- dragon ls
    elif args.subcmd == 'ls':
        if args.object == 'models':
            return cmd_ls_models(args)
        raise Exception(f'Unhandled ls type {args.subcmd}')

if __name__ == '__main__':
    exit(main())
