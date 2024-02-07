from typing import Callable, Dict

_model_repository = {}

def register_model(name:str, create_model:Callable):
    global _model_repository

    if name not in _model_repository:
        _model_repository[name] = create_model

def get_registered_models() -> Dict[str, Callable]:
    '''
    Returns a mapping of model name -> create_model callable accepting keyword args
    '''
    return _model_repository
