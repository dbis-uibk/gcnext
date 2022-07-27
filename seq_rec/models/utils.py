import importlib
import os.path

import torch
import yaml


def get_custom_model(model_name):
    r"""Automatically select model class based on model name

    Based on recbole.utils.get_model. Extended by custom models.

    Args:
        model_name (str): model name

    Returns:
        Recommender: model class
    """
    model_submodule = [
        'general_recommender', 'sequential_recommender'
    ]

    model_file_name = model_name.lower()
    model_module = None
    for submodule in model_submodule:
        module_path = '.'.join(['recbole.model', submodule, model_file_name])
        if importlib.util.find_spec(module_path, __name__):
            model_module = importlib.import_module(module_path, __name__)
            break
    if model_module is None:
        module_path = '.'.join(['models', model_file_name])
        if importlib.util.find_spec(module_path, __name__):
            model_module = importlib.import_module(module_path, __name__)

    if model_module is None:
        raise ValueError('`model_name` [{}] is not the name of an existing model.'.format(model_name))
    model_class = getattr(model_module, model_name)
    return model_class


def get_config_from_file(filepath):
    with open(filepath)as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config
