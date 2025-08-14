
import os
import yaml
import random
import importlib
from typing import Tuple
from collections import OrderedDict

import numpy as np

import torch
from torchvision.transforms import Compose


def seed_all(seed: int) -> None:
    """ Seeds python.random, torch and numpy,

    Args:
        seed (int): Desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def load_fr_model(config: dict) -> Tuple[torch.nn.Module, Compose]:
    """ Loads the model from the yaml config by calling 'module' and constructs the transforamtions

    Args:
        config (dict): FR config, containing 'module' aka the function to call, optionally 'params' with which to call the 'module' and 'transformations'

    Returns:
        Tuple[torch.nn.Module, Compose]: Returns the constructed model and given transformation.
    """

    fr_model = load_module(config)
    transform = construct_transformation(config["transformations"])

    return fr_model, transform


def parse_config_file(config_loc: str) -> dict:
    """ Constructs an argument class from the configuration located at <config_loc>.

    Args:
        config_loc (str): Location of the configuration file.

    Returns:
        Argument: Argument class 
    """
    
    assert os.path.exists(config_loc), f" Given config path ({config_loc}) does not exist!"
    config: dict = {}
    with open(config_loc, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    return config


def construct_transformation(transformation_arguments: dict) -> Compose:
    """ Constructs a composition of transformations given by the transformation arguments.

    Args:
        transformation_arguments (Arguments): Arguments of the transformation.

    Returns:
        Compose: Torchvision Compose object of given transformations.
    """

    transforms_list = []
    idx = 1
    while True:
        try:
            trans_args = transformation_arguments[f"trans_{idx}"]
            module_name, function_name = trans_args["module"].rsplit(".", 1)
            module = importlib.import_module(module_name)
            trans_function = getattr(module, function_name)
            if "params" in trans_args:
                transforms_list.append(trans_function(**(trans_args["params"])))
            else:
                transforms_list.append(trans_function())
        except KeyError:
            break
        idx += 1

    return Compose(transforms_list)


def load_module(module_args: dict, call=True):
    """ Loads module defined by given arguments: name given by using <module: 'name'> and parameters by using <params: ...>.

    Args:
        module_args (Arguments): Arguments from which to construct the desired module.
        call (Bool): Changes the return value, if call=True calls the function before returning results, otherwise not.

    Returns:
        func: Returns the returned values called by the desired module function call or the callable function if call=False.
    """
    module_name, function_name = module_args["module"].rsplit(".", 1)
    module = importlib.import_module(module_name)
    trans_function = getattr(module, function_name)
    if call:
        if hasattr(module_args, "params"):
            return trans_function(**vars(getattr(module_args, "params")))
        else:
            return trans_function()
    else:
        return trans_function


if __name__ == "__main__":

    ...