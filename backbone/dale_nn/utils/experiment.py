
__all__ = ['listdir', 'bool_arg', 'tuple_arg', 'list_arg', 'FLAGS']

import sys
import os
import argparse
import random
from pathlib import Path
import yaml

import numpy as np
import torch

def listdir(dirpath, fullpath=True):
    dirpath = Path(dirpath)
    if fullpath:
        return [dirpath/p for p in os.listdir(dirpath)]
    else:
        return os.listdir(dirpath)

# from https://github.com/fastai/fastscript/blob/master/00_core.ipynb
def bool_arg(v):
    "Use as `type` for `Param` to get `bool` behavior"
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

def tuple_arg(v):
    print("tuple_arg function:",v, type(v))
    if isinstance(v, tuple): return v
    else:return eval(v)


def list_arg(v):
    print("list_arg function",v, type(v))
    if isinstance(v, list): return v
    else:return eval(v)


class FLAGS:
    '''
    Class to hold experiment parameters. Attributes can be set directly from a config yaml file or from command line
    if defaults are defined in the script (see example use below).

    For an example template config.yaml define defaults (as below) and call write_config_file_to_save_path().

    Example use:
        In a script or notebook that is be exported to a script
    ```
    flags = FLAGS()
    # define default simuluation params. If not defined, setting from command line args will throw an error
    flags.global_seed = 1
    flags.lr = 0.01
    flags.batch_size = 64
    flags.use_wandb = False
    flags.selected_params = ['batch_size', 'lr'] # attrs used to build save_path (as well as global_seed)
    # etc.

    flags.update_from_command_line_args_if_applicable()

    if not flags.run_exists(): # note: will return False if flags.overwrite_previous_run = True
        flags.save_path.mkdir(parents=True, exist_ok=True)
    else: sys.exit()

    flags.write_config_file_to_save_path('exp_config.txt')

    ```

    '''
    def __init__(self):
        """Default attributes"""
        self.global_seed = 7
        self.results_save_dir = "./test/" # dir where the experiment is saved

        # This list dictates which params are included in savepath and selected_params_dict
        self.selected_params = []

    @property
    def save_path(self):
        """
        Builds a save path from selected_params and the seed. The format
        should be selected_params/seed1/, selected_params/seed2/ etc.

        In future, we might need to hash the selected_params if the folder
        path gets too long.

        Returns a pathlib.Path object, so has functionality like:
            flags.save_path.mkdir(parents=True, exist_ok=True)
        """
        s = ''
        for key in self.selected_params: # don't use selected_param_dict, as we might (cosmetically) care about order
            s += f'{key}-{self.param_dict[key]}_'

        head = Path(self.results_save_dir)
        middle = Path(s[:-1])
        assert len(str(middle)) < 255 # max folder_name length
        # will need to hash or something if it is too long
        tail = Path(f'global_seed-{self.global_seed}')
        return head/middle/tail

    def run_exists(self, results_file=None, config_file=None, check_full_param_dict=True):
        """
        Returns True if an experiment with same params has already been run. False if not.

        The self.save_path should be composed of the "relevant" params so it is the minimum requirement,
        but by passing a config_file and setting "check_full_param_dict" to true, the method will check
        against a full config file fields. This step is optional as new (irrelevant to old exp) fields could
        be added to the default flag attrs as new experimental conditions are run.

        Finally check for presence of results_file. Note at the moment results_file only get saved after the
        full experiment has been run, but this may change and will need to check correct number of epochs have been run.

        Args:
            check_full_param_dict: bool. Whether to check run exists based on comparison with full parameter dict
            config_file  : the savename of the flags parameters in the folder. If None, doesn't check
            results_file : path to results file that should exist. If None, doesn't check

        """
        # first just check if the save_path exists
        if not self.save_path.exists():
            return False

        # second make a more complete check on the full params dict
        if check_full_param_dict and config_file is not None:
            with open(self.save_path/config_file, 'r') as config_filestream:
                config_dict = yaml.load(config_filestream, Loader=yaml.FullLoader)
                config_dump = yaml.dump(config_dict, default_flow_style=False)
                self_dump  = yaml.dump(self.param_dict, default_flow_style=False)
            if self_dump != config_dump:
                print('INFO: Experiment run does not match full param settings')
                return False

        # third, check results exists
        if results_file is not None:
            results_filepath = self.save_path/results_file
            if not results_filepath.exists():
                print('INFO: Savepath exists, but no results file. Re-running')
                return False

        # if the other conditions haven't returned False, then run exists
        print("INFO: Experiment run already exists")
        return True

    @property
    def param_dict(self):
        """This is dict of all params"""
        param_dict = {}
        for k in self.__dict__.keys():
            if k.startswith('__'):
                continue
            elif type(self.__dict__[k]) == classmethod:
                continue # We don't want methods like this one
            else:
                param_dict[k] = self.__dict__[k]
        return param_dict

    @property
    def selected_param_dict(self):
        selected_param_dict = {}
        for k in self.param_dict.keys():
            if k in self.selected_params:
                selected_param_dict[k] = self.__dict__[k]
        return selected_param_dict

    def update_from_command_line_args(self, verbose=True):
        """
        Will throw an error if command line flag does not already exist as an attribute.

        If --config=yaml_file has been supplied, loads the yaml file and sets attributes
        """
        # check if a config flag pointing to yaml file has been supplied, e.g. from orion
        orion_conf_dict = None
        for i, arg_str in enumerate(sys.argv):
            if arg_str.startswith('--config'):
                print("FLAGS obj detected a config.yaml file:", sys.argv[i+1])
                with open(sys.argv[i+1], 'r') as config_filestream:
                    orion_conf_dict = yaml.load(config_filestream, Loader=yaml.FullLoader)
                    print("Loaded", orion_conf_dict)

        # else assume normal command line args (will throw error if defaults don't exists)
        if orion_conf_dict is None:
            parser = self.build_argparser()
            self.set_params_from_argparse(parser.parse_args(), verbose=verbose)
        else:
            for key, item in orion_conf_dict.items():
                self.__dict__[key] = item

    def build_argparser(self) -> argparse.ArgumentParser:
        """
        Constructs an argument parser from self.param_dict

        Returns an instance of argparse.ArgumentParser
        """
        parser = argparse.ArgumentParser()
        for key, item in self.param_dict.items():
            arg_type = type(item)
            if arg_type == bool:
                arg_type = bool_arg
            if arg_type == tuple:
                arg_type = tuple_arg
            if arg_type == list:
                arg_type = list_arg
            parser.add_argument(f'--{key}', default=item, type=arg_type)
        return parser

    @staticmethod
    def get_sysargv_dict():
        """Parses sys.argv elemnets beginning with -- into a dict """
        d = {}
        for arg_str in sys.argv:
            if arg_str.startswith('--'):
                key, val = arg_str[2:].split('=')
                d[key] = val
        return d

    def set_params_from_argparse(self, args:argparse.Namespace, verbose=True):
        """
        Sets attributes from an argparse.Namespace object

        vars() - returns the __dict__ attribute for a module, class, instance,
                 or any other object with a __dict__ attribute.

        Will throw an error if command line flag does not already exist as an attribute
        """
        if verbose:
            print(f'Set flags.script_name to {sys.argv[0]}')
            unchanged_keys = self.param_dict.keys() - self.get_sysargv_dict().keys()
            if len(unchanged_keys) > 0:
                unchanged_params = {k:self.param_dict[k] for k in unchanged_keys}
                print('INFO: The following settings were not changed via command line')
                for k,val in unchanged_params.items(): print('\t', k,':', val)

        self.script_name = sys.argv[0]
        for key, item in vars(args).items():
            self.__dict__[key] = item

    def write_config_file_to_save_path(self, fname="exp_config.yaml"):
        """
        Writes a yaml config file (fname) to self.save_path

        """
        config_filepath = self.save_path/fname
        with open(config_filepath, 'w') as file:
            yaml.dump(self.param_dict, stream=file, default_flow_style=False)
        print(f'Wrote config file to {config_filepath}')

    def __repr__(self):
        s = ''
        s += 'Flags parameters: \n'
        for key in sorted(self.param_dict.keys(), key=str.casefold):
            if key == 'selected_params':continue
            s += f'{key} : {self.param_dict[key]} \n'
        return s