
__all__ = ['set_seed_all', 'set_cudnn_flags', 'get_device', 'get_dataloader_xy', 'generate_rectified_normal_data',
           'vectorise', 'un_vectorise', 'get_param_types', 'get_updated_paramgraddict']

import torch
import numpy as np
import random
import types
import torch.nn.functional as F


def set_seed_all(seed):
    """
    Sets all random states
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def set_cudnn_flags():
    """Set CuDNN flags for reproducibility"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Returns torch.device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def get_dataloader_xy(dataloader):
    """
    Returns the x, y tensors of the dataloders.

    If SubsetRandomSampler then gets the correct indices
    """
    if isinstance(dataloader.batch_sampler.sampler, torch.utils.data.sampler.SubsetRandomSampler):
        inds = dataloader.batch_sampler.sampler.indices
        x = dataloader.dataset.x[inds]
        y = dataloader.dataset.y[inds]
    elif isinstance(dataloader.batch_sampler.sampler,torch.utils.data.sampler.SequentialSampler):
        x = dataloader.dataset.x
        y = dataloader.dataset.y
    elif isinstance(dataloader.batch_sampler.sampler,torch.utils.data.sampler.RandomSampler):
        x = dataloader.dataset.x
        y = dataloader.dataset.y
    return x, y

def generate_rectified_normal_data(n_samples=1e6, input_size=1000, h0_var=1,
                                   z0_var=None):
    """
    Generates input data matrices z0, h0=relu(z0) of shape (n_samples, input_size).
    z0 is ~ N(0, z0_var) or \sqrt(h0_var) * \sqrt(2*\pi / (\pi -1))

    Args:
        n_samples : n columns
        input_size : n rows
        h0_var : variance of h0
        z0_var : Optional,variance of z0, if provided h0_var is ignored.
        seed : random

    Returns:
        - h0 = max(0, z0)
        - z0 are samples from N(0, sigma_z0)
        - sigma_z0

    for $Var[h_i^\ell] = 1$, $\sigma_{z_{\ell}}^2 \leftarrow \frac{2\pi}{ \pi -1}$
    """
    n_samples = int(n_samples)
    #np.random.seed(seed)
    mu = 0
    if z0_var is None and h0_var is not None:
        sigma_z0 =  np.sqrt(h0_var) * np.sqrt(2*np.pi / (np.pi -1))
    elif z0_var is not None:
        sigma_z0 = np.sqrt(z0_var)
        if h0_var is not None:print('WARNING: IGNORED h0 var')
    else: raise

    z0 = np.random.normal(loc=0, scale=sigma_z0,
                          size=(n_samples, input_size))
    h0 = np.clip(z0, 0, np.inf)
    return z0, h0, sigma_z0


@torch.no_grad()
def vectorise(named_parameters, batched=False):
    """
    Vectorises a dict/ generator of key:tensors. Based on vectorize func from FisherCalculator

    While nn.Module.named_parameters() iterates over an Orderdered dict, and therefore keys should be ordered,
    but we convert to dict and sort the keys anyway, so will work for any (un-ordered) dict.

    Args:
        named_parameters : a dictionary or generator of {name : values}. e.g nn.Module.named_parameters
        batched : the values supplied for each parameter are considered batched in the first dimension

    Returns:
        column vector, or 2d array if batched
    """
    if type(named_parameters) is types.GeneratorType:
        named_parameters = dict(named_parameters)

    to_cat = []
    for key in sorted(named_parameters.keys()):
        value = named_parameters[key]
        assert(len(value.shape) >= 1)
        #print(key, value.device)
        if not batched and len(value.shape)<=2:
            new_shape = [np.prod(value.shape)]

        elif batched and len(value.shape)<=3:
            new_shape = [-1, np.prod(value.shape[1:])]

        else:
            print("vectorise func is not tested for the dimensions supplied")
            raise

        to_cat.append(value.view(*new_shape))

    # concat on the last dimension, then unsqueeze for e.g col vector.

    return torch.cat(to_cat,dim=-1).unsqueeze(dim=-1)

@torch.no_grad()
def un_vectorise(tensor, named_parameters, batched=False):
    """
    Assumes tensor has been vectorised from a named_parameter-like dict.
    Uses the keys of this dict to deterimine the
    Args:
        tensor: tensor to un-vectorise
        named_parameters : the original dictionary or generator that was vectorised
    """
    if type(named_parameters) is types.GeneratorType:
        named_parameters = dict(named_parameters)

    param_keys = sorted(named_parameters.keys())
    param_dict = {}
    param_sizes = []    # in the concat dimension
    orginal_shapes = [] # to use when reshaping the split tensor

    for key in param_keys:
        orginal_shape = named_parameters[key].shape
        if batched:
            size = (-1,np.prod(orginal_shape[1:]))
        else:
            size = (np.prod(orginal_shape))

        orginal_shapes.append(orginal_shape)
        param_sizes.append(size) #

    param_sizes = [int(s) for s in param_sizes]
    # have to squeeze tensor on last dim, as vectorise adds that after concating
    split_tensor = torch.split(tensor.squeeze(), param_sizes, dim=-1)

    for i, key in enumerate(param_keys):
        param_dict[key] = split_tensor[i].view(orginal_shapes[i])

    return param_dict



def get_param_types(model):
    """
    returns a list of param "types". I.e Wei, Wex, Wix, b, g, alpha
    """
    param_types = []
    for k, p in model.named_parameters():
        p_type = k.split('.')[-1]
        if p_type not in param_types:
            param_types.append(p_type)
    return param_types

def get_updated_paramgraddict(model, layerdict_paramgraddict=None, norm=False):
    """
    Returns the parameter flattened values, per layer, and per parameter in a dict of lists (norm=True means the value is
    the l2 norm over this param_type, otherwise it is the mean over this param_type).
    If no layerdict_paramgraddict is given, one is created. Does not work for single layer module
    that is not a DenseNet or FisherDenseNet currently
    """

    if hasattr(model,'layers'):
        layers=model.layers
    else:
        # In this case model is just one layer
        layers={'layer': model}
    paramkeys=get_param_types(model)

    if not layerdict_paramgraddict: # checks if None or empty list or dict
        _layerdict_paramgraddict={}
    else:
        _layerdict_paramgraddict=dict(layerdict_paramgraddict)
    for name,layer in layers.items():
        for k in paramkeys:
            paramgrad=getattr(getattr(layer,k),'grad').flatten()
            if norm:
                paramgrad=torch.norm(paramgrad)
            else:
                paramgrad=paramgrad.mean()
            if (name in _layerdict_paramgraddict.keys()):
                if (k in _layerdict_paramgraddict[name].keys()):
                    _layerdict_paramgraddict[name][k].append(paramgrad)
                else:
                    _layerdict_paramgraddict[name][k]=[paramgrad]
            else:
                _layerdict_paramgraddict[name]={k:[paramgrad]}

    return _layerdict_paramgraddict
