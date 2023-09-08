
__all__ = ['BaseDenseLayer', 'BaseWeightInitPolicy', 'BaseUpdatePolicy', 'set_attr_for_all_layers', 'HeNormalInit',
           'SGD', 'DenseLayer', 'DenseNet', 'build_densenet', 'LayerNormLayer', 'BatchNormLayer']

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from backbone import num_flat_features


class BaseDenseLayer(ABC, nn.Module):
    """
    Base class that all fully connected layers should inherit from.
    """
    def __init__(self,n_input, n_output, nonlinearity=None, weight_init_gain=None):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.nonlinearity = nonlinearity
        self.weight_init_gain = weight_init_gain

        self.weight_init_policy = BaseWeightInitPolicy()
        self.update_policy      = BaseUpdatePolicy()
        self.parent_index = None # "parent" will be the network class object
        self.parent_name = None

    @abstractmethod
    def forward(self):
        pass

    def init_weights(self, **args):
        self.weight_init_policy.init_weights(self, **args)

    #@torch.no_grad()
    def update(self, **args):
        self.update_policy.update(self, **args)

    @property
    def param_names(self):
        return [p[0] for p in self.named_parameters()]

    def __repr__(self):
        r  = ''
        r += str(self.__class__.__name__)+' '
        for key, param in self.named_parameters():
            r += key +' ' + str(list(param.shape))+' '
        if self.nonlinearity is None:
            r += 'linear'
        else:
            r += str(self.nonlinearity.__name__)
        return r


class BaseWeightInitPolicy():
    """
    Generic weight init policy.
    """
    @staticmethod
    def init_weights(layer, **args):
        raise NotImplementedError


class BaseUpdatePolicy():
    """
    Generic update policy.  Update is not a static method as we may want grad history
    etc for some update policies.
    """
    def update(self, layer, **args):
        raise NotImplementedError


def set_attr_for_all_layers(model, attr_name, class_def):
    for key, layer in model.layers.items():
        layer.__dict__[attr_name] = class_def()


class HeNormalInit(BaseWeightInitPolicy):
    """
    This weight init policy assumes layer with Parameters:
        - W, b
    """
    @staticmethod
    def init_weights(layer):
        assert layer.weight_init_gain is not None
        target_std = np.sqrt((layer.weight_init_gain / layer.n_input))
        nn.init.zeros_(layer.b)
        nn.init.normal_(layer.W, mean=0, std=target_std)


class SGD(BaseUpdatePolicy):
    def update(self, layer, **args):
        """
        Standard Stochastic gradient descent
        Args:
            lr : learning rate
        """
        with torch.no_grad():
            for key, p in layer.named_parameters():
                p -= p.grad * args['lr']


class DenseLayer(BaseDenseLayer):
    def __init__(self, n_input, n_output, nonlinearity=None, weight_init_gain=2):
        """
        n_input:      input dimension
        n_output:     output dimension
        nonlinearity: nonlinear activation function, if None then linear
        weight_init_gain: used for weight init
        """
        super().__init__(n_input, n_output, nonlinearity, weight_init_gain)

        self.W = nn.Parameter(torch.randn(n_output, n_input))
        self.b = nn.Parameter(torch.zeros(n_output, 1))

        # by default use these weight and update policies
        self.weight_init_policy = HeNormalInit()
        self.update_policy = SGD()

    def forward(self, x):
        """
        Note: in all models we transpose x to match eqs.
        """
        self.z = torch.mm(self.W, x.T) + self.b
        self.z = self.z.T # transpose again to maintain first axis as batch
        if self.nonlinearity is not None:
            self.h = self.nonlinearity(self.z)
        else:
            self.h = self.z
        return self.h


class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict() # ordered dict that respects insertion order

    @property
    def n_input(self):
        return self[0].n_input

    def update(self, **args):
        for key, layer in self.layers.items():
            layer.update(**args)

    def init_weights(self, **args):
        for key, layer in self.layers.items():
            layer.init_weights(**args)

    def forward(self, x):
        x = x.view(-1, num_flat_features(x))
        for key, layer in self.layers.items():
            x = layer.forward(x)
        return x

    def __getitem__(self, item):
        # Enables layers to be indexed
        if isinstance(item, slice):
            print("Slicing not supported yet")
            raise
        key = list(self.layers)[item]
        return self.layers[key]

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the non-activated output of the second-last layer.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (??)
        """
        for key, layer in self.layers.items()[:-1]:
            x = layer.forward(x)
        return x

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (??)
        """
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[progress: progress +
                torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)


def build_densenet(ModelClass, LayerClass, layer_dims=[784, 200, 200, 10]):
    """
    Constructs a densenet of a single layerclass.

    Args:
        ModelClass: the model class to use, typically DenseNet
        LayerClass: all the layers are of this class
        layer_dims: list of dims of each layer. Default is 2 hidden layers of size 200 (784,200,200,10).
                    and 784 input, 10 output. List elements can also be an iterable of "layer params"
                    (apart from first which is the input data dim). If list element is an iterable,
                    then the first element should be the layer "size", e.g. 200.

    Note: Automatically sets all but the last layer's activation function to Relu and sets the
          weight_init_gain of the first layer to 1, as inputs are assumed to not have gone through relu.
          Every other layer has weight_init_gain 2.
    """
    net = ModelClass()
    n_layers = len(layer_dims) - 1  # -1 as elements i, i+1 parameterise each layer
    for i in range(n_layers):
        if hasattr(layer_dims[i], '__iter__'):
            input_dim = layer_dims[i][0]  # input dim is first element of iterable, e.g. (200,1)
        else:
            input_dim = layer_dims[i]  # mlp layer or the input layer

        layer_params = layer_dims[i+1]  # this will be either an int, or iterable (ne, ni)

        if i+1 == n_layers: nonlin = None  # final layer is linear
        else: nonlin = F.relu

        #  set gain for weight init dep. on layer number
        if i == 0:  gain = 1
        else: gain = 2

        net.layers['fc'+str(i)] = LayerClass(input_dim, layer_params, nonlin, weight_init_gain=gain)
        net.layers['fc'+str(i)].parent_index = i
        net.layers['fc'+str(i)].parent_name = 'fc' + str(i)

        # Update object dict with layers dict so can access with dot notation
        net.__dict__.update(net.layers)

    return net


class LayerNormLayer(DenseLayer):
    def __init__(self, n_input, n_output, nonlinearity=None, weight_init_gain=2):
        super().__init__(n_input, n_output, nonlinearity, weight_init_gain)

        self.g = nn.Parameter(torch.ones(n_output, 1))
        self.b = nn.Parameter(torch.zeros(n_output, 1))

    def forward(self, x):
        self.zhat = torch.mm(self.W, x.T) + self.b
        # after x.T, zhat is n_output x batch
        sigma = self.zhat.std(axis=0, keepdim=True)
        mu    = self.zhat.mean(axis=0, keepdim=True)
        self.z = (self.g / sigma)* (self.zhat - mu) +self.b
        self.z = self.z.T # maintain first axis as batch
        if self.nonlinearity is not None:
            self.h = self.nonlinearity(self.z)
        else:
            self.h = self.z
        return self.h


class BatchNormLayer(DenseLayer):
    def __init__(self, n_input, n_output, nonlinearity=None, weight_init_gain=2):
        super().__init__(n_input, n_output, nonlinearity, weight_init_gain)
        self.g = nn.Parameter(torch.ones(n_output, 1))
        self.b = nn.Parameter(torch.zeros(n_output, 1))

    def forward(self, x):
        self.zhat = torch.mm(self.W, x.T) + self.b
        # after x.T, zhat is n_output x batch
        sigma = self.zhat.std(axis=1, keepdim=True)
        mu    = self.zhat.mean(axis=1, keepdim=True)
        self.z = (self.g / sigma)* (self.zhat - mu) +self.b
        self.z = self.z.T  # maintain first axis as batch
        if self.nonlinearity is not None:
            self.h = self.nonlinearity(self.z)
        else:
            self.h = self.z
        return self.h
