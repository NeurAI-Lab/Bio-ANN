
__all__ = ['calculate_ColumnEi_layer_params', 'ColumnEiWeightInitPolicy', 'PostiveWeightInitPolicy', 'ColummEiDense',
           'ColumnEiSGD']

import numpy as np
import torch
import torch.nn as nn
from .base_models import BaseDenseLayer
from .base_models import BaseWeightInitPolicy, BaseUpdatePolicy


def calculate_ColumnEi_layer_params(total, ratio):
    """
    For a ColumnEi model layer_params is a total number of units, and a ratio e.g 20, for 20:1.
    This is a util function to calculate n_e, n_i.
    Args:
        ratio : int
        total : int, total number of units (typically n_input to a layer)
    """
    frac = total / (ratio+1)
    n_i = int(np.ceil(frac))
    n_e = int(np.floor(frac * ratio))
    return n_e, n_i

class ColumnEiWeightInitPolicy(BaseWeightInitPolicy):
    """
    Weights are drawn from an exponential distribution
    """
    @staticmethod
    def init_weights(layer):
        ne = layer.ne
        ni = layer.ni

        denom = ( (2*np.pi-1)/(2*np.pi) ) * (ne + (ne**2/ni))
        sigma_we = np.sqrt( 1 / denom)
        sigma_wi = (ne/ni)*sigma_we

        We_np = np.random.exponential(scale=sigma_we, size=(layer.n_output, ne))
        Wi_np = np.random.exponential(scale=sigma_wi, size=(layer.n_output, ni))
        W = np.concatenate([We_np, Wi_np], axis=1)

        layer.W_pos.data = torch.from_numpy(W).float()

        # D matrix (last ni columns are -)
        layer.D.data = torch.eye(ne+ni).float()
        layer.D.data[:,-ni:] *= -1

        # bias
        nn.init.zeros_(layer.b)

class PostiveWeightInitPolicy(BaseWeightInitPolicy):
    """
    This is the weight init that should be used for the first layer a ColumnEi network.

    We use the bias term to center the activations, and glorot init to set the weight variance.
        bias  <- layer.n_input * sigma * mnist_mean *-1

    Weights are drawn from an exponential distribution.
    """
    def __init__(self, dataset):
        if dataset == 'MNIST':
            self.pixel_mean = .1307
        elif dataset == 'KMNIST':
            self.pixel_mean = .1918
        elif dataset == 'FashionMNIST':
            self.pixel_mean = .2860
        print(dataset, self.pixel_mean)

    def init_weights(self,layer):
        # Weights
        print(layer.n_output, layer.n_input)
        sigma = np.sqrt(1/layer.n_input)
        W_np = np.random.exponential(scale=sigma, size=(layer.n_output, layer.n_input))
        layer.W_pos.data = torch.from_numpy(W_np).float()

        # D matrix (is all positive)
        layer.D.data = torch.eye(layer.n_input).float()

        # bias
        z_mean = layer.n_input * sigma * self.pixel_mean
        nn.init.constant_(layer.b,val= -z_mean)

class ColummEiDense(BaseDenseLayer):
    """
    Class modeling EI in "parallel". Based on the RNN formulation of Song et al 2016 Plos Bio.
    """
    def __init__(self, n_input, layer_params: '(n_output, ratio)', nonlinearity=None, weight_init_gain=None, clamp=True):
        """
        layer_params: (tuple) (ouput, ratio of e (to i))
        """
        print(n_input, layer_params)
        n_output, ratio = layer_params
        self.ne, self.ni = calculate_ColumnEi_layer_params(n_input, ratio)

        # call super to set self. n_input, n_output, nonlnearity, weight_init_gain
        # and init_weights(), update() methods to impl. self.update_policy.update() etc
        super().__init__(n_input, n_output, nonlinearity, weight_init_gain)

        self.clamp = clamp

        self.W_pos = nn.Parameter(torch.empty(self.n_output, self.n_input))
        self.D     = nn.Parameter(torch.empty(self.n_input, self.n_input),requires_grad=False)
        self.b     = nn.Parameter(torch.empty(self.n_output,1))

        self.weight_init_policy = ColumnEiWeightInitPolicy()
        self.update_policy = ColumnEiSGD()

    @property
    def W(self):
        return self.W_pos@self.D

    def forward(self, x):
        self.x = x
        self.z = torch.matmul(self.W, self.x.T)
        self.z = self.z + self.b
        self.z = self.z.T   # maintain batch as first axis
        if self.nonlinearity is not None:
            self.h = self.nonlinearity(self.z)
        else:
            self.h = self.z

        return self.h

    def __repr__(self):
        r = super().__repr__()
        r += f', ne:{self.ne}, ni: {self.ni}'
        return r

class ColumnEiSGD(BaseUpdatePolicy):

    @torch.no_grad()
    def update(self, layer, **args):
        lr = args['lr']
        layer.b     -= layer.b.grad *lr
        layer.W_pos -= layer.W_pos.grad * lr

        if layer.clamp:
            layer.W_pos.data = torch.clamp(layer.W_pos, min=0)

