# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------
import abc
import math
import warnings

import numpy as np
import torch
import torch.nn as nn


def rezero_weights(m):
    """Function used to update the weights after each epoch.

    Call using :meth:`torch.nn.Module.apply` after each epoch if required
    For example: ``m.apply(rezero_weights)``

    :param m: HasRezeroWeights module
    """
    if isinstance(m, HasRezeroWeights):
        m.rezero_weights()


def normalize_sparse_weights(m):
    """Initialize the weights using kaiming_uniform initialization normalized
    to the number of non-zeros in the layer instead of the whole input size.

    Similar to torch.nn.Linear.reset_parameters() but applying weight
    sparsity to the input size
    """
    if isinstance(m, SparseWeightsBase):
        _, input_size = m.module.weight.shape
        fan = int(input_size * (1.0 - m.sparsity))
        gain = nn.init.calculate_gain("leaky_relu", math.sqrt(5))
        std = gain / np.math.sqrt(fan)
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        nn.init.uniform_(m.module.weight, -bound, bound)
        if m.module.bias is not None:
            bound = 1 / math.sqrt(fan)
            nn.init.uniform_(m.module.bias, -bound, bound)


class HasRezeroWeights(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def rezero_weights(self):
        """Set the previously selected weights to zero."""
        raise NotImplementedError


class SparseWeightsBase(nn.Module, HasRezeroWeights):
    """
    Base class for the all Sparse Weights modules.

    :param module:
      The module to sparsify the weights
    :param sparsity:
      Pct of weights that are zero in the layer.
    """

    def __init__(self, module, weight_sparsity=None, sparsity=None):
        super(SparseWeightsBase, self).__init__()
        assert weight_sparsity is not None or sparsity is not None
        if weight_sparsity is not None and sparsity is None:
            sparsity = 1.0 - weight_sparsity
            warnings.warn(
                "Parameter `weight_sparsity` is deprecated. Use `sparsity` instead.",
                DeprecationWarning,
            )

        self.module = module
        self.sparsity = sparsity

    def extra_repr(self):
        return "sparsity={}".format(self.sparsity)

    def forward(self, x):
        return self.module(x)

    @property
    def weight_sparsity(self):
        warnings.warn(
            "Parameter `weight_sparsity` is deprecated. Use `sparsity` instead.",
            DeprecationWarning,
        )
        return 1.0 - self.sparsity

    @property
    def weight(self):
        return self.module.weight

    @property
    def bias(self):
        return self.module.bias


class SparseWeights(SparseWeightsBase):
    """Enforce weight sparsity on linear module during training.

    Sample usage:

      model = nn.Linear(784, 10)
      model = SparseWeights(model, 0.4)

    :param module:
      The module to sparsify the weights
    :param sparsity:
      Pct of weights that are zero in the layer.
    :param allow_extremes:
      Allow values sparsity=0 and sparsity=1. These values are often a sign that
      there is a bug in the configuration, because they lead to Identity and
      Zero layers, respectively, but they can make sense in scenarios where the
      mask is dynamic.
    """

    def __init__(self, module, weight_sparsity=None, sparsity=None,
                 allow_extremes=False):
        assert len(module.weight.shape) == 2, "Should resemble a nn.Linear"
        super(SparseWeights, self).__init__(
            module, weight_sparsity=weight_sparsity, sparsity=sparsity
        )

        if allow_extremes:
            assert 0 <= self.sparsity <= 1
        else:
            assert 0 < self.sparsity < 1

        # For each unit, decide which weights are going to be zero
        in_features = self.module.in_features
        ne = self.module.ne
        ni = self.module.ni

        # Mask for excitatory neurons
        num_nz = int(round((1 - self.sparsity) * in_features))
        ne_zero_mask = torch.ones(ne, in_features, dtype=torch.bool)
        for ex_unit in range(ne):
            in_indices = np.random.choice(in_features, num_nz, replace=False)
            ne_zero_mask[ex_unit, in_indices] = False
        self.register_buffer("ne_zeros_mask", ne_zero_mask.half())

        # Mask for inhibitory neurons
        ni_zero_mask = torch.ones(ni, in_features, dtype=torch.bool)
        for in_unit in range(ni):
            in_indices = np.random.choice(in_features, num_nz, replace=False)
            ni_zero_mask[in_unit, in_indices] = False
        self.register_buffer("ni_zeros_mask", ni_zero_mask.half())
        self.rezero_weights()

    def rezero_weights(self):
        self.module.Wex.data.masked_fill_(self.ne_zeros_mask.bool(), 0)
        self.module.Wix.data.masked_fill_(self.ni_zeros_mask.bool(), 0)


# model = nn.Linear(784, 10)
# model = SparseWeights(model, 0.4)
# from backbone.dale_nn.ei_models import EiDense
#
# model = EiDense(784, (10, 1))
# model.Wix.data = torch.ones(model.Wix.data.shape)
# model.Wex.data = torch.ones(model.Wex.data.shape)
# # model = torch.nn.Linear(784, 10, bias=False)
# model = SparseWeights(model, 0.5, allow_extremes=True)
# print(torch.numel(model.module.Wix))
# print(torch.sum(model.module.Wix > 0))
#
# print(torch.numel(model.module.Wex))
# print(torch.sum(model.module.Wex == 0))
