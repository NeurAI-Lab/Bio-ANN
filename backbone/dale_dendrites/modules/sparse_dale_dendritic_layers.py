# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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

"""
A simple implementation of dendrite weights. This combines the output from a (sparse)
linear layer with the output from a set of dendritic segments.
"""
import abc
import torch
from backbone.dale_nn.ei_models import (
    EiDense,
    EiDense_MultipleInt_LayerNorm_WeightInitPolicy,
)
from backbone.dale_dendrites.modules.sparse_update_rules import DaleDendrites_cSGD_UpdatePolicy, HebbianUpdate
from backbone.dendrites.modules.apply_dendrites import (
    DendriticAbsoluteMaxGate1d,
)
from backbone.dendrites.modules.dendrite_segments import DendriteSegments
from backbone.dale_dendrites.modules.sparse_weights import SparseWeights


# =============================================================================
# Without Shunt
# =============================================================================
class SparseDaleDendriticLayerBase(SparseWeights, metaclass=abc.ABCMeta):
    """
    Base class for all Dendritic Layer modules.

    This combines a DendriteSegments module with Dale Principle Layer
    The output from the dendrite segments (shape of num_units x num_segments)
    is applied to the output of of the linear weights (shape of num_units).
    Thus, each linear output unit gets modulated by a set of dendritic segments.
    """
    def __init__(
            self, n_input, layer_params, num_segments, dim_context,
            module_sparsity, dendrite_sparsity, dendrite_bias=None,
            nonlinearity=None, weight_init_gain=None, i_iid_i=False, parent_idx=None,

    ):
        """
        TODO: specify the type - what is module_sparsity type?
        :param module: linear module from in-units to out-units
        :param num_segments: number of dendrite segments per out-unit
        :param dim_context: length of the context vector;
                            the same context will be applied to each segment
        :param module_sparsity: sparsity applied over linear module;
        :param dendrite_sparsity: sparsity applied transformation per unit per segment
        :param dendrite_bias: whether or not dendrite activations have an additive bias
        """
        self.dim_context = dim_context
        self.segments = None

        module = EiDense(
            n_input,
            layer_params,
            nonlinearity,
            weight_init_gain
        )

        module.weight_init_policy = EiDense_MultipleInt_LayerNorm_WeightInitPolicy(inhib_iid_init=i_iid_i)
        module.parent_index = parent_idx

        super().__init__(
            module,
            sparsity=module_sparsity,
            allow_extremes=True,
        )

        self.update_policy = DaleDendrites_cSGD_UpdatePolicy()
        self.hebbian_update_policy = HebbianUpdate()

        self.segments = DendriteSegments(
            num_units=layer_params[0],
            num_segments=num_segments,
            dim_context=dim_context,
            sparsity=dendrite_sparsity,
            bias=dendrite_bias,
        )

        # self.rezero_weights()

    def rezero_weights(self):
        """Set the previously selected weights to zero."""
        super().rezero_weights()
        if self.segments is not None:  # only none at beginning of init
            self.segments.rezero_weights()

    @abc.abstractmethod
    def apply_dendrites(self, y, dendrite_activations):
        """Apply dendrites using function specified by subclass"""
        raise NotImplementedError

    def forward(self, x, context):
        """Compute of linear layer and apply output of dendrite segments."""
        y = super().forward(x)
        dendrite_activations = self.segments(context)  # num_units x num_segments
        return self.apply_dendrites(y, dendrite_activations)

    @property
    def segment_weights(self):
        return self.segments.weights

    #@torch.no_grad()
    def hebbian_update(self, **args):
        self.hebbian_update_policy.update(self, **args)

    def update(self, **args):
        self.update_policy.update(self, **args)


class SparseDaleAbsoluteMaxGatingDendriticLayer(SparseDaleDendriticLayerBase):
    """
    This layer is similar to `GatingDendriticLayer`, but selects dendrite activations
    based on absolute max activation values instead of just max activation values. For
    example, if choosing between activations -7.4, and 6.5 for a particular unit, -7.4
    will be chosen, and its sign will be kept.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dendritic_absolute_max_gate = DendriticAbsoluteMaxGate1d()

    def apply_dendrites(self, y, dendrite_activations):
        """Apply dendrites as a gating mechanism."""
        return self.dendritic_absolute_max_gate(y, dendrite_activations).values

# model = SparseDaleAbsoluteMaxGatingDendriticLayer(
#     100,
#     (10, 1),
#     10,
#     64,
#     0.5,
#     0.1
# )
#
# model.module.init_weights()
# model.rezero_weights()
#
# print(torch.numel(model.module.Wix))
# print(torch.sum(model.module.Wix > 0))
#
# print(torch.numel(model.module.Wex))
# print(torch.sum(model.module.Wex == 0))


#
# input = torch.randn(28, 100)
# context = torch.randn(28, 64)
# model(input, context)
