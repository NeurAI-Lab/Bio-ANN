# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
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

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable
from backbone.dale_dendrites.modules.dale_dendritic_layers import (
    DaleAbsoluteMaxGatingDendriticLayer,
    DaleAbsoluteMaxGatingDendriticLayerWithShunt
)
from backbone.dale_nn.ei_models import (
    EiDense,
    EiDense_MultipleInt_LayerNorm_WeightInitPolicy,
    EiDenseWithShunt,
    EiDenseWithShunt_MultipleInt_LayerNorm_WeightInitPolicy,
    DalesANN_cSGD_UpdatePolicy,
)

from backbone.dendrites.utils.k_winners import KWinners
from backbone.dendrites.utils.sparse_weights import rezero_weights


class DaleDendriticMLP(nn.Module):
    """
    A simple but restricted MLP with two hidden layers of the same size. Each hidden
    layer contains units with dendrites. Dendrite segments receive context directly as
    input.  The class is used to experiment with different dendritic weight
    initializations and learning parameters

    :param input_size: size of the input to the network
    :param output_size: the number of units in the output layer. Must be either an
                        integer if there is a single output head, or an iterable
                        of integers if there are multiple output heads.
    :param hidden_sizes: the number of units in each hidden layer
    :param num_segments: the number of dendritic segments that each hidden unit has
    :param dim_context: the size of the context input to the network
    :param kw: whether to apply k-Winners to the outputs of each hidden layer
    :param kw_percent_on: percent of hidden units activated by K-winners. If 0, use ReLU
    :param context_percent_on: percent of non-zero units in the context input.
    :param dendrite_weight_sparsity: the sparsity level of dendritic weights.
    :param weight_sparsity: the sparsity level of feed-forward weights.
    :param weight_init: the initialization applied to feed-forward weights; must be
                        either "kaiming" (for Kaiming Uniform) of "modified" (for
                        sparse Kaiming Uniform)
    :param dendrite_init: the initialization applied to dendritic weights; similar to
                          `weight_init`
    :param freeze_dendrites: whether to set `requires_grad=False` for all dendritic
                             weights so they don't train
    :param dendritic_layer_class: dendritic layer class to use for each hidden layer
    :param output_nonlinearity: nonlinearity to apply to final output layer.
                                'None' of no nonlinearity.
                    _____
                   |_____|    # classifier layer, no dendrite input
                      ^
                      |
                  _________
    context -->  |_________|  # second linear layer with dendrites
                      ^
                      |
                  _________
    context -->  |_________|  # first linear layer with dendrites
                      ^
                      |
                    input
    """

    def __init__(
        self, input_size, output_size, hidden_sizes, num_segments, dim_context,
        kw, kw_percent_on=0.05, context_percent_on=1.0, dendrite_weight_sparsity=0.95,
        weight_sparsity=0.95, weight_init="modified", dendrite_init="modified",
        freeze_dendrites=False, output_nonlinearity=None, use_shunting=False,
        output_inhib_units=1, i_iid_i=False, apply_to_dendrites=False,
    ):

        # Forward & dendritic weight initialization must be either "kaiming" or
        # "modified"
        assert weight_init in ("kaiming", "modified")
        assert dendrite_init in ("kaiming", "modified")
        assert kw_percent_on is None or (kw_percent_on >= 0.0 and kw_percent_on < 1.0)
        assert context_percent_on >= 0.0

        if kw_percent_on == 0.0:
            kw = False

        super().__init__()

        self.apply_to_dendrites = apply_to_dendrites
        self.num_segments = num_segments
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dim_context = dim_context
        self.kw = kw
        self.kw_percent_on = kw_percent_on
        self.weight_sparsity = weight_sparsity
        self.dendrite_weight_sparsity = dendrite_weight_sparsity
        self.output_nonlinearity = output_nonlinearity
        self.hardcode_dendrites = (dendrite_init == "hardcoded")

        self._layers = nn.ModuleList()
        self._activations = nn.ModuleList()
        self._activation_counts = {}
        self._keep_probs = {}

        if self.hardcode_dendrites:
            dendrite_sparsity = 0.0
        else:
            dendrite_sparsity = self.dendrite_weight_sparsity

        if use_shunting:
            dendritic_layer_class = DaleAbsoluteMaxGatingDendriticLayerWithShunt
        else:
            dendritic_layer_class = DaleAbsoluteMaxGatingDendriticLayer

        # Allow user to specify multiple layer types, with backward compatibility.
        # Just specify dendritic_layer_class as a module, and automatically broadcast
        # to a list of modules. Or, specify a list of customized modules.
        if not isinstance(dendritic_layer_class, list):
            dendritic_layer_classes = [dendritic_layer_class
                                       for i in
                                       range(len(self.hidden_sizes))]
        else:
            dendritic_layer_classes = dendritic_layer_class

        for i in range(len(self.hidden_sizes)):
            if i == 0:
                gain = 1
            else:
                gain = 2
            self._activation_counts[f'layer_{i}'] = torch.zeros(hidden_sizes[i][0])
            curr_dend = dendritic_layer_classes[i](
                n_input=input_size,
                layer_params=hidden_sizes[i],
                num_segments=num_segments,
                dim_context=dim_context,
                dendrite_sparsity=dendrite_sparsity,
                weight_init_gain=gain,
                parent_idx=i,
                i_iid_i=i_iid_i
            )

            #TODO: Adjust the dale init weights to account for input sparsity
            if weight_init == "modified":
                # Scale weights to be sampled from the new initialization U(-h, h) where
                # h = sqrt(1 / (weight_density * previous_layer_percent_on))
                if i == 0:
                    # first hidden layer can't have kw input
                    self._init_sparse_weights(curr_dend, 0.0)
                else:
                    self._init_sparse_weights(
                        curr_dend,
                        1 - kw_percent_on if kw else 0.0
                    )

            if dendrite_init == "modified":
                self._init_sparse_dendrites(curr_dend, 1 - context_percent_on)

            if freeze_dendrites:
                # Dendritic weights will not be updated during backward pass
                for name, param in curr_dend.named_parameters():
                    if "segments" in name:
                        param.requires_grad = False

            if self.kw:
                curr_activation = KWinners(n=hidden_sizes[i][0],
                                           percent_on=kw_percent_on,
                                           k_inference_factor=1.0,
                                           boost_strength=0.0,
                                           boost_strength_factor=0.0)
            else:
                curr_activation = nn.ReLU()

            self._layers.append(curr_dend)
            self._activations.append(curr_activation)

            input_size = self.hidden_sizes[i][0]

        self._single_output_head = not isinstance(output_size, Iterable)
        if self._single_output_head:
            output_size = (output_size,)

        self._output_layers = nn.ModuleList()
        for out_size in output_size:
            output_layer = nn.Sequential()

            if use_shunting:
                output_linear = EiDenseWithShunt(input_size, (out_size, output_inhib_units))
                output_linear.weight_init_policy = EiDenseWithShunt_MultipleInt_LayerNorm_WeightInitPolicy(inhib_iid_init=i_iid_i)
                output_linear.update_policy = DalesANN_cSGD_UpdatePolicy()
            else:
                output_linear = EiDense(input_size, (out_size, output_inhib_units))
                output_linear.weight_init_policy = EiDense_MultipleInt_LayerNorm_WeightInitPolicy(inhib_iid_init=i_iid_i)
                output_linear.update_policy = DalesANN_cSGD_UpdatePolicy()

            output_layer.add_module("output_linear", output_linear)

            if self.output_nonlinearity is not None:
                output_layer.add_module("non_linearity", output_nonlinearity)
            self._output_layers.append(output_layer)

    def forward(self, x, context=None):
        assert (context is not None) or (self.num_segments == 0)

        for i, (layer, activation) in enumerate(zip(self._layers, self._activations)):
            orig_act = layer(x, context)
            # Apply Dropout
            if self.training:
                if self._keep_probs:
                    # layer_mask = torch.Tensor([Binomial(probs=prob).sample() for prob in self._keep_probs[f'layer_{i}']])
                    layer_mask = torch.rand(orig_act.shape[1]) < self._keep_probs[f'layer_{i}']
                    layer_mask = layer_mask.to(x.device)
                    orig_act *= layer_mask

            x = activation(orig_act)
            # Update the activation count
            if self.training:
                self._activation_counts[f'layer_{i}'] += (x > 0).sum(dim=0).cpu()

        if self._single_output_head:
            return self._output_layers[0](x)
        else:
            return [out_layer(x) for out_layer in self._output_layers]

    # ------ Weight initialization functions ------
    @staticmethod
    def _init_sparse_weights(m, input_sparsity):
        """
        Modified Kaiming weight initialization that considers input sparsity and weight
        sparsity.
        """
        input_density = 1.0 - input_sparsity
        weight_density = 1.0
        # weight_density = 1.0 - m.sparsity
        _, fan_in = m.W.size()
        bound = 1.0 / np.sqrt(input_density * weight_density * fan_in)
        nn.init.uniform_(m.W, -bound, bound)
        m.apply(rezero_weights)

    @staticmethod
    def _init_sparse_dendrites(m, input_sparsity):
        """
        Modified Kaiming initialization for dendrites segments that consider input
        sparsity and dendritic weight sparsity.
        """
        # Assume `m` is an instance of `DendriticLayerBase`
        if m.segments is not None:
            input_density = 1.0 - input_sparsity
            weight_density = 1.0 - m.segments.sparsity
            fan_in = m.dim_context
            bound = 1.0 / np.sqrt(input_density * weight_density * fan_in)
            nn.init.uniform_(m.segment_weights, -bound, bound)
            m.apply(rezero_weights)

    def hardcode_dendritic_weights(self, context_vectors, init):
        """
        Set up specific weights for each dendritic segment based on the value of init.

        if init == "overlapping":
            We hardcode the weights of dendrites such that each context selects 5% of
            hidden units to become active and form a subnetwork. Hidden units are
            sampled with replacement, hence subnetworks can overlap. Any context/task
            which does not use a particular hidden unit will cause it to turn off, as
            the unit's other segment(s) have -1 in all entries and will yield an
            extremely small dendritic activation.

        otherwise if init == "non_overlapping":
            We hardcode the weights of dendrites such that each unit recognizes a single
            random context vector. The first dendritic segment is initialized to contain
            positive weights from that context vector. The other segment(s) ensure that
            the unit is turned off for any other context - they contain negative weights
            for all other weights.

        :param context_vectors:
        :param init: a string "overlapping" or "non_overlapping"
        """
        if self.num_segments > 0:
            for dendrite in self._layers:
                self._hardcode_dendritic_weights(dendrite.weights, context_vectors,
                                                 init)

    @staticmethod
    def _hardcode_dendritic_weights(dendrite_weights, context_vectors, init):
        squeeze = False
        if len(dendrite_weights.shape) == 2:
            # 1 segment dendrite, so add in a segment dimension
            squeeze = True
            original_weights = dendrite_weights
            dendrite_weights = dendrite_weights.unsqueeze(dim=1)

        num_units, num_segments, dim_context = dendrite_weights.size()
        num_contexts, _ = context_vectors.size()

        if init == "overlapping":
            new_dendritic_weights = -0.95 * torch.ones((num_units, num_segments,
                                                        dim_context))

            # The number of units to allocate to each context (with replacement)
            k = int(0.05 * num_units)

            # Keep track of the number of contexts for which each segment has already
            # been chosen; this is to not overwrite a previously hardcoded segment
            num_contexts_chosen = {i: 0 for i in range(num_units)}

            for c in range(num_contexts):

                # Pick k random units to be activated by the cth context
                selected_units = torch.randperm(num_units)[:k]
                for i in selected_units:
                    i = i.item()

                    # If num_segments other contexts have already selected unit i to
                    # become active, skip
                    segment_id = num_contexts_chosen[i]
                    if segment_id == num_segments:
                        continue

                    new_dendritic_weights[i, segment_id, :] = context_vectors[c, :]
                    num_contexts_chosen[i] += 1

        elif init == "non_overlapping":
            new_dendritic_weights = torch.zeros((num_units, num_segments, dim_context))

            for i in range(num_units):
                context_perm = context_vectors[torch.randperm(num_contexts), :]
                new_dendritic_weights[i, :, :] = 1.0 * (context_perm[0, :] > 0)
                new_dendritic_weights[i, 1:, :] = -1
                new_dendritic_weights[i, 1:, :] += new_dendritic_weights[i, 0, :]
                del context_perm

        else:
            raise Exception("Invalid dendritic weight hardcode choice")

        dendrite_weights.data = new_dendritic_weights

        if squeeze:
            dendrite_weights = dendrite_weights.squeeze(dim=1)
            # dendrite weights doesn't point to the dendrite weights tensor,
            # so expicitly assign the new values
            original_weights.data = dendrite_weights

    def update(self, **args):
        for layer in self._layers:
            layer.update(**args)

        self._output_layers[0][0].update(**args)

    def hebbian_update(self, **args):
        for layer in self._layers:
            layer.hebbian_update(**args)

    def init_weights(self, **args):
        for layer in self._layers:
            layer.init_weights(**args)

        self._output_layers[0][0].init_weights(**args)

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def get_si_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for name, pp in self.named_parameters():
            if self.apply_to_dendrites or "segment" not in name:
                params.append(pp.view(-1))
        return torch.cat(params)

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)

    def get_si_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        grads = []
        for name, pp in self.named_parameters():
            if self.apply_to_dendrites or "segment" not in name:
                grads.append(pp.grad.view(-1))
        return torch.cat(grads)

# model = DaleDendriticMLP(
#     input_size=784,
#     output_size=10,
#     hidden_sizes=[(500, 50), (500, 5)],
#     num_segments=10,
#     dim_context=784,
#     kw=True,
#     weight_init="modified",
#     dendrite_init="modified",
#     apply_to_dendrites=True
# )
#
#
# for name, pram in model.named_parameters():
#     if 'segment' in name:
#         print(name)
#
# len(model.get_si_params())