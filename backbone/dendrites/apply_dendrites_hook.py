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

import torch

from backbone.dendrites.hooks import TrackStatsHookBase
from backbone.dendrites.utils.mask_utils import indices_to_mask

__all__ = [
    "ApplyDendritesHook",
]


class ApplyDendritesHook(TrackStatsHookBase):
    """
    Hook for tracking an `apply_dendrites` module.
    """

    def __init__(self, name, max_samples_to_track):
        super().__init__(name=name)

        self.num_samples = max_samples_to_track

        # Activations of num_samples x num_units
        self.dendrite_activations = torch.tensor([])

        # Mask of num_samples x num_units x num_segments
        self.winning_mask = torch.tensor([]).bool()

    def get_statistics(self):
        return (self.dendrite_activations, self.winning_mask)

    def __call__(self, module, x, y):
        """
        Save up to the last 'max_samples_to_track' of the dendrite activations and the
        corresponding winning mask.

        :param x: input to an `apply_dendrites` modules; this is tuple
                  of (y, dendrite_activations)
        :param y: dendrite_output named tuple (with values and indices)
        """

        if not self._tracking:
            return

        dendrite_activations = x[1]
        winning_mask = indices_to_mask(y.indices, shape=x[1].shape, dim=2)

        # The `self` tensors were initialized on the cpu which could differ from the
        # values collected during the forward pass.
        device = winning_mask.device
        self.winning_mask = self.winning_mask.to(device)
        self.dendrite_activations = self.dendrite_activations.to(device)

        # MetaCL creates a deepcopy of the model, but this isn't allowed on non-leaf
        # tensors. In detaching it, this will always be the case.
        dendrite_activations = dendrite_activations.detach()

        # Prepend the newest activations and winning masks.
        self.winning_mask = torch.cat((winning_mask, self.winning_mask), dim=0)
        self.dendrite_activations = torch.cat((dendrite_activations,
                                               self.dendrite_activations), dim=0)

        # Keep only the last 'num_samples'.
        self.winning_mask = self.winning_mask[:self.num_samples, ...]
        self.dendrite_activations = self.dendrite_activations[:self.num_samples, ...]
