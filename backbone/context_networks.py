# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from backbone import xavier, num_flat_features
import torch.nn.functional as F


class ContextNet(nn.Module):
    """
    Network composed of two hidden layers, each containing 100 ReLU activations.
    Designed for the MNIST dataset.
    """

    def __init__(self, input_size: int, hidden_params: tuple, output_size: int, normalize: bool = True) -> None:
        """
        Instantiates the layers of the network.
        :param input_size: the size of the input data
        :param output_size: the size of the output
        """
        super(ContextNet, self).__init__()

        self.input_size = input_size
        self.hidden_params = hidden_params
        self.output_size = output_size
        self.normalize = normalize

        lst_modules = []
        input_dim = input_size
        for n, n_hidden_units in enumerate(hidden_params):
            lst_modules.append(nn.Linear(input_dim, n_hidden_units))
            lst_modules.append(nn.ReLU())
            input_dim = n_hidden_units

        # Output Units
        lst_modules.append(nn.Linear(input_dim, output_size))
        # lst_modules.append(nn.Sigmoid())

        self.net = nn.Sequential(*lst_modules)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Calls the Xavier parameter initialization function.
        """
        self.net.apply(xavier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (output_size)
        """
        # x = x.view(-1, num_flat_features(x))
        out = self.net(x)
        if self.normalize:
            out = F.normalize(out)
        return out


class ContextNetv2(nn.Module):
    """
    Network composed of two hidden layers, each containing 100 ReLU activations.
    Designed for the MNIST dataset.
    """

    def __init__(self, input_size: int, hidden_params: tuple, feat_dim: int, n_prototypes: int) -> None:
        """
        Instantiates the layers of the network.
        :param input_size: the size of the input data
        :param output_size: the size of the output
        """
        super(ContextNetv2, self).__init__()

        self.input_size = input_size
        self.hidden_params = hidden_params
        self.feat_dim = feat_dim
        self.n_prototypes = n_prototypes

        lst_modules = []
        input_dim = input_size
        for n, n_hidden_units in enumerate(hidden_params):
            lst_modules.append(nn.Linear(input_dim, n_hidden_units))
            lst_modules.append(nn.ReLU())
            input_dim = n_hidden_units

        # Output Units
        lst_modules.append(nn.Linear(input_dim, feat_dim))
        self.feat = nn.Sequential(*lst_modules)
        self.prototype = nn.Linear(feat_dim, n_prototypes, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Calls the Xavier parameter initialization function.
        """
        self.feat.apply(xavier)
        self.prototype.apply(xavier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (output_size)
        """
        x = x.view(-1, num_flat_features(x))
        feat = self.feat(x)
        assignment = self.prototype(feat)
        return feat, assignment


class ContextNetv3(nn.Module):
    """
    Network composed of two hidden layers, each containing 100 ReLU activations.
    Designed for the MNIST dataset.
    """

    def __init__(self, input_size: int, hidden_params: tuple, feat_dim: int, n_prototypes: int) -> None:
        """
        Instantiates the layers of the network.
        :param input_size: the size of the input data
        :param output_size: the size of the output
        """
        super(ContextNetv3, self).__init__()

        self.input_size = input_size
        self.hidden_params = hidden_params
        self.feat_dim = feat_dim
        self.n_prototypes = n_prototypes

        lst_modules = []
        input_dim = input_size
        for n, n_hidden_units in enumerate(hidden_params):
            lst_modules.append(nn.Linear(input_dim, n_hidden_units))
            lst_modules.append(nn.ReLU())
            input_dim = n_hidden_units

        # Output Units
        lst_modules.append(nn.Linear(input_dim, feat_dim))
        self.feat = nn.Sequential(*lst_modules)
        self.prototype = nn.Linear(feat_dim, n_prototypes, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Calls the Xavier parameter initialization function.
        """
        self.feat.apply(xavier)
        self.prototype.apply(xavier)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (output_size)
        """
        x = x.view(-1, num_flat_features(x))
        feat = self.feat(x)
        out = self.prototype(feat)
        assignment = out.argmax(dim=1)
        context = self.prototype.weight[assignment]
        return out, context


def normalize(x):
    x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
    x_normalized = x.div(x_norm + 0.00001)
    return x_normalized


class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias=False)
        self.scale_factor = 10

    def forward(self, x):
        L_norm = torch.norm(self.L.weight, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
        cos_dist = torch.mm(x, self.L.weight.div(L_norm + 0.00001).transpose(0, 1))
        scores = self.scale_factor * (cos_dist)
        return scores


class ContextNetv4(nn.Module):
    """
    Network composed of two hidden layers, each containing 100 ReLU activations.
    Designed for the MNIST dataset.
    """

    def __init__(self, input_size: int, hidden_params: tuple, feat_dim: int, n_prototypes: int) -> None:
        """
        Instantiates the layers of the network.
        :param input_size: the size of the input data
        :param output_size: the size of the output
        """
        super(ContextNetv4, self).__init__()

        self.input_size = input_size
        self.hidden_params = hidden_params
        self.feat_dim = feat_dim
        self.n_prototypes = n_prototypes

        lst_modules = []
        input_dim = input_size
        for n, n_hidden_units in enumerate(hidden_params):
            lst_modules.append(nn.Linear(input_dim, n_hidden_units))
            lst_modules.append(nn.ReLU())
            input_dim = n_hidden_units

        # Output Units
        lst_modules.append(nn.Linear(input_dim, feat_dim))
        self.feat = nn.Sequential(*lst_modules)
        # self.prototype = nn.Linear(feat_dim, n_prototypes, bias=False)
        self.prototype = distLinear(feat_dim, n_prototypes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Calls the Xavier parameter initialization function.
        """
        self.feat.apply(xavier)
        self.prototype.apply(xavier)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (output_size)
        """
        x = x.view(-1, num_flat_features(x))
        feat = self.feat(x)
        feat_norm = torch.norm(feat, p=2, dim=1).unsqueeze(1).expand_as(feat)
        feat = feat.div(feat_norm + 0.00001)

        out = self.prototype(feat)
        assignment = out.argmax(dim=1)

        L_norm = torch.norm(self.prototype.L.weight, p=2, dim=1).unsqueeze(1).expand_as(self.prototype.L.weight.data)
        centroids = self.prototype.L.weight.div(L_norm + 0.00001)

        context = centroids[assignment]
        return assignment, context, feat

# model = ContextNetv4(784, (500, ), 128, 10)
# input = torch.randn(64, 784)
# out, context = model(input)
# context_id = assignment.argmax(dim=1)
# model.prototype.weight[context_id].shape
