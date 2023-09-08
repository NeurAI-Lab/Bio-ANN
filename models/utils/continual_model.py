# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch.optim import SGD, Adam
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device


class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                args: Namespace, transform: torchvision.transforms) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)
        # self.opt = Adam(self.net.parameters(), lr=self.args.lr)
        self.device = get_device()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        pass

    def prototype_context(self, dataset):
        if self.learn_context:
            self.train_context_fn = self.construct_prototype(
                self.clusters, self.contexts, self.contexts_n, self.subindices
            )
        else:
            # Find a context vector by computing the prototype of all training examples
            self.context_vector = self.compute_prototype(dataset.train_loader).to(self.device)
            self.contexts = torch.cat((self.contexts, self.context_vector.unsqueeze(0)))
            self.train_context_fn = self.train_prototype(self.context_vector)

    # =========================================================================
    # Functions for Creating Prototypes from data
    # =========================================================================
    def compute_prototype(self, loader):
        """
        Returns the prototype vector of all samples iterated over in `loader`.
        """
        prototype_vector = torch.zeros([])
        n_prototype = 0

        # for x, _, _ in loader:
        for x in loader:
            if isinstance(x, list):
                x = x[0]
            x = x.flatten(start_dim=1)
            n_x = x.size(0)

            prototype_vector = prototype_vector + x.sum(dim=0)
            n_prototype += n_x

        prototype_vector /= n_prototype
        return prototype_vector

    def train_prototype(self, context_vector):
        """
        Returns a function that takes a batch of training examples and returns the same
        context vector for each
        """

        def _train_prototype(data):
            context = context_vector.repeat((data.size(0), 1))
            return context

        return _train_prototype

    # =========================================================================
    # Functions for Learning Prototypes
    # =========================================================================
    def construct_prototype(self, clusters, contexts, n_samples_per_prototype,
                            subindices, max_samples_per_cluster=256):
        """
        Returns a function that takes a batch of training examples and performs a clustering
        procedure to determine the appropriate context vector. The resulting context vector
        returned by the function is either a) an existing context vector in `contexts` or
        b) simply the prototype of the batch.

        :param clusters: List of Torch Tensors where the item at position i gives the
                         exemplars representing cluster i
        :param contexts: List containing a single Torch Tensor in which row i gives the ith
                         context vector
        :param n_samples_per_context: List of ints where entry i gives the number of
                                      samples used to compute the ith context (i.e.,
                                      `contexts[0][i]`)
        :param subindices: List/Tensor/Array that can index contexts to select subindices;
                           optional
        :param max_samples_per_cluster: Integer giving the maximum number of data samples
                                        per cluster to store for computing statistical tests
        """

        def _construct_prototype(data):

            # The following variables are declared nonlocal since they are modified by this
            # (inner) function
            # nonlocal clusters
            # nonlocal contexts
            # nonlocal n_samples_per_prototype
            data = data[:, self.subindices]

            # Due to memory constraints, each Tensor in `clusters` will contain a maximum
            # number of individual exemplars which are then used to compute the prototype
            max_samples_per_cluster = 256
            cluster_id = None

            for j in range(len(self.clusters)):

                # If already clustered, skip
                if cluster_id is not None:
                    continue

                if self.should_cluster(self.clusters[j], data):
                    cluster_id = j

                    # As clusters grow, keeping all exemplars (i.e., the data samples that
                    # are used to compute prototype) in memory will be problematic; for this
                    # reason we only store `max_samples_per_cluster` examples in memory and
                    # discard the rest; the following code implements exactly this while
                    # ensuring the prototype vector incorporates all observed data samples
                    # even if not stored in memory

                    # Update prototype via weighted averaging: the two weights are 1) the
                    # number of samples that have contributed towards computing the
                    # prototype vectory in memory, and 2) the current batch size
                    n = n_samples_per_prototype[j]
                    n_cluster = self.clusters[j].size(0)
                    n_batch = data.size(0)

                    updated_prototype = n * self.contexts[0][j] + n_batch * data.mean(dim=0)
                    updated_prototype /= (n + n_batch)
                    self.contexts[0][j, :] = updated_prototype

                    n_samples_per_prototype[j] += n_batch

                    # For computation efficiency, drop some samples out of memory

                    # Randomly select which examples in memory will be stored, and which
                    # ones from the batch will be stored
                    p_cluster = n_cluster / (n_cluster + n_batch)
                    p_batch = 1.0 - p_cluster

                    n_retain = int(max_samples_per_cluster * p_cluster)
                    retain_inds = np.random.choice(range(n_cluster), size=n_retain,
                                                   replace=False)

                    n_new = int(max_samples_per_cluster * p_batch)
                    new_inds = np.random.choice(range(n_batch), size=n_new, replace=False)

                    self.clusters[j] = torch.cat((clusters[j][retain_inds],
                                             data[new_inds]))

            if cluster_id is None:

                # No existing cluster is appropriate for the given batch; create new cluster
                self.clusters.append(data[:max_samples_per_cluster, :])
                self.contexts[0] = torch.cat((contexts[0], data.mean(dim=0).unsqueeze(0)))
                n_samples_per_prototype.append(data.size(0))

                cluster_id = len(n_samples_per_prototype) - 1

            return self.contexts[0][cluster_id].repeat((data.size(0), 1))

        return _construct_prototype

    def infer_prototype(self, contexts, subindices=None):
        """
        Returns a function that takes a batch of test examples and returns a
        2D array where row i gives the the prototype vector closest to the ith
        test example.
        """
        def _infer_prototype(data, return_index=False):
            if subindices is not None:
                data = data[:, subindices]
            context = torch.cdist(contexts, data)
            index = context.argmin(dim=0)
            context = contexts[index]
            if return_index:
                return context, index
            return context

        return _infer_prototype

    # =========================================================================
    # Functions for clustering data samples
    # =========================================================================
    def should_cluster(self, set1, set2, p=0.9):
        """
        Returns True iff the multivariate two-sample test that compares samples from set1
        and set2 suggests that they "belong to the same distribution"; False otherwise.

        :param set1: 2D Torch Tensor
        :param set2: 2D Torch Tensor
        :param p: Statistical significance threshold
        """
        p_value = self.two_sample_hotelling_statistic(set1, set2)
        return p_value < p

    def two_sample_hotelling_statistic(self, set1, set2):
        """
        Returns a p-value of whether set1 and set2 share the same underlying data-generating
        process. Note that all matrix inversions in the standard formulation are replaced
        with the Moore-Penrose pseudo-inverse. More details are provided here:

            https://en.wikipedia.org/wiki/Hotelling%27s_T-squared_distribution#Two-sample_st
            atistic

        :param set1: 2D Torch Tensor
        :param set2: 2D Torch Tensor
        """

        # NOTE: The operations performs in this function require float64 datatype since
        # numerical values become extremely small. This requires additional memory and
        # slightly slows down the training process.
        set1 = set1.double()
        set2 = set2.double()

        n1 = set1.size(0)
        n2 = set2.size(0)

        mean1 = set1.mean(dim=0)
        mean2 = set2.mean(dim=0)

        # Sample covariance matrices
        cov1 = torch.matmul((set1 - mean1).T, (set1 - mean1))
        cov1 = cov1 / (n1 - 1)

        cov2 = torch.matmul((set2 - mean2).T, set2 - mean2)
        cov2 = cov2 / (n2 - 1)

        # Unbiased pooled covariance matrix
        cov = (n1 - 1) * cov1 + (n2 - 1) * cov2
        cov = cov / (n1 + n2 - 2)

        # T^2 statistic
        t_squared = torch.matmul((mean1 - mean2).unsqueeze(0), torch.pinverse(cov))
        t_squared = torch.matmul(t_squared, mean1 - mean2)
        t_squared = (n1 * n2 / (n1 + n2)) * t_squared

        # Number of features
        p = set1.size(1)
        n = n1 + n2

        # Transform to F variable
        f_statistic = (n - p - 1) / (p * (n - 2)) * t_squared
        f_statistic = f_statistic.cpu().numpy()
        p_value = f.cdf(f_statistic, p, n - p - 1)

        return p_value
