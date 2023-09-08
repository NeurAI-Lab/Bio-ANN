# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torchvision.transforms as transforms
from datasets.transforms.rotation import Rotation
from torch.utils.data import DataLoader
from backbone.MNISTMLP import MNISTMLP
import torch.nn.functional as F
from datasets.perm_mnist import store_mnist_loaders
from datasets.utils.continual_dataset import ContinualDataset
from argparse import Namespace
import numpy as np


class RotatedMNIST(ContinualDataset):
    NAME = 'rot-mnist'
    SETTING = 'domain-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 20

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        super(RotatedMNIST, self).__init__(args)
        np.random.seed(args.mnist_seed)
        self.rotations = [Rotation() for _ in range(RotatedMNIST.N_TASKS)]
        self.task_id = 0

    def get_data_loaders(self):
        # transform = transforms.Compose((Rotation(), transforms.ToTensor()))
        transform = transforms.Compose((self.rotations[self.task_id], transforms.ToTensor(), transforms.Normalize((0.13062755,), (0.30810780,))))
        train, test = store_mnist_loaders(transform, self)
        self.task_id += 1
        return train, test

    def not_aug_dataloader(self, batch_size):
        return DataLoader(self.train_loader.dataset,
                          batch_size=batch_size, shuffle=True)

    @staticmethod
    def get_backbone():
        return MNISTMLP(28 * 28, RotatedMNIST.N_CLASSES_PER_TASK)

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_denormalization_transform():
        return None
