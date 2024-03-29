# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.optim import SGD

from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets.utils.validation import ValidationDataset
from utils.status import progress_bar
from backbone.DaleNN import build_dale_nn
import torch
import numpy as np
import math


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Joint training: a strong, simple baseline.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_dale_nn_args(parser)
    return parser


class JointDale(ContinualModel):
    NAME = 'joint_dale'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(JointDale, self).__init__(backbone, loss, args, transform)
        self.old_data = []
        self.old_labels = []
        self.current_task = 0

    def end_task(self, dataset):
        if dataset.SETTING != 'domain-il':
            self.old_data.append(dataset.train_loader.dataset.data)
            self.old_labels.append(torch.tensor(dataset.train_loader.dataset.targets))
            self.current_task += 1

            # # for non-incremental joint training
            if len(dataset.test_loaders) != dataset.N_TASKS: return

            # reinit network
            if self.args.backbone == 'default':
                self.net = dataset.get_backbone()
            elif self.args.backbone == 'dale_nn':
                print('=' * 30)
                print('Building Dale NN')
                print('=' * 30)
                self.net = build_dale_nn(
                    input_dim=self.args.input_units,
                    output_dim=self.args.output_units,
                    n_e=self.args.n_e,
                    n_i=self.args.n_i,
                    n_hidden=self.args.n_hidden,
                    c_sgd=self.args.c_sgd,
                    i_iid_i=self.args.i_iid_i,
                )
                self.net.init_weights()

            self.net.to(self.device)
            self.net.train()
            self.opt = SGD(self.net.parameters(), lr=self.args.lr)

            # prepare dataloader
            all_data, all_labels = None, None
            for i in range(len(self.old_data)):
                if all_data is None:
                    all_data = self.old_data[i]
                    all_labels = self.old_labels[i]
                else:
                    all_data = np.concatenate([all_data, self.old_data[i]])
                    all_labels = np.concatenate([all_labels, self.old_labels[i]])

            temp_dataset = ValidationDataset(all_data, all_labels, transform=dataset.TRANSFORM)
            loader = torch.utils.data.DataLoader(temp_dataset, batch_size=self.args.batch_size, shuffle=True)

            # train
            for e in range(self.args.n_epochs):
                for i, batch in enumerate(loader):
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    if self.args.backbone == 'dale_nn':
                        self.net.zero_grad()
                    else:
                        self.opt.zero_grad()
                    outputs = self.net(inputs)
                    loss = self.loss(outputs, labels.long())
                    loss.backward()
                    if self.args.backbone == 'dale_nn':
                        self.net.update(lr=self.args.lr)
                    else:
                        self.opt.step()
                    progress_bar(i, len(loader), e, 'J', loss.item())
        else:
            self.old_data.append(dataset.train_loader)
            # train
            if len(dataset.test_loaders) != dataset.N_TASKS: return
            loader_caches = [[] for _ in range(len(self.old_data))]
            sources = torch.randint(5, (128,))
            all_inputs = []
            all_labels = []
            for source in self.old_data:
                for x, l, _ in source:
                    all_inputs.append(x)
                    all_labels.append(l)
            all_inputs = torch.cat(all_inputs)
            all_labels = torch.cat(all_labels)
            bs = self.args.batch_size
            for e in range(self.args.n_epochs):
                order = torch.randperm(len(all_inputs))
                for i in range(int(math.ceil(len(all_inputs) / bs))):
                    inputs = all_inputs[order][i * bs: (i+1) * bs]
                    labels = all_labels[order][i * bs: (i+1) * bs]
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    if self.args.backbone == 'dale_nn':
                        self.net.zero_grad()
                    else:
                        self.opt.zero_grad()
                    outputs = self.net(inputs)
                    loss = self.loss(outputs, labels.long())
                    loss.backward()
                    if self.args.backbone == 'dale_nn':
                        self.net.update(lr=self.args.lr)
                    else:
                        self.opt.step()
                    progress_bar(i, int(math.ceil(len(all_inputs) / bs)), e, 'J', loss.item())

    def observe(self, inputs, labels, not_aug_inputs):
        return 0
