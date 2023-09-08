# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from backbone.dendrites.utils.sparse_weights import rezero_weights


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via Bio-ANN.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_dale_active_dendrites_args(parser)
    # Replay
    parser.add_argument('--buffer_sampling', type=str, default='reservoir', choices=['uniform', 'reservoir', 'reservoir_tb'])
    parser.add_argument('--use_buffer_context', type=int, default=0, help='Penalty weight.')
    parser.add_argument('--alpha', type=float, required=True, help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True, help='Penalty weight.')
    # Synaptic Intelligence
    parser.add_argument('--sc_weight', type=float, default=0, help='surrogate loss weight parameter c')
    parser.add_argument('--gamma', type=float, default=0.1, help='xi parameter for EWC online')
    parser.add_argument('--apply_to_dendrites', type=int, default=1, help='Apply SI regularization to dendrites')
    parser.add_argument('--wei_si_weight', type=float, default=1)
    parser.add_argument('--wix_si_weight', type=float, default=1)
    # Heterogeneous Dropout
    parser.add_argument('--apply_heterogeneous_dropout', type=int, default=0)
    parser.add_argument('--dropout_alpha', type=float, nargs='*', default=[0, 0])
    parser.add_argument('--disable_dropout_on_replay', type=float, default=1)
    # Hebbian Learning
    parser.add_argument('--hebbian_update', type=int, default=0)
    parser.add_argument('--apply_hebb_to_buff_context', type=int, default=1)
    parser.add_argument('--hebbian_lr', type=float, default=0.0003)
    # Different LR for Inhibitory weights
    parser.add_argument('--lr_wix', type=float, default=0.3)
    parser.add_argument('--lr_wei', type=float, default=0.3)
    return parser


class BioANN(ContinualModel):
    NAME = 'bio_ann'

    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(BioANN, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.current_task = 0

        # Synaptic Intelligence
        self.apply_to_dendrites = self.args.apply_to_dendrites
        self.checkpoint = self.net.get_si_params(self.apply_to_dendrites).data.clone().to(self.device)
        self.big_omega = None
        self.small_omega = 0

        # Contexts
        self.dim_context = args.dim_context
        self.input_size = args.input_units
        self.learn_context = args.learn_context

        # Tensor for accumulating each task's prototype vector
        self.contexts = torch.zeros((0, self.dim_context))
        self.contexts = self.contexts.to(self.device)
        self.train_context_fn = None
        self.infer_context_fn = None

        # Multiple Learning rates
        self.lr_ei = self.args.lr
        self.lr_ix = self.args.lr

        # Dropout
        if self.args.apply_heterogeneous_dropout:
            self.net.apply_heterogeneous_dropout = True
        else:
            self.net.apply_heterogeneous_dropout = False

        if self.learn_context:

            # Store "exemplars" for each context vector as a list of Torch Tensors;
            # these are used to perform statistical tests against a new batch of data to
            # determine if that new batch corresponds to the same task
            self.clusters = []

            # `contexts` needs to be a mutable data type in order to be modified by a
            # nested function (below), so it is simply wrapped in a 1-element list
            self.contexts = [self.contexts]

            # This list keeps track of how many exemplars have been used to compute each
            # context vector since 1) we compute a weighted average, and 2) most
            # exemplars are discarded for memory efficiency
            self.contexts_n = []

            # In order to perform statistical variable transformations (below), there
            # are restrictions on the dimensionality of the input, so subindices
            # randomly sample features and discard others
            self.subindices = np.random.choice(range(self.input_size), size=self.dim_context, replace=False)
            self.subindices.sort()

        else:

            # Since the prototype vector is an element-wise mean of individual data
            # samples it's necessarily the same dimension as the input
            assert self.dim_context == self.input_size, \
                ("For prototype experiments `dim_context` must match `input_size`")

    def observe(self, inputs, labels, not_aug_inputs):

        if not hasattr(self, 'classes_so_far'):
            self.register_buffer('classes_so_far', labels.unique().to('cpu'))
        else:
            self.register_buffer('classes_so_far', torch.cat((
                self.classes_so_far, labels.to('cpu'))).unique())

        self.net.train()

        real_batch_size = inputs.shape[0]
        inputs = inputs.view(real_batch_size, -1)
        self.net.zero_grad(set_to_none=True)

        loss = 0
        buf_contexts = None
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_logits, buf_contexts = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)

            buf_inputs = buf_inputs.view(buf_inputs.shape[0], -1)

            if not self.args.use_buffer_context:
                buf_contexts = self.infer_context_fn(buf_inputs)

            buf_output = self.net(buf_inputs, buf_contexts, disable_dropout=True)
            loss = self.args.alpha * self.loss(buf_output, buf_labels) + \
                   self.args.beta * F.mse_loss(buf_output, buf_logits)

        contexts = self.train_context_fn(inputs)
        contexts = contexts.to(inputs.device)

        outputs = self.net(inputs, contexts)
        penalty = self.penalty()
        loss += self.loss(outputs, labels) + self.args.sc_weight * penalty

        if torch.isnan(loss):
            raise ValueError('NAN Loss')

        loss.backward()

        nn.utils.clip_grad.clip_grad_value_(self.net.parameters(), 1)
        self.net.update(lr=self.args.lr, lr_ei=self.lr_ei, lr_ix=self.lr_ix)

        if self.args.hebbian_update:
            if buf_contexts is not None and self.args.apply_hebb_to_buff_context and self.args.alpha > 0:
                contexts = torch.cat((contexts, buf_contexts))
            self.net.hebbian_update(lr=self.args.hebbian_lr, context=contexts)

        self.small_omega += self.args.lr * self.net.get_si_grads(self.apply_to_dendrites).data ** 2
        # self.net.apply(rezero_weights)
        self.net.rezero_weights()

        if self.args.buffer_sampling == 'reservoir':
            self.buffer.add_data(
                examples=not_aug_inputs,
                labels=labels,
                logits=outputs.data,
                contexts=contexts.detach(),
            )

        return loss.item()

    def begin_task(self, dataset) -> None:
        print('=' * 30)
        print('Creating Context Vectors')
        print('=' * 30)
        self.prototype_context(dataset)

        if self.learn_context:
            self.infer_context_fn = self.infer_prototype(self.contexts[0], self.subindices)
        else:
            self.infer_context_fn = self.infer_prototype(self.contexts)

    def end_task(self, dataset) -> None:

        if self.args.buffer_sampling == 'uniform':
            with torch.no_grad():
                self.fill_buffer(dataset, self.current_task)
        elif self.args.buffer_sampling == 'reservoir_tb':
            with torch.no_grad():
                self.reservoir_sampling(dataset, self.current_task)

        if self.learn_context:
            self.infer_context_fn = self.infer_prototype(self.contexts[0], self.subindices)
        else:
            self.infer_context_fn = self.infer_prototype(self.contexts)

        # Calculate the Dropout keep Probabilities
        for layer_idx in range(self.args.n_hidden):
            activation_counts = self.net._activation_counts[f'layer_{layer_idx}']
            max_act = torch.max(activation_counts)
            self.net._keep_probs[f'layer_{layer_idx}'] = torch.exp(-activation_counts * self.args.dropout_alpha[layer_idx] / max_act)

            activation_counts = self.net._activation_counts[f'layer_{layer_idx}_classwise']
            max_act = torch.max(activation_counts, dim=1)[0]
            self.net._keep_probs[f'layer_{layer_idx}_classwise'] = 1 - torch.exp(-activation_counts * self.args.dropout_alpha[layer_idx] / max_act[:, None])

        self.current_task += 1

        # Update Big Omega
        if self.big_omega is None:
            self.big_omega = torch.zeros_like(self.net.get_si_params(self.apply_to_dendrites)).to(self.device)
        self.big_omega += self.small_omega / ((self.net.get_si_params(self.apply_to_dendrites).data - self.checkpoint) ** 2 + self.args.gamma)

        # store parameters checkpoint and reset small_omega
        self.checkpoint = self.net.get_si_params(self.apply_to_dendrites).data.clone()
        self.small_omega = 0

        # Adjust big omega
        self.adjust_big_omega()

        # Caclulate the Dropout keep Probabilities
        for layer_idx in range(self.args.n_hidden):
            activation_counts = self.net._activation_counts[f'layer_{layer_idx}']
            max_act = torch.max(activation_counts)
            self.net._keep_probs[f'layer_{layer_idx}'] = torch.exp(-activation_counts * self.args.dropout_alpha[layer_idx] / max_act)

            activation_counts = self.net._activation_counts[f'layer_{layer_idx}_classwise']
            max_act = torch.max(activation_counts, dim=1)[0]
            self.net._keep_probs[f'layer_{layer_idx}_classwise'] = 1 - torch.exp(-activation_counts * self.args.dropout_alpha[layer_idx] / max_act[:, None])


        self.current_task += 1

        # Reduce learning rate of inhibitory units
        self.lr_ei = self.args.lr_wei
        self.lr_ix = self.args.lr_wix

    # =========================================================================
    # Synaptic Intelligence
    # =========================================================================
    def penalty(self):
        if self.big_omega is None:
            return torch.tensor(0.0).to(self.device)
        else:
            penalty = (self.adj_big_omega * ((self.net.get_si_params(self.apply_to_dendrites) - self.checkpoint) ** 2)).sum()

        return penalty

    def adjust_big_omega(self):
        self.adj_big_omega = torch.zeros_like(self.big_omega)
        start_idx = 0
        for name, param in self.net.named_parameters():
            if self.apply_to_dendrites or "segment" not in name:
                param_count = torch.numel(param)
                end_idx = start_idx + param_count
                # print(name, self.big_omega[start_idx: end_idx].mean())
                # print(name, param.shape, param_count, start_idx, end_idx)
                if 'Wei' in name:
                    self.adj_big_omega[start_idx:end_idx] = self.args.wei_si_weight * self.big_omega[start_idx: end_idx]
                elif 'Wix' in name:
                    self.adj_big_omega[start_idx:end_idx] = self.args.wix_si_weight * self.big_omega[start_idx: end_idx]
                else:
                    self.adj_big_omega[start_idx:end_idx] = self.big_omega[start_idx: end_idx]
                start_idx += param_count

    def count_params(self, param_shape):
        count = 1
        for dim in param_shape:
            count *= dim
        return count

    # =========================================================================
    # Replay Buffer
    # =========================================================================
    def fill_buffer(self, dataset, t_idx: int) -> None:
        """
        :param mem_buffer: the memory buffer
        :param dataset: the dataset from which take the examples
        :param t_idx: the task index
        """

        mode = self.net.training
        self.net.eval()
        if hasattr(self, 'context_network'):
            self.context_network.eval()

        classes_so_far = len(self.classes_so_far)
        if dataset.SETTING == 'domain-il':
            classes_so_far += t_idx * classes_so_far

        new_samples_per_class = self.buffer.buffer_size // classes_so_far

        if dataset.SETTING == 'domain-il':
            buf_samples_per_class = t_idx * new_samples_per_class
        else:
            buf_samples_per_class = new_samples_per_class

        print('=' * 30)
        print('Filling the Buffer with samples per class', buf_samples_per_class, new_samples_per_class)
        print('=' * 30)

        if t_idx > 0:
            # 1) First, subsample prior classes
            buf_x, buf_y, buf_l, buf_c = self.buffer.get_all_data()

            self.buffer.empty()
            for _y in buf_y.unique():
                idx = (buf_y == _y)
                _y_x, _y_y, _y_l, _y_c = buf_x[idx], buf_y[idx], buf_l[idx], buf_c[idx]
                sample_perm = np.random.permutation(_y_y.shape[0])
                _y_x, _y_y, _y_l, _y_c = _y_x[sample_perm], _y_y[sample_perm], _y_l[sample_perm], _y_c[sample_perm]

                self.buffer.add_data(
                    examples=_y_x[:buf_samples_per_class],
                    labels=_y_y[:buf_samples_per_class],
                    logits=_y_l[:buf_samples_per_class],
                    contexts=_y_c[:buf_samples_per_class],
                )

        # 2) Then, fill with current tasks
        loader = dataset.train_loader

        # 2.1 Extract all features
        a_x, a_y, a_l, a_c = [], [], [], []
        for x, y, not_norm_x in loader:
            x, y, not_norm_x = (a.to(self.device) for a in [x, y, not_norm_x])
            a_x.append(not_norm_x)
            a_y.append(y.to('cpu'))

            contexts = self.train_context_fn(x)
            if isinstance(contexts, tuple):
                contexts = contexts[1]

            x = x.view(x.shape[0], -1)
            logits = self.net(x, contexts)

            a_c.append(contexts.cpu())
            a_l.append(logits.cpu())

        a_x, a_y, a_l, a_c = torch.cat(a_x), torch.cat(a_y), torch.cat(a_l), torch.cat(a_c)
        sample_perm = np.random.permutation(a_y.shape[0])
        a_x, a_y, a_l, a_c = a_x[sample_perm], a_y[sample_perm], a_l[sample_perm], a_c[sample_perm]

        for _y in a_y.unique():
            idx = (a_y == _y)
            _x, _y, _l, _c = a_x[idx], a_y[idx], a_l[idx], a_c[idx]

            self.buffer.add_data(
                examples=_x[:new_samples_per_class].to(self.device),
                labels=_y[:new_samples_per_class].to(self.device),
                logits=_l[:new_samples_per_class].to(self.device),
                contexts=_c[:new_samples_per_class].to(self.device),
            )

        assert len(self.buffer.examples) <= self.buffer.buffer_size
        self.net.train(mode)

    def reservoir_sampling(self, dataset, t_idx: int) -> None:
        """
        :param mem_buffer: the memory buffer
        :param dataset: the dataset from which take the examples
        :param t_idx: the task index
        """

        mode = self.net.training
        self.net.eval()

        if hasattr(self, 'context_network'):
            self.context_network.eval()

        # 2.1 Extract all features
        for x, y, not_norm_x in dataset.train_loader:
            x, y, not_norm_x = (a.to(self.device) for a in [x, y, not_norm_x])

            contexts = self.train_context_fn(x)
            if isinstance(contexts, tuple):
                contexts = contexts[1]

            x = x.view(x.shape[0], -1)
            logits = self.net(x, contexts)

            self.buffer.add_data(
                examples=not_norm_x.to(self.device),
                labels=y.to(self.device),
                logits=logits.to(self.device),
                contexts=contexts.to(self.device),
            )

        assert len(self.buffer.examples) <= self.buffer.buffer_size
        self.net.train(mode)
