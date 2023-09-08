import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from copy import deepcopy
from torch import nn
from torch.nn import functional as F


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Complementary Learning Systems Based Experience Replay')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    # Consistency Regularization Weight
    parser.add_argument('--reg_weight', type=float, default=0.1)

    # Stable Model parameters
    parser.add_argument('--stable_model_update_freq', type=float, default=0.70)
    parser.add_argument('--stable_model_alpha', type=float, default=0.999)

    # Plastic Model Parameters
    parser.add_argument('--plastic_model_update_freq', type=float, default=0.90)
    parser.add_argument('--plastic_model_alpha', type=float, default=0.999)

    # Synaptic Consolidation
    parser.add_argument('--penalty_weight', type=float, default=10)
    parser.add_argument('--fisher_alpha', type=float, default=0.999)
    parser.add_argument('--fisher_update_freq', type=float, default=0.90)

    return parser


# =============================================================================
# Mean-ER
# =============================================================================
class SYCLSER(ContinualModel):
    NAME = 'syclser'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(SYCLSER, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        # Initialize plastic and stable model
        self.plastic_model = deepcopy(self.net).to(self.device)
        self.stable_model = deepcopy(self.net).to(self.device)
        # set regularization weight
        self.reg_weight = args.reg_weight
        # set parameters for plastic model
        self.plastic_model_update_freq = args.plastic_model_update_freq
        self.plastic_model_alpha = args.plastic_model_alpha
        # set parameters for stable model
        self.stable_model_update_freq = args.stable_model_update_freq
        self.stable_model_alpha = args.stable_model_alpha

        self.consistency_loss = nn.MSELoss(reduction='none')
        self.current_task = 0
        self.global_step = 0

        # Regularization Params
        self.logsoft = nn.LogSoftmax(dim=1)
        self.fisher_update_freq = args.fisher_update_freq
        self.fisher_alpha = args.fisher_alpha
        self.fish = None
        self.adjusted_fish = None

    def observe(self, inputs, labels, not_aug_inputs):

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        loss = 0

        if not self.buffer.is_empty():

            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)

            stable_model_logits = self.stable_model(buf_inputs)
            plastic_model_logits = self.plastic_model(buf_inputs)

            stable_model_prob = F.softmax(stable_model_logits, 1)
            plastic_model_prob = F.softmax(plastic_model_logits, 1)

            label_mask = F.one_hot(buf_labels, num_classes=stable_model_logits.shape[-1]) > 0
            sel_idx = stable_model_prob[label_mask] > plastic_model_prob[label_mask]
            sel_idx = sel_idx.unsqueeze(1)

            ema_logits = torch.where(
                sel_idx,
                stable_model_logits,
                plastic_model_logits,
            )

            l_cons = torch.mean(self.consistency_loss(self.net(buf_inputs), ema_logits.detach()))
            l_reg = self.args.reg_weight * l_cons
            loss += l_reg

            if hasattr(self, 'writer'):
                self.writer.add_scalar(f'Task {self.current_task}/l_cons', l_cons.item(), self.iteration)
                self.writer.add_scalar(f'Task {self.current_task}/l_reg', l_reg.item(), self.iteration)

            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

            # Log values
            if hasattr(self, 'writer'):
                self.writer.add_scalar(f'Task {self.current_task}/l_reg', l_reg.item(), self.iteration)

        outputs = self.net(inputs)
        ce_loss = self.loss(outputs, labels)
        penalty_loss = self.args.penalty_weight * self.penalty()
        loss = loss + ce_loss + penalty_loss

        # Log values
        if hasattr(self, 'writer'):
            self.writer.add_scalar(f'Task {self.current_task}/ce_loss', ce_loss.item(), self.iteration)
            self.writer.add_scalar(f'Task {self.current_task}/penalty_loss', penalty_loss.item(), self.iteration)
            self.writer.add_scalar(f'Task {self.current_task}/loss', loss.item(), self.iteration)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(
            examples=not_aug_inputs,
            labels=labels[:real_batch_size],
        )

        # Update the ema model
        self.global_step += 1
        if torch.rand(1) < self.plastic_model_update_freq:
            self.update_plastic_model_variables()

        if torch.rand(1) < self.stable_model_update_freq:
            self.update_stable_model_variables()

        if torch.rand(1) < self.fisher_update_freq:
            self.update_fisher()

        return loss.item()

    def update_plastic_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1), self.plastic_model_alpha)
        for ema_param, param in zip(self.plastic_model.parameters(), self.net.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def update_stable_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1),  self.stable_model_alpha)
        for ema_param, param in zip(self.stable_model.parameters(), self.net.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def update_fisher(self):
        fish = torch.zeros_like(self.net.get_params())
        inputs, labels = self.buffer.get_all_data(transform=self.transform)

        for ex, lab in zip(inputs, labels):
            self.stable_model.zero_grad()
            output = self.stable_model(ex.unsqueeze(0))
            loss = - F.nll_loss(self.logsoft(output), lab.unsqueeze(0), reduction='none')
            exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
            loss = torch.mean(loss)
            loss.backward()
            fish += exp_cond_prob * self.stable_model.get_grads() ** 2

        fish /= len(inputs)

        if self.fish is None:
            self.fish = fish
        else:
            # Use EMA to update the fisher information matrix
            fisher_alpha = min(1 - 1 / (self.global_step + 1), self.fisher_alpha)
            self.fish = (fisher_alpha * self.fish) + ((1 - fisher_alpha) * fish)

        # Adjust Fisher
        self.adjusted_fish = self.adjust_fisher()

    def penalty(self):
        if self.fish is None:
            return torch.tensor(0.0).to(self.device)
        else:
            penalty = (self.adjusted_fish * ((self.net.get_params() - self.stable_model.get_params()) ** 2)).sum()
            return penalty

    def count_params(self, param_shape):
        count = 1
        for dim in param_shape:
            count *= dim
        return count

    def adjust_fisher(self):
        adjusted_fisher = self.fish.clone()
        start_idx = 0
        total_params = 0
        for name, param in self.net.named_parameters():
            param_count = self.count_params(list(param.shape))
            if ('conv' in name or 'shortcut' in name) and len(param.shape) > 1:
                # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                # print(name, list(param.shape), start_idx, start_idx + param_count)
                # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                num_filters = param.shape[0]
                params_per_filter = param.shape[1] * param.shape[2] * param.shape[3]

                for filter_idx in range(num_filters):
                    filter_start_idx = start_idx + (filter_idx * params_per_filter)
                    filter_end_idx = start_idx + ((filter_idx + 1) * params_per_filter)
                    filter_params = adjusted_fisher[filter_start_idx: filter_end_idx]
                    imp_val = torch.mean(filter_params)
                    adjusted_fisher[filter_start_idx: filter_end_idx] = imp_val

            start_idx += param_count
            total_params += param_count

        return adjusted_fisher
