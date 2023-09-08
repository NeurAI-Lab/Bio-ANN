# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import json
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# os.chdir(r'/git/continual_learning/mammoth')
import importlib
from datasets import NAMES as DATASET_NAMES
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args, add_gcil_args
from datasets import ContinualDataset
from utils.continual_training import train as ctrain
from datasets import get_dataset
from models import get_model
from utils.training import train
from utils.best_args import best_args
from utils.conf import set_random_seed
from backbone.DaleNN import build_dale_nn
from backbone.dendrites.modules.dendritic_mlp import DendriticMLP
from backbone.dendrites.modules.dendritic_layers import \
    GatingDendriticLayer, AbsoluteMaxGatingDendriticLayer, OneSegmentDendriticLayer
from backbone.dale_dendrites.dale_dendritic_mlp import DaleDendriticMLP
from backbone.dale_dendrites.sparse_dale_dendritic_mlp import SparseDaleDendriticMLP
from backbone.dale_dendrites.modules.dale_dendritic_layers import \
    DaleAbsoluteMaxGatingDendriticLayer, DaleAbsoluteMaxGatingDendriticLayerWithShunt
from backbone.dendrites.utils.sparse_weights import rezero_weights

dict_dendritic_layers = {
    'gating_dendrite': GatingDendriticLayer,
    'absolute_max_gating': AbsoluteMaxGatingDendriticLayer,
    'one_segment': OneSegmentDendriticLayer,
}

dict_dale_dendritic_layers = {
    'absolute_max_gating_without_shunt': DaleAbsoluteMaxGatingDendriticLayer,
    'absolute_max_gating_with_shunt': DaleAbsoluteMaxGatingDendriticLayerWithShunt,
}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main():
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True, help='Model name.',
                        choices=get_all_models())
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--load_best_args', action='store_true',
                        help='Loads the best arguments for each method, '
                             'dataset and memory buffer.')
    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)

    if args.load_best_args:
        if args.dataset == 'gcil-cifar100':
            add_gcil_args(parser)

        if hasattr(mod, 'Buffer'):
            parser.add_argument('--buffer_size', type=int, required=True,
                                help='The size of the memory buffer.')
        args = parser.parse_args()
        if args.model == 'joint':
            if args.dataset == 'gcil-cifar100':
                best = best_args[args.dataset]['sgd'][args.weight_dist]
            else:
                best = best_args[args.dataset]['sgd']
        else:
            if args.dataset == 'gcil-cifar100':
                best = best_args[args.dataset][args.model][args.weight_dist]
            else:
                best = best_args[args.dataset][args.model]
        if hasattr(args, 'buffer_size'):
            best = best[args.buffer_size]
        else:
            best = best[-1]
        for key, value in best.items():
            setattr(args, key, value)
    else:
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        if args.dataset in ['gcil-cifar100', 'gcil-gcifar100']:
            add_gcil_args(parser)
        args = parser.parse_args()
        print(args)

    if args.seed is not None:
        set_random_seed(args.seed)

    if args.model == 'mer':
        setattr(args, 'batch_size', 1)

    # save arguments
    model_dir = os.path.join(args.output_dir, "saved_models", args.dataset, args.experiment_id)
    args_path = os.path.join(model_dir, "args.txt")
    os.makedirs(model_dir, exist_ok=True)

    z = vars(args).copy()
    with open(args_path, "w") as f:
        f.write("arguments: " + json.dumps(z) + "\n")

    dataset = get_dataset(args)
    if args.backbone == 'default':
        backbone = dataset.get_backbone()

    elif args.backbone == 'dale_nn':
        print('=' * 30)
        print('Building Dale NN')
        print('=' * 30)
        backbone = build_dale_nn(
            input_dim=args.input_units,
            output_dim=args.output_units,
            n_e=args.n_e,
            n_i=args.n_i,
            n_hidden=args.n_hidden,
            c_sgd=args.c_sgd,
            i_iid_i=args.i_iid_i,
            layer_type=args.layer_type
        )
        backbone.init_weights()
        print(backbone)

    elif args.backbone == 'active_dendrites':
        print('=' * 30)
        print('Building Active Dendrites')
        print('=' * 30)
        backbone = DendriticMLP(
            input_size=args.input_units,
            output_size=args.output_units,
            hidden_sizes=args.hidden_sizes,
            num_segments=args.num_segments,
            dim_context=args.dim_context,
            kw=args.kw,
            kw_percent_on=args.kw_percent_on,
            context_percent_on=args.context_percent_on,
            dendrite_weight_sparsity=args.dendrite_weight_sparsity,
            weight_sparsity=args.weight_sparsity,
            weight_init=args.weight_init,
            dendrite_init=args.dendrite_init,
            freeze_dendrites=args.freeze_dendrites,
            dendritic_layer_class=dict_dendritic_layers[args.dendritic_layer_class],
            output_nonlinearity=args.output_nonlinearity,
        )
        print(backbone)

    elif args.backbone == 'dale_active_dendrites':
        print('=' * 30)
        print('Building Dale Active Dendrites')
        print('=' * 30)
        backbone = DaleDendriticMLP(
            input_size=args.input_units,
            output_size=args.output_units,
            hidden_sizes=args.n_hidden * [(args.n_e, args.n_i)],
            num_segments=args.num_segments,
            dim_context=args.dim_context,
            kw=args.kw,
            kw_percent_on=args.kw_percent_on,
            context_percent_on=args.context_percent_on,
            dendrite_weight_sparsity=args.dendrite_weight_sparsity,
            weight_sparsity=args.weight_sparsity,
            weight_init=args.weight_init,
            dendrite_init=args.dendrite_init,
            freeze_dendrites=args.freeze_dendrites,
            output_nonlinearity=args.output_nonlinearity,
            use_shunting=args.use_shunting,
            output_inhib_units=args.output_inhib_units,
            i_iid_i=args.i_iid_i,
        )

        backbone.init_weights()
        print(backbone)

    elif args.backbone == 'sparse_dale_active_dendrites':
        print('=' * 30)
        print('Building Sparse Dale Active Dendrites')
        print('=' * 30)
        backbone = SparseDaleDendriticMLP(
            input_size=args.input_units,
            output_size=args.output_units,
            hidden_sizes=args.n_hidden * [(args.n_e, args.n_i)],
            num_segments=args.num_segments,
            dim_context=args.dim_context,
            kw=args.kw,
            kw_percent_on=args.kw_percent_on,
            context_percent_on=args.context_percent_on,
            dendrite_weight_sparsity=args.dendrite_weight_sparsity,
            weight_sparsity=args.weight_sparsity,
            weight_init=args.weight_init,
            dendrite_init=args.dendrite_init,
            freeze_dendrites=args.freeze_dendrites,
            output_nonlinearity=args.output_nonlinearity,
            use_shunting=args.use_shunting,
            output_inhib_units=args.output_inhib_units,
            i_iid_i=args.i_iid_i,
            apply_heterogeneous_dropout=args.apply_heterogeneous_dropout
        )

        backbone.init_weights()
        backbone.rezero_weights()
        print(backbone)

    else:
        raise ValueError('Invalid Backbone selection')


    params = count_parameters(backbone)
    print('!' * 30)
    print(f'Number of parameters = {params}')
    print('!' * 30)

    loss = dataset.get_loss()

    # Add context Network for Active-DANN with learnable context
    model = get_model(args, backbone, loss, dataset.get_transform())

    results_dir = os.path.join(args.output_dir, "results", dataset.SETTING, args.dataset, args.model, args.experiment_id, "mean_accs.csv")
    if os.path.exists(results_dir):
        print('*' * 30)
        print('Experiment Already trained')
        print(results_dir)
        print('*' * 30)

    else:
        if isinstance(dataset, ContinualDataset):
            train(model, dataset, args)
        else:
            assert not hasattr(model, 'end_task')
            ctrain(args)


if __name__ == '__main__':
    main()
