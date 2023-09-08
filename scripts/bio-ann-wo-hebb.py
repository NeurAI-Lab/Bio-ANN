import yaml
import os

dataset_dict = {
    'perm-mnist-n': 'pm',
    'rot-mnist-n': 'rm',
    'seq-mnist': 'sm',
}

params = {
    'perm-mnist-n':
        {
            5: {
                'n_epoch': 3,
                'batch_size': 128,
                'weight_sparsity': 0.5,
                'kw_percent_on': '0.05 0.05',
                'lr': 0.3,
                'lr_wix': 0.03,
                'lr_wei': 0.03,
            },
            10: {
                'n_epoch': 3,
                'batch_size': 128,
                'weight_sparsity': 0.5,
                'kw_percent_on': '0.05 0.05',
                'lr': 0.3,
                'lr_wix': 0.03,
                'lr_wei': 0.03,
            },
            20: {
                'n_epoch': 3,
                'batch_size': 128,
                'weight_sparsity': 0.5,
                'kw_percent_on': '0.05 0.05',
                'lr': 0.3,
                'lr_wix': 0.03,
                'lr_wei': 0.003,
            },
        },
    'rot-mnist-n':
        {
            5: {
                'n_epoch': 3,
                'batch_size': 128,
                'weight_sparsity': 0.5,
                'kw_percent_on': '0.05 0.05',
                'lr': 0.3,
                'lr_wix': 0.03,
                'lr_wei': 0.003,
            },
            10: {
                'n_epoch': 3,
                'batch_size': 128,
                'weight_sparsity': 0.5,
                'kw_percent_on': '0.05 0.05',
                'lr': 0.3,
                'lr_wix': 0.03,
                'lr_wei': 0.003,
            },
            20: {
                'n_epoch': 3,
                'batch_size': 128,
                'weight_sparsity': 0.5,
                'kw_percent_on': '0.05 0.05',
                'lr': 0.3,
                'lr_wix': 0.003,
                'lr_wei': 0.0003,
            },
        },
    'seq-mnist':
        {
            5: {
                'n_epoch': 5,
                'batch_size': 128,
                'weight_sparsity': 0.5,
                'kw_percent_on': '0.05 0.05',
                'lr': 0.3,
                'lr_wix': 0.03,
                'lr_wei': 0.003,
            },
        }
}

lst_dataset = [('perm-mnist-n', 5), ('rot-mnist-n', 5), ('perm-mnist-n', 10), ('rot-mnist-n', 10), ('perm-mnist-n', 20), ('rot-mnist-n', 20)]
lst_replay = [((1, 0.5), 0)]

context_percent_on = 0.1
dendrite_weight_sparsity = 0.5
n_e, n_i = (1844, 204)

num_runs = 3
start_seed = 10
count = 0

lst_hebbian_lr = [3e-8]
lst_dropout_alpha = ['0.1 0.1']
lst_sc_weight = [0.1]
lst_si_scale = [(10, 10), ]

for seed in range(start_seed, start_seed + num_runs):
    for dataset, n_tasks in lst_dataset:
        train_params = params[dataset][n_tasks]
        for (alpha, beta), use_buffer_context in lst_replay:
            for hebbian_lr in lst_hebbian_lr:
                for dropout_alpha in lst_dropout_alpha:
                    for sc_weight in lst_sc_weight:
                        for wix_si_weight, wei_si_weight in lst_si_scale:
                            exp_id = f"adann-{dataset_dict[dataset]}-{n_tasks}-{alpha}-{beta}-{hebbian_lr}-{''.join(dropout_alpha.split(' '))}-{sc_weight}-{wix_si_weight}-{wei_si_weight}-s{seed}"
                            job_args = f"python main.py  \
                                --experiment_id {exp_id} \
                                --seed {seed} \
                                --model bio_ann \
                                --backbone sparse_dale_active_dendrites \
                                --n_e {n_e} \
                                --n_i {n_i} \
                                --buffer_size 500 \
                                --alpha {alpha} \
                                --beta {beta} \
                                --dataset {dataset} \
                                --num_tasks {n_tasks} \
                                --num_segments {n_tasks} \
                                --dim_context 784 \
                                --kw_percent_on {train_params['kw_percent_on']} \
                                --context_percent_on {context_percent_on} \
                                --dendrite_weight_sparsity {dendrite_weight_sparsity} \
                                --weight_sparsity {train_params['weight_sparsity']} \
                                --lr {train_params['lr']} \
                                --lr_wei {train_params['lr_wei']} \
                                --lr_wix {train_params['lr_wix']} \
                                --n_epochs {train_params['n_epoch']} \
                                --batch_size {train_params['batch_size']} \
                                --minibatch_size {train_params['batch_size']} \
                                --hebbian_update 0 \
                                --hebbian_lr {hebbian_lr} \
                                --apply_heterogeneous_dropout 1 \
                                --dropout_alpha {dropout_alpha} \
                                --sc_weight {sc_weight} \
                                --wix_si_weight {wix_si_weight} \
                                --wei_si_weight {wei_si_weight} \
                                --gamma 0.1 \
                                --use_buffer_context {use_buffer_context} \
                                --output_dir output/bio_ann/ \
                                --tensorboard \
                                --csv_log \
                                --mnist_seed 0 \
                                "
                            count += 1
                            os.system(job_args)

print('%s jobs counted' % count)
