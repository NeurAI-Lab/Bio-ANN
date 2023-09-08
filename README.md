# Bio-ANN

The official repository for TMLR and COLLAs'23 paper "A Study of Biologically Plausible Neural Network: The Role and Interactions of Brain-Inspired Mechanisms in Continual Learning""
We extended the [CLS-ER Repo](https://github.com/NeurAI-Lab/CLS-ER) framework with our method (Bio-ANN)

## Usage
Please see the scripts folder for specific commands

    ```
    python main.py  \
        --experiment_id {exp_id} \
        --model bio_ann \
        --backbone sparse_dale_active_dendrites \
        --buffer_size {buffer_size} \
        --alpha {alpha} \
        --beta {beta} \
        --dataset {dataset} \
        --num_tasks {n_tasks} \
        --num_segments {n_tasks} \
        --hebbian_update 1 \
        --hebbian_lr {hebbian_lr} \
        --apply_heterogeneous_dropout 1 \
        --dropout_alpha {dropout_alpha} \
        --sc_weight {sc_weight} \
        --wix_si_weight {wix_si_weight} \
        --wei_si_weight {wei_si_weight} \
        --output_dir {/path/to/output/folder} \
        --tensorboard \
        --csv_log \
        --mnist_seed 0 \

## Requirements

- torch==1.7.0

- torchvision==0.9.0

- quadprog==0.1.7

## Cite Our Work

If you find the code useful in your research, please consider citing our paper:

    @article{sarfraz2023study,
      title={A Study of Biologically Plausible Neural Network: The Role and Interactions of Brain-Inspired Mechanisms in Continual Learning},
      author={Sarfraz, Fahad and Arani, Elahe and Zonooz, Bahram},
      journal={Transactions on Machine Learning Research},
      year={2023}
    }
