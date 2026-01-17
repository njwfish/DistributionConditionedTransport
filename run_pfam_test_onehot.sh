#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition ou_bcs_low
#SBATCH --mem 40GB
#SBATCH -o logs/otoh_new_oh2
#SBATCH -e logs/etoh_new_oh2

export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="c72e34cd8dc67f7220e3517232e86861cd5c537b"

python test_pfam_evaluation_onehot.py \
    --test_pt_file data/pfam/pfam_tokenized_data_test_1000.pt \
    --train_pt_file data/pfam/pfam_tokenized_data_1000.pt \
    --output_dir outputs/pfam_onehot_4cabc7320b996f6208ab773561a32e35 \
    -n 50
    #--oracle_mode
    #--random_mapping
    #--output_dir outputs/pfam_onehot_92bed41b2e6c11218f65a5b7b317f5be