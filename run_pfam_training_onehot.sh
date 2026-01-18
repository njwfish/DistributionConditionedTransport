#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition ou_bcs_low
#SBATCH --mem 100GB
#SBATCH -o logs/ouf_oh_dfm_clan_10
#SBATCH -e logs/euf_oh_dfm_clan_10

export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="c72e34cd8dc67f7220e3517232e86861cd5c537b"

python main.py experiment=pfam_dfm_onehot_clan experiment.data_file=pfam_tokenized_data_clan_10.pt
