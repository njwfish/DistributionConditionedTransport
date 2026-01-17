#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition ou_bcs_normal
#SBATCH --mem 50GB
#SBATCH -o logs/o1
#SBATCH -e logs/e1

export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="c72e34cd8dc67f7220e3517232e86861cd5c537b"

python inspect_trellis_dataset.py