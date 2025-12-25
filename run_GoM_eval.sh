#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --partition ou_bcs_high
#SBATCH --mem 20GB
#SBATCH -o logs/o
#SBATCH -e logs/e

export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="c72e34cd8dc67f7220e3517232e86861cd5c537b"

python main.py experiment=snapMMD