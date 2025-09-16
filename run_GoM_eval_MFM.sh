#!/bin/bash
#SBATCH -t 00:05:00
#SBATCH --gres shard:1
#SBATCH --constraint any-A100
#SBATCH --partition abugoot
#SBATCH --mem 1GB
#SBATCH -o logs/oge_MFM_PBMC
#SBATCH -e logs/ege_MFM_PBMC

export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="c72e34cd8dc67f7220e3517232e86861cd5c537b"

python GoM_eval_MFM.py