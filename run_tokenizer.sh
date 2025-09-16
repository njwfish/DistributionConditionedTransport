#!/bin/bash
#SBATCH -t 00:20:00
#SBATCH --gres shard:1
#SBATCH --constraint any-A100
#SBATCH --partition abugoot
#SBATCH --mem 5GB
#SBATCH -o logs/o
#SBATCH -e logs/e

export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="c72e34cd8dc67f7220e3517232e86861cd5c537b"

python main.py experiment=synthetic_protein training.num_epochs=1000 training.early_stopping=false