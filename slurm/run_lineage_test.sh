#!/bin/bash
#SBATCH --chdir=/orcd/archive/abugoot/001/Projects/paolo/main_tde
#SBATCH -t 00:10:00
#SBATCH --gres shard:1
#SBATCH --constraint any-A100
#SBATCH --partition abugoot
#SBATCH --mem 100GB
#SBATCH -o logs/ol
#SBATCH -e logs/el

export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="c72e34cd8dc67f7220e3517232e86861cd5c537b"

python main.py experiment=lineage