#!/bin/bash
#SBATCH -t 03:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition abugoot
#SBATCH --mem 50GB
#SBATCH -o logs/o_virus_time_only_posthoc
#SBATCH -e logs/e_virus_time_only_posthoc

export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="c72e34cd8dc67f7220e3517232e86861cd5c537b"

#python main.py experiment=synthetic_protein training.num_epochs=1000 training.early_stopping=false
python main.py experiment=virus_time_only_posthoc training.num_epochs=1000 training.early_stopping=true