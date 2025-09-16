#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition abugoot
#SBATCH --mem 50GB
#SBATCH -o logs/o_time_eval_quick
#SBATCH -e logs/e_time_eval_quick

export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="c72e34cd8dc67f7220e3517232e86861cd5c537b"

#python main.py experiment=synthetic_protein training.num_epochs=1000 training.early_stopping=false
python virus_month_idx_eval2.py