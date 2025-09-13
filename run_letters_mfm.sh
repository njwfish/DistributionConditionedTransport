#!/bin/bash
#SBATCH -t 02:00:00
#SBATCH --gres shard:1
#SBATCH --constraint any-A100
#SBATCH --partition abugoot
#SBATCH --mem 10GB
#SBATCH -o logs/om_letters 
#SBATCH -e logs/em_letters 

export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="c72e34cd8dc67f7220e3517232e86861cd5c537b"

# Ensure we run from the project root and import local modules first
PROJECT_ROOT="/orcd/archive/abugoot/001/Projects/paolo/tde_main"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"


python main.py experiment=letters_mfm training.num_epochs=10000 +model.source_only=true wandb=offline
