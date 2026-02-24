#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition ou_bcs_normal
#SBATCH --mem 100GB
#SBATCH -o logs/otr_train_scgen3_%a
#SBATCH -e logs/etr_train_scgen3_%a
#SBATCH --array=0-4:1

export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="c72e34cd8dc67f7220e3517232e86861cd5c537b"

# Array of split names
split_names=("replicas-1" "replicas-2" "pdo21" "pdo27" "pdo75")

split=${split_names[$SLURM_ARRAY_TASK_ID]}

echo "Running job for split: ${split}"

python -u train_scgen.py --split_name ${split}