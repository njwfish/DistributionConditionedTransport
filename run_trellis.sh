#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition ou_bcs_low
#SBATCH --mem 100GB
#SBATCH -o logs/otr_fm_ctt_%a
#SBATCH -e logs/etr_fm_ctt_%a
#SBATCH --array=0-4:1

export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="c72e34cd8dc67f7220e3517232e86861cd5c537b"

# Array of split names
split_names=("replicas-1" "replicas-2" "pdo21" "pdo27" "pdo75")

# Array of predictor loss weights
seeds=(0)

num_seeds=${#seeds[@]}

# Calculate which split and seed to use based on array task ID
# 5 splits x 1 seed = 5 total combinations
split_idx=$((SLURM_ARRAY_TASK_ID / num_seeds))
seed_idx=$((SLURM_ARRAY_TASK_ID % num_seeds))

split=${split_names[$split_idx]}
seed=${seeds[$seed_idx]}

echo "Running job for split: ${split}, seed: ${seed}"

python main.py experiment=trellis_fm_ctt experiment.split_name=${split} seed=${seed}
