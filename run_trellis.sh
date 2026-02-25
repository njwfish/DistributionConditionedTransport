#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition ou_bcs_low
#SBATCH --mem 50GB
#SBATCH -o logs/otr_stratified_baseline_%a
#SBATCH -e logs/etr_stratified_baseline_%a
#SBATCH --array=0-9:1

export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="c72e34cd8dc67f7220e3517232e86861cd5c537b"

# Array of split names
split_names=("replicas-1" "replicas-2" "pdo21" "pdo27" "pdo75")

# Ablation parameters
consecutive_ratios=(0.5 0.2)
predictor_loss_weights=(0.0)

num_consecutive_ratios=${#consecutive_ratios[@]}
num_predictor_loss_weights=${#predictor_loss_weights[@]}
num_combinations=$((num_consecutive_ratios * num_predictor_loss_weights))

# Calculate which parameters to use based on array task ID
# 5 splits x 2 consecutive_ratios x 2 predictor_loss_weights = 20 total combinations
split_idx=$((SLURM_ARRAY_TASK_ID / num_combinations))
remaining=$((SLURM_ARRAY_TASK_ID % num_combinations))
consecutive_ratio_idx=$((remaining / num_predictor_loss_weights))
predictor_loss_weight_idx=$((remaining % num_predictor_loss_weights))

split=${split_names[$split_idx]}
consecutive_ratio=${consecutive_ratios[$consecutive_ratio_idx]}
predictor_loss_weight=${predictor_loss_weights[$predictor_loss_weight_idx]}

echo "Running job for split: ${split}, consecutive_ratio: ${consecutive_ratio}, predictor_loss_weight: ${predictor_loss_weight}"

python main.py experiment=trellis_stratified \
    experiment.split_name=${split} \
    sampling.consecutive_ratio=${consecutive_ratio} \
    experiment.predictor_loss_weight=${predictor_loss_weight}
