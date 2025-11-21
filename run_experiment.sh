#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH --gres shard:1
#SBATCH --constraint any-A100
#SBATCH --partition abugoot
#SBATCH --mem 10GB
#SBATCH -o logs/oe0_%a
#SBATCH -e logs/ee0_%a
#SBATCH --array=0-19:1

export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="c72e34cd8dc67f7220e3517232e86861cd5c537b"

# Define dataset names (only GoM and PBMC)
datasets=("GoM" "PBMC")

#weight_modes=("uniform" "exponential")
weight_modes=("uniform")
#selective_pairing_modes=("single_step" "unidirectional")
selective_pairing_modes=("null")

# Define seeds
seeds=(0 1 2 3 4)

alpha=(0.001 0.01)

# Calculate indices for this array task
# Total combinations: 4 datasets × 2 weight_modes × 3 selective_pairing_modes × 10 seeds = 240 jobs
num_datasets=${#datasets[@]}                    # 4
num_weight_modes=${#weight_modes[@]}            # 2
num_selective_pairing_modes=${#selective_pairing_modes[@]} # 3
num_seeds=${#seeds[@]}                          # 10
num_alpha=${#alpha[@]}                          # 5

# Calculate which combination this job should run
dataset_idx=$((SLURM_ARRAY_TASK_ID / (num_weight_modes * num_selective_pairing_modes * num_seeds * num_alpha)))
weight_mode_idx=$(((SLURM_ARRAY_TASK_ID % (num_weight_modes * num_selective_pairing_modes * num_seeds * num_alpha)) / (num_selective_pairing_modes * num_seeds * num_alpha)))
selective_pairing_mode_idx=$(((SLURM_ARRAY_TASK_ID % (num_selective_pairing_modes * num_seeds * num_alpha)) / (num_seeds * num_alpha)))
seed_idx=$(((SLURM_ARRAY_TASK_ID % (num_seeds * num_alpha)) / num_alpha))
alpha_idx=$((SLURM_ARRAY_TASK_ID % num_alpha))

# Get the actual values for this combination
dataset_name=${datasets[$dataset_idx]}
weight_mode=${weight_modes[$weight_mode_idx]}
selective_pairing_mode=${selective_pairing_modes[$selective_pairing_mode_idx]}
seed=${seeds[$seed_idx]}
alpha=${alpha[$alpha_idx]}

echo "Running job ${SLURM_ARRAY_TASK_ID}: dataset=${dataset_name}, weight_mode=${weight_mode}, selective_pairing_mode=${selective_pairing_mode}, seed=${seed}"

# Set ot_coupling based on dataset: true for PBMC, false for others
if [ "$dataset_name" == "PBMC" ]; then
    ot_coupling=true
else
    ot_coupling=false
fi

# Run the unified hyperparameter experiment with the specified hyperparameters
python main.py experiment=snapMMD_energy_cotrain_pu dataset_name=${dataset_name} experiment.predictor_loss_weight=${alpha} experiment.selective_pairing_mode=${selective_pairing_mode} seed=${seed} experiment.ot_coupling=${ot_coupling}