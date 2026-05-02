#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition ou_bcs_high
#SBATCH --mem 128GB
#SBATCH -o logs/o_eval_scgen_new12_%a
#SBATCH -e logs/e_eval_scgen_new12_%a
#SBATCH --array=1


export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="c72e34cd8dc67f7220e3517232e86861cd5c537b"

# Array of split names and metrics
split_names=("replicas-1" "replicas-2" "pdo21" "pdo27" "pdo75")
metrics=("mmd_energy" "mmd_rbf" "swd")

# Calculate which split and metric to use
# We have 5 splits × 4 metrics = 20 combinations
# Layout: jobs 0-3 = replicas-1 with all metrics, jobs 4-7 = replicas-2 with all metrics, etc.
split_idx=$((SLURM_ARRAY_TASK_ID / 3))
metric_idx=$((SLURM_ARRAY_TASK_ID % 3))

split=${split_names[$split_idx]}
metric=${metrics[$metric_idx]}

echo "Running job ${SLURM_ARRAY_TASK_ID}: split=${split}, metric=${metric}"

python -u evaluate_trellis_scgen_shift.py --split_name ${split} --metric ${metric}