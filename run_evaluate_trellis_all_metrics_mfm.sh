#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition ou_bcs_normal
#SBATCH --mem 32GB
#SBATCH -o logs/o_mfm_fm_%a
#SBATCH -e logs/e_mfm_fm_%a
#SBATCH --array=10

export HYDRA_FULL_ERROR=1

# Array of split names (5 splits)
split_names=("replicas-1" "replicas-2" "pdo21" "pdo27" "pdo75")

# Array of metrics (3 metrics)
metrics=("mmd_energy" "mmd_rbf" "swd")

# Compute split and metric indices from array task ID
# 15 jobs total: 5 splits × 3 metrics
split_idx=$((SLURM_ARRAY_TASK_ID / 3))
metric_idx=$((SLURM_ARRAY_TASK_ID % 3))

split=${split_names[$split_idx]}
metric=${metrics[$metric_idx]}

echo "Job ${SLURM_ARRAY_TASK_ID}: Evaluating split=${split}, metric=${metric}"

# Build the base command
cmd="python -u evaluate_trellis_experimental.py \
    --match experiment.name=trellis_mfm_gnn \
    --match experiment.split_name=${split} \
    --metric ${metric} \
    --compute_baseline \
    --outputs_dir outputs_trellis_fm_working_dont_ever_touch_this_01_27_2025
    "
    #--outputs_dir outputs_trellis_fm_working_dont_ever_touch_this_01_27_2025 \

# Run the command
eval $cmd
