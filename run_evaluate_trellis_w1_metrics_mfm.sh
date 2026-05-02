#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition ou_bcs_normal
#SBATCH --mem 64GB
#SBATCH -o logs/o_drugclass_mfm_w1_%a
#SBATCH -e logs/e_drugclass_mfm_w1_%a
#SBATCH --array=0,3

export HYDRA_FULL_ERROR=1

# Array of split names (5 splits)
split_names=("replicas-1" "replicas-2" "pdo21" "pdo27" "pdo75")

# Array of metrics (3 metrics)
metrics=("w1")

# Compute split and metric indices from array task ID
# 15 jobs total: 5 splits × 3 metrics
split_idx=$((SLURM_ARRAY_TASK_ID / 1))
metric_idx=$((SLURM_ARRAY_TASK_ID % 1))

split=${split_names[$split_idx]}
metric=${metrics[$metric_idx]}

echo "Job ${SLURM_ARRAY_TASK_ID}: Evaluating split=${split}, metric=${metric}"

# Build the base command
cmd="python -u evaluate_trellis_experimental.py \
    --match experiment.name=trellis_mfm_knn \
    --match experiment.split_name=${split} \
    --metric ${metric} \
    --compute_baseline \
    --outputs_dir outputs
    "
    #--outputs_dir outputs_trellis_fm_working_dont_ever_touch_this_01_27_2025 \

# Run the command
eval $cmd
