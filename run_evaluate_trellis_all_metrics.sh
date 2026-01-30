#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition ou_bcs_low
#SBATCH --mem 32GB
#SBATCH -o logs/o_ours_fm_%a
#SBATCH -e logs/e_ours_fm_%a
#SBATCH --array=1

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
    --match experiment.name=trellis_a2a \
    --match experiment.split_name=${split} \
    --metric ${metric} \
    --predictor_loss mse \
    --cross_validate \
    --predict_delta \
    --compute_baseline \
    --predictor_type ridge \
    --use_predictor \
    "

#--outputs_dir outputs_trellis_fm_working_dont_ever_touch_this_01_27_2025 \

#    --outputs_dir outputs_trellis_fm_working_dont_ever_touch_this_01_27_2025 \

# Add patient_cv flag only for non-replicas splits (pdo21, pdo27, pdo75)
if [[ "$split" != "replicas-1" && "$split" != "replicas-2" ]]; then
    cmd="$cmd --patient_cv"
    echo "Using patient_cv for split: ${split}"
else
    echo "Skipping patient_cv for replicas split: ${split}"
fi

# Run the command
eval $cmd
