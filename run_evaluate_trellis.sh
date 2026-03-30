#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition ou_bcs_normal
#SBATCH --mem 50GB
#SBATCH -o logs/o002_%a
#SBATCH -e logs/e002_%a
#SBATCH --array=0-4

export HYDRA_FULL_ERROR=1

# Array of split names
split_names=("replicas-1" "replicas-2" "pdo21" "pdo27" "pdo75")

split=${split_names[$SLURM_ARRAY_TASK_ID]}

echo "Evaluating model for split: ${split}"

## Standard trellis with ridge predictor
cmd="python -u evaluate_trellis_experimental.py \
    --match experiment.name=trellis_a2a \
    --match experiment.split_name=${split} \
    --metric mmd \
    --predictor_loss mse \
    --cross_validate \
    --predict_delta \
    --compute_baseline \
    --predictor_type ridge"

if [[ "$split" != "replicas-1" && "$split" != "replicas-2" ]]; then
    cmd="$cmd --patient_cv"
    echo "Using patient_cv for split: ${split}"
else
    echo "Skipping patient_cv for replicas split: ${split}"
fi

eval $cmd