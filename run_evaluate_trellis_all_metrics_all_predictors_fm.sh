#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition ou_bcs_low
#SBATCH --mem 100GB
#SBATCH -o logs/o_eval_predictor_ablation_fm_%a
#SBATCH -e logs/e_eval_predictor_ablation_fm_%a
#SBATCH --array=0-44

export HYDRA_FULL_ERROR=1

# Array of split names (5 splits)
split_names=("replicas-1" "replicas-2" "pdo21" "pdo27" "pdo75")

# Array of metrics (3 metrics)
metrics=("mmd_energy" "mmd_rbf" "swd")

# Predictor types (3 predictors)
predictor_types=("ridge" "random_forest" "mlp")

# Total: 5 splits × 3 metrics × 3 predictors = 45 jobs
num_predictor_types=${#predictor_types[@]}
num_metrics=${#metrics[@]}

split_idx=$((SLURM_ARRAY_TASK_ID / (num_metrics * num_predictor_types)))
remaining=$((SLURM_ARRAY_TASK_ID % (num_metrics * num_predictor_types)))
metric_idx=$((remaining / num_predictor_types))
predictor_type_idx=$((remaining % num_predictor_types))

split=${split_names[$split_idx]}
metric=${metrics[$metric_idx]}
predictor_type=${predictor_types[$predictor_type_idx]}

echo "Job ${SLURM_ARRAY_TASK_ID}: Evaluating split=${split}, metric=${metric}, predictor_type=${predictor_type}"

# Build the base command
cmd="python -u evaluate_trellis_experimental.py \
    --match experiment.name=trellis_a2a \
    --match experiment.split_name=${split} \
    --metric ${metric} \
    --predictor_loss mse \
    --cross_validate \
    --predict_delta \
    --compute_baseline \
    --predictor_type ${predictor_type} \
    --use_predictor \
    "

# Add patient_cv flag only for non-replicas splits (pdo21, pdo27, pdo75)
if [[ "$split" != "replicas-1" && "$split" != "replicas-2" ]]; then
    cmd="$cmd --patient_cv"
    echo "Using patient_cv for split: ${split}"
else
    echo "Skipping patient_cv for replicas split: ${split}"
fi

# Run the command
eval $cmd
