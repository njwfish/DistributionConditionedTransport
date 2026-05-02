#!/bin/bash
# Evaluates all 6 tx/unidirectional ablation experiments across all 5 splits and 3 metrics.
#
# Experiments covered:
#   trellis_a2a_transformer           (fm + tx encoder)
#   trellis_a2a_unidirectional        (fm + unidirectional)
#   trellis_a2a_swd_transformer       (swd + tx encoder)
#   trellis_a2a_swd_unidirectional    (swd + unidirectional)
#   trellis_a2a_energy_transformer    (energy + tx encoder)
#   trellis_a2a_energy_unidirectional (energy + unidirectional)
#
# 6 experiments × 5 splits × 3 metrics = 90 total jobs
#
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition ou_bcs_low
#SBATCH --mem 32GB
#SBATCH -o logs/oval_tx_uni_ablation_%a
#SBATCH -e logs/eval_tx_uni_ablation_%a
#SBATCH --array=0-89:1

export HYDRA_FULL_ERROR=1

experiment_names=(
  "trellis_a2a_transformer"
  "trellis_a2a_unidirectional"
  "trellis_a2a_swd_transformer"
  "trellis_a2a_swd_unidirectional"
  "trellis_a2a_energy_transformer"
  "trellis_a2a_energy_unidirectional"
)

split_names=("replicas-1" "replicas-2" "pdo21" "pdo27" "pdo75")

metrics=("mmd_energy" "mmd_rbf" "swd")

num_splits=${#split_names[@]}
num_metrics=${#metrics[@]}

# Decompose SLURM_ARRAY_TASK_ID into (experiment, split, metric) indices
exp_idx=$((SLURM_ARRAY_TASK_ID / (num_splits * num_metrics)))
remainder=$((SLURM_ARRAY_TASK_ID % (num_splits * num_metrics)))
split_idx=$((remainder / num_metrics))
metric_idx=$((remainder % num_metrics))

experiment_name=${experiment_names[$exp_idx]}
split=${split_names[$split_idx]}
metric=${metrics[$metric_idx]}

echo "Job ${SLURM_ARRAY_TASK_ID}: experiment=${experiment_name}, split=${split}, metric=${metric}"

cmd="python -u evaluate_trellis_experimental.py \
    --match experiment.name=${experiment_name} \
    --match experiment.split_name=${split} \
    --metric ${metric} \
    --predictor_loss mse \
    --cross_validate \
    --predict_delta \
    --predictor_type ridge \
    --use_predictor \
    "

if [[ "$split" != "replicas-1" && "$split" != "replicas-2" ]]; then
    cmd="$cmd --patient_cv"
    echo "Using patient_cv for split: ${split}"
else
    echo "Skipping patient_cv for replicas split: ${split}"
fi

eval $cmd
