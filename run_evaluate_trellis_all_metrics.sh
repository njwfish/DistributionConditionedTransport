#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition ou_bcs_low
#SBATCH --mem 100GB
#SBATCH -o logs/o_eval_strat_baseline_%a
#SBATCH -e logs/e_eval_strat_baseline_%a
#SBATCH --array=0-29

export HYDRA_FULL_ERROR=1

# Array of split names (5 splits)
split_names=("replicas-1" "replicas-2" "pdo21" "pdo27" "pdo75")

# Ablation parameters (must match run_trellis.sh)
consecutive_ratios=(0.5 0.2)
predictor_loss_weights=(0.0)

# Array of metrics (3 metrics)
metrics=("mmd_energy" "mmd_rbf" "swd")

# Total: 5 splits × 2 consecutive_ratios × 2 predictor_loss_weights × 3 metrics = 60 jobs
num_metrics=${#metrics[@]}
num_predictor_loss_weights=${#predictor_loss_weights[@]}
num_consecutive_ratios=${#consecutive_ratios[@]}

split_idx=$((SLURM_ARRAY_TASK_ID / (num_consecutive_ratios * num_predictor_loss_weights * num_metrics)))
remaining=$((SLURM_ARRAY_TASK_ID % (num_consecutive_ratios * num_predictor_loss_weights * num_metrics)))
consecutive_ratio_idx=$((remaining / (num_predictor_loss_weights * num_metrics)))
remaining=$((remaining % (num_predictor_loss_weights * num_metrics)))
predictor_loss_weight_idx=$((remaining / num_metrics))
metric_idx=$((remaining % num_metrics))

split=${split_names[$split_idx]}
consecutive_ratio=${consecutive_ratios[$consecutive_ratio_idx]}
predictor_loss_weight=${predictor_loss_weights[$predictor_loss_weight_idx]}
metric=${metrics[$metric_idx]}

echo "Job ${SLURM_ARRAY_TASK_ID}: Evaluating split=${split}, consecutive_ratio=${consecutive_ratio}, predictor_loss_weight=${predictor_loss_weight}, metric=${metric}"

# Build the base command
cmd="python -u evaluate_trellis_experimental.py \
    --match experiment.name=trellis_a2a_stratified \
    --match experiment.split_name=${split} \
    --match sampling.consecutive_ratio=${consecutive_ratio} \
    --match experiment.predictor_loss_weight=${predictor_loss_weight} \
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
