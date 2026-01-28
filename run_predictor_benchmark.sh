#!/bin/bash
#SBATCH -t 02:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition ou_bcs_normal
#SBATCH --mem 50GB
#SBATCH -o logs/predictor1_benchmark_%a
#SBATCH -e logs/predictor1_benchmark_err_%a
#SBATCH --array=0-11

# ============================================================================
# Predictor Benchmark Script
# ============================================================================
# This script benchmarks 12 combinations of predictor configurations:
#   - 3 predictor types: linear, ridge, random_forest
#   - 2 CV types: regular cross-validation, patient-specific cross-validation
#   - 2 cheat modes: normal mode, cheat mode (train on train+test)
#
# Fixed parameters:
#   - predictor_loss: mse
#   - predict_delta: enabled
#   - metric: mmd
#   - patient_holdout_fraction: 1.0 (default)
#   - folds_per_patient: 1 (default)
#
# Combination mapping (SLURM_ARRAY_TASK_ID):
#   0: linear   + regular CV + normal
#   1: ridge    + regular CV + normal
#   2: rf       + regular CV + normal
#   3: linear   + patient CV + normal
#   4: ridge    + patient CV + normal
#   5: rf       + patient CV + normal
#   6: linear   + regular CV + cheat
#   7: ridge    + regular CV + cheat
#   8: rf       + regular CV + cheat
#   9: linear   + patient CV + cheat
#  10: ridge    + patient CV + cheat
#  11: rf       + patient CV + cheat
# ============================================================================

export HYDRA_FULL_ERROR=1

# Configuration: Change this to test different splits
# Options: "replicas-1", "replicas-2", "pdo21", "pdo27", "pdo75"
SPLIT_NAME="${SPLIT_NAME:-pdo27}"

# Predictor types array
predictor_types=("linear" "ridge" "random_forest")

# Decode the array task ID into configuration choices
idx=$SLURM_ARRAY_TASK_ID

# idx % 3 = predictor type (0=linear, 1=ridge, 2=random_forest)
predictor_idx=$((idx % 3))
predictor_type=${predictor_types[$predictor_idx]}

# (idx / 3) % 2 = CV type (0=regular, 1=patient)
cv_type=$(( (idx / 3) % 2 ))

# (idx / 6) = cheat mode (0=off, 1=on)
cheat_mode=$((idx / 6))

# Build the command
cmd="python -u evaluate_trellis_experimental.py"
cmd+=" --match \"experiment.name=trellis_a2a\""
cmd+=" --match \"experiment.split_name=${SPLIT_NAME}\""
cmd+=" --metric mmd"
cmd+=" --predictor_loss mse"
cmd+=" --predict_delta"
cmd+=" --use_predictor"
cmd+=" --cross_validate"
cmd+=" --compute_baseline"
cmd+=" --predictor_type ${predictor_type}"

# Add patient CV flag if needed
if [ $cv_type -eq 1 ]; then
    cmd+=" --patient_cv"
    cv_name="patient_cv"
else
    cv_name="regular_cv"
fi

# Add cheat mode flag if needed
if [ $cheat_mode -eq 1 ]; then
    cmd+=" --cheat_mode"
    mode_name="cheat"
else
    mode_name="normal"
fi

# Print configuration
echo "============================================================================"
echo "PREDICTOR BENCHMARK - Job ${SLURM_ARRAY_TASK_ID}"
echo "============================================================================"
echo "Split:          ${SPLIT_NAME}"
echo "Predictor type: ${predictor_type}"
echo "CV type:        ${cv_name}"
echo "Mode:           ${mode_name}"
echo "============================================================================"
echo ""
echo "Running command:"
echo "${cmd}"
echo ""
echo "============================================================================"

# Run the command
eval $cmd

echo ""
echo "============================================================================"
echo "Job ${SLURM_ARRAY_TASK_ID} completed"
echo "Configuration: ${predictor_type} + ${cv_name} + ${mode_name}"
echo "============================================================================"
