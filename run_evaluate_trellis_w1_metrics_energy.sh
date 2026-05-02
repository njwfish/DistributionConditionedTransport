#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition ou_bcs_normal
#SBATCH --mem 32GB
#SBATCH -o logs/oval_drugclass_energy_%a
#SBATCH -e logs/eval_drugclass_energy_%a
#SBATCH --array=0-14:1

export HYDRA_FULL_ERROR=1

# Slurm runs a copy under /var/spool/slurmd/... so BASH_SOURCE is not the repo path.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${REPO_ROOT}" || { echo "Failed: cd ${REPO_ROOT}" >&2; exit 1; }
if [[ ! -f "${REPO_ROOT}/evaluate_trellis_experimental.py" ]]; then
  echo "REPO_ROOT=${REPO_ROOT} is not the DCT repo (missing evaluate_trellis_experimental.py)." >&2
  echo "Submit from the repo: cd .../DistributionConditionedTransport && sbatch run_evaluate_trellis_w1_metrics_*.sh" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${REPO_ROOT}/slurm/activate_dct_training_env.sh"

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
    --match experiment.name=trellis_a2a_energy \
    --match experiment.split_name=${split} \
    --metric ${metric} \
    --predictor_loss mse \
    --cross_validate \
    --predict_delta \
    --predictor_type ridge \
    --use_predictor \
    --aggregate_by_drug_class \
    --outputs_dir outputs_energy_and_swd_generators_never_delete_01_29_2026 \
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
