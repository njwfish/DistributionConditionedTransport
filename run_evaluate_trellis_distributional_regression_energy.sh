#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition ou_bcs_low
#SBATCH --mem 32GB
#SBATCH -o logs/trellis_distreg_energy_%a
#SBATCH -e logs/trellis_distreg_energy_err_%a
#SBATCH --array=0-19:1

export HYDRA_FULL_ERROR=1

# Slurm runs a copy under /var/spool/slurmd/... so BASH_SOURCE is not the repo path.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${REPO_ROOT}" || { echo "Failed: cd ${REPO_ROOT}" >&2; exit 1; }
if [[ ! -f "${REPO_ROOT}/evaluate_trellis_distributional_regression.py" ]]; then
  echo "REPO_ROOT=${REPO_ROOT} is not the DCT repo (missing evaluate_trellis_distributional_regression.py)." >&2
  echo "Submit from the repo: cd .../DistributionConditionedTransport && sbatch run_evaluate_trellis_distributional_regression_energy.sh" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${REPO_ROOT}/slurm/activate_dct_training_env.sh"

split_names=("replicas-1" "replicas-2" "pdo21" "pdo27" "pdo75")
num_sample_shards=4
split_idx=$((SLURM_ARRAY_TASK_ID / num_sample_shards))
sample_shard=$((SLURM_ARRAY_TASK_ID % num_sample_shards))
split=${split_names[$split_idx]}

echo "Job ${SLURM_ARRAY_TASK_ID}: distributional-regression eval for split=${split}, sample_shard=${sample_shard}/${num_sample_shards}"

python -u evaluate_trellis_distributional_regression.py \
  --split_name "${split}" \
  --experiment_name trellis_a2a_energy \
  --outputs_dir outputs_energy_and_swd_generators_never_delete_01_29_2026 \
  --results_dir trellis_distributional_regression_results \
  --num_sample_shards "${num_sample_shards}" \
  --sample_shard "${sample_shard}" \
  --k_max 64 \
  --k_values 1 2 4 8 16 32 64 \
  --m 20 \
  --sampler_epochs 500 \
  --sampler_hidden 128 \
  --sampler_lr 1e-3 \
  --sampler_batch_size 256 \
  --swd_subsample_rounds 100
