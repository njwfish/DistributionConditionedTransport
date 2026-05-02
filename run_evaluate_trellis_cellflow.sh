#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition ou_bcs_normal
#SBATCH --mem 100GB
#SBATCH -o logs/oval03_cellflow_%a
#SBATCH -e logs/eval03_cellflow_%a
#SBATCH --array=0-4:1

export HYDRA_FULL_ERROR=1

REPO_DIR="${SLURM_SUBMIT_DIR:-}"
if [[ -z "${REPO_DIR}" || ! -f "${REPO_DIR}/activate_cellflow_env.sh" ]]; then
  REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "${REPO_DIR}/activate_cellflow_env.sh"

# Array of split names
split_names=("replicas-1" "replicas-2" "pdo21" "pdo27" "pdo75")

split=${split_names[$SLURM_ARRAY_TASK_ID]}

echo "Evaluating CellFlow for split: ${split}"

python - <<'PY'
import jax
print("JAX devices:", jax.devices())
PY

python -u evaluate_cellflow.py \
    --split_name "${split}" \
    --cache_path "logs/cellflow_eval_${split}.jsonl" \
    --clear_jax_cache_every 1
