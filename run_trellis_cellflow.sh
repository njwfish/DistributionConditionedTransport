#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition ou_bcs_normal
#SBATCH --mem 100GB
#SBATCH -o logs/otr_train_cellflow_%a
#SBATCH -e logs/etr_train_cellflow_%a
#SBATCH --array=0-4:1

export HYDRA_FULL_ERROR=1

REPO_DIR="/orcd/data/omarabu/001/paolo/dct_trellis"
source "${REPO_DIR}/activate_cellflow_env.sh"

# Array of split names
split_names=("replicas-1" "replicas-2" "pdo21" "pdo27" "pdo75")

split=${split_names[$SLURM_ARRAY_TASK_ID]}

echo "Running CellFlow training for split: ${split}"

python - <<'PY'
import jax
print("JAX devices:", jax.devices())
PY

python -u train_cellflow.py \
    --split_name "${split}" \
    --num_iterations 20000
