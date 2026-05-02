#!/bin/bash
#SBATCH -t 06:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition ou_bcs_normal
#SBATCH --mem 100GB
#SBATCH -o logs/o_eval_cellot_%a
#SBATCH -e logs/e_eval_cellot_%a
#SBATCH --array=1

# 5 splits x 3 metrics = 15 jobs (indices 0-14)

split_names=("replicas-1" "replicas-2" "pdo21" "pdo27" "pdo75")
metrics=("mmd_energy" "mmd_rbf" "swd")

num_metrics=${#metrics[@]}

split_idx=$((SLURM_ARRAY_TASK_ID / num_metrics))
metric_idx=$((SLURM_ARRAY_TASK_ID % num_metrics))

split=${split_names[$split_idx]}
metric=${metrics[$metric_idx]}

echo "Running job ${SLURM_ARRAY_TASK_ID}: split=${split}, metric=${metric}"

python -u evaluate_cellot.py \
    --split_name ${split} \
    --metric ${metric} \
    --compute_baseline
