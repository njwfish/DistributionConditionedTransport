#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition ou_bcs_normal
#SBATCH --mem 50GB
#SBATCH -o logs/otr_eval_mfm0_%a
#SBATCH -e logs/etr_eval_mfm0_%a
#SBATCH --array=2

export HYDRA_FULL_ERROR=1

# Array of split names
split_names=("replicas-1" "replicas-2" "pdo21" "pdo27" "pdo75")

split=${split_names[$SLURM_ARRAY_TASK_ID]}

echo "Evaluating model for split: ${split}"

## Standard trellis with ridge predictor
python -u evaluate_trellis.py \
    --match "experiment.name=trellis_mfm_gnn" \
    --match "experiment.split_name=${split}" \
    --metric mmd \
    --compute_baseline \