#!/bin/bash
#SBATCH -t 04:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition ou_bcs_low
#SBATCH --mem 50GB
#SBATCH -o logs/otr_cellot_%a
#SBATCH -e logs/etr_cellot_%a
#SBATCH --array=0-54

# 5 splits x 11 treatments = 55 jobs (indices 0-54)

split_names=("replicas-1" "replicas-2" "pdo21" "pdo27" "pdo75")
treatments=("O" "S" "VS" "L" "V" "F" "C" "SF" "CS" "CF" "CSF")

num_treatments=${#treatments[@]}

split_idx=$((SLURM_ARRAY_TASK_ID / num_treatments))
treat_idx=$((SLURM_ARRAY_TASK_ID % num_treatments))

split=${split_names[$split_idx]}
treatment=${treatments[$treat_idx]}

echo "Running job ${SLURM_ARRAY_TASK_ID}: split=${split}, treatment=${treatment}"

python -u train_cellot.py \
    --split_name ${split} \
    --treatment ${treatment} \
    --batch_size 256
