#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH --gres shard:1
#SBATCH --constraint any-A100
#SBATCH --partition abugoot
#SBATCH --mem 20GB
#SBATCH -o logs/o_script2
#SBATCH -e logs/e_script2


export PYTHONPATH=/orcd/archive/abugoot/001/Projects/paolo/CoupledDistributionEmbeddings:${PYTHONPATH}

python -m virus_eval.script2_train_timepoint_models virus_eval/dataset_offset_1.npz