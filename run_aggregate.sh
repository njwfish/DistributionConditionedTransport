#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH --partition ou_bcs_normal
#SBATCH --mem 10GB
#SBATCH -o logs/o_aggregate1
#SBATCH -e logs/e_aggregate1

#python parse_evaluation_logs.py --generator fm
#python aggregate_by_metric.py

#python parse_ablation_logs.py
python aggregate_ablation.py
