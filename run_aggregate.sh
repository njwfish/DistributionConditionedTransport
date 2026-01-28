#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH --partition ou_bcs_normal
#SBATCH --mem 50GB
#SBATCH -o logs/o_aggregate_energy1
#SBATCH -e logs/e_aggregate_energy1

#python parse_evaluation_logs.py --generator energy
python aggregate_results.py --generator energy