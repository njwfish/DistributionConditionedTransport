#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH --gres shard:1
#SBATCH --constraint any-A100
#SBATCH --partition abugoot
#SBATCH --mem 20GB
#SBATCH -o logs/o_script3
#SBATCH -e logs/e_script3


export PYTHONPATH=/orcd/archive/abugoot/001/Projects/paolo/CoupledDistributionEmbeddings:${PYTHONPATH}

python -m virus_eval.script3_eval_timepoint_with_models /orcd/archive/abugoot/001/Projects/paolo/CoupledDistributionEmbeddings/outputs/virus_c0d130ccd49ba40073008dbad4dcc93e -n 1