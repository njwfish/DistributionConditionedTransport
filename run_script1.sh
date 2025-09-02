#!/bin/bash
#SBATCH -t 05:00:00
#SBATCH --gres shard:1
#SBATCH --constraint any-A100
#SBATCH --partition abugoot
#SBATCH --mem 60GB
#SBATCH -o logs/o_script1
#SBATCH -e logs/e_script1


export PYTHONPATH=/orcd/archive/abugoot/001/Projects/paolo/CoupledDistributionEmbeddings:${PYTHONPATH}

python -m virus_eval.script1_generate_dataset \
    outputs/virus_c0d130ccd49ba40073008dbad4dcc93e \
    --offset 1 \
    --epochs 200 