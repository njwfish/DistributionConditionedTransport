#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH --gres shard:1
#SBATCH --constraint any-A100
#SBATCH --partition abugoot
#SBATCH --mem 20GB
#SBATCH -o logs/o_script3_fc_2
#SBATCH -e logs/e_script3_fc_2


export PYTHONPATH=/orcd/archive/abugoot/001/Projects/paolo/CoupledDistributionEmbeddings:${PYTHONPATH}

python -m virus_eval.script3_eval_timepoint_with_models /orcd/archive/abugoot/001/Projects/paolo/CoupledDistributionEmbeddings/outputs/virus_c0d130ccd49ba40073008dbad4dcc93e -n 1
python -m virus_eval.script3_forecasting_eval \
  /orcd/archive/abugoot/001/Projects/paolo/CoupledDistributionEmbeddings/outputs/virus_c0d130ccd49ba40073008dbad4dcc93e \
  --dataset_path /orcd/archive/abugoot/001/Projects/paolo/CoupledDistributionEmbeddings/virus_eval/dataset_offset_1.npz \
  --heldout_path /orcd/archive/abugoot/001/Projects/paolo/CoupledDistributionEmbeddings/data/spikeprot0430/virus_tokenized_data_for_tde_downsampled100.pt \
  --models_dir /orcd/archive/abugoot/001/Projects/paolo/CoupledDistributionEmbeddings/virus_eval \
  --num_control_sets 5 --set_size 16