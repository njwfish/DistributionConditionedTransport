#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH --gres shard:1
#SBATCH --constraint any-A100
#SBATCH --partition abugoot
#SBATCH --mem 16GB
#SBATCH -o logs/os_%a
#SBATCH -e logs/es_%a
#SBATCH --array=0-3:1

# Datasets and hyperparameters to analyze
# Define dataset names
datasets=("GoM" "PBMC" "LV" "Repressilator")

weight_modes=("uniform")
selective_pairing_modes=("single_step")
generator_source_only=true

# Calculate indices for this array task (no seed dimension; we aggregate across seeds)
num_datasets=${#datasets[@]}                    # 4
num_weight_modes=${#weight_modes[@]}            # 2
num_selective_pairing_modes=${#selective_pairing_modes[@]} # 3

dataset_idx=$((SLURM_ARRAY_TASK_ID / (num_weight_modes * num_selective_pairing_modes)))
weight_mode_idx=$(((SLURM_ARRAY_TASK_ID % (num_weight_modes * num_selective_pairing_modes)) / num_selective_pairing_modes))
selective_pairing_mode_idx=$((SLURM_ARRAY_TASK_ID % num_selective_pairing_modes))

dataset_name=${datasets[$dataset_idx]}
weight_mode=${weight_modes[$weight_mode_idx]}
selective_pairing_mode=${selective_pairing_modes[$selective_pairing_mode_idx]}

# Match the training settings you used in run_snapMMD.sh / config
# If you changed these in your run, update the values below to match.
weight_mode=${weight_mode}
selective_pairing_mode=${selective_pairing_mode}
num_samples=null
exponential_weight_scale=1.44

echo "Analyzing dataset=${dataset_name}, weight_mode=${weight_mode}, selective_pairing_mode=${selective_pairing_mode} across all seeds"

python snapMMD_analyze_results_flexible_2.py \
  --config snapMMD_analysis_config.yaml \
  --disable-emd \
  --set dataset_name=${dataset_name} \
  --set match_criteria.sampling.weight_mode=${weight_mode} \
  --set match_criteria.sampling.selective_pairing_mode=${selective_pairing_mode} \
  --set match_criteria.sampling.num_samples=${num_samples} \
  --set match_criteria.sampling.exponential_weight_scale=${exponential_weight_scale} \
  --set match_criteria.experiment.generator_source_only=${generator_source_only}