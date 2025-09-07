#!/bin/bash
#SBATCH -t 02:00:00
#SBATCH --gres shard:1
#SBATCH --constraint any-A100
#SBATCH --partition abugoot
#SBATCH --mem 16GB
#SBATCH -o logs/o007s3_%a
#SBATCH -e logs/e007s3_%a
#SBATCH --array=0-4:1

# Define dataset names (only GoM and PBMC)
#datasets=("GoM" "PBMC")
#weight_modes=("uniform" "exponential")
datasets=("PBMC")
weight_modes=("uniform")
selective_pairing_modes=("single_step")

alpha=(0.0001 0.0005 0.001 0.005 0.01)


# Calculate indices for this array task (no seed dimension; we aggregate across seeds)
num_datasets=${#datasets[@]}                    # 4
num_weight_modes=${#weight_modes[@]}            # 2
num_selective_pairing_modes=${#selective_pairing_modes[@]} # 3
num_alpha=${#alpha[@]}                          # 5


dataset_idx=$((SLURM_ARRAY_TASK_ID / (num_weight_modes * num_selective_pairing_modes * num_alpha)))
weight_mode_idx=$(((SLURM_ARRAY_TASK_ID % (num_weight_modes * num_selective_pairing_modes * num_alpha)) / (num_selective_pairing_modes * num_alpha)))
selective_pairing_mode_idx=$(((SLURM_ARRAY_TASK_ID % (num_selective_pairing_modes * num_alpha)) / num_alpha))
alpha_idx=$((SLURM_ARRAY_TASK_ID % num_alpha))

dataset_name=${datasets[$dataset_idx]}
weight_mode=${weight_modes[$weight_mode_idx]}
selective_pairing_mode=${selective_pairing_modes[$selective_pairing_mode_idx]}
alpha=${alpha[$alpha_idx]}
# Match the training settings you used in run_snapMMD.sh / config
# If you changed these in your run, update the values below to match.
predictor_loss_weight=${alpha}
use_predicted_latent=false
train_predictor_posthoc=false
weight_mode=${weight_mode}
selective_pairing_mode=${selective_pairing_mode}
num_samples=null
exponential_weight_scale=1.44

echo "Analyzing dataset=${dataset_name}, weight_mode=${weight_mode}, selective_pairing_mode=${selective_pairing_mode} across all seeds"

python snapMMD_analyze_results_flexible_attempt2.py \
  --use-ridge-predictor \
  --config snapMMD_analysis_config.yaml \
  --set dataset_name=${dataset_name} \
  --set match_criteria.experiment.predictor_loss_weight=${predictor_loss_weight} \
  --set match_criteria.experiment.use_predicted_latent=${use_predicted_latent} \
  --set match_criteria.experiment.train_predictor_posthoc=${train_predictor_posthoc} \
  --set match_criteria.sampling.weight_mode=${weight_mode} \
  --set match_criteria.sampling.selective_pairing_mode=${selective_pairing_mode} \
  --set match_criteria.sampling.num_samples=${num_samples} \
  --set match_criteria.sampling.exponential_weight_scale=${exponential_weight_scale} \
