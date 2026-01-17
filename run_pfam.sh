#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition ou_bcs_low
#SBATCH --mem 40GB
#SBATCH -o logs/of
#SBATCH -e logs/ef

export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="c72e34cd8dc67f7220e3517232e86861cd5c537b"

#python -u test_pfam_print.py
#python -u visualize_pfam_embeddings.py
#python -u print_progen2_vocab.py
#python print_pfam_sequences.py
#python -u test_pfam_print.py

python -u count_pfams_in_pt.py
#python split_pt_file.py data/pfam/pfam_tokenized_data_clan.pt --train_size 200
#python -u analyze_pfam_lengths.py
#python count_pfam_stats.py
#python filter_pfam.py
#python -u test_pfam_evaluation.py \
#    --test_pt_file data/pfam/pfam_tokenized_data_test_1000.pt \
#    --output_dir outputs/pfam_11cf78c4746b7d595cc9fdc9db7176ad \
#    -n 50
    #--output_dir outputs/pfam_10aaa4008673874ff5daad8002b8575d
    #--output_dir outputs_progen2_firstrun/pfam_28ad2d74a123e25b88f8d360d7d47170