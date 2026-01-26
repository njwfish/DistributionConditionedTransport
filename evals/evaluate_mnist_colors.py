"""
Evaluation script for MNIST-Colors experiments.

Metrics:
- SWD: Sliced Wasserstein Distance on image features
- Color MSE: Mean squared error between mean colors of generated and target images
"""

import sys
from pathlib import Path

# Add parent dir to path
parent_dir = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, parent_dir)

import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import hydra
import pickle
import gc

from utils.experiment_utils import get_all_experiments_info, load_best_model
from generator.losses import sliced_wasserstein_distance
from datasets.mnist_colors import MNISTColorsDataset


def compute_color_mse(generated, target):
    """Compute MSE between mean colors of generated and target images.

    Args:
        generated: (batch, 3, H, W) tensor
        target: (batch, 3, H, W) tensor

    Returns:
        MSE value
    """
    gen_mean_color = generated.mean(dim=(2, 3))  # (batch, 3)
    tgt_mean_color = target.mean(dim=(2, 3))
    return ((gen_mean_color - tgt_mean_color) ** 2).mean().item()


def compute_swd_images(generated, target, n_projections=100):
    """Compute SWD between flattened images.

    Args:
        generated: (batch, 3, H, W) tensor
        target: (batch, 3, H, W) tensor
        n_projections: Number of random projections for SWD

    Returns:
        SWD value
    """
    gen_flat = generated.reshape(generated.shape[0], -1)
    tgt_flat = target.reshape(target.shape[0], -1)
    return sliced_wasserstein_distance(gen_flat, tgt_flat, n_projections=n_projections).item()


def compute_structural_similarity(generated, target):
    """Compute mean absolute pixel difference (inverted to similarity).

    Args:
        generated: (batch, 3, H, W) tensor
        target: (batch, 3, H, W) tensor

    Returns:
        Similarity value (higher = more similar)
    """
    return 1.0 - torch.abs(generated - target).mean().item()


def load_model(cfg, exp_dir, device):
    """Load encoder and generator from checkpoint."""
    enc = hydra.utils.instantiate(cfg['encoder'])
    gen = hydra.utils.instantiate(cfg['generator'])

    state = load_best_model(exp_dir)
    enc.load_state_dict(state['encoder_state_dict'])
    gen.load_state_dict(state['generator_state_dict'])

    enc.eval().to(device)
    gen.eval().to(device)

    return enc, gen


def evaluate_distribution_encoder(enc, gen, dataset, device, n_eval_batches=50, set_size=64):
    """Evaluate model with distribution encoder (encodes samples directly)."""
    swds = []
    color_mses = []
    similarities = []

    with torch.no_grad():
        for _ in range(n_eval_batches):
            # Get random batch
            idx = np.random.randint(len(dataset))
            batch = dataset[idx]

            source = batch['source_samples'].float().to(device).unsqueeze(0)  # (1, set_size, C, H, W)
            target = batch['target_samples'].float().to(device).unsqueeze(0)

            # Encode both sets
            combined = torch.cat([source, target], dim=0)  # (2, set_size, C, H, W)
            latents = enc(combined)  # (2, latent_dim)
            source_lat, target_lat = latents[0:1], latents[1:2]

            # Generate samples - need to reshape for generator
            source_flat = source.squeeze(0)  # (set_size, C, H, W)
            generated = gen.sample(source_flat, source_lat, target_lat)

            # Squeeze extra batch dimension if present: (1, set_size, C, H, W) -> (set_size, C, H, W)
            if generated.dim() == 5 and generated.shape[0] == 1:
                generated = generated.squeeze(0)

            # Metrics
            target_flat = target.squeeze(0)
            swds.append(compute_swd_images(generated, target_flat))
            color_mses.append(compute_color_mse(generated, target_flat))
            similarities.append(compute_structural_similarity(generated, target_flat))

    return {
        'swd': np.mean(swds),
        'swd_std': np.std(swds),
        'color_mse': np.mean(color_mses),
        'color_mse_std': np.std(color_mses),
        'similarity': np.mean(similarities),
        'similarity_std': np.std(similarities),
    }


def get_training_colors(exp_config, n_unique_sets):
    """Regenerate the training colors used during training.

    CRITICAL: For embedding encoder, the embedding indices correspond to specific
    colors generated with the training seed. We must use the same seed to get
    the correct color values.
    """
    seed = exp_config.get('seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate colors the same way as MNISTColorsDataset
    colors = torch.rand(n_unique_sets, 3)
    return colors


def find_nearest_color_idx(target_color, training_colors):
    """Find the index of the nearest training color using Euclidean distance.

    Args:
        target_color: RGB color tensor (3,)
        training_colors: Training colors tensor (n_unique_sets, 3)

    Returns:
        Index of the nearest training color
    """
    # Compute Euclidean distances
    distances = torch.norm(training_colors - target_color.unsqueeze(0), dim=1)
    return torch.argmin(distances).item()


def evaluate_embedding_encoder(enc, gen, dataset, exp_config, device, n_eval_batches=50, set_size=64):
    """Evaluate model with embedding encoder (looks up embeddings by index).

    For out-of-distribution colors, finds the nearest training color and uses
    that embedding index (analogous to MVN/GMM approach).
    """
    swds = []
    color_mses = []
    similarities = []

    # Get number of embeddings from the encoder
    n_unique_sets = enc.n_unique_sets if hasattr(enc, 'n_unique_sets') else enc.embedding.num_embeddings

    # Get training colors for nearest neighbor lookup
    training_colors = get_training_colors(exp_config, n_unique_sets)

    with torch.no_grad():
        for _ in range(n_eval_batches):
            # Get random batch
            idx = np.random.randint(len(dataset))
            batch = dataset[idx]

            source = batch['source_samples'].float().to(device)  # (set_size, C, H, W)
            target = batch['target_samples'].float().to(device)

            # Get evaluation colors from the dataset
            source_color = dataset.colors[batch['source_idx']]
            target_color = dataset.colors[batch['target_idx']]

            # Find nearest training colors (for OOD generalization)
            source_idx = find_nearest_color_idx(source_color, training_colors)
            target_idx = find_nearest_color_idx(target_color, training_colors)

            # Get embeddings by nearest training color index
            source_lat = enc(torch.tensor([source_idx]).long().to(device))
            target_lat = enc(torch.tensor([target_idx]).long().to(device))

            # Generate samples
            generated = gen.sample(source, source_lat, target_lat)

            # Squeeze extra batch dimension if present: (1, set_size, C, H, W) -> (set_size, C, H, W)
            if generated.dim() == 5 and generated.shape[0] == 1:
                generated = generated.squeeze(0)

            # Metrics
            swds.append(compute_swd_images(generated, target))
            color_mses.append(compute_color_mse(generated, target))
            similarities.append(compute_structural_similarity(generated, target))

    return {
        'swd': np.mean(swds),
        'swd_std': np.std(swds),
        'color_mse': np.mean(color_mses),
        'color_mse_std': np.std(color_mses),
        'similarity': np.mean(similarities),
        'similarity_std': np.std(similarities),
    }


def get_experiment_info(exp):
    """Extract key info from experiment config."""
    cfg = exp['config']
    encoder_type = 'embedding' if 'Embedding' in exp['encoder'] else 'distribution'

    gen_target = cfg['generator'].get('_target_', '')
    if 'flow_matching' in gen_target.lower():
        gen_type = 'FlowMatching'
    elif 'direct' in gen_target.lower():
        loss_type = cfg['generator'].get('loss_type', 'direct')
        gen_type = f'Direct_{loss_type}'
    elif 'energy' in gen_target.lower():
        gen_type = 'Energy'
    else:
        gen_type = gen_target.split('.')[-1] if gen_target else 'Unknown'

    return {
        'encoder_type': encoder_type,
        'generator_type': gen_type,
        'n_unique_sets': cfg['dataset'].get('n_unique_sets', 'N/A'),
        'seed': cfg.get('seed', 42),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate MNIST-Colors experiments')
    parser.add_argument('--output_dir', default='../outputs', help='Directory containing experiments')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--n_eval_batches', type=int, default=100, help='Number of evaluation batches')
    parser.add_argument('--set_size', type=int, default=64, help='Number of samples per set')
    parser.add_argument('--save_path', default='mnist_colors_eval_results.pkl', help='Where to save results')
    parser.add_argument('--num_epochs', type=int, default=None, help='Filter by training epochs')
    parser.add_argument('--n_unique_sets', type=int, default=None, help='Filter by n_unique_sets')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Get experiments
    print("Loading experiment configs...")
    configs = get_all_experiments_info(args.output_dir)

    # Filter to MNIST-Colors experiments with completed training
    exps = []
    for c in configs:
        if 'mnist_colors' not in c['name']:
            continue
        if not Path(c['dir'], 'best_model.pt').exists():
            continue
        # Filter by num_epochs if specified
        if args.num_epochs is not None:
            cfg_epochs = c['config'].get('training', {}).get('num_epochs')
            if cfg_epochs != args.num_epochs:
                continue
        # Filter by n_unique_sets if specified
        if args.n_unique_sets is not None:
            cfg_n_unique = c['config'].get('dataset', {}).get('n_unique_sets')
            if cfg_n_unique != args.n_unique_sets:
                continue
        exps.append(c)

    print(f"Found {len(exps)} MNIST-Colors experiments with best_model.pt")

    if not exps:
        print("No completed experiments found!")
        return

    # Create evaluation dataset
    print("Creating evaluation dataset...")
    eval_ds = MNISTColorsDataset(
        n_sets=10000,
        set_size=args.set_size,
        seed=999,  # Different seed for evaluation
        train=False,  # Use test set
    )
    print(f"Evaluation dataset: {len(eval_ds)} sets")

    results = []

    for exp in tqdm(exps, desc="Evaluating models"):
        cfg = exp['config']
        info = get_experiment_info(exp)

        print(f"\nEvaluating: {exp['name']}")
        print(f"  Encoder: {info['encoder_type']}, Generator: {info['generator_type']}")
        print(f"  n_unique_sets: {info['n_unique_sets']}, seed: {info['seed']}")

        try:
            enc, gen = load_model(cfg, exp['dir'], device)

            if info['encoder_type'] == 'embedding':
                metrics = evaluate_embedding_encoder(
                    enc, gen, eval_ds, cfg, device, args.n_eval_batches, args.set_size
                )
            else:
                metrics = evaluate_distribution_encoder(
                    enc, gen, eval_ds, device, args.n_eval_batches, args.set_size
                )

            result = {
                'experiment': exp['name'],
                'experiment_dir': exp['dir'],
                **info,
                **metrics
            }
            results.append(result)

            print(f"  SWD: {metrics['swd']:.4f}, Color MSE: {metrics['color_mse']:.4f}")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()

    # Save results
    print(f"\nSaving results to {args.save_path}...")
    with open(args.save_path, 'wb') as f:
        pickle.dump(results, f)

    # Also save as CSV
    df = pd.DataFrame(results)
    csv_path = args.save_path.replace('.pkl', '.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if len(df) > 0:
        summary = df.groupby(['encoder_type', 'generator_type', 'n_unique_sets'])[
            ['swd', 'color_mse', 'similarity']
        ].mean()
        print(summary.round(4))
    else:
        print("No results to summarize.")

    print(f"\nTotal experiments evaluated: {len(results)}")


if __name__ == '__main__':
    main()
