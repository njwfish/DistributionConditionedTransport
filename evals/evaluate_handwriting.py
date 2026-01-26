"""
Evaluation script for Handwriting experiments.

Metrics:
- SWD: Sliced Wasserstein Distance on image features
- MSE: Mean squared error between generated and target images
- Similarity: 1 - mean absolute pixel difference
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
from datasets.handwriting import HandwritingDataset


def compute_mse(generated, target):
    """Compute MSE between generated and target images.

    Args:
        generated: (batch, C, H, W) tensor
        target: (batch, C, H, W) tensor

    Returns:
        MSE value
    """
    return ((generated - target) ** 2).mean().item()


def compute_swd_images(generated, target, n_projections=100):
    """Compute SWD between flattened images.

    Args:
        generated: (batch, C, H, W) tensor
        target: (batch, C, H, W) tensor
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
        generated: (batch, C, H, W) tensor
        target: (batch, C, H, W) tensor

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
    mses = []
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
            mses.append(compute_mse(generated, target_flat))
            similarities.append(compute_structural_similarity(generated, target_flat))

    return {
        'swd': np.mean(swds),
        'swd_std': np.std(swds),
        'mse': np.mean(mses),
        'mse_std': np.std(mses),
        'similarity': np.mean(similarities),
        'similarity_std': np.std(similarities),
    }


def evaluate_embedding_encoder(enc, gen, dataset, device, n_eval_batches=50, set_size=64):
    """Evaluate model with embedding encoder (looks up embeddings by index).

    The handwriting dataset now returns integer indices directly.
    """
    swds = []
    mses = []
    similarities = []

    n_unique_sets = enc.n_unique_sets if hasattr(enc, 'n_unique_sets') else enc.embedding.num_embeddings

    with torch.no_grad():
        for _ in range(n_eval_batches):
            # Get random batch
            idx = np.random.randint(len(dataset))
            batch = dataset[idx]

            source = batch['source_samples'].float().to(device)  # (set_size, C, H, W)
            target = batch['target_samples'].float().to(device)

            # Get indices directly from batch (now returns integers)
            source_idx = batch['source_idx']
            target_idx = batch['target_idx']

            # Clamp indices to valid range
            source_idx = min(source_idx, n_unique_sets - 1)
            target_idx = min(target_idx, n_unique_sets - 1)

            # Get embeddings by index
            source_lat = enc(torch.tensor([source_idx]).long().to(device))
            target_lat = enc(torch.tensor([target_idx]).long().to(device))

            # Generate samples
            generated = gen.sample(source, source_lat, target_lat)

            # Squeeze extra batch dimension if present: (1, set_size, C, H, W) -> (set_size, C, H, W)
            if generated.dim() == 5 and generated.shape[0] == 1:
                generated = generated.squeeze(0)

            # Metrics
            swds.append(compute_swd_images(generated, target))
            mses.append(compute_mse(generated, target))
            similarities.append(compute_structural_similarity(generated, target))

    return {
        'swd': np.mean(swds),
        'swd_std': np.std(swds),
        'mse': np.mean(mses),
        'mse_std': np.std(mses),
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

    # Get frac_unique_writers or n_unique_writers
    frac_unique_writers = cfg['dataset'].get('frac_unique_writers', None)
    n_unique_writers = cfg['dataset'].get('n_unique_writers', None)

    return {
        'encoder_type': encoder_type,
        'generator_type': gen_type,
        'frac_unique_writers': frac_unique_writers,
        'n_unique_writers': n_unique_writers,
        'seed': cfg.get('seed', 42),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Handwriting experiments')
    parser.add_argument('--output_dir', default='../outputs', help='Directory containing experiments')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--n_eval_batches', type=int, default=100, help='Number of evaluation batches')
    parser.add_argument('--set_size', type=int, default=64, help='Number of samples per set')
    parser.add_argument('--save_path', default='handwriting_eval_results.pkl', help='Where to save results')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Get experiments
    print("Loading experiment configs...")
    configs = get_all_experiments_info(args.output_dir)

    # Filter to handwriting experiments with completed training
    exps = []
    for c in configs:
        if 'handwriting' not in c['name']:
            continue
        if not Path(c['dir'], 'best_model.pt').exists():
            continue
        exps.append(c)

    print(f"Found {len(exps)} Handwriting experiments with best_model.pt")

    if not exps:
        print("No completed experiments found!")
        return

    # Create evaluation dataset with all writers available
    print("Creating evaluation dataset...")
    eval_ds = HandwritingDataset(
        set_size=args.set_size,
        seed=999,  # Different seed for evaluation
        train=True,  # Use training writers for evaluation
        frac_unique_writers=1.0,  # Use all writers for evaluation
    )
    print(f"Evaluation dataset: {len(eval_ds)} writers")

    results = []

    for exp in tqdm(exps, desc="Evaluating models"):
        cfg = exp['config']
        info = get_experiment_info(exp)

        print(f"\nEvaluating: {exp['name']}")
        print(f"  Encoder: {info['encoder_type']}, Generator: {info['generator_type']}")
        print(f"  frac_unique_writers: {info['frac_unique_writers']}, n_unique_writers: {info['n_unique_writers']}, seed: {info['seed']}")

        try:
            enc, gen = load_model(cfg, exp['dir'], device)

            # Create dataset with same writer configuration as training
            train_ds = HandwritingDataset(
                set_size=args.set_size,
                seed=cfg.get('seed', 42),  # Same seed as training
                train=True,
                frac_unique_writers=info['frac_unique_writers'],
                n_unique_writers=info['n_unique_writers'],
            )

            if info['encoder_type'] == 'embedding':
                metrics = evaluate_embedding_encoder(
                    enc, gen, train_ds, device, args.n_eval_batches, args.set_size
                )
            else:
                metrics = evaluate_distribution_encoder(
                    enc, gen, train_ds, device, args.n_eval_batches, args.set_size
                )

            result = {
                'experiment': exp['name'],
                'experiment_dir': exp['dir'],
                **info,
                **metrics
            }
            results.append(result)

            print(f"  SWD: {metrics['swd']:.4f}, MSE: {metrics['mse']:.4f}")

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
        summary = df.groupby(['encoder_type', 'generator_type', 'frac_unique_writers'])[
            ['swd', 'mse', 'similarity']
        ].mean()
        print(summary.round(4))
    else:
        print("No results to summarize.")

    print(f"\nTotal experiments evaluated: {len(results)}")


if __name__ == '__main__':
    main()
