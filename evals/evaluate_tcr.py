"""
Evaluation script for TCR repertoire models.

For between-patient models:
1. Compute distribution embeddings for all train patient timepoints
2. Train a linear predictor: pred_next_embedding = Linear(current_embedding)
3. On test patients, predict next timepoint embedding and sample from it
4. Compute SW/MMD/Energy distances between sampled and actual target distributions
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
import hydra
from transformers import EsmModel, AutoTokenizer
from geomloss import SamplesLoss
from sklearn.linear_model import RidgeCV

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.tcr import TCRDataset

# Default sample size for embedding and evaluation
DEFAULT_SAMPLE_SIZE = 512


def get_esm_embeddings(esm_model, input_ids, attention_mask):
    """Compute mean-pooled ESM embeddings."""
    with torch.no_grad():
        hidden_states = esm_model(input_ids, attention_mask=attention_mask).last_hidden_state
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def retokenize_sequences(sequences, esm_tokenizer, max_length=64, device='cuda', decode_tokenizer=None):
    """
    Decode generated sequences and retokenize for ESM embedding.

    Args:
        sequences: tensor of shape (batch_size, num_samples, seq_len) or (num_samples, seq_len)
        esm_tokenizer: Tokenizer for ESM model (used for re-tokenizing)
        max_length: Maximum sequence length
        device: Device to put tensors on
        decode_tokenizer: Tokenizer to decode the input sequences (default: esm_tokenizer)
                         Use the generator's tokenizer for Progen models.
    """
    if decode_tokenizer is None:
        decode_tokenizer = esm_tokenizer

    if sequences.dim() == 2:
        sequences = sequences.unsqueeze(0)

    batch_size, num_samples, seq_len = sequences.shape

    all_seqs = []
    for b in range(batch_size):
        for s in range(num_samples):
            seq_ids = sequences[b, s].cpu().tolist()
            seq_str = decode_tokenizer.decode(seq_ids, skip_special_tokens=True)
            all_seqs.append(seq_str)

    tokens = esm_tokenizer(
        all_seqs,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_length
    )
    return tokens['input_ids'].to(device), tokens['attention_mask'].to(device)


class LinearPredictor(nn.Module):
    """Simple linear predictor for next timepoint embedding."""
    def __init__(self, latent_dim):
        super().__init__()
        self.linear = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        return self.linear(x)


class DeltaPredictor(nn.Module):
    """Predictor that learns delta (target - source) and adds it to source.

    Wraps a base predictor that outputs the delta, then adds source to get the target.
    This is often easier to learn when source and target are similar.
    """
    def __init__(self, base_predictor):
        super().__init__()
        self.base_predictor = base_predictor

    def forward(self, x):
        # Predict delta and add to source
        delta = self.base_predictor(x)
        return x + delta


def compute_energy_distance(sampled_embs, target_embs):
    """Compute energy distance between two sets of embeddings."""
    loss_fn = SamplesLoss("energy", p=2)
    return loss_fn(sampled_embs, target_embs).item()


def compute_sliced_wasserstein(sampled_embs, target_embs):
    """Compute Sinkhorn (approximated Wasserstein) distance."""
    loss_fn = SamplesLoss("sinkhorn", blur=0.01, scaling=0.9)
    return loss_fn(sampled_embs, target_embs).item()


def compute_mmd(sampled_embs, target_embs, sigma=1.0):
    """Compute Maximum Mean Discrepancy with RBF kernel."""
    def rbf_kernel(x, y, sigma):
        dist = torch.cdist(x, y, p=2)
        return torch.exp(-dist ** 2 / (2 * sigma ** 2))

    k_xx = rbf_kernel(sampled_embs, sampled_embs, sigma)
    k_yy = rbf_kernel(target_embs, target_embs, sigma)
    k_xy = rbf_kernel(sampled_embs, target_embs, sigma)

    n = sampled_embs.size(0)
    m = target_embs.size(0)

    mmd = (k_xx.sum() / (n * n) + k_yy.sum() / (m * m) - 2 * k_xy.sum() / (n * m))
    return mmd.item()


METRIC_FUNCTIONS = {
    'energy': compute_energy_distance,
    'sliced_wasserstein': compute_sliced_wasserstein,
    'mmd': compute_mmd,
}


def load_model(checkpoint_dir, device='cuda'):
    """Load encoder and generator from checkpoint directory."""
    config_path = os.path.join(checkpoint_dir, 'config.yaml')
    config = OmegaConf.load(config_path)

    # Instantiate encoder and generator
    encoder = hydra.utils.instantiate(config.encoder)
    generator = hydra.utils.instantiate(config.generator)

    # Find best checkpoint or latest
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        checkpoint_path = best_model_path
    else:
        # Find the highest epoch checkpoint
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        epochs = [int(f.split('_')[-1].replace('.pt', '')) for f in checkpoints]
        max_epoch = max(epochs)
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{max_epoch}.pt')

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    generator.load_state_dict(checkpoint['generator_state_dict'])

    encoder.to(device).eval()
    generator.to(device).eval()

    return encoder, generator, config


def compute_all_embeddings(encoder, dataset, device='cuda', sample_size=None):
    """Compute distribution embeddings for all repertoires in the dataset.

    Args:
        encoder: The distribution encoder model
        dataset: TCRDataset to compute embeddings for
        device: Device to use for computation
        sample_size: Number of sequences to use per repertoire. If None, uses DEFAULT_SAMPLE_SIZE.
    """
    if sample_size is None:
        sample_size = DEFAULT_SAMPLE_SIZE

    embeddings = {}

    # Get unique repertoire indices
    unique_indices = set()
    for source_idx, target_idx in dataset.pair_indices:
        unique_indices.add(source_idx)
        unique_indices.add(target_idx)

    print(f"Computing embeddings for {len(unique_indices)} unique repertoires using {sample_size} samples each...")

    for idx in tqdm(sorted(unique_indices)):
        item = dataset.data[idx]
        metadata = dataset.metadata[idx]

        # Use all sequences for computing embedding (not just set_size)
        # But we need to batch if there are too many
        all_esm_ids = item['samples']['esm_input_ids']
        all_esm_mask = item['samples']['esm_attention_mask']

        # Sample a fixed subset for consistent embedding
        n_seqs = all_esm_ids.size(0)
        actual_sample_size = min(n_seqs, sample_size)
        indices = np.random.choice(n_seqs, actual_sample_size, replace=False)

        esm_ids = all_esm_ids[indices].unsqueeze(0).to(device)
        esm_mask = all_esm_mask[indices].unsqueeze(0).to(device)

        with torch.no_grad():
            samples = {
                'esm_input_ids': esm_ids,
                'esm_attention_mask': esm_mask,
            }
            embedding = encoder(samples)

        embeddings[idx] = {
            'embedding': embedding.cpu(),
            'metadata': metadata,
        }

    return embeddings


def build_predictor_training_data(embeddings, dataset):
    """Build training pairs for the predictor (consecutive timepoints within patient)."""
    source_embeddings = []
    target_embeddings = []

    for subject_id, indices in dataset.patient_indices.items():
        if len(indices) < 2:
            continue

        # Sort indices by timepoint
        idx_time_pairs = [(idx, dataset.metadata[idx][1]) for idx in indices]
        idx_time_pairs.sort(key=lambda x: x[1])

        # Create pairs of consecutive timepoints
        for i in range(len(idx_time_pairs) - 1):
            source_idx = idx_time_pairs[i][0]
            target_idx = idx_time_pairs[i + 1][0]

            if source_idx in embeddings and target_idx in embeddings:
                source_embeddings.append(embeddings[source_idx]['embedding'])
                target_embeddings.append(embeddings[target_idx]['embedding'])

    if not source_embeddings:
        return None, None

    source_embeddings = torch.cat(source_embeddings, dim=0)
    target_embeddings = torch.cat(target_embeddings, dim=0)

    return source_embeddings, target_embeddings


def train_predictor(source_embeddings, target_embeddings, latent_dim,
                   alphas=None, device='cuda', predict_delta=False):
    """Train the predictor using sklearn's RidgeCV with LOO cross-validation.

    Uses closed-form Ridge regression with efficient LOO CV for regularization selection.
    Fits one RidgeCV model per output dimension, then transfers weights to a PyTorch
    LinearPredictor for compatibility with the rest of the codebase.

    Args:
        source_embeddings: torch.Tensor of shape (n_samples, latent_dim)
        target_embeddings: torch.Tensor of shape (n_samples, latent_dim)
        latent_dim: Dimension of the latent space
        alphas: List of regularization values to try. If None, uses a default grid.
        device: Device for the resulting PyTorch model
        predict_delta: If True, train to predict (target - source) instead of target directly

    Returns:
        LinearPredictor with weights initialized from RidgeCV
    """
    # Default regularization grid (log-spaced from 1e-4 to 1e4)
    if alphas is None:
        alphas = np.logspace(-4, 4, 20)

    # Convert to numpy
    X = source_embeddings.cpu().numpy()
    Y = target_embeddings.cpu().numpy()

    # If predicting delta, train on (target - source) instead of target
    if predict_delta:
        Y = Y - X
        print("Training predictor to predict DELTA (target - source)")

    n_samples, n_dims = Y.shape
    print(f"Training RidgeCV predictor on {n_samples} samples, {n_dims} dimensions")
    print(f"Regularization grid: {alphas[0]:.2e} to {alphas[-1]:.2e} ({len(alphas)} values)")

    # Train one RidgeCV model per output dimension
    # RidgeCV with cv=None uses efficient LOO cross-validation
    weights = []
    biases = []
    selected_alphas = []

    for dim in range(n_dims):
        model = RidgeCV(alphas=alphas, cv=None, store_cv_results=True)
        model.fit(X, Y[:, dim])
        weights.append(model.coef_)
        biases.append(model.intercept_)
        selected_alphas.append(model.alpha_)

    # Print summary of selected alphas
    alphas_arr = np.array(selected_alphas)
    print(f"Selected alphas: min={alphas_arr.min():.2e}, max={alphas_arr.max():.2e}, "
          f"median={np.median(alphas_arr):.2e}")

    # Stack weights and biases: weights is (n_dims, latent_dim), biases is (n_dims,)
    W = np.stack(weights, axis=0)  # (latent_dim, latent_dim)
    b = np.array(biases)  # (latent_dim,)

    # Create PyTorch LinearPredictor and set weights
    predictor = LinearPredictor(latent_dim).to(device)
    with torch.no_grad():
        predictor.linear.weight.copy_(torch.tensor(W, dtype=torch.float32))
        predictor.linear.bias.copy_(torch.tensor(b, dtype=torch.float32))

    # Wrap in DeltaPredictor if predicting delta
    if predict_delta:
        predictor = DeltaPredictor(predictor)

    # Compute training MSE (predictor now handles delta internally if applicable)
    predictor.eval()
    with torch.no_grad():
        pred_target = predictor(source_embeddings.to(device))
        train_mse = ((pred_target - target_embeddings.to(device)) ** 2).mean().item()
    print(f"Training MSE: {train_mse:.6f}")

    return predictor


def evaluate_model(
    encoder,
    generator,
    predictor,
    train_dataset,
    test_dataset,
    esm_model,
    esm_tokenizer,
    latent_dim,
    num_samples=16,
    metrics=('energy', 'sliced_wasserstein', 'mmd'),
    device='cuda',
    use_predictor=True,
    within_patient=False,
):
    """
    Evaluate the model on test data.

    Both models are evaluated on the same task: predict next timepoint (consecutive pairs).

    For within-patient models:
    - Use zeros as target embedding (model learns "move to next time" without target conditioning)

    For between-patient models:
    - Use predictor to predict target embedding from source, or oracle (actual target embedding)
    """
    results = {metric: [] for metric in metrics}
    results['pairs'] = []

    # Build test pairs - consecutive timepoints only (t -> t+1)
    # This is the "predict next timepoint" task for both models
    test_pairs = []
    for subject_id, indices in test_dataset.patient_indices.items():
        if len(indices) < 2:
            continue

        # Sort by timepoint
        idx_time_pairs = [(idx, test_dataset.metadata[idx][1]) for idx in indices]
        idx_time_pairs.sort(key=lambda x: x[1])

        # Only consecutive pairs (t -> t+1)
        for i in range(len(idx_time_pairs) - 1):
            source_idx = idx_time_pairs[i][0]
            target_idx = idx_time_pairs[i + 1][0]
            test_pairs.append((source_idx, target_idx))

    print(f"Evaluating on {len(test_pairs)} test pairs...")

    for source_idx, target_idx in tqdm(test_pairs):
        source_item = test_dataset.data[source_idx]
        target_item = test_dataset.data[target_idx]
        source_meta = test_dataset.metadata[source_idx]
        target_meta = test_dataset.metadata[target_idx]

        # Sample source sequences for encoding
        n_source = source_item['samples']['esm_input_ids'].size(0)
        n_target = target_item['samples']['esm_input_ids'].size(0)

        # Use DEFAULT_SAMPLE_SIZE for embedding
        embed_size = min(n_source, DEFAULT_SAMPLE_SIZE)
        source_embed_indices = np.random.choice(n_source, embed_size, replace=False)

        # For generation input and comparison, use num_samples
        source_indices = np.random.choice(n_source, min(n_source, num_samples), replace=False)
        target_indices = np.random.choice(n_target, min(n_target, num_samples), replace=False)

        # Prepare source batch for embedding
        source_embed_batch = {
            'esm_input_ids': source_item['samples']['esm_input_ids'][source_embed_indices].unsqueeze(0).to(device),
            'esm_attention_mask': source_item['samples']['esm_attention_mask'][source_embed_indices].unsqueeze(0).to(device),
        }

        # Prepare source batch for generation (smaller, used as input to generator)
        # Include both ESM and Progen keys to support both generator types
        # ESM DFM expects 3D: (batch_size, set_size, seq_len)
        source_batch = {
            'esm_input_ids': source_item['samples']['esm_input_ids'][source_indices].unsqueeze(0).to(device),
            'esm_attention_mask': source_item['samples']['esm_attention_mask'][source_indices].unsqueeze(0).to(device),
        }
        # Progen expects 2D: (batch_size, seq_len) - use single source sequence
        if 'progen_input_ids' in source_item['samples']:
            # Pick first source sequence for Progen (it generates num_samples from single source)
            source_batch['progen_input_ids'] = source_item['samples']['progen_input_ids'][source_indices[0]].unsqueeze(0).to(device)
            source_batch['progen_attention_mask'] = source_item['samples']['progen_attention_mask'][source_indices[0]].unsqueeze(0).to(device)

        # Prepare target batch for computing actual target embedding (for oracle comparison)
        target_embed_size = min(n_target, DEFAULT_SAMPLE_SIZE)
        target_embed_indices = np.random.choice(n_target, target_embed_size, replace=False)
        target_embed_batch = {
            'esm_input_ids': target_item['samples']['esm_input_ids'][target_embed_indices].unsqueeze(0).to(device),
            'esm_attention_mask': target_item['samples']['esm_attention_mask'][target_embed_indices].unsqueeze(0).to(device),
        }

        with torch.no_grad():
            # Encode source distribution using many samples
            source_embedding = encoder(source_embed_batch)

            if within_patient:
                # Within-patient model: use zeros as target embedding
                # The model learns "move to next time" without target conditioning
                target_embedding = torch.zeros_like(source_embedding)
            elif use_predictor and predictor is not None:
                # Predict target embedding (predictor handles delta internally if applicable)
                target_embedding = predictor(source_embedding)
            else:
                # Oracle: use actual target embedding (from many samples)
                target_embedding = encoder(target_embed_batch)

            # Sample sequences
            sampled_sequences = generator.sample(
                source_batch, source_embedding, target_embedding,
                num_samples=num_samples
            )

            # Retokenize for ESM (use generator's tokenizer for decoding if available)
            decode_tokenizer = getattr(generator, 'tokenizer', esm_tokenizer)
            sampled_ids, sampled_mask = retokenize_sequences(
                sampled_sequences, esm_tokenizer, max_length=64, device=device,
                decode_tokenizer=decode_tokenizer
            )

            # Get ESM embeddings for sampled sequences
            sampled_embs = get_esm_embeddings(esm_model, sampled_ids, sampled_mask)

            # Get ESM embeddings for actual target sequences
            target_ids = target_item['samples']['esm_input_ids'][target_indices].to(device)
            target_mask = target_item['samples']['esm_attention_mask'][target_indices].to(device)
            target_embs = get_esm_embeddings(esm_model, target_ids, target_mask)

        # Compute metrics
        for metric in metrics:
            metric_fn = METRIC_FUNCTIONS[metric]
            value = metric_fn(sampled_embs, target_embs)
            results[metric].append(value)

        results['pairs'].append({
            'source': source_meta,
            'target': target_meta,
        })

    # Aggregate results
    output = {'per_pair': results}
    for metric in metrics:
        values = np.array(results[metric])
        output[f'{metric}_mean'] = values.mean()
        output[f'{metric}_sem'] = values.std() / np.sqrt(len(values))

    return output


def main():
    parser = argparse.ArgumentParser(description='Evaluate TCR model')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Path to checkpoint directory')
    parser.add_argument('--num_samples', type=int, default=512,
                       help='Number of samples to generate per pair for evaluation')
    parser.add_argument('--embed_size', type=int, default=512,
                       help='Number of samples to use for embedding computation')
    parser.add_argument('--use_predictor', action='store_true',
                       help='Use predictor for target embedding (vs oracle)')
    parser.add_argument('--predict_delta', action='store_true',
                       help='Train predictor to predict delta (target-source) instead of target directly')
    parser.add_argument('--within_patient', action='store_true',
                       help='Evaluate within-patient model (uses zero target embedding)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Set global sample size
    global DEFAULT_SAMPLE_SIZE
    DEFAULT_SAMPLE_SIZE = args.embed_size

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device

    # Load model
    print("Loading model...")
    encoder, generator, config = load_model(args.checkpoint_dir, device)

    # Load ESM model for embedding
    print("Loading ESM model...")
    esm_model = EsmModel.from_pretrained('facebook/esm2_t6_8M_UR50D').to(device)
    esm_model.eval()
    esm_tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')

    # Load datasets
    print("Loading datasets...")
    dataset_config = config.dataset

    # Create train dataset for predictor training
    train_dataset = TCRDataset(
        seed=args.seed,
        set_size=dataset_config.set_size,
        data_dir=dataset_config.data_dir,
        data_file=dataset_config.data_file,
        max_length=dataset_config.max_length,
        within_patient=True,  # Always use within-patient for predictor training
        split='train',
        test_fraction=dataset_config.test_fraction,
    )

    # Create test dataset
    test_dataset = TCRDataset(
        seed=args.seed,
        set_size=dataset_config.set_size,
        data_dir=dataset_config.data_dir,
        data_file=dataset_config.data_file,
        max_length=dataset_config.max_length,
        within_patient=True,  # Test on within-patient transitions
        split='test',
        test_fraction=dataset_config.test_fraction,
    )

    latent_dim = config.experiment.latent_dim

    # For within-patient models, no predictor needed (uses zero target embedding)
    predictor = None
    if not args.within_patient:
        # Compute embeddings for all train repertoires
        print("Computing train embeddings...")
        train_embeddings = compute_all_embeddings(encoder, train_dataset, device)

        # Build predictor training data
        print("Building predictor training data...")
        source_embs, target_embs = build_predictor_training_data(train_embeddings, train_dataset)

        if source_embs is not None and args.use_predictor:
            print(f"Training RidgeCV predictor on {source_embs.size(0)} pairs...")
            predictor = train_predictor(
                source_embs, target_embs, latent_dim,
                device=device,
                predict_delta=args.predict_delta
            )

    # Evaluate
    print("\nEvaluating...")
    if args.within_patient:
        mode = "within_patient"
    else:
        if args.use_predictor:
            mode = "predictor_delta" if args.predict_delta else "predictor"
        else:
            mode = "oracle"
    print(f"Mode: {mode}")

    results = evaluate_model(
        encoder, generator, predictor,
        train_dataset, test_dataset,
        esm_model, esm_tokenizer,
        latent_dim=latent_dim,
        num_samples=args.num_samples,
        metrics=('energy', 'sliced_wasserstein', 'mmd'),
        device=device,
        use_predictor=args.use_predictor,
        within_patient=args.within_patient,
    )

    # Print results
    print("\n" + "=" * 60)
    print(f"Results ({mode} mode):")
    print("=" * 60)
    for metric in ['energy', 'sliced_wasserstein', 'mmd']:
        mean = results[f'{metric}_mean']
        sem = results[f'{metric}_sem']
        print(f"  {metric}: {mean:.4f} ± {sem:.4f}")

    # Save results
    output_path = os.path.join(args.checkpoint_dir, f'eval_results_{mode}.pt')
    torch.save(results, output_path)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
