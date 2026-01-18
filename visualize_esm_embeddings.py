#!/usr/bin/env python
"""
Visualize ESM embeddings for protein sequences from the test dataset.

This script:
1. Selects N random elements (families/clans) from the test data
2. For each element, computes mean-pooled ESM embeddings for all sequences
   (averaged across residues, but NOT across sequences)
3. Applies PCA to reduce dimensionality to 2D
4. Plots the first two principal components, color-coded by element

Usage:
    python visualize_esm_embeddings.py \
        --test_pt_file data/pfam/pfam_tokenized_data_clan_eval.pt \
        --num_elements 10 \
        --output_file esm_embeddings_pca.png
"""

import argparse
import os
import sys
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

# Add the project root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Default data file
DEFAULT_TEST_FILE = 'data/pfam/pfam_tokenized_data_clan_eval.pt'


def get_esm_model_and_device(device):
    """Load ESM model for computing embeddings."""
    from transformers import EsmModel
    from utils.hf_local import resolve_local_or_repo
    
    model_name = 'facebook/esm2_t6_8M_UR50D'
    resolved_name = resolve_local_or_repo(model_name)
    
    model = EsmModel.from_pretrained(resolved_name)
    model.to(device)
    model.eval()
    
    return model


def compute_sequence_embeddings(esm_model, esm_input_ids, esm_attention_mask, device):
    """
    Compute mean-pooled ESM embedding for each sequence (averaged across residues).
    
    Args:
        esm_model: ESM model
        esm_input_ids: [num_seqs, seq_len] tensor of token IDs
        esm_attention_mask: [num_seqs, seq_len] attention mask
        device: torch device
        
    Returns:
        [num_seqs, hidden_dim] tensor - one embedding per sequence
    """
    esm_input_ids = esm_input_ids.to(device)
    esm_attention_mask = esm_attention_mask.to(device)
    
    with torch.no_grad():
        outputs = esm_model(input_ids=esm_input_ids, attention_mask=esm_attention_mask)
        hidden_states = outputs.last_hidden_state  # [num_seqs, seq_len, hidden_dim]
        
        # Mean-pool across residues (sequence length dimension)
        mask = esm_attention_mask.unsqueeze(-1).float()  # [num_seqs, seq_len, 1]
        seq_embeddings = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        # [num_seqs, hidden_dim]
        
    return seq_embeddings.cpu()


def main():
    parser = argparse.ArgumentParser(description='Visualize ESM embeddings with PCA')
    parser.add_argument('--test_pt_file', type=str, default=DEFAULT_TEST_FILE,
                        help=f'Path to the test .pt file (default: {DEFAULT_TEST_FILE})')
    parser.add_argument('--num_elements', type=int, default=10,
                        help='Number of random elements to visualize (default: 10)')
    parser.add_argument('--output_file', type=str, default='esm_embeddings_pca.png',
                        help='Output file for the plot (default: esm_embeddings_pca.png)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (default: cuda if available, else cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--max_seqs_per_element', type=int, default=None,
                        help='Maximum sequences per element to include (default: all)')
    parser.add_argument('--figsize', type=float, nargs=2, default=[10, 8],
                        help='Figure size (width, height) in inches (default: 10 8)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for output figure (default: 150)')
    
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Set device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Handle relative paths
    if not os.path.isabs(args.test_pt_file):
        args.test_pt_file = os.path.join(SCRIPT_DIR, args.test_pt_file)
    
    # Verify path exists
    if not os.path.exists(args.test_pt_file):
        print(f"Error: Test .pt file not found: {args.test_pt_file}")
        sys.exit(1)
    
    # Load test data
    print(f"Loading test data from: {args.test_pt_file}")
    test_data = torch.load(args.test_pt_file, weights_only=False)
    print(f"Test data contains {len(test_data)} elements")
    
    # Select random elements
    num_elements = min(args.num_elements, len(test_data))
    selected_indices = random.sample(range(len(test_data)), num_elements)
    print(f"Selected {num_elements} random elements: {selected_indices}")
    
    # Load ESM model
    print("Loading ESM model...")
    esm_model = get_esm_model_and_device(device)
    
    # Compute embeddings for all sequences in selected elements
    all_embeddings = []
    all_labels = []
    element_names = []
    
    print("Computing ESM embeddings...")
    for i, idx in enumerate(tqdm(selected_indices, desc="Processing elements")):
        print(f"Processing element {idx}")
        print("-"*100)
        element = test_data[idx]
        samples = element['samples']
        
        # Get element name
        element_name = element.get('pfam', element.get('clan', f'Element_{idx}'))
        element_names.append(element_name)
        
        esm_input_ids = samples['esm_input_ids']
        esm_attention_mask = samples['esm_attention_mask']
        
        # Optionally limit sequences per element
        num_seqs = esm_input_ids.shape[0]
        if args.max_seqs_per_element is not None and num_seqs > args.max_seqs_per_element:
            seq_indices = random.sample(range(num_seqs), args.max_seqs_per_element)
            esm_input_ids = esm_input_ids[seq_indices]
            esm_attention_mask = esm_attention_mask[seq_indices]
            num_seqs = args.max_seqs_per_element
        
        # Compute embeddings
        embeddings = compute_sequence_embeddings(
            esm_model, esm_input_ids, esm_attention_mask, device
        )
        
        all_embeddings.append(embeddings)
        all_labels.extend([i] * embeddings.shape[0])
        
        print(f"  Element {idx} ({element_name}): {embeddings.shape[0]} sequences")
    
    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_labels = np.array(all_labels)
    
    print(f"\nTotal sequences: {len(all_labels)}")
    print(f"Embedding dimension: {all_embeddings.shape[1]}")
    
    # Apply PCA
    print("Applying PCA...")
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.4f}")
    
    # Use a colormap with distinct colors
    cmap = plt.cm.get_cmap('tab10' if num_elements <= 10 else 'tab20')
    colors = [cmap(i / num_elements) for i in range(num_elements)]
    
    # Compute global axis limits with some padding
    x_min, x_max = embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max()
    y_min, y_max = embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max()
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05
    xlim = (x_min - x_padding, x_max + x_padding)
    ylim = (y_min - y_padding, y_max + y_padding)
    
    # =========================================================================
    # Figure 1: Combined plot with all elements
    # =========================================================================
    print("Creating combined plot...")
    fig1, ax1 = plt.subplots(figsize=tuple(args.figsize))
    
    # Plot each element's sequences
    for i in range(num_elements):
        mask = all_labels == i
        ax1.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=f'{element_names[i]} (n={mask.sum()})',
            alpha=0.7,
            s=50,
            edgecolors='white',
            linewidth=0.5
        )
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    ax1.set_title(f'ESM Embeddings PCA - {num_elements} Elements from Test Data')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    
    # Add legend
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    # Save combined plot
    plt.savefig(args.output_file, dpi=args.dpi, bbox_inches='tight')
    print(f"Combined plot saved to: {args.output_file}")
    
    # =========================================================================
    # Figure 2: Panel of individual subplots
    # =========================================================================
    print("Creating panel of individual plots...")
    
    # Calculate grid dimensions
    ncols = min(4, num_elements)  # Max 4 columns
    nrows = (num_elements + ncols - 1) // ncols  # Ceiling division
    
    # Create figure with subplots
    panel_figsize = (4 * ncols, 4 * nrows)
    fig2, axes = plt.subplots(nrows, ncols, figsize=panel_figsize, squeeze=False)
    
    # Flatten axes for easy iteration
    axes_flat = axes.flatten()
    
    # Plot each element in its own subplot
    for i in range(num_elements):
        ax = axes_flat[i]
        mask = all_labels == i
        
        # Plot background (all other points in light gray)
        other_mask = ~mask
        ax.scatter(
            embeddings_2d[other_mask, 0],
            embeddings_2d[other_mask, 1],
            c='lightgray',
            alpha=0.3,
            s=20,
            edgecolors='none'
        )
        
        # Plot this element's points
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]],
            alpha=0.8,
            s=50,
            edgecolors='white',
            linewidth=0.5
        )
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f'{element_names[i]} (n={mask.sum()})', fontsize=10)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=8)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=8)
        ax.tick_params(labelsize=7)
    
    # Hide unused subplots
    for i in range(num_elements, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    fig2.suptitle(f'ESM Embeddings PCA - Individual Elements\n(gray points show other elements for context)', 
                  fontsize=12, y=1.02)
    plt.tight_layout()
    
    # Save panel plot
    panel_output_file = args.output_file.rsplit('.', 1)[0] + '_panel.' + args.output_file.rsplit('.', 1)[1]
    plt.savefig(panel_output_file, dpi=args.dpi, bbox_inches='tight')
    print(f"Panel plot saved to: {panel_output_file}")
    
    # Also save the embedding data for potential further analysis
    data_output_file = args.output_file.rsplit('.', 1)[0] + '_data.npz'
    np.savez(
        data_output_file,
        embeddings_2d=embeddings_2d,
        embeddings_full=all_embeddings,
        labels=all_labels,
        element_names=element_names,
        selected_indices=selected_indices,
        pca_components=pca.components_,
        pca_explained_variance_ratio=pca.explained_variance_ratio_
    )
    print(f"Embedding data saved to: {data_output_file}")
    
    print("\nDone! Generated files:")
    print(f"  - Combined plot: {args.output_file}")
    print(f"  - Panel plot: {panel_output_file}")
    print(f"  - Data file: {data_output_file}")
    
    plt.show()


if __name__ == '__main__':
    main()
