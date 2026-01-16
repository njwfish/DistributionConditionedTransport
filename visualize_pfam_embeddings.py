#!/usr/bin/env python3
"""
Script to visualize PFAM sequences using ESM embeddings and PCA.
Loads the pre-tokenized dataset and computes ESM embeddings for each sequence,
then plots a 2D PCA scatterplot colored by PFAM family.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoModel
from tqdm import tqdm
import argparse


def load_dataset(data_path: str = 'data/pfam/pfam_tokenized_data.pt'):
    """Load the pre-tokenized PFAM dataset."""
    print(f"Loading dataset from {data_path}...")
    data = torch.load(data_path)
    print(f"Loaded {len(data)} PFAM families")
    return data


def compute_esm_embeddings(data, esm_name: str = 'facebook/esm2_t6_8M_UR50D', 
                           device: str = 'cuda', batch_size: int = 32,
                           max_seqs_per_family: int = None):
    """
    Compute ESM embeddings for all sequences in the dataset.
    
    Args:
        data: List of family dictionaries with tokenized sequences
        esm_name: Name of the ESM model to use
        device: Device to run inference on
        batch_size: Batch size for inference
        max_seqs_per_family: Maximum sequences to use per family (for faster testing)
    
    Returns:
        embeddings: numpy array of shape (n_sequences, embedding_dim)
        labels: list of PFAM family labels for each sequence
    """
    print(f"Loading ESM model: {esm_name}...")
    model = AutoModel.from_pretrained(esm_name, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    
    all_embeddings = []
    all_labels = []
    
    print("Computing embeddings...")
    with torch.no_grad():
        for family_data in tqdm(data, desc="Processing families"):
            pfam = family_data['pfam']
            input_ids = family_data['samples']['esm_input_ids']
            attention_mask = family_data['samples']['esm_attention_mask']
            
            # Optionally limit sequences per family
            if max_seqs_per_family is not None:
                input_ids = input_ids[:max_seqs_per_family]
                attention_mask = attention_mask[:max_seqs_per_family]
            
            n_seqs = input_ids.shape[0]
            
            # Process in batches
            for i in range(0, n_seqs, batch_size):
                batch_input_ids = input_ids[i:i+batch_size].to(device)
                batch_attention_mask = attention_mask[i:i+batch_size].to(device)
                
                # Get ESM embeddings
                outputs = model(input_ids=batch_input_ids, 
                               attention_mask=batch_attention_mask)
                
                # Use mean pooling over sequence length (excluding padding)
                # outputs.last_hidden_state shape: (batch, seq_len, hidden_dim)
                hidden_states = outputs.last_hidden_state
                
                # Mean pool over non-padding tokens
                mask_expanded = batch_attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                mean_embeddings = sum_embeddings / sum_mask
                
                all_embeddings.append(mean_embeddings.cpu().numpy())
                all_labels.extend([pfam] * batch_input_ids.shape[0])
    
    embeddings = np.vstack(all_embeddings)
    print(f"Computed embeddings for {len(all_labels)} sequences")
    print(f"Embedding shape: {embeddings.shape}")
    
    return embeddings, all_labels


def plot_pca(embeddings, labels, output_path: str = 'pfam_pca.png', 
             n_components: int = 2, figsize: tuple = (12, 10)):
    """
    Create a PCA visualization of the embeddings colored by PFAM family.
    
    Args:
        embeddings: numpy array of embeddings
        labels: list of PFAM labels
        output_path: path to save the figure
        n_components: number of PCA components (2 or 3)
        figsize: figure size
    """
    print(f"Running PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    embeddings_pca = pca.fit_transform(embeddings)
    
    # Get unique labels and create color map
    unique_labels = sorted(list(set(labels)))  # Sort for consistency
    n_families = len(unique_labels)
    print(f"Number of unique PFAM families: {n_families}")
    
    # Create a color map
    if n_families <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_families))
    elif n_families <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_families))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, n_families))
    
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
    point_colors = [label_to_color[label] for label in labels]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each family separately for legend
    for i, label in enumerate(unique_labels):
        mask = np.array([l == label for l in labels])
        ax.scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1], 
                   c=[label_to_color[label]], label=label, alpha=0.7, s=50)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('PFAM Sequences Embedded with ESM (PCA Visualization)')
    
    # Put legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {output_path}")
    
    # Also show explained variance
    print(f"Explained variance ratio: PC1={pca.explained_variance_ratio_[0]*100:.2f}%, "
          f"PC2={pca.explained_variance_ratio_[1]*100:.2f}%")
    
    return embeddings_pca, pca, unique_labels, label_to_color


def plot_pca_individual(embeddings_pca, labels, unique_labels, label_to_color, pca,
                        output_path: str = 'pfam_pca_individual.png',
                        nrows: int = 5, ncols: int = 8, figsize: tuple = (24, 15)):
    """
    Create a grid of individual PCA plots, one per PFAM family.
    All subplots share the same axis ranges for easy comparison.
    
    Args:
        embeddings_pca: PCA-transformed embeddings
        labels: list of PFAM labels
        unique_labels: sorted list of unique PFAM labels
        label_to_color: dict mapping labels to colors
        pca: fitted PCA object (for axis labels)
        output_path: path to save the figure
        nrows: number of rows in the grid
        ncols: number of columns in the grid
        figsize: figure size
    """
    n_families = len(unique_labels)
    print(f"Creating individual plots for {n_families} families in {nrows}x{ncols} grid...")
    
    # Compute global axis limits with some padding
    x_min, x_max = embeddings_pca[:, 0].min(), embeddings_pca[:, 0].max()
    y_min, y_max = embeddings_pca[:, 1].min(), embeddings_pca[:, 1].max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    xlim = (x_min - x_pad, x_max + x_pad)
    ylim = (y_min - y_pad, y_max + y_pad)
    
    # Create the figure with subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    
    # Convert labels to numpy array for efficient masking
    labels_array = np.array(labels)
    
    # Plot each family in its own subplot
    for idx, label in enumerate(unique_labels):
        ax = axes[idx]
        mask = labels_array == label
        n_seqs = mask.sum()
        
        # Plot all points in light gray as background
        ax.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                   c='lightgray', alpha=0.3, s=10, zorder=1)
        
        # Plot this family's points in color
        ax.scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1], 
                   c=[label_to_color[label]], alpha=0.8, s=30, zorder=2)
        
        # Set consistent axis limits
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        # Set title with family name and count
        ax.set_title(f'{label}\n(n={n_seqs})', fontsize=9)
        
        # Remove tick labels for cleaner look
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)
    
    # Hide empty subplots
    for idx in range(n_families, nrows * ncols):
        axes[idx].axis('off')
    
    # Add overall axis labels
    fig.text(0.5, 0.02, f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', 
             ha='center', fontsize=12)
    fig.text(0.02, 0.5, f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', 
             va='center', rotation='vertical', fontsize=12)
    
    # Add overall title
    fig.suptitle('PFAM Families - Individual PCA Plots (gray = all sequences)', 
                 fontsize=14, y=0.98)
    
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Individual plots saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize PFAM embeddings using ESM and PCA')
    parser.add_argument('--data_path', type=str, default='data/pfam/pfam_tokenized_data.pt',
                        help='Path to the tokenized PFAM dataset')
    parser.add_argument('--esm_model', type=str, default='facebook/esm2_t6_8M_UR50D',
                        help='ESM model name')
    parser.add_argument('--output', type=str, default='pfam_pca.png',
                        help='Output path for the figure')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--max_seqs_per_family', type=int, default=None,
                        help='Maximum sequences per family (for faster testing)')
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    # Load dataset
    data = load_dataset(args.data_path)
    
    # Compute embeddings
    embeddings, labels = compute_esm_embeddings(
        data, 
        esm_name=args.esm_model,
        device=args.device,
        batch_size=args.batch_size,
        max_seqs_per_family=args.max_seqs_per_family
    )
    
    # Save embeddings for later use
    embeddings_path = args.output.replace('.png', '_embeddings.npz')
    np.savez(embeddings_path, embeddings=embeddings, labels=np.array(labels))
    print(f"Embeddings saved to {embeddings_path}")
    
    # Plot PCA (combined view)
    embeddings_pca, pca, unique_labels, label_to_color = plot_pca(
        embeddings, labels, output_path=args.output
    )
    
    # Plot individual PFAM families in a grid
    individual_output = args.output.replace('.png', '_individual.png')
    plot_pca_individual(
        embeddings_pca, labels, unique_labels, label_to_color, pca,
        output_path=individual_output
    )
    
    print("Done!")


if __name__ == "__main__":
    main()
