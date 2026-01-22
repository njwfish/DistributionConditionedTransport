#!/usr/bin/env python
"""
Plot PCA of ESM2 embeddings for TCR sequences.

Creates:
1. A scatterplot of all data color-coded by subject_id
2. A panel of scatterplots (8x5) with one per subject, color-coded by timepoint

Usage:
    python plot_pca_embeddings.py
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_file = os.path.join(base_dir, 'data/tcr/tcr_esm_embeddings.pt')
    
    # Load embeddings
    logger.info(f'Loading embeddings from {embeddings_file}')
    data = torch.load(embeddings_file)
    
    embeddings_list = data['embeddings']
    metadata_list = data['metadata']
    
    logger.info(f'Loaded {len(embeddings_list)} repertoires')
    
    # Collect all embeddings and labels
    all_embeddings = []
    all_subject_ids = []
    all_timepoints = []
    repertoire_indices = []  # Track which repertoire each embedding came from
    
    for rep_idx, (emb_data, (subject_id, timepoint)) in enumerate(zip(embeddings_list, metadata_list)):
        emb = emb_data['embeddings'].numpy()
        n_seqs = emb.shape[0]
        
        all_embeddings.append(emb)
        all_subject_ids.extend([subject_id] * n_seqs)
        all_timepoints.extend([timepoint] * n_seqs)
        repertoire_indices.extend([rep_idx] * n_seqs)
    
    all_embeddings = np.vstack(all_embeddings)
    all_subject_ids = np.array(all_subject_ids)
    all_timepoints = np.array(all_timepoints)
    
    logger.info(f'Total sequences: {len(all_embeddings):,}')
    logger.info(f'Embedding shape: {all_embeddings.shape}')
    
    # Compute PCA
    logger.info('Computing PCA...')
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(all_embeddings)
    
    logger.info(f'Explained variance ratio: {pca.explained_variance_ratio_}')
    
    # Get unique subjects and sort them
    unique_subjects = sorted(set(all_subject_ids), key=lambda x: (int(x.split('-')[1]) if '-' in x else 0))
    n_subjects = len(unique_subjects)
    logger.info(f'Number of unique subjects: {n_subjects}')
    
    # Compute global axis limits with some padding
    x_min, x_max = pca_coords[:, 0].min(), pca_coords[:, 0].max()
    y_min, y_max = pca_coords[:, 1].min(), pca_coords[:, 1].max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    xlim = (x_min - x_pad, x_max + x_pad)
    ylim = (y_min - y_pad, y_max + y_pad)
    
    # =========================================================================
    # Figure 1: All data color-coded by subject_id
    # =========================================================================
    logger.info('Creating figure 1: all data by subject...')
    
    fig1, ax1 = plt.subplots(figsize=(12, 10))
    
    # Create color map for subjects
    cmap_subjects = plt.cm.get_cmap('tab20', n_subjects)
    subject_to_color = {subj: cmap_subjects(i) for i, subj in enumerate(unique_subjects)}
    
    # Plot each subject with a different color
    for subj in unique_subjects:
        mask = all_subject_ids == subj
        ax1.scatter(
            pca_coords[mask, 0], 
            pca_coords[mask, 1],
            c=[subject_to_color[subj]],
            label=subj,
            alpha=0.5,
            s=1,
            rasterized=True  # For faster rendering with many points
        )
    
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax1.set_title('TCR ESM2 Embeddings - All Subjects')
    
    # Legend outside the plot
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, markerscale=5)
    
    fig1.tight_layout()
    fig1_path = os.path.join(base_dir, 'pca_all_subjects.png')
    fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
    logger.info(f'Saved: {fig1_path}')
    plt.close(fig1)
    
    # =========================================================================
    # Figure 2: Panel of scatterplots, one per subject, color-coded by timepoint
    # =========================================================================
    logger.info('Creating figure 2: panel by subject with timepoint coloring...')
    
    n_rows, n_cols = 8, 5
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(20, 32))
    axes = axes.flatten()
    
    # Get global timepoint range for consistent color scaling
    global_min_time = all_timepoints.min()
    global_max_time = all_timepoints.max()
    
    for idx, subj in enumerate(unique_subjects):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        mask = all_subject_ids == subj
        
        subj_coords = pca_coords[mask]
        subj_times = all_timepoints[mask]
        
        # Get unique timepoints for this subject
        unique_times = sorted(set(subj_times))
        n_times = len(unique_times)
        
        if n_times == 1:
            # Single timepoint: use middle of colormap
            colors = np.full(len(subj_times), 0.5)
        else:
            # Normalize timepoints to [0, 1] for this subject
            time_min, time_max = min(unique_times), max(unique_times)
            colors = (subj_times - time_min) / (time_max - time_min)
        
        scatter = ax.scatter(
            subj_coords[:, 0],
            subj_coords[:, 1],
            c=colors,
            cmap='coolwarm',
            vmin=0,
            vmax=1,
            alpha=0.6,
            s=2,
            rasterized=True
        )
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f'{subj} (t={unique_times[0]}-{unique_times[-1]}, n={len(subj_coords):,})', fontsize=9)
        ax.tick_params(labelsize=7)
        
        # Add colorbar showing timepoints
        if n_times > 1:
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels([str(unique_times[0]), str(unique_times[-1])])
            cbar.ax.tick_params(labelsize=6)
    
    # Hide unused subplots
    for idx in range(n_subjects, len(axes)):
        axes[idx].axis('off')
    
    fig2.suptitle('TCR ESM2 Embeddings by Subject (color = timepoint: blue=early, red=late)', 
                  fontsize=14, y=1.01)
    fig2.tight_layout()
    
    fig2_path = os.path.join(base_dir, 'pca_by_subject_timepoint.png')
    fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
    logger.info(f'Saved: {fig2_path}')
    plt.close(fig2)
    
    # =========================================================================
    # Figure 3 (bonus): Overlay plot showing temporal progression per subject
    # =========================================================================
    logger.info('Creating figure 3: temporal progression overlay...')
    
    fig3, ax3 = plt.subplots(figsize=(12, 10))
    
    # For each subject, compute mean position at each timepoint and draw arrows
    for subj in unique_subjects:
        mask = all_subject_ids == subj
        subj_coords = pca_coords[mask]
        subj_times = all_timepoints[mask]
        
        unique_times = sorted(set(subj_times))
        
        if len(unique_times) > 1:
            # Compute mean position at each timepoint
            mean_positions = []
            for t in unique_times:
                t_mask = subj_times == t
                mean_pos = subj_coords[t_mask].mean(axis=0)
                mean_positions.append(mean_pos)
            
            mean_positions = np.array(mean_positions)
            
            # Plot trajectory
            color = subject_to_color[subj]
            ax3.plot(mean_positions[:, 0], mean_positions[:, 1], 
                    c=color, alpha=0.7, linewidth=1.5, label=subj)
            
            # Mark start and end
            ax3.scatter(mean_positions[0, 0], mean_positions[0, 1], 
                       c=[color], s=50, marker='o', edgecolors='black', linewidths=0.5)
            ax3.scatter(mean_positions[-1, 0], mean_positions[-1, 1], 
                       c=[color], s=100, marker='*', edgecolors='black', linewidths=0.5)
    
    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim)
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax3.set_title('TCR ESM2 Embeddings - Mean Trajectories Over Time\n(circle=start, star=end)')
    ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    
    fig3.tight_layout()
    fig3_path = os.path.join(base_dir, 'pca_temporal_trajectories.png')
    fig3.savefig(fig3_path, dpi=150, bbox_inches='tight')
    logger.info(f'Saved: {fig3_path}')
    plt.close(fig3)
    
    logger.info('Done!')


if __name__ == '__main__':
    main()
