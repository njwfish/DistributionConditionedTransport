#########################################
### beware not thoroughly tested yet ###
### will do soon --gokul ###############
########################################

import os
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from typing import Optional, List

def GetRASiData(root, file_path):
    """Load and preprocess RASi lineage tracing data"""
    root = Path(root)
    out_file = root / 'RASi_preprocessed.h5ad'
    
    if out_file.exists():
        return sc.read(out_file)
    
    # Load the data
    ad = sc.read_h5ad(file_path)
    
    # Filter for common clones (appear in day6, day15, and day22)
    clones_day6 = ad[ad.obs['timepoint'] == 'day6'].obs['clone_id'].unique()
    clones_day15 = ad[ad.obs['timepoint'] == 'day15'].obs['clone_id'].unique()
    clones_day22 = ad[ad.obs['timepoint'] == 'day22'].obs['clone_id'].unique()
    common_clones = np.intersect1d(np.intersect1d(clones_day6, clones_day15), clones_day22)
    
    with_clone = ad[ad.obs['clone_id'].isin(common_clones)]
    
    # Preprocessing
    sc.pp.normalize_total(with_clone, target_sum=1e4)
    sc.pp.log1p(with_clone)
    sc.pp.highly_variable_genes(with_clone, n_top_genes=10000)
    rna_feats = with_clone[:, with_clone.var.highly_variable]
    sc.pp.scale(rna_feats, max_value=10)
    
    root.mkdir(parents=True, exist_ok=True)
    rna_feats.write(out_file)
    return rna_feats

class RASiDataset(Dataset):
    """Dataset for RASi lineage-traced scRNA-seq"""

    def __init__(
        self,
        set_size: int = 100,
        min_cells: int = 3,
        n_pcs: int = 50,
        root: str = './data',
        file_path: str = '/orcd/data/omarabu/001/Omnicell_datasets/RASi_timeseries/RASi_perturbseq_10_04_2025.h5ad',
        data_shape: List[int] = [10000],
        seed: Optional[int] = None
    ):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.set_size = set_size
        self.min_cells = min_cells
        self.root = root
        self.data_shape = data_shape

        adata = GetRASiData(root, file_path)
        sc.tl.pca(adata, n_comps=n_pcs)
        self.adata = adata

        self.data, self.metadata = self.generate_clone_sets()
        self.n_sets = len(self.data)
        self.src_samples, self.tgt_samples, self.src_metadata, self.tgt_metadata = self.generate_clone_set_pairs()

    def generate_clone_sets(self):
        """Generate clone sets grouped by clone_id and timepoint"""
        feats = self.adata.obsm['X_pca']
        df = self.adata.obs[['clone_id', 'timepoint']].astype(str)
        df['cluster_id'] = df.agg('--'.join, axis=1)

        clusters = df['cluster_id'].unique()
        F = feats.shape[1]
        tensor_list = []
        metadata = []

        for cluster in clusters:
            idxs = np.where(df['cluster_id'] == cluster)[0]
            n_cells = len(idxs)
            num_sets = -(-n_cells // self.set_size)  # ceil(n_cells / set_size)

            for set_num in range(num_sets):
                start = set_num * self.set_size
                end = min(start + self.set_size, n_cells)
                selected = feats[idxs[start:end]]

                if len(selected) >= self.min_cells:
                    if len(selected) < self.set_size:
                        pad_size = self.set_size - len(selected)
                        pad_idxs = np.random.choice(idxs[:end], pad_size, replace=True)
                        selected = np.vstack([selected, feats[pad_idxs]])

                    tensor_list.append(torch.tensor(selected, dtype=torch.float32))
                    metadata.append(cluster.split('--') + [f"set{set_num+1}"])

        tensor = torch.stack(tensor_list) if tensor_list else torch.empty((0, self.set_size, F))
        return tensor, metadata

    def generate_clone_set_pairs(self):
        """
        Generate paired clone sets for clones that appear at consecutive time points.
        Pairs: day6->day15, day15->day22, day6->day22
        
        Returns:
            src_tensor: Tensor of source clone sets.
            tgt_tensor: Tensor of target clone sets.
            src_metadata: List of metadata corresponding to source clone sets.
            tgt_metadata: List of metadata corresponding to target clone sets.
        """
        src_tensors = []
        tgt_tensors = []
        src_metadata = []
        tgt_metadata = []
        
        # Create a dictionary mapping (clone_id, timepoint) to indices in the dataset
        clone_time_dict = {}
        for idx, meta in enumerate(self.metadata):
            # meta is a list: [clone_id, timepoint, set_label]
            clone_id = meta[0]
            timepoint = meta[1]
            key = (clone_id, timepoint)
            if key not in clone_time_dict:
                clone_time_dict[key] = []
            clone_time_dict[key].append(idx)
        
        # Define time point transitions
        transitions = [
            ('day6', 'day15'),
            ('day15', 'day22'),
        ]
        
        # For each clone, pair sets across consecutive timepoints
        for (clone_id, timepoint) in clone_time_dict:
            for src_time, tgt_time in transitions:
                if timepoint == src_time:
                    target_key = (clone_id, tgt_time)
                    if target_key in clone_time_dict:
                        # Pair all combinations from source and target timepoints
                        for src_idx in clone_time_dict[(clone_id, src_time)]:
                            for tgt_idx in clone_time_dict[target_key]:
                                src_tensors.append(self.data[src_idx])
                                tgt_tensors.append(self.data[tgt_idx])
                                src_metadata.append(self.metadata[src_idx])
                                tgt_metadata.append(self.metadata[tgt_idx])
        
        src_tensor = torch.stack(src_tensors) if src_tensors else torch.empty((0, self.set_size, self.data.shape[-1]))
        tgt_tensor = torch.stack(tgt_tensors) if tgt_tensors else torch.empty((0, self.set_size, self.data.shape[-1]))
        
        return src_tensor, tgt_tensor, src_metadata, tgt_metadata

    def __len__(self):
        return len(self.src_samples)

    def __getitem__(self, idx):
        return {
            'source_samples': self.src_samples[idx],
            'target_samples': self.tgt_samples[idx],
            'source_metadata': self.src_metadata[idx],
            'target_metadata': self.tgt_metadata[idx]
        }