import os
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Tuple
import json
import hashlib

def GetLTSeqDataWithSplit(root: str, val_fraction: float = 0.2, seed: Optional[int] = None) -> Tuple:
    """
    Load or generate LTSeq data with train/validation split.
    
    Args:
        root: Root directory for data storage
        val_fraction: Fraction of clones to use for validation
        seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_adata, val_adata, train_clones, val_clones)
    """
    root = Path(root)
    
    # Create a unique suffix based on split parameters for file naming
    split_params = f"val{val_fraction}_seed{seed}"
    split_hash = hashlib.md5(split_params.encode()).hexdigest()[:8]
    
    # File paths for split data
    train_file = root / f'LTSeq_train_{split_hash}.h5ad'
    val_file = root / f'LTSeq_val_{split_hash}.h5ad' 
    split_info_file = root / f'LTSeq_split_{split_hash}.json'
    
    # Check if split files already exist
    if train_file.exists() and val_file.exists() and split_info_file.exists():
        print(f"Loading existing train/val split from {root}")
        train_adata = sc.read(train_file)
        val_adata = sc.read(val_file)
        
        with open(split_info_file, 'r') as f:
            split_info = json.load(f)
        train_clones = set(split_info['train_clones'])
        val_clones = set(split_info['val_clones'])
        
        return train_adata, val_adata, train_clones, val_clones
    
    print(f"Generating new train/val split and saving to {root}")
    
    # Get the base preprocessed data
    base_adata = GetLTSeqData(root)
    
    # Set seed for reproducible splits
    if seed is not None:
        np.random.seed(seed)
    
    # Get unique clones and split them
    unique_clones = base_adata.obs['clone'].unique()
    unique_clones = unique_clones[unique_clones != 0]  # Remove background clone
    
    # Shuffle clones
    np.random.shuffle(unique_clones)
    
    # Split into train/val
    n_val = int(len(unique_clones) * val_fraction)
    val_clones = set(unique_clones[:n_val])
    train_clones = set(unique_clones[n_val:])
    
    # Filter data by clone assignment
    train_mask = base_adata.obs['clone'].isin(train_clones)
    val_mask = base_adata.obs['clone'].isin(val_clones)
    
    train_adata = base_adata[train_mask].copy()
    val_adata = base_adata[val_mask].copy()
    
    # Save the split data
    root.mkdir(parents=True, exist_ok=True)
    train_adata.write(train_file)
    val_adata.write(val_file)
    
    # Save split information
    split_info = {
        'train_clones': list(train_clones),
        'val_clones': list(val_clones),
        'val_fraction': val_fraction,
        'seed': seed,
        'n_train_clones': len(train_clones),
        'n_val_clones': len(val_clones),
        'n_train_cells': train_adata.n_obs,
        'n_val_cells': val_adata.n_obs
    }
    
    with open(split_info_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"Split info: {len(train_clones)} train clones ({train_adata.n_obs} cells), "
          f"{len(val_clones)} val clones ({val_adata.n_obs} cells)")
    
    return train_adata, val_adata, train_clones, val_clones

def GetLTSeqData(root):
    root = Path(root)
    out_file = root / 'LTSeq_preprocessed.h5ad'
    if out_file.exists():
        return sc.read(out_file)

    # note: this will break if O2 is down lol
    urls = {
        'counts': 'https://kleintools.hms.harvard.edu/paper_websites/state_fate2020/stateFate_inVitro_normed_counts.mtx.gz',
        'meta': 'https://kleintools.hms.harvard.edu/paper_websites/state_fate2020/stateFate_inVitro_metadata.txt.gz',
        'genes': 'https://kleintools.hms.harvard.edu/paper_websites/state_fate2020/stateFate_inVitro_gene_names.txt.gz',
        'clones': 'https://kleintools.hms.harvard.edu/paper_websites/state_fate2020/stateFate_inVitro_clone_matrix.mtx.gz',
    }

    root.mkdir(parents=True, exist_ok=True)

    for name, url in urls.items():
        fn = root / Path(url).name
        if not fn.exists():
            subprocess.run(['wget', url], cwd=root)
            subprocess.run(['gzip', '-d', fn.name], cwd=root)

    # wrangling
    counts = sc.read_mtx(root / 'stateFate_inVitro_normed_counts.mtx')
    clones = sc.read_mtx(root / 'stateFate_inVitro_clone_matrix.mtx')
    meta = pd.read_csv(root / 'stateFate_inVitro_metadata.txt', sep='\t')
    genes = pd.read_csv(root / 'stateFate_inVitro_gene_names.txt', header=None, sep='\t')

    counts.var_names = [g.upper() for g in genes[0].values]
    clone_ids = [np.argmax(clones.X[i, :]) for i in range(clones.shape[0])]

    counts.obs['clone'] = clone_ids
    counts.obs['time'] = meta['Time point'].values
    counts.obs['well'] = meta['Well'].values
    counts.obs['type'] = meta['Cell type annotation'].values
    counts.obs['SPRING1'] = meta['SPRING-x'].values
    counts.obs['SPRING2'] = meta['SPRING-y'].values

    with_clone = counts[counts.obs['clone'] != 0]

    # preprocessing
    sc.pp.normalize_total(with_clone, target_sum=1e4)
    sc.pp.log1p(with_clone)
    sc.pp.highly_variable_genes(with_clone, n_top_genes=10000)
    rna_feats = with_clone[:, with_clone.var.highly_variable]
    sc.pp.scale(rna_feats, max_value=10)

    rna_feats.write(out_file)
    return rna_feats

class LTSeqDataset(Dataset):
    """Dataset for lineage-traced scRNA-seq from Weinreb, et al., 2020"""

    def __init__(
        self,
        set_size: int = 100,
        min_cells: int = 3,
        n_pcs: int = 50,
        root: str = './data',
        data_shape: List[int] = [10000],
        val_fraction: float = 0.2,
        seed: Optional[int] = None
    ):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.set_size = set_size
        self.min_cells = min_cells
        # self.n_pcs = n_pcs
        self.root = root
        self.data_shape = data_shape
        self.val_fraction = val_fraction
        self._seed = seed

        # Load train/val split data (or generate if not exists)
        train_adata, val_adata, self.train_clones, self.val_clones = GetLTSeqDataWithSplit(
            root, val_fraction, seed
        )
        
        # Apply PCA to both datasets
        sc.tl.pca(train_adata, n_comps=n_pcs)
        sc.tl.pca(val_adata, n_comps=n_pcs)
        
        self.train_adata = train_adata
        self.val_adata = val_adata
        
        # Generate data for both splits using the loaded split data
        self.train_data, self.train_metadata = self.generate_clone_sets_from_adata(self.train_adata)
        self.val_data, self.val_metadata = self.generate_clone_sets_from_adata(self.val_adata)
        
        # Generate pairs for both splits
        self.train_src, self.train_tgt, self.train_src_meta, self.train_tgt_meta = self.generate_clone_set_pairs_from_data(
            self.train_data, self.train_metadata, self.train_clones
        )
        self.val_src, self.val_tgt, self.val_src_meta, self.val_tgt_meta = self.generate_clone_set_pairs_from_data(
            self.val_data, self.val_metadata, self.val_clones
        )
        
        # Default to training data (for backward compatibility)
        self.data = self.train_data
        self.metadata = self.train_metadata
        self.src_samples = self.train_src
        self.tgt_samples = self.train_tgt
        self.src_metadata = self.train_src_meta
        self.tgt_metadata = self.train_tgt_meta
        self.n_sets = len(self.data)

    def generate_clone_sets_from_adata(self, adata):
        """Generate clone sets from an already filtered adata object."""
        feats = adata.obsm['X_pca']
        df = adata.obs[['clone', 'time']].astype(str)
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

    def generate_clone_set_pairs_from_data(self, data, metadata, allowed_clones):
        """
        Generate paired clone sets for clones that appear at two consecutive time points.
        
        Args:
            data: The tensor data for the split
            metadata: The metadata for the split
            allowed_clones: Set of allowed clone IDs
            
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
        
        # create a dictionary mapping (clone, time) to the indices in the dataset
        clone_time_dict = {}
        for idx, meta in enumerate(metadata):
            # meta is a list: [clone, time, set_label]
            try:
                time_val = int(meta[1][0])
                clone_id = int(meta[0])
                # Only include clones that are in the allowed set (should already be filtered, but double-check)
                if clone_id not in allowed_clones:
                    continue
            except ValueError:
                continue
            key = (meta[0], time_val)
            if key not in clone_time_dict:
                clone_time_dict[key] = []
            clone_time_dict[key].append(idx)
        
        # For each clone and for times 2, 4, 6 attempt to find a matching t+2
        for (clone, t) in clone_time_dict:
            if t in [2, 4]:
                target_key = (clone, t + 2)
                if target_key in clone_time_dict:
                    # Pair all combinations from source (time t) and target (time t+2)
                    for src_idx in clone_time_dict[(clone, t)]:
                        for tgt_idx in clone_time_dict[target_key]:
                            src_tensors.append(data[src_idx])
                            tgt_tensors.append(data[tgt_idx])
                            src_metadata.append(metadata[src_idx])
                            tgt_metadata.append(metadata[tgt_idx])
        
        if src_tensors:
            src_tensor = torch.stack(src_tensors)
            tgt_tensor = torch.stack(tgt_tensors)
        else:
            # Handle empty case
            sample_shape = data.shape[1:] if len(data) > 0 else (self.set_size, 50)  # Default shape
            src_tensor = torch.empty((0,) + sample_shape)
            tgt_tensor = torch.empty((0,) + sample_shape)
        
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
    
    def set_mode(self, mode='train'):
        """Switch between training and validation modes."""
        if mode == 'train':
            self.data = self.train_data
            self.metadata = self.train_metadata
            self.src_samples = self.train_src
            self.tgt_samples = self.train_tgt
            self.src_metadata = self.train_src_meta
            self.tgt_metadata = self.train_tgt_meta
        elif mode == 'val':
            self.data = self.val_data
            self.metadata = self.val_metadata
            self.src_samples = self.val_src
            self.tgt_samples = self.val_tgt
            self.src_metadata = self.val_src_meta
            self.tgt_metadata = self.val_tgt_meta
        else:
            raise ValueError("mode must be 'train' or 'val'")
        
        self.n_sets = len(self.data)
    
    def get_train_dataset(self):
        """Return a copy of this dataset set to training mode."""
        dataset_copy = LTSeqDataset.__new__(LTSeqDataset)  # Create without calling __init__
        dataset_copy.__dict__.update(self.__dict__)  # Copy all attributes
        dataset_copy.set_mode('train')
        return dataset_copy
    
    def get_val_dataset(self):
        """Return a copy of this dataset set to validation mode."""
        dataset_copy = LTSeqDataset.__new__(LTSeqDataset)  # Create without calling __init__
        dataset_copy.__dict__.update(self.__dict__)  # Copy all attributes
        dataset_copy.set_mode('val')
        return dataset_copy
    
    @property
    def train_size(self):
        """Number of training pairs."""
        return len(self.train_src)
    
    @property
    def val_size(self):
        """Number of validation pairs."""
        return len(self.val_src)
    
    @property
    def n_train_clones(self):
        """Number of training clones."""
        return len(self.train_clones)
    
    @property
    def n_val_clones(self):
        """Number of validation clones."""
        return len(self.val_clones)
    
    def get_split_info(self):
        """Get information about the current train/val split."""
        split_params = f"val{self.val_fraction}_seed{self._seed}"
        split_hash = hashlib.md5(split_params.encode()).hexdigest()[:8]
        
        return {
            'val_fraction': self.val_fraction,
            'split_hash': split_hash,
            'n_train_clones': len(self.train_clones),
            'n_val_clones': len(self.val_clones),
            'train_size': len(self.train_src),
            'val_size': len(self.val_src),
            'train_file': f'LTSeq_train_{split_hash}.h5ad',
            'val_file': f'LTSeq_val_{split_hash}.h5ad',
            'split_info_file': f'LTSeq_split_{split_hash}.json'
        }
    
    def get_split_file_paths(self):
        """Get the file paths for the current split."""
        info = self.get_split_info()
        root = Path(self.root)
        return {
            'train_file': root / info['train_file'],
            'val_file': root / info['val_file'], 
            'split_info_file': root / info['split_info_file']
        }
