import scanpy as sc
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, List
import os

class CrossTissueDataset(Dataset):
    """Eraslan et al 2022 cross tissue atlas, grouped by donor_id."""

    def __init__(
        self,
        split: str = 'train',  # 'train' or 'test'
        set_size: int = 128,
        min_cells: int = 3,
        root: str = './data',
        seed: Optional[int] = None,
        data_shape: Optional[List[int]] = None,
        n_unique_sets: Optional[int] = None,
    ):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        assert split in ['train', 'test'], "split must be train or test"
        
        self.split = split
        self.set_size = set_size
        self.min_cells = min_cells
        self.root = Path(root)

        # Load data
        # print(os.listdir(self.root))
        adata = sc.read(self.root / 'eraslan2022.h5ad')
        
        # rescale PCs to unit variance zero mean
        adata.obsm['X_pca'] = (adata.obsm['X_pca'] - np.mean(adata.obsm['X_pca'], axis=0)) / np.std(adata.obsm['X_pca'], axis=0)
        
        self.adata = adata

        # Get sorted donor IDs and split into train/test (8 and 8)
        all_donors = sorted(adata.obs['donor_id'].unique())
        mid = len(all_donors) // 2
        train_donors = all_donors[:mid]
        test_donors = all_donors[mid:]
        
        self.donors = train_donors if split == 'train' else test_donors
        
        # Filter adata to only include donors in this split
        self.adata = adata[adata.obs['donor_id'].isin(self.donors)]
        
        # Store features and build donor index mapping
        self.feats = self.adata.obsm['X_pca']
        self.donor_ids = self.adata.obs['donor_id'].values
        
        # Build index mapping for each donor
        self.donor_indices = {}
        for donor in self.donors:
            idxs = np.where(self.donor_ids == donor)[0]
            if len(idxs) >= self.min_cells:
                self.donor_indices[donor] = idxs
        
        # List of valid donors (those with enough cells)
        self.valid_donors = list(self.donor_indices.keys())
        self.n_donors = len(self.valid_donors)

    def __len__(self):
        return self.n_donors

    def _sample_donor(self, donor):
        """sample set_size cells from the given donor."""
        idxs = self.donor_indices[donor]
        n_cells = len(idxs)
        
        if n_cells >= self.set_size:
            # Sample without replacement
            selected_idxs = np.random.choice(idxs, self.set_size, replace=False)
        else:
            # Sample with replacement if not enough cells
            selected_idxs = np.random.choice(idxs, self.set_size, replace=True)
        
        return torch.tensor(self.feats[selected_idxs], dtype=torch.float32), selected_idxs

    def __getitem__(self, idx):
        # Get source donor
        source_donor = self.valid_donors[idx]
        
        # Random target donor
        target_idx = np.random.randint(0, self.n_donors)
        target_donor = self.valid_donors[target_idx]
        
        source_samples, source_adata_indices = self._sample_donor(source_donor)
        target_samples, target_adata_indices = self._sample_donor(target_donor)
        
        return {
            'source_samples': source_samples,
            'target_samples': target_samples,
            'source_metadata': {'donor_id': source_donor, 'adata_indices': source_adata_indices},
            'target_metadata': {'donor_id': target_donor, 'adata_indices': target_adata_indices},
            'source_idx': idx,
            'target_idx': target_idx
        }