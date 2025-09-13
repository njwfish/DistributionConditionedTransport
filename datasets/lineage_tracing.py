import os
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import torch
from torch.utils.data import Dataset
from typing import Optional, List

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

        adata = GetLTSeqData(root)
        sc.tl.pca(adata, n_comps=n_pcs)
        self.adata = adata

        self.data, self.metadata = self.generate_clone_sets()
        self.n_sets = len(self.data)
        self.src_samples, self.tgt_samples, self.src_metadata, self.tgt_metadata = self.generate_clone_set_pairs()

    def generate_clone_sets(self):
        feats = self.adata.obsm['X_pca']
        df = self.adata.obs[['clone', 'time']].astype(str)
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
        generate paired clone sets for clones that appear at two consecutive time points.
        
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
        for idx, meta in enumerate(self.metadata):
            # meta is a list: [clone, time, set_label]
            try:
                time_val = int(meta[1][0])
            except ValueError:
                continue
            key = (meta[0], time_val)
            if key not in clone_time_dict:
                clone_time_dict[key] = []
            clone_time_dict[key].append(idx)
        
        # For each clone and for times 2, 4, 6 attempt to find a matching t+1
        for (clone, t) in clone_time_dict:
            if t in [2, 4]:
                target_key = (clone, t + 2)
                if target_key in clone_time_dict:
                    # Pair all combinations from source (time t) and target (time t+1)
                    for src_idx in clone_time_dict[(clone, t)]:
                        for tgt_idx in clone_time_dict[target_key]:
                            src_tensors.append(self.data[src_idx])
                            tgt_tensors.append(self.data[tgt_idx])
                            src_metadata.append(self.metadata[src_idx])
                            tgt_metadata.append(self.metadata[tgt_idx])
        
        src_tensor = torch.stack(src_tensors)
        tgt_tensor = torch.stack(tgt_tensors)
        
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