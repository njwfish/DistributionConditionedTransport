import os
import anndata as ad
import scanpy as sc
import pandas as pd
import json
import numpy as np
from pathlib import Path
import subprocess
import torch
from torch.utils.data import Dataset
from typing import Optional, List
from collections import defaultdict
import random

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

        self.seed = seed

        self.set_size = set_size
        self.min_cells = min_cells
        self.root = root
        self.data_shape = data_shape

        processed_dir = os.path.join(root, 'processed')
        os.makedirs(processed_dir, exist_ok=True)

        # paths
        adata_path = os.path.join(processed_dir, f'adata_pca_{n_pcs}.h5ad')
        data_path = os.path.join(processed_dir, f'clones_data_s{set_size}_m{min_cells}.pt')
        meta_path = os.path.join(processed_dir, f'clones_meta_s{set_size}_m{min_cells}.json')

        # try to load the processed adata
        if os.path.exists(adata_path):
            print(f'loading cached adata from {adata_path}  !!')
            self.adata = ad.read_h5ad(adata_path)
        else:
            print('computing pcs')
            adata = GetLTSeqData(root)
            sc.tl.pca(adata, n_comps=n_pcs)
            self.adata = adata
            adata.write(adata_path) # save it for next time!

        # now let's get our clone sets :)
        if os.path.exists(data_path) and os.path.exists(meta_path):
            print(f'loading cached clone sets from {processed_dir}!')
            self.data = torch.load(data_path)
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            print('generating clone sets')
            self.data, self.metadata = self.generate_clone_sets()
            # save our hard work!
            torch.save(self.data, data_path)
            with open(meta_path, 'w') as f:
                json.dump(self.metadata, f)

        # finish setting up!
        self.n_sets = len(self.data)
        # self.src_samples, self.tgt_samples, self.src_metadata, self.tgt_metadata = self.generate_clone_set_pairs()
        self.paired_data = self.generate_clone_set_pairs(test_split=0.9)

        self.train_srcs, self.train_tgts, self.train_src_meta, self.train_tgt_meta = self.paired_data['train']
        self.test_srcs, self.test_tgts, self.test_src_meta, self.test_tgt_meta = self.paired_data['test']

        self.mode = 'train'

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
            # a little trick for ceiling division!
            num_sets = (n_cells + self.set_size - 1) // self.set_size

            for set_num in range(num_sets):
                start = set_num * self.set_size
                end = min(start + self.set_size, n_cells)
                current_idxs = idxs[start:end]
                selected = feats[current_idxs]

                if len(selected) >= self.min_cells:
                    # pad if the set is too small!
                    if len(selected) < self.set_size:
                        pad_size = self.set_size - len(selected)
                        pad_idxs = np.random.choice(current_idxs, pad_size, replace=True)
                        selected = np.vstack([selected, feats[pad_idxs]])

                    tensor_list.append(torch.tensor(selected, dtype=torch.float32))
                    metadata.append(cluster.split('--') + [f"set{set_num+1}"])

        tensor = torch.stack(tensor_list) if tensor_list else torch.empty((0, self.set_size, F))
        return tensor, metadata

    def generate_clone_set_pairs(self, test_split: float = 0.9):
        clone_map = {}
        for i, meta in enumerate(self.metadata):
            # meta is [clone, time, set_label]
            try:
                time = int(meta[1][0])
                key = (meta[0], time)
                if key not in clone_map:
                    clone_map[key] = []
                clone_map[key].append(i)
            except (ValueError, IndexError):
                continue

        # find all clones that we can actually make pairs from!
        pairable_clones = set()
        for clone, t in clone_map:
            if t in [2, 4] and (clone, t + 2) in clone_map:
                pairable_clones.add(clone)

        # now, let's split these special clones into train and test sets!
        clones = sorted(list(pairable_clones)) # sort for reproducibility!
        np.random.seed(self.seed)
        np.random.shuffle(clones) 
        
        split_idx = int(len(clones) * (1 - test_split))
        train_clones = set(clones[:split_idx])
        test_clones = set(clones[split_idx:])

        print(f'splitting {len(clones)} clones into {len(train_clones)} train and {len(test_clones)} test')

        # helper
        def _make_pairs(clones_to_use):
            srcs, tgts, src_meta, tgt_meta = [], [], [], []
            for clone in clones_to_use:
                for t in [2, 4]: # our source time points!
                    src_key = (clone, t)
                    tgt_key = (clone, t + 2)
                    if src_key in clone_map and tgt_key in clone_map:
                        for src_i in clone_map[src_key]:
                            for tgt_i in clone_map[tgt_key]:
                                srcs.append(self.data[src_i])
                                tgts.append(self.data[tgt_i])
                                src_meta.append(self.metadata[src_i])
                                tgt_meta.append(self.metadata[tgt_i])
            
            return torch.stack(srcs), torch.stack(tgts), src_meta, tgt_meta

        train_data = _make_pairs(sorted(list(train_clones)))
        test_data = _make_pairs(sorted(list(test_clones)))
        
        return {'train': train_data, 'test': test_data}

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_srcs)
        else:
            return len(self.test_srcs)

    def __getitem__(self, idx):
        if self.mode == 'train':
            srcs = self.train_srcs
            tgts = self.train_tgts
            src_meta = self.train_src_meta
            tgt_meta = self.train_tgt_meta
        else: # test mode
            srcs = self.test_srcs
            tgts = self.test_tgts
            src_meta = self.test_src_meta
            tgt_meta = self.test_tgt_meta

        return {
            'source_samples': srcs[idx],
            'target_samples': tgts[idx],
            'source_metadata': src_meta[idx],
            'target_metadata': tgt_meta[idx]
        }

class LTSeqDatasetUnstructured(Dataset):
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

        self.seed = seed

        self.set_size = set_size
        self.min_cells = min_cells
        self.root = root
        self.data_shape = data_shape

        processed_dir = os.path.join(root, 'processed')
        os.makedirs(processed_dir, exist_ok=True)

        # paths
        adata_path = os.path.join(processed_dir, f'adata_pca_{n_pcs}.h5ad')
        data_path = os.path.join(processed_dir, f'clones_data_s{set_size}_m{min_cells}.pt')
        meta_path = os.path.join(processed_dir, f'clones_meta_s{set_size}_m{min_cells}.json')

        # try to load the processed adata
        if os.path.exists(adata_path):
            print(f'loading cached adata from {adata_path}  !!')
            self.adata = ad.read_h5ad(adata_path)
        else:
            print('computing pcs')
            adata = GetLTSeqData(root)
            sc.tl.pca(adata, n_comps=n_pcs)
            self.adata = adata
            adata.write(adata_path) # save it for next time!

        # now let's get our clone sets :)
        if os.path.exists(data_path) and os.path.exists(meta_path):
            print(f'loading cached clone sets from {processed_dir}!')
            self.data = torch.load(data_path)
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            print('generating clone sets')
            self.data, self.metadata = self.generate_clone_sets()
            # save our hard work!
            torch.save(self.data, data_path)
            with open(meta_path, 'w') as f:
                json.dump(self.metadata, f)

        # finish setting up!
        self.n_sets = len(self.data)
        # self.src_samples, self.tgt_samples, self.src_metadata, self.tgt_metadata = self.generate_clone_set_pairs()
        self.paired_data = self.generate_clone_set_pairs(test_split=0.9)

        self.train_srcs, self.train_tgts, self.train_src_meta, self.train_tgt_meta = self.paired_data['train']
        self.test_srcs, self.test_tgts, self.test_src_meta, self.test_tgt_meta = self.paired_data['test']

        self.guess_srcs, self.guess_tgts, self.guess_src_meta, self.guess_tgt_meta = self.clone_set_random_pairing()

        self.mode = 'train'

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
            # a little trick for ceiling division!
            num_sets = (n_cells + self.set_size - 1) // self.set_size

            for set_num in range(num_sets):
                start = set_num * self.set_size
                end = min(start + self.set_size, n_cells)
                current_idxs = idxs[start:end]
                selected = feats[current_idxs]

                if len(selected) >= self.min_cells:
                    # pad if the set is too small!
                    if len(selected) < self.set_size:
                        pad_size = self.set_size - len(selected)
                        pad_idxs = np.random.choice(current_idxs, pad_size, replace=True)
                        selected = np.vstack([selected, feats[pad_idxs]])

                    tensor_list.append(torch.tensor(selected, dtype=torch.float32))
                    metadata.append(cluster.split('--') + [f"set{set_num+1}"])

        tensor = torch.stack(tensor_list) if tensor_list else torch.empty((0, self.set_size, F))
        return tensor, metadata

    def generate_clone_set_pairs(self, test_split: float = 0.9):
        clone_map = {}
        for i, meta in enumerate(self.metadata):
            # meta is [clone, time, set_label]
            try:
                time = int(meta[1][0])
                key = (meta[0], time)
                if key not in clone_map:
                    clone_map[key] = []
                clone_map[key].append(i)
            except (ValueError, IndexError):
                continue

        # find all clones that we can actually make pairs from!
        pairable_clones = set()
        for clone, t in clone_map:
            if t in [2, 4] and (clone, t + 2) in clone_map:
                pairable_clones.add(clone)

        # now, let's split these special clones into train and test sets!
        clones = sorted(list(pairable_clones)) # sort for reproducibility!
        np.random.seed(self.seed)
        np.random.shuffle(clones)
        
        split_idx = int(len(clones) * (1 - test_split))
        train_clones = set(clones[:split_idx])
        test_clones = set(clones[split_idx:])

        print(f'splitting {len(clones)} clones into {len(train_clones)} train and {len(test_clones)} test')

        # helper
        def _make_pairs(clones_to_use):
            srcs, tgts, src_meta, tgt_meta = [], [], [], []
            for clone in clones_to_use:
                for t in [2, 4]: # our source time points!
                    src_key = (clone, t)
                    tgt_key = (clone, t + 2)
                    if src_key in clone_map and tgt_key in clone_map:
                        for src_i in clone_map[src_key]:
                            for tgt_i in clone_map[tgt_key]:
                                srcs.append(self.data[src_i])
                                tgts.append(self.data[tgt_i])
                                src_meta.append(self.metadata[src_i])
                                tgt_meta.append(self.metadata[tgt_i])
            
            return torch.stack(srcs), torch.stack(tgts), src_meta, tgt_meta

        # generate pairs for each split! woohoo!
        train_data = _make_pairs(sorted(list(train_clones)))
        test_data = _make_pairs(sorted(list(test_clones)))
        
        return {'train': train_data, 'test': test_data}
    
    def clone_set_random_pairing(self):

        clone_map = defaultdict(list)
        time_map = defaultdict(list)
        for i, (clone, time_str, _) in enumerate(self.metadata):
            try:
                t = int(time_str[0])
                clone_map[(clone, t)].append(i)
                time_map[t].append(i)
            except (ValueError, IndexError):
                continue

        lonely_clones = {c for c, t in clone_map if t in [2, 4] and (c, t + 2) not in clone_map}

        srcs, tgts, src_meta, tgt_meta = [], [], [], []
        for c in lonely_clones:
            for t in [2, 4]:
                tgt_t = t + 2
                if (c, t) in clone_map and (c, tgt_t) not in clone_map:
                    if time_map[tgt_t]:
                        for src_i in clone_map[(c, t)]:
                            tgt_i = random.choice(time_map[tgt_t])
                            srcs.append(self.data[src_i])
                            tgts.append(self.data[tgt_i])
                            src_meta.append(self.metadata[src_i])
                            tgt_meta.append(self.metadata[tgt_i])

        return torch.stack(srcs), torch.stack(tgts), src_meta, tgt_meta


    def __len__(self):
        if self.mode == 'train':
            return len(self.train_srcs)
        else:
            return len(self.test_srcs)

    def __getitem__(self, idx):
        if self.mode == 'train':
            # 50/50 chance to get a guessing pair instead!
            use_guess = random.random() < 0.5
            
            if use_guess:
                guess_idx = random.randint(0, len(self.guess_srcs) - 1)
                srcs = self.guess_srcs[guess_idx]
                tgts = self.guess_tgts[guess_idx]
                src_meta = self.guess_src_meta[guess_idx]
                tgt_meta = self.guess_tgt_meta[guess_idx]
            else:
                srcs = self.train_srcs[idx]
                tgts = self.train_tgts[idx]
                src_meta = self.train_src_meta[idx]
                tgt_meta = self.train_tgt_meta[idx]

        else:
            srcs = self.test_srcs[idx]
            tgts = self.test_tgts[idx]
            src_meta = self.test_src_meta[idx]
            tgt_meta = self.test_tgt_meta[idx]

        return {
            'source_samples': srcs,
            'target_samples': tgts,
            'source_metadata': src_meta,
            'target_metadata': tgt_meta
        }