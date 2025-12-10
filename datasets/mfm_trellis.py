import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List, Any
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import scipy as sp
import os
import hydra
from hydra.core.global_hydra import GlobalHydra
import ot


"""
Lightning datamodule for the organoid drug-screen (trellis) dataset.
In this code base, we use the name "trellis" as a short hand for this dataset.
"""

import time
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import pickle
import yaml as yml
import numpy as np
import torch
from collections import defaultdict


class trellis_dataset(Dataset):
    def __init__(
        self,
        control=set(["DMSO", "AH", "H2O"]),
        treatment=["O", "S", "VS", "L", "V", "F", "C", "SF", "CS", "CF", "CSF"],
        culture=["PDO", "PDOF", "F"],
        cell_type=["PDOs", "Fibs"],
        split_name='pdo21',
        seed=0,
    ):

        assert split_name in ["replicas-1", "replicas-2", "pdo21", "pdo27", "pdo75"], "split not recognized"
        if split_name == "replicas-1":
            self.split_source = (
                "organoid_data_preprocessed/replica_holdout/replica_1_holdout/data_splits_replicas_1.pickle"
            )
            data_path = "organoid_data_preprocessed/replica_holdout/replica_1_holdout/trellis_replicas_1_normalized.npy"
        elif split_name == "replicas-2":
            self.split_source = (
                "organoid_data_preprocessed/replica_holdout/replica_2_holdout/data_splits_replicas_2.pickle"
            )
            data_path = "organoid_data_preprocessed/replica_holdout/replica_2_holdout/trellis_replicas_2_normalized.npy"
        elif split_name == "pdo21":
            self.split_source = (
                "data//split_patient_test_pdo21.pickle"
            )
            data_path = "/orcd/data/omarabu/001/paolo/CoupledDistributionEmbeddings/organoid_data_preprocessed/patient_holdout/trellis_patients_pdo21_normalized.npy"
        elif split_name == "pdo27":
            self.split_source = (
                "data//split_patient_test_pdo27.pickle"
            )
            data_path = "/orcd/data/omarabu/001/paolo/CoupledDistributionEmbeddings/organoid_data_preprocessed/patient_holdout/trellis_patients_pdo27_normalized.npy"
        elif split_name == "pdo75":
            self.split_source = (
                "data/split_patient_test_pdo75.pickle"
            )
            data_path = "/orcd/data/omarabu/001/paolo/CoupledDistributionEmbeddings/organoid_data_preprocessed/patient_holdout/trellis_patients_pdo75_normalized.npy"
        else:
            raise ValueError("split not recognized")


        with open(self.split_source, "rb") as handle:
            self.data_splits = pickle.load(handle)
            # from these split we will get the background cells
        self.data = np.load(data_path)[:, :-1]

        # TODO: for the replica splits (I don't think for the patient splits) there is also a val part of the split that we can use.
        split=self.data_splits["train"]
        
        self.control = control  # identify x0
        self.treatment = treatment
        self.culture = culture
        self.cell_type = cell_type

        self.split_name = split_name

        self.split = self.__filter_control__(split)

        # construct dataset
        start = time.time()
        self.construct_data()
        end = time.time()
        print("done. Time (s):", print(end - start))


    def construct_data(self):
        self.samples_tmp, self.culture, self.x0, self.x1, self.cell_cond, self.treat_cond, self.patients = self.select_experiments()

        self.samples = self.samples_tmp

    def select_experiments(self):
        samples_tmp, cultures, sources, targets, cell_conds, treat_conds, patients = [], [], [], [], [], [], []

        for i in range(len(self.split)):
            #exp = self.split[i]

            if self.split_name == "replicas-1" or self.split_name == "replicas-2":
                exp = self.split[i]
                pdo_num = -1 # no pdo number meta data in these splits

            else:
                exp_patient = self.split[i]

                pdo_num = exp_patient.keys()
                assert len(pdo_num) == 1, "More than one pdo number!"

                pdo_num = list(pdo_num)[0]
                exp = exp_patient[pdo_num]

            x0_treatment = list(set(exp.keys()).intersection(self.control))[0]
            treatkeys = [key for key in exp.keys() if key not in self.control]

            for t in treatkeys:
                concentration = list(exp[t].keys())
                max_conc = str(max(map(int, concentration)))

                cultures_keys = list(exp[t][max_conc].keys())
                for culture in cultures_keys:

                    x0_pdos_idx, x1_pdos_idx, x0_fibs_idx, x1_fibs_idx = [], [], [], []
                    if culture in ["PDOF", "PDO"]:
                        x0_pdos_idx = exp[x0_treatment]["0"][culture][self.cell_type[0]].copy().tolist()
                        x1_pdos_idx = exp[t][max_conc][culture][self.cell_type[0]].copy().tolist()

                    if culture in ["PDOF", "F"]:
                        x0_fibs_idx = exp[x0_treatment]["0"][culture][self.cell_type[1]].copy().tolist()
                        x1_fibs_idx = exp[t][max_conc][culture][self.cell_type[1]].copy().tolist()

                    # concat x0 and x1 idcs
                    x0_idx = x0_pdos_idx + x0_fibs_idx
                    x1_idx = x1_pdos_idx + x1_fibs_idx

                    # create data
                    x0 = np.array(self.data[x0_idx])
                    x1 = np.array(self.data[x1_idx])

                    # get cell type one-hot encoding for x0 populations
                    x0_cell_pdos_idx = range(0, len(x0_pdos_idx))
                    x0_cell_fibs_idx = range(len(x0_pdos_idx), len(x0_idx))
                    cond_cell = np.zeros((x0.shape[0], len(self.cell_type)))
                    cond_cell[x0_cell_pdos_idx, 0] = 1
                    cond_cell[x0_cell_fibs_idx, 1] = 1

                    # get treatment one-hot encoding
                    treat_idx = self.treatment.index(t)
                    cond_treat = torch.nn.functional.one_hot(
                        torch.tensor(treat_idx).long(), num_classes=len(self.treatment)
                    )
                    cond_treat = cond_treat.expand(x0.shape[0], -1).detach().numpy()

                    samples_tmp.append(
                        (
                            culture,
                            x0,
                            x1,
                            cond_cell,
                            cond_treat,
                            str(pdo_num),
                        )
                    )

                    patients.append(str(pdo_num))
                    cultures.append(culture)
                    targets.append(x1)
                    cell_conds.append(cond_cell)
                    treat_conds.append(cond_treat)
            sources.append(x0)

        self.num_samples = len(samples_tmp)
        return samples_tmp, cultures, sources, targets, cell_conds, treat_conds, patients

    def __filter_control__(self, split):
        split_lst = []
        for ls in split:
            #keyset = set(ls.keys())
            if self.has_empty_element(ls):
                continue
            split_lst.append(ls)
        return split_lst

    def has_empty_element(self, nested_dict):
        for key, value in nested_dict.items():
            if isinstance(value, dict):  # Check if the item is a dictionary
                if not value:  # Check if the dictionary is empty
                    return (
                        True  # Return True immediately upon finding an empty dictionary
                    )
                else:
                    # Recursively check further in the dictionary
                    if self.has_empty_element(value):
                        return True
        return False  # Return False if no empty dictionary is found after checking all items


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        culture, x0, x1, cell_cond, treat_cond, patient = self.samples[idx]
        return (
            idx,
            culture,
            x0,
            x1,
            cell_cond,
            treat_cond,
            patient,
        )




class MFM_Trellis(Dataset):
    """Unified dataset for all SnapMMD datasets (GoM, LV, PBMC, Repressilator)."""
    
    # Dataset-specific configurations
    DATASET_CONFIGS = {
        'GoM': {
            'data_dir': 'data/realdata',
            'data_name': 'GoM_data.npz',
            'data_shape': [2],
            'has_x_scaling': False
        },
        'LV': {
            'data_dir': 'data/classic',
            'data_name': 'LV_data.npz',
            'data_shape': [2],
            'has_x_scaling': False
        },
        'PBMC': {
            'data_dir': 'data/realdata',
            'data_name': 'processed_pbmc_data_sub500_every_2_until20.npz',
            'data_shape': [30],
            'has_x_scaling': True
        },
        'Repressilator': {
            'data_dir': 'data/classic',
            'data_name': 'Repressilator_data.npz',
            'data_shape': [3],
            'has_x_scaling': False
        }
    }
    
    def __init__(
            self,
            dataset_name: str,
            testing_method: str = "forecast",
            seed: Optional[int] = None,
            set_size: int = 32,
            ot_coupling: bool = False,
            **kwargs,  # absorb any extra keyword args without failing
            ):
        """
        Args:
            dataset_name: Name of the dataset to load ('GoM', 'LV', 'PBMC', 'Repressilator')
            testing_method: Testing method to use
            seed: Random seed for reproducibility
            set_size: Number of samples per parameter set
            **kwargs: Absorb additional keyword arguments that may be supplied by Hydra. These are ignored to ensure
                robustness when the class is referenced in Hydra configs for purposes other than direct instantiation.
        """
        if dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(self.DATASET_CONFIGS.keys())}")
        
        self.dataset_name = dataset_name
        self.config = self.DATASET_CONFIGS[dataset_name]
        
        if seed is not None:
            np.random.seed(seed)
        
        self.testing_method = testing_method
        self.ot_coupling = ot_coupling
        # TODO: hmmm, maybe I interpreted set_size slightly wrong. Is it supposed to be a subset of the whole population at a given time point or just the size of the population?
        self.set_size = set_size
        
        # Resolve base directory robustly with or without Hydra
        if GlobalHydra.instance().is_initialized():
            base_dir = hydra.utils.get_original_cwd()
        else:
            base_dir = os.getcwd()
        data_path = os.path.join(base_dir, self.config['data_dir'], self.config['data_name'])
        self.full_dataset = np.load(data_path)
        
        # NOTE: indexing [:-1] is to remove the last time point, which is the target for forecasting benchmarks.
        self.data = self.full_dataset['Xs'][:-1]
        self.initial_conditions = self.full_dataset['y0']
        self.time_steps = self.full_dataset['dts']
        self.time_scale = self.full_dataset['time_scale']
        self.N_steps = self.full_dataset['N_steps']
        
        # Handle PBMC-specific X_scaling field
        if self.config['has_x_scaling']:
            self.X_scaling = self.full_dataset['X_scaling']
            
    
    def d_fun(self, source_idx, target_idx):
        return (target_idx - source_idx)/self.time_scale
        
    
    # TODO: make really really sure that you are not training on the test data.
    def __len__(self):
        if self.testing_method == "forecast":
            num = self.data.shape[0]
            return num**2 - num

        # TODO: implement interpolation as an alternative task to forecasting.
        else:
            raise NotImplementedError(f"Testing method '{self.testing_method}' not implemented")

    
    def __getitem__(self, idx):
        if self.testing_method == "forecast":
            # Map linear index to ordered pair (source_idx, target_idx) with source_idx != target_idx
            # Using n*(n-1) indexing that skips the diagonal
            # TODO: need to change this if you ever want to sample pairs from identical time-points.
            # TODO: make sure this is correct.
            n = self.data.shape[0]
            i = idx // (n - 1)
            j = idx % (n - 1)
            if j >= i:
                j += 1
            source_idx, target_idx = i, j
            
            source_samples = torch.tensor(self.data[source_idx], dtype=torch.float)
            target_samples = torch.tensor(self.data[target_idx], dtype=torch.float)
            
            subset_indices = np.random.choice(source_samples.shape[0], size=self.set_size, replace=False)
            
            source_samples = source_samples[subset_indices]
            target_samples = target_samples[subset_indices]

            if self.ot_coupling:
                # NOTE: converted to numpy to avoid CUDA issues.  
                # Compute OT coupling using POT with NumPy backend to avoid CUDA init in DataLoader workers
                source_np = source_samples.cpu().numpy()
                target_np = target_samples.cpu().numpy()
                cost = ot.dist(source_np, target_np, metric="sqeuclidean")
                G = ot.emd([], [], cost)
                # G = ot.sinkhorn([], [], cost, 1e-1)
                # G = ot.bregman.empirical_sinkhorn(src, tgt, 1e-1)

                # use all elements from ot plan
                # TODO: is random shuffling needed here?
                choices = np.arange(G.shape[0] * G.shape[1])
                idx0, idx1 = np.divmod(choices, G.shape[1])

                # OT paired samples
                source_samples = source_samples[idx0]
                target_samples = target_samples[idx1]
            
            
            return {
                'source_samples': source_samples,
                'target_samples': target_samples,
                'source_idx': source_idx,
                'target_idx': target_idx,                
                'd': self.d_fun(source_idx, target_idx),
            }
        # TODO: implement interpolation as an alternative task to forecasting.
        else:
            raise NotImplementedError(f"Testing method '{self.testing_method}' not implemented")
    
    # TODO: remove this? I don't remember putting it in.
    @property
    def data_shape(self):
        """Get the data shape for this dataset."""
        return self.config['data_shape'] 