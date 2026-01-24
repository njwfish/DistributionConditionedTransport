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
import time
import pickle


class trellis_dataset(Dataset):
    def __init__(
        self,
        control=set(["DMSO", "AH", "H2O"]),
        treatment=["O", "S", "VS", "L", "V", "F", "C", "SF", "CS", "CF", "CSF"],
        culture=["PDO", "PDOF", "F"],
        cell_type=["PDOs", "Fibs"],
        split_name='pdo21',
        set_size=32,
        seed=0,
        **kwargs,
    ):
        
        assert split_name in ["replicas-1", "replicas-2", "pdo21", "pdo27", "pdo75"], "split not recognized"
        if split_name == "replicas-1":
            split_source = (
                "organoid_data_preprocessed/replica_holdout/replica_1_holdout/data_splits_replicas_1.pickle"
            )
            data_path = "organoid_data_preprocessed/replica_holdout/replica_1_holdout/trellis_replicas_1_normalized.npy"
        elif split_name == "replicas-2":
            split_source = (
                "organoid_data_preprocessed/replica_holdout/replica_2_holdout/data_splits_replicas_2.pickle"
            )
            data_path = "organoid_data_preprocessed/replica_holdout/replica_2_holdout/trellis_replicas_2_normalized.npy"
        elif split_name == "pdo21":
            split_source = (
                "organoid_data_preprocessed/patient_holdout/split_patient_test_pdo21.pickle"
            )
            data_path = "organoid_data_preprocessed/patient_holdout/trellis_patients_pdo21_normalized.npy"
        elif split_name == "pdo27":
            split_source = (
                "organoid_data_preprocessed/patient_holdout/split_patient_test_pdo27.pickle"
            )
            data_path = "organoid_data_preprocessed/patient_holdout/trellis_patients_pdo27_normalized.npy"
        elif split_name == "pdo75":
            split_source = (
                "organoid_data_preprocessed/patient_holdout/split_patient_test_pdo75.pickle"
            )
            data_path = "organoid_data_preprocessed/patient_holdout/trellis_patients_pdo75_normalized.npy"
        else:
            raise ValueError("split not recognized")


        if GlobalHydra.instance().is_initialized():
            base_dir = hydra.utils.get_original_cwd()
        else:
            base_dir = os.getcwd()
        split_path = os.path.join(base_dir, split_source)
        with open(split_path, "rb") as handle:
            self.data_splits = pickle.load(handle)
            # from these split we will get the background cells
        data_path = os.path.join(base_dir, data_path)
        self.data = np.load(data_path)[:, :-1]

        # Load both train and test splits
        split_train = self.data_splits["train"]
        split_test = self.data_splits["test"]
        
        self.set_size = set_size
        self.control = control  # identify x0
        self.treatment = treatment
        self.cell_type = cell_type
        self.split_name = split_name

        self.split_train = self.__filter_control__(split_train)
        self.split_test = self.__filter_control__(split_test)

        # construct dataset
        start = time.time()
        self.construct_data()
        end = time.time()

    def get_train_predictor_bool(self, source_is_train, source_sample_idx, source_is_x0,
                                  target_is_train, target_sample_idx, target_is_x0):
        """
        Returns True if source and target form the 'natural' (x0 -> x1) pairing 
        from the same train sample.
        """
        return (source_is_train and target_is_train and 
                source_sample_idx == target_sample_idx and 
                source_is_x0 and not target_is_x0)

    def construct_data(self):
        # Process train split
        self.samples_train, _, _, _, _, _, _ = self.select_experiments(self.split_train)
        
        # Process test split
        self.samples_test, _, _, _, _, _, _ = self.select_experiments(self.split_test)
        
        # Compute total number of x populations for indexing
        # From train: x0 and x1 for each sample = 2 * n_train
        # From test: only x0 for each sample = n_test (x1's are never touched)
        self.n_train = len(self.samples_train)
        self.n_test = len(self.samples_test)
        self.total_x_populations = 2 * self.n_train + self.n_test

    def select_experiments(self, split):
        samples_tmp, cultures, sources, targets, cell_conds, treat_conds, patients = [], [], [], [], [], [], []

        for i in range(len(split)):
            if self.split_name == "replicas-1" or self.split_name == "replicas-2":
                exp = split[i]
                pdo_num = -1 # no pdo number meta data in these splits
            else:
                exp_patient = split[i]

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

        return samples_tmp, cultures, sources, targets, cell_conds, treat_conds, patients

    def __filter_control__(self, split):
        split_lst = []
        for ls in split:
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

    def _flat_idx_to_x_info(self, flat_idx):
        """
        Map a flat index to x population info.
        
        Flat index layout:
        - [0, 2*n_train): train samples, alternating x0 and x1
          - flat_idx = 2*i => train sample i, x0
          - flat_idx = 2*i+1 => train sample i, x1
        - [2*n_train, 2*n_train + n_test): test samples, only x0
          - flat_idx = 2*n_train + j => test sample j, x0
        
        Returns: (x, cell_cond, treat_cond, culture, patient, is_from_train, sample_idx, is_x0)
        """
        if flat_idx < 2 * self.n_train:
            # From train data
            sample_idx = flat_idx // 2
            is_x0 = (flat_idx % 2 == 0)
            culture, x0, x1, cell_cond, treat_cond, patient = self.samples_train[sample_idx]
            x = x0 if is_x0 else x1
            return x, cell_cond, treat_cond, culture, patient, True, sample_idx, is_x0
        else:
            # From test data (only x0)
            test_sample_idx = flat_idx - 2 * self.n_train
            culture, x0, x1, cell_cond, treat_cond, patient = self.samples_test[test_sample_idx]
            return x0, cell_cond, treat_cond, culture, patient, False, test_sample_idx, True

    def __len__(self):
        # Total pairs = (total x populations)^2
        return self.total_x_populations ** 2

    def __getitem__(self, idx):
        N = self.total_x_populations
        
        # Decompose idx into source and target flat indices
        source_flat_idx = idx // N
        target_flat_idx = idx % N
        
        # Get source info
        source_x, source_cell_cond, source_treat_cond, source_culture, source_patient, \
            source_is_train, source_sample_idx, source_is_x0 = self._flat_idx_to_x_info(source_flat_idx)
        
        # Get target info
        target_x, target_cell_cond, target_treat_cond, target_culture, target_patient, \
            target_is_train, target_sample_idx, target_is_x0 = self._flat_idx_to_x_info(target_flat_idx)
        
        source_samples = torch.tensor(source_x, dtype=torch.float)
        target_samples = torch.tensor(target_x, dtype=torch.float)
        
        # Use source's conditioning info
        treat_cond = torch.tensor(source_treat_cond, dtype=torch.float)
        cell_cond = torch.tensor(source_cell_cond, dtype=torch.float)
        
        source_subset_indices = np.random.choice(source_samples.shape[0], size=self.set_size, replace=False)
        target_subset_indices = np.random.choice(target_samples.shape[0], size=self.set_size, replace=False)
            
        source_samples = source_samples[source_subset_indices]
        target_samples = target_samples[target_subset_indices]
        
        return {
            'source_samples': source_samples,
            'target_samples': target_samples,
            #'cell_cond': cell_cond,
            #'treat_cond': treat_cond,
            'patient': source_patient,
            'culture': source_culture,
            'train_predictor_bool': self.get_train_predictor_bool(
                source_is_train, source_sample_idx, source_is_x0,
                target_is_train, target_sample_idx, target_is_x0
            )
        }
