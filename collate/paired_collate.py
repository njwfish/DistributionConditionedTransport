import torch
import random
from typing import List, Dict, Any, Union


class PairedCollate:
    """
    A collate function class that creates source/target pairs from batches.
    Provides an object-based interface consistent with SetMixer.
    
    Example usage with DataLoader:
        paired_collate = PairedCollate(method='random')
        dataset = MNISTDataset(...)
        dataloader = DataLoader(
            dataset, 
            batch_size=32, 
            collate_fn=paired_collate.collate_fn
        )
    """
    def __init__(self, method: str = 'random', shift: int = 1, allow_cyclic: bool = True):
        """
        Args:
            method: Pairing method - 'random' or 'shift'
            shift: Only used for shift method - how many positions to shift
            allow_cyclic: Only used for shift method - whether to allow cyclic pairing (default: True)
        """
        self.method = method
        self.shift = shift
        self.allow_cyclic = allow_cyclic
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply pairing to a batch."""
        if self.method == 'random':
            return random_permutation_collate_fn(batch)
        elif self.method == 'shift':
            return shift_pairing_collate_fn(batch, shift=self.shift, allow_cyclic=self.allow_cyclic)
        else:
            raise ValueError(f"Unknown pairing method: {self.method}")
    
    def collate_fn(self, batch: list):
        """
        Custom collate function that recursively collates batches of nested dictionaries and tensors,
        then applies pairing.
        """
        def recursive_collate(batch_part):
            if isinstance(batch_part[0], torch.Tensor):
                return torch.stack(batch_part)
            elif isinstance(batch_part[0], dict):
                return {key: recursive_collate([b[key] for b in batch_part]) for key in batch_part[0]}
            else:
                return batch_part  # for strings, lists, or other non-tensor data
            
        # First collate, then convert to list format for pairing
        collated = recursive_collate(batch)
        
        # Convert collated dict format to list of dicts for pairing functions
        if isinstance(collated, dict):
            batch_size = None
            keys = collated.keys()
            
            # Find batch size from first tensor
            for key, value in collated.items():
                if isinstance(value, torch.Tensor) and len(value.shape) >= 1:
                    batch_size = value.shape[0]
                    break
            
            if batch_size is None:
                return collated
                
            # Convert to list of dicts
            batch_list = []
            for i in range(batch_size):
                item = {}
                for key in keys:
                    if isinstance(collated[key], torch.Tensor):
                        item[key] = collated[key][i]
                    elif isinstance(collated[key], list):
                        item[key] = collated[key][i]
                    else:
                        item[key] = collated[key]
                batch_list.append(item)
                
            # Apply pairing
            return self(batch_list)
        else:
            return self(batch)


def shift_pairing_collate_fn(batch: List[Dict[str, Any]], shift: int = 1, allow_cyclic: bool = True) -> Dict[str, Any]:
    """
    Collate function that pairs sets by shifting indices within the batch.
    
    Takes a batch where each item has 'samples' and creates source/target pairs
    by shifting the set indices (ensuring i != i').
    
    Args:
        batch: List of dictionaries, each containing 'samples' with shape (set_size, dim)
        shift: How many positions to shift for target pairing (default: 1)
        allow_cyclic: Whether to allow cyclic pairing (default: True)
    
    Returns:
        Dictionary with:
            - 'source_samples': shape (k, set_size, dim) if allow_cyclic=True, else (k-shift, set_size, dim)
            - 'target_samples': shape (k, set_size, dim) if allow_cyclic=True, else (k-shift, set_size, dim)
    """
    if len(batch) == 0:
        return {}
    
    k = len(batch)  # number of sets
    
    if allow_cyclic:
        # Original behavior: use modular arithmetic for cyclic pairing
        source_indices = list(range(k))
        target_indices = [(i + shift) % k for i in range(k)]
    else:
        # New behavior: no cyclic pairing, last 'shift' elements remain unpaired
        num_pairs = k - shift
        if num_pairs <= 0:
            # Not enough elements to create any pairs without cyclic pairing
            return {}
        
        source_indices = list(range(k-shift))
        target_indices = [(i + shift) for i in range(k-shift)]
    
    result = {}
    keys = batch[0].keys()
    
    for key in keys:
        if key == 'samples':
            # Stack all samples first: (k, set_size, dim)
            all_samples = torch.stack([batch[i][key] for i in range(k)], dim=0)
            
            # Create source and target by indexing
            result['source_samples'] = all_samples[source_indices]  # Same as all_samples
            result['target_samples'] = all_samples[target_indices]  # Shifted version
            
        elif key == 'raw_texts':
            # Handle raw texts with shifted pairing
            all_texts = [batch[i][key] for i in range(k)]
            result['source_raw_texts'] = [all_texts[i] for i in source_indices]
            result['target_raw_texts'] = [all_texts[i] for i in target_indices]
            
        else:
            # Handle other keys normally - just stack them
            values = [batch[i][key] for i in range(k)]
            if isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values, dim=0)
            else:
                result[key] = values
    
    return result


def random_permutation_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function that pairs sets using random permutation within the batch.
    
    Takes a batch of k sets and creates source/target pairs using random permutation
    (ensuring i != i' for all pairs).
    
    Args:
        batch: List of dictionaries, each containing 'samples' with shape (set_size, dim)
    
    Returns:
        Dictionary with:
            - 'source_samples': shape (k, set_size, dim)
            - 'target_samples': shape (k, set_size, dim) with random permutation
    """
    if len(batch) == 0:
        return {}
    
    k = len(batch)  # number of sets
    
    # Create a random permutation ensuring no self-pairing
    if k == 1:
        # Special case: if only one set, we have to pair it with itself
        target_indices = [0]
    else:
        # Generate a derangement (permutation with no fixed points)
        target_indices = list(range(k))
        while True:
            random.shuffle(target_indices)
            # Check if it's a valid derangement (no i maps to itself)
            if all(i != target_indices[i] for i in range(k)):
                break
    
    source_indices = list(range(k))
    
    result = {}
    keys = batch[0].keys()
    
    for key in keys:
        if key == 'samples':
            # Handle samples (tensor or dict)
            if isinstance(batch[0][key], torch.Tensor):
                # Stack all samples: (k, set_size, dim)
                all_samples = torch.stack([batch[i][key] for i in range(k)], dim=0)
                result['source_samples'] = all_samples[source_indices]
                result['target_samples'] = all_samples[target_indices]
            elif isinstance(batch[0][key], dict):
                # Handle dictionary samples
                source_samples = [batch[i][key] for i in source_indices]
                target_samples = [batch[i][key] for i in target_indices]
                result['source_samples'] = _collate_dict_samples(source_samples)
                result['target_samples'] = _collate_dict_samples(target_samples)
            else:
                result['source_samples'] = [batch[i][key] for i in source_indices]
                result['target_samples'] = [batch[i][key] for i in target_indices]
                
        elif key == 'raw_texts':
            # Handle raw texts with permuted pairing
            all_texts = [batch[i][key] for i in range(k)]
            result['source_raw_texts'] = [all_texts[i] for i in source_indices]
            result['target_raw_texts'] = [all_texts[i] for i in target_indices]
            
        else:
            # Handle other keys normally
            values = [batch[i][key] for i in range(k)]
            if isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values, dim=0)
            else:
                result[key] = values
    
    return result


def _collate_dict_samples(dict_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Helper function to collate dictionary samples.
    
    Args:
        dict_samples: List of dictionaries with same keys
        
    Returns:
        Dictionary with collated values
    """
    if len(dict_samples) == 0:
        return {}
    
    result = {}
    keys = dict_samples[0].keys()
    
    for key in keys:
        values = [sample[key] for sample in dict_samples]
        
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values, dim=0)
        else:
            result[key] = values
    
    return result


def cross_dataset_pairing_collate_fn(source_dataset, target_dataset):
    """
    Factory function that creates a collate function for pairing samples 
    from two different datasets.
    
    Args:
        source_dataset: Dataset to sample sources from
        target_dataset: Dataset to sample targets from
        
    Returns:
        Collate function that pairs samples from the two datasets
    """
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Note: This assumes the batch comes from source_dataset and we'll
        randomly sample from target_dataset to create pairs.
        """
        if len(batch) == 0:
            return {}
        
        k = len(batch)  # number of sets in source batch
        
        # Sample random indices from target dataset
        target_indices = [random.randint(0, len(target_dataset) - 1) for _ in range(k)]
        
        # Get target samples
        target_items = [target_dataset[idx] for idx in target_indices]
        
        result = {}
        
        # Process source samples (from batch)
        source_keys = batch[0].keys()
        for key in source_keys:
            if key == 'samples':
                if isinstance(batch[0][key], torch.Tensor):
                    result['source_samples'] = torch.stack([batch[i][key] for i in range(k)], dim=0)
                elif isinstance(batch[0][key], dict):
                    result['source_samples'] = _collate_dict_samples([batch[i][key] for i in range(k)])
                else:
                    result['source_samples'] = [batch[i][key] for i in range(k)]
            elif key == 'raw_texts':
                result['source_raw_texts'] = [batch[i][key] for i in range(k)]
        
        # Process target samples
        for key in target_items[0].keys():
            if key == 'samples':
                if isinstance(target_items[0][key], torch.Tensor):
                    result['target_samples'] = torch.stack([target_items[i][key] for i in range(k)], dim=0)
                elif isinstance(target_items[0][key], dict):
                    result['target_samples'] = _collate_dict_samples([target_items[i][key] for i in range(k)])
                else:
                    result['target_samples'] = [target_items[i][key] for i in range(k)]
            elif key == 'raw_texts':
                result['target_raw_texts'] = [target_items[i][key] for i in range(k)]
        
        return result
    
    return collate_fn


# Alias for the most commonly used function
random_pairing_collate_fn = random_permutation_collate_fn


def test_paired_collate():
    """Test the PairedCollate class."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test random pairing
    paired_collate = PairedCollate(method='random')
    
    # Test with simple tensor data converted to dict format
    batch_as_list = [
        {'samples': torch.randn(20, 3), 'metadata': i}
        for i in range(10)
    ]
    
    result = paired_collate.collate_fn(batch_as_list)
    
    # Check that it returns paired source/target structure
    assert 'source_samples' in result
    assert 'target_samples' in result
    assert result['source_samples'].shape == (10, 20, 3)
    assert result['target_samples'].shape == (10, 20, 3)
    assert 'metadata' in result
    
    # Test shift pairing
    shift_collate = PairedCollate(method='shift', shift=2)
    result_shift = shift_collate.collate_fn(batch_as_list)
    
    assert 'source_samples' in result_shift
    assert 'target_samples' in result_shift
    assert result_shift['source_samples'].shape == (10, 20, 3)
    assert result_shift['target_samples'].shape == (10, 20, 3)
    
    # Verify shift logic: target[i] should be source[(i+2)%10]
    for i in range(10):
        expected_target_idx = (i + 2) % 10
        # The source_samples should just be the original order
        # The target_samples should be shifted
        assert torch.equal(result_shift['source_samples'][i], batch_as_list[i]['samples'])
        assert torch.equal(result_shift['target_samples'][i], batch_as_list[expected_target_idx]['samples'])
    
    print("All PairedCollate tests passed!")


if __name__ == "__main__":
    test_paired_collate() 