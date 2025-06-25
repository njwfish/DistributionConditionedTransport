import torch

def generate_k_sparse_dirichlet_probs(
        batch_size: int, 
        k: int = 2, 
        alpha: float = 1.0,
        batch_size_out: int = None,
        device: torch.device = None
    ) -> torch.Tensor:
    """
    Generate mixing probabilities where each new set is a mixture of k randomly chosen
    source sets, with mixture weights drawn from a Dirichlet distribution.
    Vectorized implementation.
    
    Args:
        batch_size: Number of sets
        k: Number of source sets to mix for each new set (k <= batch_size)
        alpha: Concentration parameter for Dirichlet distribution
              alpha < 1: Sparse (prefer mixing from one dominant set)
              alpha = 1: Uniform (all mixing proportions equally likely)
              alpha > 1: Dense (prefer more even mixing)
        device: torch device
    
    Returns:
        Tensor of shape (batch_size, batch_size) with mixing probabilities
    """
    if batch_size_out is None:
        batch_size_out = batch_size
    if k > batch_size:
        raise ValueError(f"k ({k}) cannot be larger than batch_size ({batch_size})")
    
    # Generate all source set indices at once (batch_size, k)
    source_sets = torch.argsort(torch.rand(batch_size_out, batch_size, device=device), dim=1)[:, :k]
    
    # Generate all Dirichlet weights at once (batch_size, k)
    weights = torch.distributions.Dirichlet(
        torch.ones(batch_size_out, k, device=device) * alpha
    ).sample()
    
    # Create sparse mixing matrix using scatter
    mix_probs = torch.zeros(batch_size_out, batch_size, device=device)
    mix_probs.scatter_(1, source_sets, weights)
    
    return mix_probs

import torch
import torch.nn.functional as F

import torch

def rowwise_unique(mat, k):
    n_rows, n_cols = mat.shape
    # Sort along rows
    sorted_mat, _ = torch.sort(mat, dim=1)
    
    # Find where values change within each row
    diffs = torch.ones_like(sorted_mat, dtype=torch.bool)
    diffs[:, 1:] = sorted_mat[:, 1:] != sorted_mat[:, :-1]
    
    # Get indices of the unique elements
    idx = torch.arange(n_cols, device=mat.device).expand(n_rows, -1)
    idx = idx.masked_fill(~diffs, n_cols)  # mask duplicates with large index
    idx_sorted, sort_idx = torch.sort(idx, dim=1)
    
    # Use the sorted indices to gather unique values
    sorted_mat_gathered = torch.gather(sorted_mat, 1, sort_idx)
    
    # Take first k columns
    result = sorted_mat_gathered[:, :k]
    
    # If there are fewer uniques than k, repeat the last valid element
    # Find how many valid (non-masked) values per row
    valid_counts = (idx_sorted < n_cols).sum(dim=1)
    needs_padding = valid_counts < k
    if needs_padding.any():
        last_valid_idx = valid_counts.clamp(max=n_cols-1) - 1
        last_valid_vals = torch.gather(sorted_mat, 1, last_valid_idx.unsqueeze(1)).expand(-1, k)
        mask = torch.arange(k, device=mat.device).expand(n_rows, -1) >= valid_counts.unsqueeze(1)
        result = torch.where(mask, last_valid_vals, result)
    
    return result



def mix_batch_sets(
        data, 
        mix_probs: torch.Tensor = None, 
        mixed_set_size: int = None,
        n_mixed_sets: int = None,
        replacement: bool = True,
        k: int = None,
        alpha: float = 1.0
    ):
    """
    Mix sets by sampling points from existing sets according to mixing probabilities.
    Handles both raw tensor inputs and dictionary inputs with metadata.
    For dictionary inputs, automatically detects which keys contain sample-specific data
    based on tensor shapes matching (batch_size, set_size, ...).
    
    Args:
        data: Either:
            - Tensor of shape (batch_size, set_size, features)
            - Dictionary with 'samples' key and optional metadata
        mix_probs: Optional mixing probability matrix of shape (batch_size, batch_size).
                  Each row represents sampling probabilities from the original sets to create
                  a new mixed set (rows sum to 1).
                  If None, uniform probabilities will be used.
        mixed_set_size: Size of the output sets. If None, same as input set_size.
        replacement: Whether to sample with replacement.
        k: If mix_probs is None, number of source sets to mix (passed to generate_k_sparse_dirichlet_probs)
        alpha: If mix_probs is None, Dirichlet concentration parameter (passed to generate_k_sparse_dirichlet_probs)
    
    Returns:
        Mixed data in the same format as input:
        - If input is tensor: tensor of shape (batch_size, mixed_set_size, features)
        - If input is dict: dictionary with mixed sample-specific data and unchanged metadata
    """
    # Handle dictionary input
    if isinstance(data, dict):
        samples = data['samples']
        if isinstance(samples, torch.Tensor):
            batch_size, set_size = samples.shape[:2]
            device = samples.device
        else:
            input_id_keys = [x for x in samples.keys() if 'input_ids' in x]
            device = samples[input_id_keys[0]].device
            batch_size, set_size = samples[input_id_keys[0]].shape[:2]

        mixed_set_size = mixed_set_size if mixed_set_size is not None else set_size
        n_mixed_sets = n_mixed_sets if n_mixed_sets is not None else batch_size
        
        # If no mix_probs provided, generate k-sparse Dirichlet probabilities
        if mix_probs is None:
            k = k if k is not None else batch_size // 2
            mix_probs = generate_k_sparse_dirichlet_probs(batch_size, k, alpha, batch_size_out=n_mixed_sets, device=device)
            
        # Generate source indices once to use for all sample-specific data
        source_set_indices = torch.multinomial(
            mix_probs,
            num_samples=mixed_set_size,
            replacement=replacement
        )
        source_point_indices = torch.randint(
            0, set_size, 
            (n_mixed_sets, mixed_set_size),
            device=device
        )
        source_set_unique = rowwise_unique(source_set_indices, k)

        # Create output dictionary
        mixed_data = {}
        
        # Mix all sample-specific data using the same indices
        for key, value in data.items():
            if isinstance(value, torch.Tensor) and len(value.shape) >= 2:
                # Check if first two dimensions match batch_size and set_size
                if value.shape[:2] == (batch_size, set_size):
                    mixed_data[key] = value[source_set_indices, source_point_indices]
                elif value.shape[0] == batch_size:
                    mixed_data[key] = value[source_set_unique]
                else:
                    mixed_data[key] = value
            elif isinstance(value, dict):
                nested_d = {}
                for k2, v2 in data[key].items():
                    if isinstance(v2, torch.Tensor) and len(v2.shape) >= 2:
                        # Check if first two dimensions match batch_size and set_size
                        if v2.shape[:2] == (batch_size, set_size):
                            nested_d[k2] = v2[source_set_indices, source_point_indices]
                        elif v2.shape[0] == batch_size:
                            source_set_unique = rowwise_unique(source_set_indices, k)
                            nested_d[k2] = v2[source_set_unique]
                        else:
                            nested_d[k2] = v2
                mixed_data[key] = nested_d

            elif key == 'raw_texts':
                mixed_raw = [
                                [data['raw_texts'][ssi.item()][spi.item()] for ssi, spi in zip(set_row, point_row)]
                                for set_row, point_row in zip(source_set_indices.T, source_point_indices.T)
                            ]
                
                mixed_data[key] = mixed_raw
            else:
                mixed_data[key] = value
        
        # add weights to mixed_data by indexing into mix_probs
        mixed_data['weights'] = torch.gather(mix_probs, 1, source_set_unique)
                
        return mixed_data
        
    # Handle tensor input
    elif isinstance(data, torch.Tensor):
        batch_size, set_size, features = data.shape
        device = data.device
        
        # If no mix_probs provided, generate k-sparse Dirichlet probabilities
        if mix_probs is None:
            k = k if k is not None else batch_size // 2
            mix_probs = generate_k_sparse_dirichlet_probs(batch_size, k, alpha, batch_size_out=n_mixed_sets, device=device)

        if mixed_set_size is None:
            mixed_set_size = set_size
            
        # Sample source sets and points in one g
        source_set_indices = torch.multinomial(
            mix_probs,
            num_samples=mixed_set_size,
            replacement=replacement
        )

        source_point_indices = torch.randint(
            0, set_size, 
            size=(n_mixed_sets, mixed_set_size),
            device=device,
        )
        
        # Gather points using a single indexing operation
        mixed_data = data[source_set_indices, source_point_indices]
        
        return mixed_data
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

def test_mix_batch_sets():
    """Run a series of tests to verify the mixing function works correctly."""
    
    # Test 1: Basic shape preservation with tensor
    batch_size, set_size, features = 4, 10, 3
    data = torch.randn(batch_size, set_size, features)
    mixed = mix_batch_sets(data)
    assert mixed.shape == data.shape, f"Expected shape {data.shape}, got {mixed.shape}"
    
    # Test 2: Dictionary input with sample-specific metadata
    data_dict = {
        'samples': torch.randn(batch_size, set_size, features),  # Should be mixed
        'point_features': torch.randn(batch_size, set_size, 2),  # Should be mixed
        'set_features': torch.randn(batch_size, 5),  # Should not be mixed
        'global_features': torch.randn(10),  # Should not be mixed
        'metadata': list(range(batch_size))  # Should not be mixed
    }
    mixed_dict = mix_batch_sets(data_dict)
    
    # Check shapes and mixing behavior
    assert mixed_dict['samples'].shape == (batch_size, set_size, features)
    assert mixed_dict['point_features'].shape == (batch_size, set_size, 2)
    assert torch.equal(mixed_dict['set_features'], data_dict['set_features'])
    assert torch.equal(mixed_dict['global_features'], data_dict['global_features'])
    assert mixed_dict['metadata'] == data_dict['metadata']
    
    # Test 3: Verify points come from original sets
    data = torch.arange(batch_size * set_size).reshape(batch_size, set_size, 1).float()
    mixed = mix_batch_sets(data)
    assert torch.all(torch.isin(mixed, data)), "Mixed data contains values not in original data"
    
    # Test 4: Test deterministic mixing with extreme probabilities
    identity_probs = torch.eye(batch_size)
    mixed = mix_batch_sets(data, identity_probs)
    for i in range(batch_size):
        set_values = mixed[i]
        original_set_values = data[i]
        assert torch.all(torch.isin(set_values, original_set_values)), \
            f"Set {i} contains values from other sets when it shouldn't"
    
    # Test 5: Test mixing proportions
    batch_size = 2
    set_size = 1000
    data = torch.arange(batch_size * set_size).reshape(batch_size, set_size, 1).float()
    mix_probs = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
    mixed = mix_batch_sets(data, mix_probs)
    
    for i in range(batch_size):
        mixed_set = mixed[i]
        counts = [(mixed_set < set_size).float().mean(),
                 (mixed_set >= set_size).float().mean()]
        expected = mix_probs[i]
        assert torch.allclose(torch.tensor(counts), expected, atol=0.05), \
            f"Set {i} mixing proportions {counts} differ significantly from expected {expected}"
    
    print("All tests passed!")

def test_mixing():
    """Test the mixing functionality."""
    batch_size = 1000  # Large batch to test vectorization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test k-sparse Dirichlet probability generation
    k = 3
    alpha = 0.5
    mix_probs = generate_k_sparse_dirichlet_probs(batch_size, k, alpha, device=device)
    
    # Test shape and basic properties
    assert mix_probs.shape == (batch_size, batch_size)
    assert torch.allclose(mix_probs.sum(dim=1), torch.ones(batch_size, device=device))
    assert torch.all((mix_probs >= 0) & (mix_probs <= 1))
    
    # Verify sparsity
    assert torch.sum(mix_probs > 0, dim=1).max() <= k
    
    # Test actual mixing with different parameters
    data = torch.randn(batch_size, 50, 10, device=device)  # 1000 sets of 50 points with 10 features
    
    # Test 1: Basic mixing
    mixed = mix_batch_sets(data, k=3, alpha=0.5)
    assert mixed.shape == data.shape
    
    # Test 2: Different output size
    mixed = mix_batch_sets(data, k=3, alpha=0.5, mixed_set_size=30)
    assert mixed.shape == (batch_size, 30, 10)
    
    # Test 3: No replacement
    mixed = mix_batch_sets(data, k=3, alpha=0.5, replacement=False)
    assert mixed.shape == data.shape
    
    # Test 4: Extreme alpha values
    mixed_sparse = mix_batch_sets(data, k=3, alpha=0.1)  # Very sparse
    mixed_uniform = mix_batch_sets(data, k=3, alpha=5.0)  # More uniform
    
    print("All tests passed!")

class SetMixer:
    """
    A transform that applies Dirichlet mixing followed by random pairing for 
    coupled distribution embeddings. Handles both raw tensor inputs and dictionary
    inputs with metadata from SetDataset classes.
    
    For dictionary inputs, automatically detects which keys contain sample-specific data
    based on tensor shapes matching (batch_size, set_size, ...).
    
    Always applies: Mix sets → Random pair for source/target → Return coupled batch
    
    Example usage with DataLoader:
        mixer = SetMixer(k=3, alpha=0.5)
        dataset = MNISTDataset(...)  # or any other SetDataset
        dataloader = DataLoader(
            dataset, 
            batch_size=32, 
            collate_fn=mixer.collate_fn
        )
    """
    def __init__(
            self, 
            k: int = 2, 
            alpha: float = 1.0, 
            mixed_set_size: int = None,
            n_mixed_sets: int = None,
            replacement: bool = True,
            mix_prob: float = 1.0
        ):
        self.k = k
        self.alpha = alpha
        self.mixed_set_size = mixed_set_size
        self.n_mixed_sets = n_mixed_sets
        self.replacement = replacement
        self.mix_prob = mix_prob
    
    def _apply_random_pairing(self, mixed_data):
        """Apply random pairing to mixed data to create source/target pairs."""
        # Convert mixed data to list format for pairing functions
        if isinstance(mixed_data, dict):
            # Convert dict format to list of dicts for pairing
            batch_size = None
            keys = mixed_data.keys()
            
            # Find batch size from first tensor
            for key, value in mixed_data.items():
                if isinstance(value, torch.Tensor) and len(value.shape) >= 1:
                    batch_size = value.shape[0]
                    break
            
            if batch_size is None:
                return mixed_data
                
            # Convert to list of dicts
            batch_list = []
            for i in range(batch_size):
                item = {}
                for key in keys:
                    if isinstance(mixed_data[key], torch.Tensor):
                        item[key] = mixed_data[key][i]
                    elif isinstance(mixed_data[key], list):
                        item[key] = mixed_data[key][i]
                    else:
                        item[key] = mixed_data[key]
                batch_list.append(item)
                
        elif isinstance(mixed_data, torch.Tensor):
            # Convert tensor to list of dicts
            batch_size = mixed_data.shape[0]
            batch_list = [{'samples': mixed_data[i]} for i in range(batch_size)]
        else:
            return mixed_data
            
        # Apply random permutation pairing
        from .paired_collate import random_permutation_collate_fn
        return random_permutation_collate_fn(batch_list)
    
    def __call__(self, batch_data):
        """Apply mixing to a batch of sets, followed by random pairing."""
        if self.mix_prob < 1.0 and torch.rand(1) > self.mix_prob:
            # If not mixing, still apply pairing
            return self._apply_random_pairing(batch_data)
            
        # Apply mixing first
        mixed_data = mix_batch_sets(
            batch_data,
            k=self.k,
            alpha=self.alpha,
            mixed_set_size=self.mixed_set_size,
            n_mixed_sets=self.n_mixed_sets,
            replacement=self.replacement
        )
        
        # Then apply random pairing
        return self._apply_random_pairing(mixed_data)
    
    def collate_fn(self, batch: list):
        """
        Custom collate function that recursively collates batches of nested dictionaries and tensors.
        Supports standard tensors and nested structures like {'samples': {...}}.
        """

        def recursive_collate(batch_part):
            if isinstance(batch_part[0], torch.Tensor):
                return torch.stack(batch_part)
            elif isinstance(batch_part[0], dict):
                return {key: recursive_collate([b[key] for b in batch_part]) for key in batch_part[0]}
            else:
                return batch_part  # for strings, lists, or other non-tensor data
            
        collated = recursive_collate(batch)
        return self(collated)

    def prescribed_mixing(self, batch_data, mix_probs):
        """Apply prescribed mixing probabilities, followed by random pairing."""
        mixed_data = mix_batch_sets(
            batch_data, 
            mix_probs,
            mixed_set_size=self.mixed_set_size,
            n_mixed_sets=self.n_mixed_sets,
            replacement=self.replacement,
            k=self.k,
            alpha=self.alpha
        )
        
        # Apply random pairing
        return self._apply_random_pairing(mixed_data)




def test_set_mixer():
    """Test the SetMixer transform."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test basic tensor mixing + pairing
    mixer = SetMixer(k=2, alpha=0.5)
    batch = torch.randn(10, 20, 3)  # 10 sets, 20 points, 3 features
    mixed = mixer(batch)
    # Should return source_samples and target_samples after pairing
    assert 'source_samples' in mixed
    assert 'target_samples' in mixed
    assert mixed['source_samples'].shape == batch.shape
    assert mixed['target_samples'].shape == batch.shape
    
    # Test dictionary input with various types of metadata
    batch_dict = {
        'samples': torch.randn(10, 20, 3),  # Should be mixed
        'point_features': torch.randn(10, 20, 2),  # Should be mixed
        'set_features': torch.randn(10, 5),  # Should not be mixed
        'global_features': torch.randn(10),  # Should not be mixed
        'metadata': list(range(10))  # Should not be mixed
    }
    
    # Test mixing with automatic detection of sample-specific data
    mixer = SetMixer(k=2, alpha=0.5)
    mixed_dict = mixer(batch_dict)
    
    # Check that it returns paired source/target structure
    assert 'source_samples' in mixed_dict
    assert 'target_samples' in mixed_dict
    assert mixed_dict['source_samples'].shape == (10, 20, 3)
    assert mixed_dict['target_samples'].shape == (10, 20, 3)
    
    # Test collate_fn with dictionaries
    batch_as_list = [
        {
            'samples': torch.randn(20, 3),
            'point_features': torch.randn(20, 2),
            'set_features': torch.randn(5),
            'metadata': i,
        }
        for i in range(10)
    ]
    mixed = mixer.collate_fn(batch_as_list)
    assert 'source_samples' in mixed
    assert 'target_samples' in mixed
    assert mixed['source_samples'].shape == (10, 20, 3)
    assert mixed['target_samples'].shape == (10, 20, 3)
    
    # Test probabilistic mixing (should still apply pairing even when not mixing)
    mixer_prob = SetMixer(k=2, alpha=0.5, mix_prob=0.0)
    unmixed = mixer_prob(batch_dict)
    assert 'source_samples' in unmixed
    assert 'target_samples' in unmixed
    
    print("All SetMixer tests passed!")

if __name__ == "__main__":
    test_mixing()
    test_mix_batch_sets()
    test_set_mixer()
