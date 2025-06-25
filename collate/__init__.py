# Collate functions for distribution embeddings

# Paired collate functions for coupled distribution embeddings
from .paired_collate import (
    PairedCollate,  # New object-based interface
    shift_pairing_collate_fn,
    random_permutation_collate_fn,
    random_pairing_collate_fn,  # alias for random_permutation_collate_fn
    cross_dataset_pairing_collate_fn
)

# Set mixing collate functions (now with optional pairing support)
from .mixing_collate import (
    SetMixer,
    mix_batch_sets,
    generate_k_sparse_dirichlet_probs
)

__all__ = [
    # Paired collate - object-based interface
    'PairedCollate',
    
    # Paired collate functions (for backward compatibility)
    'shift_pairing_collate_fn',
    'random_permutation_collate_fn',
    'random_pairing_collate_fn',  # alias
    'cross_dataset_pairing_collate_fn',
    
    # Mixing collate functions (with optional pairing)
    'SetMixer',
    'mix_batch_sets',
    'generate_k_sparse_dirichlet_probs'
] 