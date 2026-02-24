"""
Train scGen VAE on the trellis dataset.

Usage:
    python train_scgen.py --split_name pdo21 --set_size 32 --max_epochs 100
"""

import argparse
import sys
import os
import numpy as np
import anndata
import torch

# Add scgen to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scgen"))
from scgen import SCGEN

from datasets.trellis import trellis_dataset


def build_anndata_from_train_samples(samples_train):
    """
    Pool all cells (both x0 and x1) from all train samples into a single AnnData.

    Each element of samples_train is a tuple:
        (culture, x0, x1, cond_cell_x0, cond_cell_x1, cond_treat, patient)

    We collect x0 (control) and x1 (treated) cells from every sample, and store
    metadata about which condition (x0/x1), treatment, culture, and patient each
    cell came from.
    """
    all_cells = []
    condition_labels = []
    treatment_labels = []
    culture_labels = []
    patient_labels = []

    treatments = ["O", "S", "VS", "L", "V", "F", "C", "SF", "CS", "CF", "CSF"]

    for culture, x0, x1, cond_cell_x0, cond_cell_x1, cond_treat, patient in samples_train:
        # Decode treatment from one-hot (same for all cells in this sample)
        treat_idx = np.argmax(cond_treat[0])
        treat_name = treatments[treat_idx]

        # x0 = control cells
        all_cells.append(x0)
        condition_labels.extend(["control"] * x0.shape[0])
        treatment_labels.extend([treat_name] * x0.shape[0])
        culture_labels.extend([culture] * x0.shape[0])
        patient_labels.extend([patient] * x0.shape[0])

        # x1 = treated cells
        all_cells.append(x1)
        condition_labels.extend(["treated"] * x1.shape[0])
        treatment_labels.extend([treat_name] * x1.shape[0])
        culture_labels.extend([culture] * x1.shape[0])
        patient_labels.extend([patient] * x1.shape[0])

    X = np.concatenate(all_cells, axis=0)

    adata = anndata.AnnData(
        X=X,
        obs={
            "condition": condition_labels,
            "treatment": treatment_labels,
            "culture": culture_labels,
            "patient": patient_labels,
        },
    )

    print(f"Built AnnData: {adata.shape[0]} cells x {adata.shape[1]} features")
    print(f"  Conditions: {dict(zip(*np.unique(condition_labels, return_counts=True)))}")
    print(f"  Treatments: {dict(zip(*np.unique(treatment_labels, return_counts=True)))}")
    print(f"  Cultures:   {dict(zip(*np.unique(culture_labels, return_counts=True)))}")

    return adata


def main():
    parser = argparse.ArgumentParser(description="Train scGen VAE on trellis dataset")
    parser.add_argument("--split_name", type=str, required=True,
                        choices=["replicas-1", "replicas-2", "pdo21", "pdo27", "pdo75"],
                        help="Dataset split name")
    parser.add_argument("--set_size", type=int, default=32,
                        help="Set size for dataset (only affects __getitem__, not VAE training)")

    # Training parameters (passed to model.train() only if specified; otherwise scGen defaults are used)
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=131072, help="Training batch size")
    parser.add_argument("--early_stopping", type=bool, default=True, help="Enable early stopping")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save trained model (default: scgen_model_{split_name}/)")

    args = parser.parse_args()

    # --- 1. Instantiate dataset ---
    print("=" * 60)
    print(f"Loading trellis dataset (split={args.split_name}, set_size={args.set_size})")
    print("=" * 60)

    dataset = trellis_dataset(split_name=args.split_name, set_size=args.set_size)
    train_samples = dataset.samples_train

    print(f"Number of train samples (experiment pairs): {len(train_samples)}")

    # --- 2. Build AnnData from train samples ---
    print("\n" + "=" * 60)
    print("Building AnnData from train samples")
    print("=" * 60)

    adata = build_anndata_from_train_samples(train_samples)

    # --- 3. Setup and train scGen ---
    print("\n" + "=" * 60)
    print("Setting up scGen VAE")
    print("=" * 60)

    # Register with scGen. We use "condition" as batch_key and "treatment" as labels_key.
    # Note: neither is actually used by the VAE during training -- they are just required
    # by the scvi-tools API and used only for post-hoc methods like predict()/batch_removal().
    SCGEN.setup_anndata(adata, batch_key="condition", labels_key="treatment")

    model = SCGEN(adata)

    print(f"\nModel summary (defaults: n_hidden=800, n_latent=100, n_layers=2, dropout=0.2):")
    print(f"  n_input:     {adata.shape[1]}")
    print(f"  total cells: {adata.shape[0]}")

    print("\n" + "=" * 60)
    print("Training scGen VAE")
    print("=" * 60)

    # scvi default is train_size=0.9 and drop_last=False.
    # Avoid final training minibatch of size 1, which breaks BatchNorm.
    n_train = int(np.ceil(0.9 * adata.shape[0]))
    batch_size = args.batch_size
    if n_train % batch_size == 1:
        batch_size -= 1
        print(
            f"Adjusted batch_size from {args.batch_size} to {batch_size} "
            f"to avoid a singleton final training batch."
        )

    train_kwargs = {}
    if args.max_epochs is not None:
        train_kwargs["max_epochs"] = args.max_epochs
    if batch_size is not None:
        train_kwargs["batch_size"] = batch_size
    if args.early_stopping:
        train_kwargs["early_stopping"] = True

    model.train(**train_kwargs)

    # --- 4. Save model ---
    save_path = f"scgen_model_{args.split_name}"
    print(f"\nSaving model to: {save_path}")
    model.save(save_path, overwrite=True)

    # --- 5. Quick sanity check: get latent representations ---
    print("\n" + "=" * 60)
    print("Sanity check: computing latent representations")
    print("=" * 60)

    latent = model.get_latent_representation()
    print(f"Latent shape: {latent.shape}")
    print(f"Latent mean:  {latent.mean(axis=0)[:5]}...")
    print(f"Latent std:   {latent.std(axis=0)[:5]}...")

    print("\nDone!")


if __name__ == "__main__":
    main()
