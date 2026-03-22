"""
Train a treatment-conditioned CellFlow model on the Trellis dataset.

This script:
1. Loads a Trellis split.
2. Flattens all train cells into a single AnnData object.
3. Uses only treatment identity as the perturbation covariate.
4. Trains one joint CellFlow model across all treatments.
5. Saves the trained model and training metadata.

The script assumes `cellflow` is already installed in the environment.

Usage:
    python train_cellflow.py --split_name pdo21 --num_iterations 20000
    python train_cellflow.py --split_name replicas-1 --num_iterations 50000
"""

import argparse
import json
from pathlib import Path

import anndata
import numpy as np

from cellflow.model import CellFlow

from datasets.trellis import trellis_dataset

TREATMENTS = ["O", "S", "VS", "L", "V", "F", "C", "SF", "CS", "CF", "CSF"]


def decode_treatment(cond_treat: np.ndarray) -> str:
    treat_idx = int(np.argmax(cond_treat[0]))
    return TREATMENTS[treat_idx]


def build_anndata(samples) -> anndata.AnnData:
    all_cells = []
    control_labels = []
    treatment_labels = []

    for _culture, x0, x1, _c0, _c1, cond_treat, _patient in samples:
        treatment_name = decode_treatment(cond_treat)

        all_cells.append(x0)
        control_labels.extend([True] * x0.shape[0])
        treatment_labels.extend(["control"] * x0.shape[0])

        all_cells.append(x1)
        control_labels.extend([False] * x1.shape[0])
        treatment_labels.extend([treatment_name] * x1.shape[0])

    X = np.concatenate(all_cells, axis=0).astype(np.float32)
    obs = {
        "control": control_labels,
        "treatment": treatment_labels,
    }
    return anndata.AnnData(X=X, obs=obs)


def write_metadata(path: Path, metadata: dict) -> None:
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description="Train CellFlow on the Trellis dataset")
    parser.add_argument(
        "--split_name",
        type=str,
        required=True,
        choices=["replicas-1", "replicas-2", "pdo21", "pdo27", "pdo75"],
        help="Dataset split name",
    )
    parser.add_argument(
        "--set_size",
        type=int,
        default=32,
        help="Set size for the Trellis dataset object",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        required=True,
        help="Number of CellFlow training iterations. This is required because CellFlow itself does not define a training default.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save the trained model. Defaults to cellflow_model_{split_name}",
    )
    args = parser.parse_args()

    save_dir = Path(args.save_dir or f"cellflow_model_{args.split_name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Loading Trellis dataset: split={args.split_name}")
    print("=" * 60)
    dataset = trellis_dataset(split_name=args.split_name, set_size=args.set_size)

    print(f"Train samples: {len(dataset.samples_train)}")
    print(f"Test samples:  {len(dataset.samples_test)}")

    adata_train = build_anndata(dataset.samples_train)

    unique_treatments, counts = np.unique(adata_train.obs["treatment"], return_counts=True)
    treatment_counts = {str(k): int(v) for k, v in zip(unique_treatments, counts, strict=False)}

    print("\n" + "=" * 60)
    print("Prepared AnnData")
    print("=" * 60)
    print(f"Cells:    {adata_train.n_obs}")
    print(f"Features: {adata_train.n_vars}")
    print(f"Treatment counts: {treatment_counts}")

    print("\n" + "=" * 60)
    print("Preparing CellFlow model")
    print("=" * 60)
    cf = CellFlow(adata_train)
    cf.prepare_data(
        sample_rep="X",
        control_key="control",
        perturbation_covariates={"treatment": ["treatment"]},
    )
    cf.prepare_model()

    print("\n" + "=" * 60)
    print("Training CellFlow")
    print("=" * 60)
    cf.train(
        num_iterations=args.num_iterations,
    )

    print(f"\nSaving model to: {save_dir}")
    cf.save(str(save_dir), overwrite=True)

    metadata = {
        "split_name": args.split_name,
        "set_size": args.set_size,
        "solver": "otfm",
        "num_iterations": args.num_iterations,
        "perturbation_covariates": {"treatment": ["treatment"]},
        "split_covariates": [],
        "treatments": TREATMENTS,
        "train_n_obs": int(adata_train.n_obs),
        "train_n_vars": int(adata_train.n_vars),
        "treatment_counts": treatment_counts,
    }
    write_metadata(save_dir / "training_metadata.json", metadata)

    print("\nDone.")


if __name__ == "__main__":
    main()
