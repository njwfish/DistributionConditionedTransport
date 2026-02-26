"""
Train CellOT models on the trellis dataset.

Trains a separate CellOT model for each drug treatment, learning a neural
optimal transport map from control cells (x0) to treated cells (x1).

Usage:
    python train_cellot.py --split_name pdo21
    python train_cellot.py --split_name replicas-1 --treatment O
"""

import argparse
import os
import sys
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cellot"))
from cellot.train.train import train_cellot
from cellot.utils.helpers import load_config, nest_dict
from cellot.data.utils import cast_dataset_to_loader

from datasets.trellis import trellis_dataset

TREATMENTS = ["O", "S", "VS", "L", "V", "F", "C", "SF", "CS", "CF", "CSF"]


class NumpyDataset(Dataset):
    def __init__(self, data):
        self.data = data.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def pool_cells_by_treatment(samples):
    x0_by_treat, x1_by_treat = {}, {}
    for _culture, x0, x1, _c0, _c1, cond_treat, _patient in samples:
        t = int(np.argmax(cond_treat[0]))
        x0_by_treat.setdefault(t, []).append(x0)
        x1_by_treat.setdefault(t, []).append(x1)
    return {
        t: (np.concatenate(x0_by_treat[t]), np.concatenate(x1_by_treat[t]))
        for t in sorted(x0_by_treat)
    }


def make_cellot_loader(src_train, tgt_train, src_val, tgt_val, batch_size=256):
    """Build the nested DataLoader structure that train_cellot expects."""
    datasets = nest_dict({
        "train.source": NumpyDataset(src_train),
        "train.target": NumpyDataset(tgt_train),
        "test.source": NumpyDataset(src_val),
        "test.target": NumpyDataset(tgt_val),
    }, as_dot_dict=True)
    return cast_dataset_to_loader(
        datasets, batch_size=batch_size, shuffle=True, drop_last=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Train CellOT on trellis")
    parser.add_argument(
        "--split_name", type=str, required=True,
        choices=["replicas-1", "replicas-2", "pdo21", "pdo27", "pdo75"],
    )
    parser.add_argument(
        "--treatment", type=str, default=None,
        help="Single treatment to train (e.g. 'O'). Default: all.",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    if args.treatment and args.treatment not in TREATMENTS:
        parser.error(
            "Unknown treatment '{}'. Choose from: {}".format(
                args.treatment, TREATMENTS
            )
        )

    save_dir = args.save_dir or "cellot_models/{}".format(args.split_name)
    config_path = os.path.join("cellot", "configs", "models", "cellot.yaml")
    config = load_config(config_path)

    print("=" * 60)
    print("Training CellOT for split: {}".format(args.split_name))
    print("=" * 60)

    dataset = trellis_dataset(
        split_name=args.split_name, set_size=32,
    )
    cells = pool_cells_by_treatment(dataset.samples_train)
    input_dim = next(iter(cells.values()))[0].shape[1]

    print("\nCells per treatment:")
    for t_idx, (x0, x1) in cells.items():
        tag = TREATMENTS[t_idx]
        print("  {:>3}: {:>6} ctrl, {:>6} treated".format(
            tag, x0.shape[0], x1.shape[0]))

    items = cells.items()
    if args.treatment:
        t_idx = TREATMENTS.index(args.treatment)
        if t_idx not in cells:
            print("No training data for '{}'.".format(args.treatment))
            return
        items = [(t_idx, cells[t_idx])]

    for t_idx, (x0, x1) in items:
        t_name = TREATMENTS[t_idx]
        print("\n" + "=" * 60)
        print("Training: {}".format(t_name))
        print("  Control: {}, Treated: {}".format(x0.shape, x1.shape))

        rng = np.random.RandomState(0)
        nv_src = max(1, int(len(x0) * args.val_fraction))
        nv_tgt = max(1, int(len(x1) * args.val_fraction))
        sp = rng.permutation(len(x0))
        tp = rng.permutation(len(x1))

        loader = make_cellot_loader(
            x0[sp[nv_src:]], x1[tp[nv_tgt:]],
            x0[sp[:nv_src]], x1[tp[:nv_tgt]],
            batch_size=args.batch_size,
        )

        outdir = Path(save_dir) / t_name
        outdir.mkdir(parents=True, exist_ok=True)

        train_cellot(
            outdir, config,
            loader=loader,
            model_kwargs={"input_dim": input_dim},
        )
        print("  Saved to: {}".format(outdir))

    print("\n" + "=" * 60)
    print("Done. Models under: {}".format(save_dir))


if __name__ == "__main__":
    main()
