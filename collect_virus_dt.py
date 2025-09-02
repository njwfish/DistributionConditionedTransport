import os
from typing import List

import hydra
from omegaconf import DictConfig
import numpy as np
from tqdm import tqdm


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    """
    Instantiate the virus dataset exactly as configured via Hydra (same as in main.py),
    iterate through all items to collect the 'd' field, and save the list as a .npz file.

    Run with the same overrides as used in run_virus.sh to mirror that setup, e.g.:
      python collect_virus_dt.py experiment=virus predictor=dt_mlp_sinusoidal \
        sampling=bidirectional experiment.use_predicted_latent=false \
        experiment.predictor_loss_weight=0 seed=0
    """

    # Instantiate dataset exactly as in main.py
    dataset = hydra.utils.instantiate(cfg.dataset)

    # Collect 'dt' from each dataset item
    d_values = []
    for idx in tqdm(range(len(dataset)), desc="Collecting d values"):
        item = dataset[idx]
        if "d" not in item:
            raise KeyError("Dataset item does not contain 'd'.")
        d_values.append(item["d"])

    # Determine output path; default to dataset's data_dir if available
    data_dir = getattr(dataset, "data_dir", os.getcwd())
    output_path = os.path.join(data_dir, "virus_all_d_values.npz")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save as compressed NPZ
    np.savez_compressed(output_path, precomputed_d_values=np.array(d_values))
    print(f"Saved {len(d_values)} d values to {output_path}")


if __name__ == "__main__":
    main()


