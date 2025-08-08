import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Parameters
root_dir = "figures"  # folder containing all subdirectories
datasets = ["GoM", "pbmc"]
condition_methods = ["dt_mlp_sinusoidal", "dt_ridge_sinusoidal"]
samplers = ["bidirectional", "unidirectional", "dt_equals_one"]
predictor_loss_weights = ["0.001", "0.01", "0.1", "1", "10"]

# Create panels
for dataset in datasets:
    fig, axes = plt.subplots(len(predictor_loss_weights),
                             len(condition_methods) * len(samplers),
                             figsize=(18, 15))

    for row_idx, a_val in enumerate(predictor_loss_weights):
        col_idx = 0
        for cond in condition_methods:
            for sampler in samplers:
                subdir = f"{dataset}_{cond}_{sampler}_{a_val}"
                filename = f"{dataset}_{cond}_{sampler}_{a_val}_results_CDE.png"
                full_path = os.path.join(root_dir, subdir, filename)

                ax = axes[row_idx, col_idx]
                if os.path.isfile(full_path):
                    img = mpimg.imread(full_path)
                    ax.imshow(img)
                    ax.set_title(f"condition method = {cond}\n"
                                 f"sampling method = {sampler}\n"
                                 f"a = {a_val}",
                                 fontsize=8)
                else:
                    print(f"Warning: Missing file {full_path}")
                ax.axis("off")
                col_idx += 1

    plt.suptitle(f"{dataset} — 5x6 panel", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    save_path = f"{dataset}_panel.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved {dataset} panel to {save_path}")
