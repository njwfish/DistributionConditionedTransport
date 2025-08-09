import os
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# Parameters
root_dir = "figures"  # folder containing all subdirectories
datasets = ["GoM", "pbmc"]
condition_methods = ["dt_mlp_sinusoidal", "dt_ridge_sinusoidal"]
samplers = ["bidirectional", "unidirectional", "dt_equals_one"]
predictor_loss_weights = ["0.001", "0.01", "0.1", "1", "10"]

# Function to extract MMD and EMD mean ± std from log
def extract_scores_from_log(log_file_path, dataset):
    """Return (mmd_mean, mmd_std, emd_mean, emd_std), EMD values None for pbmc."""
    mmd_mean, mmd_std, emd_mean, emd_std = None, None, None, None
    if not os.path.isfile(log_file_path):
        return mmd_mean, mmd_std, emd_mean, emd_std

    try:
        with open(log_file_path, "r") as f:
            content = f.read()

            # Match patterns like: "MMD: mean=0.1234, std=0.0567"
            mmd_match = re.search(r"MMD.*?([\d.]+)\s*\±\s*([\d.]+)", content)
            if mmd_match:
                mmd_mean = float(mmd_match.group(1))
                mmd_std = float(mmd_match.group(2))

            if dataset != "pbmc":
                emd_match = re.search(r"EMD.*?([\d.]+)\s*\±\s*([\d.]+)", content)
                if emd_match:
                    emd_mean = float(emd_match.group(1))
                    emd_std = float(emd_match.group(2))
    except Exception as e:
        print(f"Warning: could not parse {log_file_path}: {e}")

    return mmd_mean, mmd_std, emd_mean, emd_std

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

                # Find the .log file (only one expected per subdir)
                log_files = glob.glob(os.path.join(root_dir, subdir, "*.log"))
                log_path = log_files[0] if log_files else None
                mmd_mean, mmd_std, emd_mean, emd_std = extract_scores_from_log(log_path, dataset) if log_path else (None, None, None, None)

                ax = axes[row_idx, col_idx]
                if os.path.isfile(full_path):
                    img = mpimg.imread(full_path)
                    ax.imshow(img)

                    # Build title
                    title_lines = [
                        f"condition method = {cond}",
                        f"sampling method = {sampler}",
                        f"a = {a_val}"
                    ]
                    if mmd_mean is not None and mmd_std is not None:
                        title_lines.append(f"MMD = {mmd_mean:.4f} ± {mmd_std:.4f}")
                    if emd_mean is not None and emd_std is not None:
                        title_lines.append(f"EMD = {emd_mean:.4f} ± {emd_std:.4f}")

                    ax.set_title("\n".join(title_lines), fontsize=8)
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
