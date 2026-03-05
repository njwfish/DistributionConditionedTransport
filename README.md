# Distribution Conditioned Transport

A framework for learning latent representations of distributions and transport maps between them. An encoder maps sets of samples to a latent space; a generator then transports samples from a source distribution to a target distribution conditioned on their latent embeddings.

---

## Overview

Distribution Conditioned Transport (DCT) is a meta-learning framework for distribution-to-distribution transport. Given a batch of source–target distribution pairs, the model:

1. **Encodes** each distribution (a set of samples) into a latent vector `z`.
2. **Transports** source samples to the target distribution conditioned on `(z_source, z_target)`.

This enables few-shot generalization: at inference time, the encoder can embed a previously unseen distribution from a small sample set, and the generator can transport samples to it without retraining.

---

## Repository Structure

```
CoupledDistributionEmbeddings/
├── config/                        # Hydra configuration files
│   ├── config.yaml                # Top-level defaults
│   ├── dataset/                   # Dataset configs
│   ├── encoder/                   # Encoder configs (gnn, embedding, esm, ...)
│   ├── generator/                 # Generator configs (flow_matching, sinkhorn, mmd, ...)
│   ├── model/                     # Backbone network configs
│   ├── coupling/                  # Sample-pairing / coupling configs
│   ├── loss/                      # Loss manager configs
│   ├── predictor/                 # Predictor head configs
│   ├── sampling/                  # Dataloader sampler configs
│   ├── optimizer/                 # Optimizer configs
│   ├── scheduler/                 # LR scheduler configs
│   ├── training/                  # Training loop configs
│   ├── wandb/                     # Weights & Biases configs
│   └── experiment/                # Complete experiment configs (entry points)
├── datasets/                      # PyTorch Dataset implementations
│   ├── distribution_datasets.py   # MVN and GMM synthetic datasets
│   ├── mnist_colors.py            # Colored MNIST image dataset
│   ├── handwriting.py             # Handwritten character dataset
│   ├── snapMMD_unified.py         # SnapMMD benchmark (GoM, LV, PBMC, Repressilator)
│   ├── lineage_tracing.py         # Single-cell lineage tracing data (StateFate)
│   ├── batch_integration.py       # Single-cell batch integration
│   ├── tcr.py                     # T-cell receptor repertoire sequences
│   ├── virus_time_only.py         # Viral sequence evolution (time)
│   └── supervised_datasets.py     # Supervised transport variants
├── encoder/                       # Encoder modules
│   ├── encoders.py                # DistributionEncoder (GNN), EmbeddingEncoder
│   ├── conv_gnn.py                # Convolutional GNN encoder (for images)
│   ├── transformer_encoder.py     # Transformer set encoder
│   ├── esm_baseline.py            # ESM2 protein language model encoder
│   └── kernel_mean.py             # Kernel mean embedding encoder
├── generator/                     # Generator / transport modules
│   ├── flow_matching.py           # Conditional flow matching (NeuralODE)
│   ├── direct.py                  # Direct generators (Sinkhorn, Wasserstein, Energy)
│   ├── dfm_esm2.py                # Discrete flow matching with ESM2 (proteins/TCR)
│   ├── causal_transformer.py      # Autoregressive sequence generator
│   └── losses.py                  # Transport loss functions (SWD, MMD, Sinkhorn, Energy)
├── coupling/                      # Source–target pairing strategies
│   ├── ot.py                      # Optimal transport coupling
│   └── edit_distance.py           # Edit-distance coupling (sequences)
├── predictor/                     # Latent space predictor heads
├── loss/                          # Loss manager implementations
├── model/                         # Backbone network architectures (MLP, UNet, ...)
├── utils/                         # Utilities (hashing, seeding, visualization, ...)
├── evals/                         # Evaluation scripts and result files
│   ├── evaluate_mvn_gmm.py        # MVN/GMM evaluation (W2, SWD, MMD, Energy)
│   ├── evaluate_mnist_colors.py   # MNIST-Colors evaluation
│   ├── evaluate_handwriting.py    # Handwriting evaluation
│   └── evaluate_supervised_comparison.py
├── scripts/                       # Experiment management utilities
│   ├── sweep_mvn_gmm.sh           # SLURM sweep launcher (MVN/GMM)
│   ├── experiment_status_table.py # Check experiment completion status
│   └── check_missing.py           # Find missing experiment configurations
├── notebooks/                     # Jupyter notebooks for analysis and visualization
├── outputs/                       # Experiment outputs (auto-generated)
├── main.py                        # Training entry point
├── training.py                    # Trainer class
├── snapmmd_eval.py                # SnapMMD evaluation script
├── layers.py                      # Shared layer primitives (MLP, MeanPooledFC, ...)
└── requirements.txt               # Python dependencies
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (strongly recommended)
- Conda (recommended)

### Environment Setup

```bash
conda activate distemb
```

To install from scratch:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch`, `torchvision` — deep learning
- `hydra-core`, `omegaconf` — configuration management
- `hydra-submitit-launcher` — SLURM job submission via Hydra
- `wandb` — experiment tracking
- `geomloss` — GPU-accelerated optimal transport losses
- `torchdyn` — Neural ODE integration for flow matching
- `anndata`, `scanpy` — single-cell data handling
- `transformers` — protein language models (ESM2)
- `scikit-learn`, `pandas`, `seaborn` — analysis

---

## Core Concepts

### Encoders

| Config key | Class | Description |
|---|---|---|
| `gnn` | `DistributionEncoderGNN` | GNN over sample set; permutation-invariant via mean pooling |
| `tx` | `DistributionEncoderTx` | Transformer set encoder |
| `embedding` | `EmbeddingEncoder` | Learned lookup table (requires known distribution indices) |
| `esm` | ESM2-based | Protein language model for sequence sets |
| `kernel_mean` | `KernelMeanEmbedding` | Kernel mean embedding |

The `DistributionEncoder` variants operate on raw sample sets (`[B, N, D]`) and are compatible with out-of-distribution generalization. The `EmbeddingEncoder` requires distribution indices and uses `loss: multimarginal`.

### Generators

| Config key | Type | Description |
|---|---|---|
| `flow_matching` | Continuous | Conditional flow matching via NeuralODE |
| `sinkhorn` | Direct | Sinkhorn (entropic OT) transport loss |
| `wasserstein` | Direct | Wasserstein-2 transport loss |
| `energy` | Direct | Energy distance minimization |
| `mmd` | Direct | Maximum mean discrepancy |
| `esm_dfm` | Discrete | Discrete flow matching with ESM2 (protein/TCR sequences) |

### Source–Target Pairing (Coupling)

At training time, each batch contains a set of distributions. The coupling strategy determines how source–target pairs are formed:

- **No coupling** (`coupling: none`): pairs are formed randomly within the batch
- **OT coupling** (`coupling: sinkhorn`): pairs are formed by solving an optimal transport problem over distribution distances

---

## Running Experiments

All experiments are configured via Hydra. The entry point is `main.py`.

### Synthetic Benchmarks (MVN / GMM)

```bash
# Multivariate Normal, GNN encoder, flow matching generator
python main.py experiment=mvn_base generator=flow_matching +dataset.n_unique_sets=100

# Gaussian Mixture Model, GNN encoder, Sinkhorn generator
python main.py experiment=gmm_base generator=sinkhorn +dataset.n_unique_sets=200

# Embedding encoder variant (requires multimarginal loss)
python main.py experiment=mvn_emb_base generator=flow_matching +dataset.n_unique_sets=100
```

### Image Datasets

```bash
# Colored MNIST
python main.py experiment=mnist_colors_base generator=flow_matching +dataset.n_unique_sets=200

# Handwritten characters
python main.py experiment=handwriting_base generator=flow_matching +dataset.n_unique_sets=200
```

### SnapMMD Benchmark

```bash
# GoM, LV, PBMC, or Repressilator (set via dataset_name)
python main.py experiment=snapMMD dataset_name=PBMC
python main.py experiment=snapMMD dataset_name=GoM
```

### Biological Applications

```bash
# Single-cell lineage tracing (StateFate)
python main.py experiment=lineage_supervised
python main.py experiment=lineage_semisupervised_fm

# T-cell receptor repertoire transport (ESM2 + discrete flow matching)
python main.py experiment=tcr_esm_dfm

# Viral sequence evolution
python main.py experiment=virus_time_only

# Single-cell batch integration
python main.py experiment=batchint_fm
```

### Key Overridable Parameters

| Parameter | Description | Example |
|---|---|---|
| `generator` | Generator type | `generator=flow_matching` |
| `encoder` | Encoder type | `encoder=gnn` |
| `dataset.n_unique_sets` | Number of unique distributions | `+dataset.n_unique_sets=1000` |
| `experiment.batch_size` | Training batch size | `experiment.batch_size=128` |
| `experiment.lr` | Learning rate | `experiment.lr=1e-4` |
| `seed` | Random seed | `seed=0` |

---

## SLURM Cluster Execution

Experiments are designed for the Harvard FAS RC cluster using Hydra's Submitit launcher. The SLURM launcher is configured in `config/hydra/launcher/`.

```bash
# Launch an MVN sweep across generators and n_unique_sets
bash scripts/sweep_mvn_gmm.sh

# Launch MNIST-Colors sweep
bash scripts/sweep_mnist_colors.sh

# Check which experiments have completed
python scripts/experiment_status_table.py

# Find missing/incomplete experiments
python scripts/check_missing.py
```

Monitor running jobs:

```bash
squeue -u $USER
tail -f logs/<job_name>_<job_id>.out
```

---

## Output Structure

Each experiment saves to a timestamped directory:

```
outputs/<experiment_name>/<YYYY-MM-DD_HH-MM-SS>/
├── config.yaml          # Full resolved configuration
├── checkpoints/
│   ├── best_model.pt    # Best checkpoint by validation loss
│   └── latest.pt        # Most recent checkpoint
└── train.log            # Training log
```

---

## Evaluation

### MVN and GMM

```bash
python evals/evaluate_mvn_gmm.py \
  --output_dir outputs \
  --experiment mvn \
  --n_out_dist 1000 \
  --batch_size 100 \
  --set_size 1000 \
  --save_path evals/mvn_eval_results.pkl
```

Metrics computed:
- **W2²** — Analytic Wasserstein-2 squared distance (Bures metric; MVN only)
- **SWD** — Sliced Wasserstein distance
- **MMD** — Maximum mean discrepancy (RBF kernel, median bandwidth heuristic)
- **Energy** — Energy distance

Results are saved as both `.pkl` and `.csv`.

### MNIST-Colors and Handwriting

```bash
python evals/evaluate_mnist_colors.py --output_dir outputs --save_path evals/mnist_colors_eval_results.pkl
python evals/evaluate_handwriting.py  --output_dir outputs --save_path evals/handwriting_eval_results.pkl
```

### SnapMMD

```bash
python snapmmd_eval.py --output_dir outputs/<snapMMD_experiment_dir>
```

---

## Experiment Tracking

All runs log to [Weights & Biases](https://wandb.ai). W&B mode and project are configured in `config/wandb/`. To disable W&B:

```bash
python main.py experiment=mvn_base wandb.mode=disabled
```

---

## Reproducibility

- All randomness is controlled by the `seed` parameter.
- Configuration hashes are logged to W&B and used for output directory naming to prevent duplicate runs.
- The training script skips re-running experiments whose output directory already exists and contains a valid checkpoint.

