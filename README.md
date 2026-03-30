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
| `kernel_mean` | `KMEEncoder` | Kernel mean embedding (random Fourier features) |
| `embedding` | `EmbeddingEncoder` | Learned lookup table (requires known distribution indices) |
| `esm` | ESM2-based | Protein language model for sequence sets |

The `DistributionEncoder` variants (`gnn`, `tx`, `kernel_mean`) operate on raw sample sets (`[B, N, D]`) and are compatible with out-of-distribution generalization. The `EmbeddingEncoder` requires distribution indices and uses `loss: multimarginal`.

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

Experiments are designed for SLURM clusters using Hydra's Submitit launcher. The SLURM launcher is configured in `config/hydra/launcher/`.

### Encoder Comparison Sweep

The main sweep compares three distribution encoders (GNN, KME, Transformer) across generators, datasets, and training modes. Scripts pack 8 experiments per GPU for efficient use of H100 nodes.

```bash
# Full sweep: GNN encoder (MVN + GMM, unsupervised + supervised)
bash scripts/sweep_mvn_gmm.sh

# KME + Transformer encoder sweep (all packed onto H100s)
bash scripts/sweep_packed_kme.sh  # KME only
bash scripts/sweep_packed_tx.sh   # Transformer only

# Or submit KME separately (1 job per experiment)
bash scripts/sweep_mvn_gmm_kme.sh
```

Each sweep covers:
- **Datasets**: MVN, GMM
- **Training modes**: unsupervised (any-to-any), supervised (source-only)
- **Generators**: flow_matching, mmd, wasserstein
- **Scales**: n_unique_sets = 10, 100, 1000, 10000
- **Seeds**: 40, 41, 42

### Other Sweeps

```bash
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

### MVN and GMM (Unsupervised)

Evaluates all trained models across encoder types (GNN, KME, ResNet+Tx, Embedding) and generator types. Automatically discovers experiments from the `outputs/` directory.

```bash
# Evaluate MVN experiments (all encoders, all generators)
python evals/evaluate_mvn_gmm.py \
  --output_dir outputs \
  --experiment mvn \
  --num_epochs 200 \
  --generators flow_matching mmd wasserstein \
  --n_out_dist 10000 \
  --batch_size 500 \
  --save_path evals/mvn_eval_results.pkl

# Evaluate GMM experiments
python evals/evaluate_mvn_gmm.py \
  --output_dir outputs \
  --experiment gmm \
  --num_epochs 200 \
  --generators flow_matching mmd wasserstein \
  --n_out_dist 10000 \
  --batch_size 500 \
  --save_path evals/gmm_eval_results.pkl

# Load a specific epoch checkpoint instead of best_model.pt
python evals/evaluate_mvn_gmm.py \
  --experiment mvn \
  --checkpoint_epoch 200 \
  --save_path evals/mvn_epoch200_results.pkl
```

Results are saved as both `.pkl` and `.csv` with columns: `encoder_type` (gnn/kme/resnet_tx/embedding), `generator_type`, `n_unique_sets`, `seed`, and metric values.

### Supervised vs Semi-supervised Comparison

Evaluates supervised (source-only) models against semi-supervised (any-to-any + ridge predictor) on a 2D grid, testing in-distribution and out-of-distribution generalization.

```bash
# MVN supervised comparison (all encoders)
python evals/evaluate_supervised_comparison.py \
  --output_dir outputs \
  --experiment mvn \
  --num_epochs 200 \
  --n_grid_points 21 \
  --eval_set_size 10000 \
  --save_dir evals/supervised_comparison_mvn

# GMM supervised comparison
python evals/evaluate_supervised_comparison.py \
  --output_dir outputs \
  --experiment gmm \
  --num_epochs 200 \
  --save_dir evals/supervised_comparison_gmm

# Evaluate a specific generator/seed
python evals/evaluate_supervised_comparison.py \
  --experiment mvn \
  --generator flow_matching \
  --seed 40 \
  --save_dir evals/supervised_comparison_mvn_fm
```

### Bulk Evaluation on SLURM

```bash
# Submit evaluation as a GPU job (recommended for large sweeps)
bash scripts/submit_evaluation.sh
```

### Metrics

- **W2²** — Analytic Wasserstein-2 squared distance (Bures metric; MVN only)
- **SWD** — Sliced Wasserstein distance
- **MMD** — Maximum mean discrepancy (RBF kernel, median bandwidth heuristic)
- **Energy** — Energy distance

### Visualization

Analysis notebooks in `notebooks/`:

| Notebook | Description |
|---|---|
| `evaluate_mvn_gmm.ipynb` | Unsupervised results: bar charts and tables by encoder x generator x K |
| `supervised.ipynb` | Supervised generalization: IID vs OOD performance by encoder type |
| `analyze_mvn_results.ipynb` | Detailed MVN analysis with heatmaps |

All notebooks automatically detect and compare encoder types (GNN, KME, ResNet+Tx).

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

