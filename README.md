# Coupled Generative Distribution Embeddings

Coupled Distribution Embeddings (CDEs) extend the generative distribution embeddings framework to learn transport maps between distributions. In CDEs, an encoder learns representations of distribution sets, and generators learn to transport samples from source distributions to target distributions based on their embeddings. This repository implements CDE architectures with multiple generator types, data collation strategies, and applications to synthetic and real-world datasets.

## Key Features

- **Coupled Transport Learning**: Learn to transport samples from source to target distributions
- **Multiple Generator Types**: Direct generators (SW, MMD, Sinkhorn losses) and Flow Matching
- **Flexible Data Collation**: Random and shift pairing strategies, with optional Dirichlet mixing
- **Reproducible Experiments**: Hash-based output tracking and Hydra configuration management
- **Scalable Training**: Slurm integration for large-scale experiments

## Setup and Configuration

### Project Structure

```
.
├── config/                      # Main Hydra configuration files
│   ├── dataset/                 # Dataset configurations
│   ├── encoder/                 # Encoder model configurations
│   ├── generator/               # Generator configurations (direct, flow_matching)
│   ├── model/                   # Model architecture configurations
│   ├── collate/                 # Data collation strategies (pairing, mixing)
│   ├── optimizer/               # Optimizer configurations
│   ├── scheduler/               # Learning rate scheduler configurations
│   ├── training/                # Training configurations
│   ├── wandb/                   # Weights & Biases configurations
│   ├── hydra/                   # Hydra-specific configurations (incl. Slurm)
│   ├── experiment/              # Complete experiment configurations
│   │   ├── mvn_coupled.yaml     # Pure coupled embeddings experiment
│   │   ├── mvn_mixed_coupled.yaml # Mixed + coupled experiment
│   │   └── ...                  # Other domain-specific experiments
│   └── config.yaml              # Base configuration file
├── datasets/                    # Dataset implementations
├── encoder/                     # Encoder models (GNN, ResNet, etc.)
├── generator/                   # Generator implementations
│   ├── direct.py                # Direct transport generators
│   ├── flow_matching.py         # Flow matching generator
│   └── losses.py                # Loss functions (SW, MMD, Sinkhorn)
├── collate/                     # Data collation strategies
│   ├── paired_collate.py        # Pairing strategies (random, shift)
│   └── mixing_collate.py        # Dirichlet mixing + pairing
├── model/                       # Model architectures (GNN, MLP, etc.)
├── utils/                       # Utility functions
├── notebooks/                   # Jupyter notebooks for analysis
├── outputs/                     # Experiment outputs directory
├── main.py                      # Main training script
├── training.py                  # Training implementation
└── requirements.txt             # Project dependencies
```

### Installation and Dependencies

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- PyTorch and torchvision
- torchdyn (for Flow Matching NeuralODE integration)
- NumPy and matplotlib
- Hydra for configuration management
- Weights & Biases for experiment tracking
- geomloss for optimal transport losses
- scikit-learn and pandas for data processing

## Core Concepts

### Coupled Distribution Embeddings

Unlike traditional generative distribution embeddings that learn unconditional generators, CDEs learn **transport maps** between distributions:

1. **Encoder**: Maps sets of samples to latent embeddings: `{x₁, ..., xₙ} → z`
2. **Pairing**: Creates source-target pairs from batches for training transport
3. **Generator**: Learns transport maps: `(x_source, z_source, z_target) → x_transported`

The key insight is that generators start from **source samples** rather than noise, learning to transport them to match the target distribution.

### Data Collation Strategies

#### Pairing Strategies

- **Random Pairing**: Randomly permutes within batch to create source-target pairs
- **Shift Pairing**: Systematic shift (e.g., target[i] = source[(i+1) % batch_size])

#### Mixing + Pairing

- **Dirichlet Mixing**: First creates diverse distributions by mixing k source sets with Dirichlet weights
- **Then Pairs**: Applies random pairing to mixed distributions for transport learning

```yaml
# Pure random pairing
collate:
  _target_: collate.paired_collate.PairedCollate
  method: random

# Dirichlet mixing + random pairing  
collate:
  _target_: collate.mixing_collate.SetMixer
  k: 3                    # Mix from 3 source sets
  alpha: 0.5              # Sparse Dirichlet (prefer one dominant source)
```

## Generator Types

### 1. Direct Generators (`generator/direct.py`)

Direct generators learn transport maps using distributional losses:

```python
class DirectGenerator(nn.Module):
    def loss(self, source_samples, target_samples, source_latent, target_latent):
        # Transport source samples using latent embeddings
        transported = self.forward(source_samples, source_latent, target_latent)
        # Compare transported vs target samples
        return self.loss_fn(transported, target_samples)
```

**Supported Loss Functions:**
- **Sliced Wasserstein Distance (SWD)**: Fast approximation using random projections
- **Maximum Mean Discrepancy (MMD)**: Kernel-based distributional distance  
- **Sinkhorn**: Regularized optimal transport distance

**Configuration:**
```yaml
generator:
  _target_: generator.direct.DirectGenerator
  loss_type: swd  # or 'mmd', 'sinkhorn'
  loss_params:
    n_projections: 100  # for SWD
```

### 2. Flow Matching Generator (`generator/flow_matching.py`)

Flow matching learns continuous transport paths using neural ODEs:

```python
class FlowMatchingGenerator(nn.Module):
    def loss(self, source_samples, target_samples, source_latent, target_latent):
        # Sample random time t ∈ [0,1]
        t = self.sample_time(batch_size, device)
        # Linear interpolant: x_t = (1-t)*source + t*target  
        x_t = self.interpolant(source_samples, target_samples, t)
        # True velocity: v = target - source
        v_true = self.velocity_field(source_samples, target_samples, t)
        # Predicted velocity field
        v_pred = self.model(x_t, t, source_latent, target_latent)
        # MSE loss between predicted and true velocity
        return torch.mean((v_pred - v_true) ** 2)
```

**Key Features:**
- **Neural ODE Integration**: Uses torchdyn for high-quality adaptive integration
- **Linear Interpolant**: Simple and effective x_t = (1-t)*source + t*target path
- **Velocity Field Prediction**: Model learns v_t(x_t, t, z_source, z_target) 
- **Trajectory Support**: Optional full trajectory return for analysis

**Configuration:**
```yaml
generator:
  _target_: generator.flow_matching.FlowMatchingGenerator
  sigma_min: 1e-4  # Numerical stability noise
```

**Usage:**
```python
# Standard sampling (final transported samples)
generated = generator.sample(source_samples, source_latent, target_latent)

# Return full transport trajectory for analysis
trajectory = generator.sample(
    source_samples, source_latent, target_latent, 
    return_trajectory=True
)
```

## Basic Usage

### Quick Start

```bash
# Run coupled MVN experiment with Flow Matching
python main.py experiment=mvn_coupled

# Run with Dirichlet mixing + pairing
python main.py experiment=mvn_mixed_coupled generator=flow_matching model=diffusion_mlp

# Override specific parameters
python main.py experiment=mvn_coupled generator=wasserstein model=direct_mlp

```

### Configuration Examples

#### Pure Coupled Embeddings
```yaml
# config/experiment/mvn_coupled.yaml
defaults:
  - /dataset: mvn
  - /encoder: gnn  
  - /generator: flow_matching
  - /collate: random_pairing  # PairedCollate with random method

latent_dim: 16
batch_size: 256
```

#### Mixing + Coupling  
```yaml
# config/experiment/mvn_mixed_coupled.yaml
defaults:
  - /dataset: mvn
  - /encoder: gnn
  - /generator: flow_matching  
  - /collate: dirichlet_k_mixing  # SetMixer with k=3, alpha=0.5

n_mix: 3      # Number of sets to mix
alpha: 0.5    # Dirichlet concentration (sparse mixing)
```


## Experiment Management

### Output Structure

Each experiment creates a hash-based directory:

```
outputs/
└── mvn_coupled_exp_f7a3b9d2/     # experiment_name_[config_hash]
    ├── config.yaml               # Complete configuration
    ├── best_model.pt             # Best checkpoint
    ├── coupled_samples_0_epoch_*.png  # Visualizations showing source→target→generated
    └── metrics.json              # Training metrics
```

### CLI Tools

```bash
# List all experiments
python experiment_cli.py list

# Show specific experiment details  
python experiment_cli.py show mvn_coupled_exp_f7a3b9d2

# Compare different generator approaches
python experiment_cli.py compare \
  mvn_coupled_direct_a1b2c3 \
  mvn_coupled_flow_d4e5f6
```

## Visualization and Analysis

The training system automatically generates coupled transport visualizations:

- **Source samples**: Original distribution samples (blue)
- **Target samples**: Target distribution samples (green)  
- **Generated samples**: Transported samples (red)

**Numerical Data**: Shows scatter plots, pairplots, and statistical comparisons
**Text Data**: CSV files with source/target/generated text samples
**Image Data**: Three-panel grids showing the transport pipeline

## Available Experiments

### 1. Multivariate Normal Distributions

Test transport learning between MVN distributions:

```bash
# Pure coupled learning
python main.py experiment=mvn_coupled

# With Dirichlet mixing for diversity
python main.py experiment=mvn_mixed_coupled
```

### 2. Gaussian Mixture Models

Complex multi-modal transport learning:

```bash
python main.py experiment=gmm_coupled
```

### 3. Text Transport (PubMed)

Learn to transport between different document topic distributions:

```bash
python main.py experiment=pubmed_coupled encoder=bert generator=gpt2
```

### 4. Biological Applications

Apply transport learning to genomics and proteomics:

```bash
# Essential genes perturbation transport
python main.py experiment=essential_genes_coupled

# DNA sequence transport
python main.py experiment=dna_coupled

# Protein sequence transport  
python main.py experiment=virus_coupled
```

## Model Architectures

### Encoders
- **GNN**: Graph neural networks for set encoding
- **ResNet**: Residual networks with set aggregation
- **BERT**: For text document sets
- **ESM**: For protein sequence sets

### Models (used within generators)
- **MLP**: Multi-layer perceptrons
- **GNN**: Graph networks  
- **UNet**: For diffusion-style models
- **Transformer**: For sequence generation

### Loss Functions
- **Transport Losses**: SWD, MMD, Sinkhorn for direct generators
- **Flow Matching**: MSE loss on velocity field predictions
- **Reconstruction**: For VAE-style models

## Contributing

This codebase supports research in distribution transport learning. Key areas for extension:

1. **New Generator Types**: Implement other transport learning approaches
2. **Advanced Pairing**: Develop smarter source-target pairing strategies  
3. **Domain Applications**: Apply to new scientific domains
4. **Evaluation Metrics**: Add transport-specific evaluation measures

## Citation

If you use this codebase in your research, please cite:

```
@software{coupled_distribution_embeddings,
  title = {Coupled Distribution Embeddings},
  author = {[Authors]},
  year = {2024},
  url = {https://github.com/[repo]/CoupledDistributionEmbeddings}
}
```
