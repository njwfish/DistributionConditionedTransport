import torch
import torch.nn as nn

class EnergyGenerator(nn.Module):
    def __init__(
        self, 
        model, 
        sigma_min=0.1,
        noise_dim=16,
        m=16,
        lambda_energy=1.0,
    ):
        """
        Energy Generator for coupled distribution embeddings using energy score.
        
        Args:
            model: Neural network that predicts x₀̂ = f_θ(x_t, t, z, ε)
            sigma_min: Minimum noise level for numerical stability
            noise_dim: Dimension of noise vector
            m: Number of samples for m-sample approximation in energy score
            lambda_energy: Weight for interaction term in energy score
            condition_on_m: Whether to condition on M_t in the model
        """
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min
        self.noise_dim = noise_dim
        self.m = m
        self.lambda_energy = lambda_energy

    def forward(self, source_samples, source_latent, target_latent):
        """
        Generate samples using energy score denoising
        
        Args:
            source_samples: [batch_size, ...] starting points (noisy samples)
            source_latent: [batch_size, latent_dim] source distribution embedding
            target_latent: [batch_size, latent_dim] target distribution embedding
        """
        batch_size = source_samples.shape[0]
        
        # Validate latent dimensions
        if source_latent.dim() != 2 or target_latent.dim() != 2:
            raise ValueError(
                f"EnergyGenerator.forward expects 2D latents shaped (batch_size, latent_dim)."
                f" Got source_latent.shape={tuple(source_latent.shape)},"
                f" target_latent.shape={tuple(target_latent.shape)}"
            )
        
        # Normalize latents per-sample before using them
        source_latent = source_latent / torch.norm(source_latent, dim=-1, keepdim=True).clamp_min(1e-12)
        target_latent = target_latent / torch.norm(target_latent, dim=-1, keepdim=True).clamp_min(1e-12)
        
        # Expand latents to match the number of source samples
        if source_latent.shape[0] != batch_size:
            source_latent = source_latent.unsqueeze(1).repeat(1, source_samples.shape[0] // source_latent.shape[0], 1).view(-1, source_latent.shape[-1])
        if target_latent.shape[0] != batch_size:
            target_latent = target_latent.unsqueeze(1).repeat(1, source_samples.shape[0] // target_latent.shape[0], 1).view(-1, target_latent.shape[-1])
        
        # Sample noise for the energy model
        noise = torch.randn(batch_size, self.noise_dim, device=source_samples.device)
        
        # Predict clean samples
        with torch.no_grad():
            predicted_samples = self.model(source_samples, source_latent, target_latent, noise)
        
        return predicted_samples

    def _compute_energy_score(self, predictions, target, λ):
        """
        Compute energy score for a single modality.
        
        Args:
            predictions: [n, m, d] - m predictions per sample
            target: [n, d] - target values
            λ: energy score regularization parameter
            
        Returns:
            energy_score: scalar loss value
        """
        n, m = predictions.shape[:2]  # Handle arbitrary dimensions after m
        
        # Flatten spatial dimensions for distance computation
        predictions_flat = predictions.view(n, m, -1)  # [n, m, d_flat]
        target_flat = target.view(n, -1)  # [n, d_flat]
        
        # Confinement term: distance to target
        target_expanded = target_flat.unsqueeze(1).expand(-1, m, -1)  # [n, m, d_flat]
        term_conf = (predictions_flat - target_expanded).norm(dim=2).mean(dim=1)  # [n]
        
        # Interaction term (efficient batched computation)
        # Using ||a-b||² = ||a||² + ||b||² - 2⟨a,b⟩ identity
        sq = predictions_flat.pow(2).sum(dim=2)  # [n, m] - squared norms
        inn = torch.bmm(predictions_flat, predictions_flat.transpose(1,2))  # [n, m, m] - inner products
        sqd = sq.unsqueeze(2) + sq.unsqueeze(1) - 2*inn  # [n, m, m] - squared distances
        sqd = torch.clamp(sqd, min=1e-6)  # avoid sqrt(0)
        d = sqd.sqrt()  # [n, m, m] - distances
        
        # Mean of off-diagonal pairwise distances
        m_mask = torch.ones(m, m, device=predictions.device) - torch.eye(m, device=predictions.device)
        mean_pd = (d * m_mask).sum(dim=(1,2)) / (m * (m - 1))  # [n]
        term_int = (λ / 2.0) * mean_pd  # [n]
        
        return (term_conf - term_int).mean()

    def loss(self, source_samples, target_samples, source_latent, target_latent):
        """
        Energy score loss using m-sample approximation
        L = mean_i [(1/m) ∑_j ||x0_true_i - x̂_ij|| - (λ/(2(m-1))) ∑_{j≠j'} ||x̂_ij - x̂_ij'||]
        
        Args:
            source_samples: samples from source distribution
            target_samples: samples from target distribution (ground truth)
            source_latent: source distribution embedding  
            target_latent: target distribution embedding
        """
        batch_size = source_samples.shape[0]
        n = target_samples.shape[0]  # number of data points
        m = self.m  # number of samples per data point
        λ = self.lambda_energy

        # Validate latent dimensions
        if source_latent.dim() != 2 or target_latent.dim() != 2:
            raise ValueError(
                f"EnergyGenerator.loss expects 2D latents shaped (batch_size, latent_dim)."
                f" Got source_latent.shape={tuple(source_latent.shape)},"
                f" target_latent.shape={tuple(target_latent.shape)}"
            )

        # Normalize latents per-sample before using them
        source_latent = source_latent / torch.norm(source_latent, dim=-1, keepdim=True).clamp_min(1e-12)
        target_latent = target_latent / torch.norm(target_latent, dim=-1, keepdim=True).clamp_min(1e-12)

        # Expand latents to match the number of source samples
        if source_latent.shape[0] != batch_size:
            source_latent = source_latent.unsqueeze(1).repeat(1, source_samples.shape[0] // source_latent.shape[0], 1).view(-1, source_latent.shape[-1])
        if target_latent.shape[0] != batch_size:
            target_latent = target_latent.unsqueeze(1).repeat(1, source_samples.shape[0] // target_latent.shape[0], 1).view(-1, target_latent.shape[-1])
        
        
        # Replicate inputs m times for m-sample approximation
        x_rep = source_samples.unsqueeze(1).expand(-1, m, -1).reshape(n * m, -1)
        source_latent = source_latent.unsqueeze(1).expand(-1, m, -1).reshape(n * m, -1)
        target_latent = target_latent.unsqueeze(1).expand(-1, m, -1).reshape(n * m, -1)
        
        # Sample noise for each replicated input
        noise = torch.randn(n * m, self.noise_dim, device=source_samples.device)

        
        # Get m predictions per data point: [n*m, x_dim] -> [n, m, x_dim]
        x0_preds = self.model(x_rep, source_latent, target_latent, noise)
        x0_preds = x0_preds.reshape(n, m, -1)

        return self._compute_energy_score(x0_preds, target_samples, λ)

    def sample(self, source_samples, source_latent, target_latent):
        """
        Generate samples by applying energy-based denoising
        
        Args:
            source_samples: [batch_size, ...] starting points
            source_latent: [batch_size, latent_dim] source embedding
            target_latent: [batch_size, latent_dim] target embedding
        """
        
        # Validate latent dimensions
        if source_latent.dim() != 2 or target_latent.dim() != 2:
            raise ValueError(
                f"EnergyGenerator.sample expects 2D latents shaped (batch_size, latent_dim)."
                f" Got source_latent.shape={tuple(source_latent.shape)},"
                f" target_latent.shape={tuple(target_latent.shape)}"
            )

        # Normalize latents per-sample before using them
        source_latent = source_latent / torch.norm(source_latent, dim=-1, keepdim=True).clamp_min(1e-12)
        target_latent = target_latent / torch.norm(target_latent, dim=-1, keepdim=True).clamp_min(1e-12)

        num_samples = source_samples.shape[0] // source_latent.shape[0]
        generated = self.forward(source_samples, source_latent, target_latent)
        
        # Reshape result: [batch_size, ...] -> [num_sets, num_samples, ...]
        return generated.reshape(source_latent.shape[0], num_samples, *source_samples.shape[1:])
