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
        
        # Expand latents to match the number of source samples
        if source_latent.shape[0] != batch_size:
            source_latent = source_latent.unsqueeze(1).repeat(1, source_samples.shape[0] // source_latent.shape[0], 1).view(-1, source_latent.shape[-1])
        if target_latent.shape[0] != batch_size:
            target_latent = target_latent.unsqueeze(1).repeat(1, source_samples.shape[0] // target_latent.shape[0], 1).view(-1, target_latent.shape[-1])
        
        # Create context by combining source and target latents
        z = torch.cat([source_latent, target_latent], dim=-1)
        
        # Sample noise for the energy model
        noise = torch.randn(batch_size, self.noise_dim, device=source_samples.device)
        
        # Predict clean samples
        with torch.no_grad():
            predicted_samples = self.model(source_samples, z, noise)
        
        return predicted_samples

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
        x0_preds = x0_preds.view(n, m, -1)
        
        # Confinement term: ||x0_true - x̂||
        x0_true_rep = target_samples.unsqueeze(1).expand(-1, m, -1)
        term_conf = (x0_preds - x0_true_rep).norm(dim=2).mean(dim=1)  # [n]
        
        # Interaction term: pairwise distances between predictions
        if m > 1:
            pdists = torch.stack([torch.pdist(x0_preds[i], p=2) for i in range(n)], dim=0)
            mean_pd = pdists.mean(dim=1)  # [n]
            term_int = (λ / 2.0) * mean_pd  # [n]
        else:
            term_int = torch.zeros_like(term_conf)
        
        # Energy score: confinement - interaction
        energy_loss = (term_conf - term_int).mean()
        
        return energy_loss

    def sample(self, source_samples, source_latent, target_latent):
        """
        Generate samples by applying energy-based denoising
        
        Args:
            source_samples: [batch_size, ...] starting points
            source_latent: [batch_size, latent_dim] source embedding
            target_latent: [batch_size, latent_dim] target embedding
        """
        num_samples = source_samples.shape[0] // source_latent.shape[0]
        generated = self.forward(source_samples, source_latent, target_latent)
        
        # Reshape result: [batch_size, ...] -> [num_sets, num_samples, ...]
        return generated.reshape(source_latent.shape[0], num_samples, *source_samples.shape[1:])
