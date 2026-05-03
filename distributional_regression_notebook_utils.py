import numpy as np
import torch
import torch.nn as nn


class ConditionalSampler(nn.Module):
    """Copied from lt_distributional_regression.ipynb."""

    def __init__(self, x_dim, z_dim, y_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, y_dim),
        )
        self.z_dim = z_dim

    def forward(self, x, z=None):
        if z is None:
            z = torch.randn(x.shape[0], self.z_dim, device=x.device)
        return self.net(torch.cat([x, z], dim=-1))

    def sample(self, x, k=1):
        # returns (k, n, y_dim)
        n = x.shape[0]
        x_rep = x.unsqueeze(0).expand(k, -1, -1).reshape(k * n, -1)
        z = torch.randn(k * n, self.z_dim, device=x.device)
        return self.net(torch.cat([x_rep, z], dim=-1)).reshape(k, n, -1)


def energy_score_loss(samples, y_true, k=8):
    """Copied from lt_distributional_regression.ipynb."""
    # samples: (k, n, d), y_true: (n, d)
    # ES = E||Y' - y|| - 0.5 * E||Y' - Y''||
    diff_to_true = torch.cdist(samples.permute(1, 0, 2), y_true.unsqueeze(1))
    term1 = diff_to_true.squeeze(-1).mean(dim=1)

    pairwise = torch.cdist(samples.permute(1, 0, 2), samples.permute(1, 0, 2))
    term2 = pairwise.sum(dim=(1, 2)) / (k * (k - 1) + 1e-8)

    return (term1 - 0.5 * term2).mean()


def fit_conditional_sampler(
    X_train,
    Y_train,
    z_dim=16,
    hidden=256,
    lr=1e-3,
    epochs=500,
    batch_size=256,
    k=8,
    device="cuda",
):
    """Copied from lt_distributional_regression.ipynb with an empty-data guard."""
    x_dim = X_train.shape[1]
    y_dim = Y_train.shape[1]
    model = ConditionalSampler(x_dim, z_dim, y_dim, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    X = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y = torch.tensor(Y_train, dtype=torch.float32, device=device)
    n = X.shape[0]
    if n == 0:
        raise ValueError("Cannot fit ConditionalSampler with zero training examples.")

    model.train()
    for epoch in range(epochs):
        idx = torch.randint(0, n, (batch_size,), device=device)
        xb, yb = X[idx], Y[idx]
        samples = model.sample(xb, k=k)
        loss = energy_score_loss(samples, yb, k=k)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (epoch + 1) % 100 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch + 1}/{epochs}  loss={loss.item():.4f}")

    model.eval()
    return model
