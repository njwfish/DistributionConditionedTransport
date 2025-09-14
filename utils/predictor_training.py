import numpy as np, torch
from typing import List, Tuple, Optional
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def build_latent_transition_dataset(
    encoder: torch.nn.Module,
    data: dict,
    num_sets: int,
    set_size: int,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    encoder.eval()
    rng = np.random.default_rng(seed)
    """
    Returns (X_ridge, y_ridge) of source/target latents built from consecutive training timepoints.
    Final held-out timepoint in training['X_val_true'] is NOT used as a target in the pairs.
    """
    Xs = data['Xs']
    with torch.no_grad():
        # Collect latents per timepoint for all training steps + held-out final
        latents_by_time: List[List[np.ndarray]] = []
        for X_np in list(Xs[:-1]):
            time_latents = []
            for _ in range(num_sets):
                subset_indices = rng.choice(X_np.shape[0], size=set_size, replace=True)
                subset = torch.tensor(X_np[subset_indices], dtype=torch.float32, device=device).unsqueeze(0)
                lat = encoder(subset).cpu().numpy()
                lat = np.squeeze(lat, axis=0).astype(np.float64)
                time_latents.append(lat)
            
            latents_by_time.append(time_latents)
            
    # Build source/target pairs from consecutive TRAINING timepoints only (exclude transition into held-out final)
    X_pairs, y_pairs = [], []
    
    for t in range(len(latents_by_time) - 1):
        cur, nxt = latents_by_time[t], latents_by_time[t + 1]
        for a,b in zip(cur, nxt):
            X_pairs.append(a); y_pairs.append(b)

    return np.vstack(X_pairs), np.vstack(y_pairs)

def train_ridge_on_latents(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: float = 1.0,
    seed: int = 0,
    ):

    model = Ridge(alpha=alpha, random_state=seed).fit(X, Y)
    return model

def get_ridge(
    encoder,
    data,
    num_sets: int = 1,
    set_size: int = 32,
    alpha: float = 1.0,
    device = "cpu",
    seed: int = 0,
):
    X, Y = build_latent_transition_dataset(
        encoder, data, num_sets=num_sets, set_size=set_size, device=device, seed=seed
    )
    model = train_ridge_on_latents(X, Y, alpha=alpha, seed=seed)
    return model