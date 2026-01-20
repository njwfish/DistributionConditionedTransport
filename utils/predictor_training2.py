import numpy as np, torch
from typing import List, Tuple
from sklearn.linear_model import Ridge


def build_latent_transition_dataset(
    encoder: torch.nn.Module,
    data: dict,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    seed: int = 0,
    two_step: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (X_ridge, y_ridge) of source/target latents built from training timepoints only.

    Encodes the entire data at each timepoint at once (no chunking).

    Modes:
    - two_step=False (default): input = latent at t; output = latent at t+1
    - two_step=True: input = concat(latent at t, latent at t+1); output = latent at t+2

    The transition into any held-out final timepoint is excluded.
    """
    encoder.eval()
    Xs = data['Xs']
    with torch.no_grad():
        # Collect one latent per timepoint (encode entire data at once)
        latents_by_time: List[np.ndarray] = []
        for X_np in list(Xs[:-1]):
            X_tensor = torch.tensor(X_np, dtype=torch.float32, device=device).unsqueeze(0)
            lat = encoder(X_tensor).cpu().numpy()
            lat = np.squeeze(lat, axis=0).astype(np.float64)
            latents_by_time.append(lat)
            
    # Build source/target pairs from TRAINING timepoints only (exclude transition into held-out final)
    X_pairs, y_pairs = [], []
    
    if not two_step:
        for t in range(len(latents_by_time) - 1):
            cur, nxt = latents_by_time[t], latents_by_time[t + 1]
            X_pairs.append(cur)
            y_pairs.append(nxt)
    else:
        # Need triples of consecutive timepoints (t, t+1, t+2)
        for t in range(len(latents_by_time) - 2):
            cur, nxt, nxt2 = latents_by_time[t], latents_by_time[t + 1], latents_by_time[t + 2]
            ab = np.concatenate([cur, nxt], axis=-1)
            X_pairs.append(ab)
            y_pairs.append(nxt2)

    return np.vstack(X_pairs), np.vstack(y_pairs)

def train_ridge_on_latents(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: float = 1.0,
    seed: int = 0,
    ):

    #print(X.shape, Y.shape)
    model = Ridge(alpha=alpha, random_state=seed).fit(X, Y)
    return model

def get_ridge(
    encoder,
    data,
    alpha: float = 1.0,
    device = "cpu",
    seed: int = 0,
    two_step: bool = False,
):
    X, Y = build_latent_transition_dataset(
        encoder, data, device=device, seed=seed, two_step=two_step
    )
    model = train_ridge_on_latents(X, Y, alpha=alpha, seed=seed)
    return model