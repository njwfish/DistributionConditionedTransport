import numpy as np, torch
from typing import List, Tuple
from sklearn.linear_model import Ridge


def build_latent_transition_dataset(
    encoder: torch.nn.Module,
    data: dict,
    num_sets: int,
    set_size: int,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    seed: int = 0,
    two_step: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    encoder.eval()
    rng = np.random.default_rng(seed)
    """
    Returns (X_ridge, y_ridge) of source/target latents built from training timepoints only.

    Modes:
    - two_step=False (default): input = latent at t; output = latent at t+1
    - two_step=True: input = concat(latent at t, latent at t+1); output = latent at t+2

    The transition into any held-out final timepoint is excluded.
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
            
    # Build source/target pairs from TRAINING timepoints only (exclude transition into held-out final)
    X_pairs, y_pairs = [], []
    
    if not two_step:
        for t in range(len(latents_by_time) - 1):
            cur, nxt = latents_by_time[t], latents_by_time[t + 1]
            # cross product across set samples at consecutive timepoints
            for a in cur:
                for b in nxt:
                    X_pairs.append(a); y_pairs.append(b)
    else:
        # Need triples of consecutive timepoints (t, t+1, t+2)
        for t in range(len(latents_by_time) - 2):
            cur, nxt, nxt2 = latents_by_time[t], latents_by_time[t + 1], latents_by_time[t + 2]
            # cross product across set samples: (a at t, b at t+1) -> c at t+2
            for a in cur:
                for b in nxt:
                    ab = np.concatenate([a, b], axis=-1)
                    for c in nxt2:
                        X_pairs.append(ab); y_pairs.append(c)

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
    num_sets: int = 1,
    set_size: int = 32,
    alpha: float = 1.0,
    device = "cpu",
    seed: int = 0,
    two_step: bool = False,
):
    X, Y = build_latent_transition_dataset(
        encoder, data, num_sets=num_sets, set_size=set_size, device=device, seed=seed, two_step=two_step
    )
    model = train_ridge_on_latents(X, Y, alpha=alpha, seed=seed)
    return model