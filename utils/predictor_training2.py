import numpy as np, torch
from typing import List, Tuple
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import TimeSeriesSplit

def build_latent_transition_dataset(
    encoder: torch.nn.Module,
    data: dict,
    num_sets: int,
    set_size: int,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    seed: int = 0,
    pairing: str = "cartesian",  # "zip" or "cartesian" or "mean"
) -> Tuple[np.ndarray, np.ndarray]:
    Xs = data["Xs"]
    rng = np.random.default_rng(seed)
    encoder.eval()
    with torch.no_grad():
        latents_by_time: List[List[np.ndarray]] = []
        for X_np in list(Xs[:-1]):  # exclude held-out final
            time_latents = []
            for _ in range(num_sets):
                idx = rng.choice(X_np.shape[0], size=set_size, replace=True)
                subset = torch.tensor(X_np[idx], dtype=torch.float32, device=device).unsqueeze(0)
                lat = encoder(subset).detach().cpu().numpy()
                lat = np.squeeze(lat, axis=0).astype(np.float64)  # (d,)
                time_latents.append(lat)
            latents_by_time.append(time_latents)

    X_pairs, y_pairs = [], []
    for t in range(len(latents_by_time) - 1):
        cur, nxt = latents_by_time[t], latents_by_time[t + 1]
        if pairing == "zip":
            for a, b in zip(cur, nxt):
                X_pairs.append(a); y_pairs.append(b)
        elif pairing == "cartesian":
            for a in cur:
                for b in nxt:
                    X_pairs.append(a); y_pairs.append(b)
        elif pairing == "mean":
            X_pairs.append(np.mean(cur, axis=0)); y_pairs.append(np.mean(nxt, axis=0))
        else:
            raise ValueError("pairing must be 'zip', 'cartesian', or 'mean'")
    return np.vstack(X_pairs), np.vstack(y_pairs)

def train_ridge_on_latents(
    X: np.ndarray,
    Y: np.ndarray,
    alphas: np.ndarray = np.logspace(-6, 3, 30),
    n_splits: int = 5,
):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    base = RidgeCV(alphas=alphas, fit_intercept=True, store_cv_values=False, cv=tscv)
    model = Pipeline([
        ("x_scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge_y_scaled", TransformedTargetRegressor(
            regressor=base,
            transformer=StandardScaler(with_mean=True, with_std=True)
        )),
    ])
    model.fit(X, Y)
    # access chosen alpha via:
    # chosen_alpha = model.named_steps["ridge_y_scaled"].regressor_.alpha_
    return model

def get_ridge(encoder, data, num_sets=4, set_size=32, device="cpu", pairing="cartesian"):
    X, Y = build_latent_transition_dataset(encoder, data, num_sets=num_sets, set_size=set_size, device=torch.device(device), pairing=pairing)
    model = train_ridge_on_latents(X, Y)
    return model
