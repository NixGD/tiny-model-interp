"""Common utilities for model analysis."""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.progress import track
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from analysis.char_classes import CharClass
from tiny_model.model import GPT, CacheKey, GPTConfig, ModelOut
from tiny_model.utils import REPO_ROOT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_path: str | Path = REPO_ROOT / "models/web-char-11-20-rezero-b/ckpt.pt") -> GPT:
    """Load a trained GPT model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model_args = checkpoint["model_args"]
    model = GPT(GPTConfig(**model_args))
    state_dict = checkpoint["model"]
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def get_batch(
    batch_size: int,
    block_size: int = 256,
    data_path: Path = REPO_ROOT / "data/fineweb_char" / "val.bin",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get a batch of data, returning input tokens and target tokens."""
    memmap = np.memmap(data_path, dtype=np.uint16, mode="r")

    def get_seq(i: int) -> torch.Tensor:
        return torch.from_numpy((memmap[i : i + block_size]).astype(np.int64))

    ix = torch.randint(len(memmap) - block_size, (batch_size,)).tolist()
    xs = torch.stack([get_seq(i) for i in ix]).to(DEVICE)
    ys = torch.stack([get_seq(i + 1) for i in ix]).to(DEVICE)
    return xs, ys


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to numpy array."""
    return tensor.detach().cpu().numpy()


def flatten_keep_last(array: np.ndarray) -> np.ndarray:
    """Flatten an array and keep the last dimension."""
    return array.reshape(-1, array.shape[-1])


def save_fig(path: Path | None = None):
    if path is not None:
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {path}")


def variance_curve_pca(X: np.ndarray, max_components: int) -> np.ndarray:
    """Compute variance curve for PCA."""
    n_components_pca = min(max_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components_pca)
    X_centered = X - X.mean(axis=0)
    pca.fit(X_centered)
    return np.cumsum(pca.explained_variance_ratio_)


def extract_activations(output: ModelOut, cache_key: CacheKey) -> np.ndarray:
    """Extract and flatten activations from model output cache."""
    return flatten_keep_last(to_numpy(output.cache.get_value(cache_key)))


Metric = Literal["prob", "logit_diff"]


def get_metric(output: ModelOut, char_class: CharClass, metric: Metric) -> np.ndarray:
    """Get the appropriate metric for a character class."""
    fn = char_class.get_probabilities if metric == "prob" else char_class.get_logit_diff
    return to_numpy(fn(output.logits)).flatten()


#### r2 curve (r^2 for different number of components in the basis) ####


def compute_pls_r2_curve(
    output: ModelOut, cache_key: CacheKey, char_class: CharClass, metric: Metric, max_components: int
) -> list[float]:
    """Returns curve of R² explained by n PLS components for the given metric and model activations."""
    activations = extract_activations(output, cache_key)
    predictions_flat = get_metric(output, char_class, metric)

    max_components = min(max_components, activations.shape[1])
    pls = PLSRegression(n_components=max_components)
    pls.fit(activations, predictions_flat)
    acts_centered = activations - pls._x_mean

    def get_r2_score(n_components: int) -> float:
        """Get the predictions from PLS components."""
        coef_n = (pls._y_std / pls._x_std) * (pls.y_loadings_[:, :n_components] @ pls.x_rotations_[:, :n_components].T)
        y_pred = (acts_centered @ coef_n.T).flatten() + pls._y_mean
        return r2_score(predictions_flat, y_pred)

    return [get_r2_score(n_comp) for n_comp in range(1, max_components + 1)]


def compute_pca_r2_curve(
    output: ModelOut, cache_key: CacheKey, char_class: CharClass, metric: Metric, max_components: int
) -> list[float]:
    """Compute the amount of y-variance explained by the first n-PCA components of X.

    Notably this is different from the variance curve of PCA, which is the amount of X-variance explained by the first n-PCA components.
    """
    activations = extract_activations(output, cache_key)
    activations_centered = activations - activations.mean(axis=0)
    metric_values = get_metric(output, char_class, metric)

    def _get_r2_score(n_components: int) -> float:
        pca = PCA(n_components=n_components)
        acts_in_pca = pca.fit_transform(activations_centered)
        reg = LinearRegression()
        reg.fit(acts_in_pca, metric_values)
        return reg.score(acts_in_pca, metric_values)

    return [_get_r2_score(n_comp) for n_comp in track(range(1, max_components + 1), description="PCA R² curve")]
