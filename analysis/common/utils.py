"""Common utilities for model analysis."""

import re
from collections import Counter
from pathlib import Path
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import IterableDataset, load_dataset
from rich.progress import track
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from analysis.common.char_classes import CharClass
from tiny_model.model import GPT, AlphaCache, CacheEnabledSpec, CacheKey, GPTConfig, HookFn, ModelOut
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


@torch.no_grad()
def run_batches(
    model: GPT,
    num_batches: int,
    batch_size: int,
    block_size: int = 256,
    data_path: Path = REPO_ROOT / "data/fineweb_char" / "val.bin",
    cache_enabled: CacheEnabledSpec = True,
    alphas_enabled: bool = False,
    hooks: dict[CacheKey, HookFn] | None = None,
    show_progress: bool = True,
) -> tuple[ModelOut, torch.Tensor, torch.Tensor]:
    """Run multiple batches and return concatenated ModelOut with all results.

    Does not support gradients.
    """
    outputs: list[ModelOut] = []
    x_batches: list[torch.Tensor] = []
    y_batches: list[torch.Tensor] = []

    iterator = range(num_batches)
    if show_progress:
        iterator = track(iterator, description="Running batches")

    for _ in iterator:
        x, y = get_batch(batch_size, block_size, data_path)
        out = model(x, targets=y, cache_enabled=cache_enabled, alphas_enabled=alphas_enabled, hooks=hooks)
        outputs.append(out)
        x_batches.append(x)
        y_batches.append(y)

    # Concatenate all results
    all_logits = torch.cat([o.logits for o in outputs], dim=0)
    all_x = torch.cat(x_batches, dim=0)
    all_y = torch.cat(y_batches, dim=0)

    # Average loss if computed
    all_loss: torch.Tensor | None = None
    if outputs[0].loss is not None:
        all_loss = torch.stack([o.loss for o in outputs]).mean()  # type: ignore

    # Concatenate caches
    all_cache = AlphaCache.concat([o.cache for o in outputs], dim=0)
    return ModelOut(logits=all_logits, loss=all_loss, cache=all_cache), all_x, all_y


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


def get_most_common_words(
    n_words: int = 100,
    n_samples: int = 10_000,
    dataset_name: str = "HuggingFaceFW/fineweb",
    dataset_config: str = "sample-10BT",
    min_word_length: int = 2,
    seed: int = 42,
) -> list[str]:
    """Extract the most common words from the dataset.

    Args:
        n_words: Number of most common words to return
        n_samples: Number of dataset examples to process
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration/subset
        min_word_length: Minimum word length to include
        seed: Random seed for shuffling

    Returns:
        List of the n most common words (without leading space)
    """
    print(f"Loading {dataset_name} ({dataset_config}) to extract {n_words} most common words...")

    dataset = cast(IterableDataset, load_dataset(dataset_name, dataset_config, split="train", streaming=True))
    dataset = dataset.shuffle(seed=seed, buffer_size=1000)  # type: ignore
    dataset = dataset.take(n_samples)  # type: ignore

    word_counts: Counter[str] = Counter()

    for example in track(dataset, description="Counting words", total=n_samples):
        text = example["text"]
        # Extract words (alphanumeric sequences)
        words = re.findall(r"\b[a-zA-Z]+\b", text)
        # Filter by minimum length
        words = [w for w in words if len(w) >= min_word_length]
        word_counts.update(words)

    most_common = [word for word, _ in word_counts.most_common(n_words)]
    print(f"Extracted {len(most_common)} most common words from {n_samples} examples")
    print(f"Top 10: {most_common[:10]}")

    return most_common
