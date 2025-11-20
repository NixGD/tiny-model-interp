"""Find task-relevant subspaces using LRP attributions and PCA.

Simplified version focusing on:
1. Computing LRP attribution vectors
2. Applying PCA transformations (centered)
3. Scatter plots colored by prediction value
4. Variance explained curves

TODO: Try ICA later - it might find sparser task-relevant subspaces
"""

from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA, FastICA

from analysis.char_classes import CharClass, create_char_classes
from analysis.utils import (
    compute_pca_r2_curve,
    compute_pls_r2_curve,
    flatten_keep_last,
    get_batch,
    load_model,
    save_fig,
    to_numpy,
)
from tiny_model.model import GPT, CacheKey
from tiny_model.tokenizer.char_tokenizer import CharTokenizer
from tiny_model.utils import REPO_ROOT

OUTPUT_DIR = REPO_ROOT / "analysis/out/lrp_subspace"


def compute_lrp_attributions(
    model: GPT,
    x_batch: torch.Tensor,
    cache_key: CacheKey,
    char_class: CharClass,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute LRP attribution vectors for a character class prediction.

    Args:
        model: GPT model with LXT enabled
        x_batch: input tokens of shape (batch_size, seq_len)
        cache_key: which layer/activation to extract attributions from
        char_class: character class to compute attributions for

    Returns:
        attributions: array of shape (batch_size * seq_len, hidden_dim)
        predictions: array of shape (batch_size * seq_len,)
    """
    model.set_lxt_enabled(True)
    model.eval()
    model.zero_grad()

    # Forward pass with cache
    output = model(x_batch, cache_enabled=True, alphas_enabled=True)
    cache = output.cache

    # Compute logit difference as loss
    loss_value = char_class.get_logit_diff(output.logits)

    # Backward pass
    loss = loss_value.mean()
    loss.backward()

    # Extract attributions (gradients)
    grad = cache.get_grad(cache_key)
    assert grad is not None
    acts = cache.get_value(cache_key)
    attributions = grad * acts

    attributions_flat = flatten_keep_last(to_numpy(attributions))
    predictions_flat = to_numpy(loss_value).flatten()
    return attributions_flat, predictions_flat


def collect_attribution_dataset(
    model: GPT,
    cache_key: CacheKey,
    char_class: CharClass,
    n_samples: int = 10000,
    batch_size: int = 32,
    block_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect attribution vectors across many examples.

    Args:
        model: GPT model with LXT enabled
        data: memory-mapped dataset
        cache_key: which layer to extract attributions from
        char_class: character class to analyze
        n_samples: total number of samples to collect
        batch_size: sequences per batch
        block_size: sequence length

    Returns:
        all_attributions: array of shape (n_samples, hidden_dim)
        all_predictions: array of shape (n_samples,)
    """
    samples_per_batch = batch_size * block_size
    n_batches = (n_samples + samples_per_batch - 1) // samples_per_batch

    all_attributions = []
    all_predictions = []

    print(f"Collecting {n_samples} attribution samples...")
    for i in range(n_batches):
        x_batch, _ = get_batch(batch_size, block_size)
        attributions, predictions = compute_lrp_attributions(model, x_batch, cache_key, char_class)
        all_attributions.append(attributions)
        all_predictions.append(predictions)

        if (i + 1) % 5 == 0 or i == n_batches - 1:
            print(f"  {i + 1}/{n_batches} batches")

    all_attributions = np.vstack(all_attributions)[:n_samples]
    all_predictions = np.concatenate(all_predictions)[:n_samples]

    print(f"Shape: {all_attributions.shape}")
    print(f"Prediction range: [{all_predictions.min():.3f}, {all_predictions.max():.3f}]")

    return all_attributions, all_predictions


def plot_scatter(
    activations_dict: dict[str, np.ndarray],
    predictions: np.ndarray,
    dims: tuple[int, int] = (0, 1),
    n_plot: int = 3000,
    out_path: Path = OUTPUT_DIR / "scatter.png",
) -> None:
    """Plot scatter of attributions in different bases.

    Args:
        attributions_dict: dict mapping method name to transformed attributions
        predictions: prediction values for coloring
        dims: which two dimensions to plot
        n_plot: number of points to plot
    """
    methods = list(activations_dict.keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4))
    if n_methods == 1:
        axes = [axes]

    # Subsample
    indices = np.random.choice(len(predictions), min(n_plot, len(predictions)), replace=False)
    preds_sub = predictions[indices]

    scatter = None
    for ax, method in zip(axes, methods, strict=False):
        attrs_sub = activations_dict[method][indices]

        scatter = ax.scatter(
            attrs_sub[:, dims[0]],
            attrs_sub[:, dims[1]],
            c=preds_sub,
            cmap="RdYlGn",
            alpha=0.4,
            s=3,
            vmin=np.percentile(preds_sub, 5),
            vmax=np.percentile(preds_sub, 95),
        )

        ax.set_xlabel(f"Component {dims[0]}")
        if ax == axes[0]:
            ax.set_ylabel(f"Component {dims[1]}")
        ax.set_title(method.upper())
        ax.grid(True, alpha=0.3)

    if scatter is not None:
        fig.colorbar(scatter, ax=axes, label="Logit Diff (uppercase)", pad=0.02)
    plt.tight_layout()
    save_fig(out_path)
    plt.close()


def plot_prediction_r2(
    r2_dict: dict[str, list[float]],
    save_path: str | None = None,
) -> None:
    """Plot R² curves for different methods.

    Args:
        r2_dict: dict mapping method name to {n_components: r2} dict
        save_path: path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {"pca": "blue", "pls": "red"}
    labels = {"pca": "PCA (centered)", "pls": "PLS"}

    for method, r2_scores in r2_dict.items():
        ax.plot(
            range(1, len(r2_scores) + 1),
            np.array(r2_scores),
            "o-",
            label=labels.get(method, method),
            color=colors.get(method, "gray"),
        )

    ax.set_xlabel("Number of Components")
    ax.set_ylabel("R² (%)")
    ax.set_title("Prediction R² from LRP Attribution Subspace")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(None, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved R² plot to: {save_path}")

    plt.close()


if __name__ == "__main__":
    model = load_model()
    model.set_lxt_enabled(True)

    tokenizer = CharTokenizer()
    char_class = create_char_classes(tokenizer)["uppercase"]

    torch.manual_seed(0)

    xs, ys = get_batch(batch_size=500, block_size=model.config.block_size)
    cache_key = CacheKey("resid_mid", 2)
    attributions, predictions = compute_lrp_attributions(model, xs, cache_key, char_class)

    print(f"Attributions shape: {attributions.shape}")
    print(f"Predictions shape: {predictions.shape}")

    pca = PCA(n_components=10)
    pca.fit(attributions)
    ica = FastICA(n_components=10, random_state=0, max_iter=200)
    ica.fit(attributions)
    attrs_dict: dict[str, np.ndarray] = {
        "original": attributions,
        "pca": pca.transform(attributions),
        "ica": cast(np.ndarray, ica.transform(attributions)),
    }

    plot_scatter(attrs_dict, predictions, dims=(0, 1), out_path=OUTPUT_DIR / "scatter_attributions.png")

    activations = model(xs, cache_enabled=True).cache.get_value(cache_key)
    activations_np = flatten_keep_last(to_numpy(activations))
    pca_activations = pca.transform(activations_np)
    ica_activations = ica.transform(activations_np)

    plot_scatter(
        {"original": activations_np, "pca": pca_activations, "ica": ica_activations},
        predictions,
        dims=(0, 1),
        out_path=OUTPUT_DIR / "scatter_activations.png",
    )

    with torch.no_grad():
        out = model(xs, targets=ys, cache_enabled=True, alphas_enabled=True)

    model = None
    max_components = 10
    r2_dict = {
        "pls": compute_pls_r2_curve(out, cache_key, char_class, "logit_diff", max_components),
        "pca": compute_pca_r2_curve(out, cache_key, char_class, "logit_diff", max_components),
    }
    plot_prediction_r2(r2_dict, save_path=str(OUTPUT_DIR / "prediction_r2.png"))
