"""Variance/R² analysis showing components needed to explain variance."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

from tiny_model.char_tokenizer import CharTokenizer
from tiny_model.model import GPT, CacheKey, Out
from tiny_model.utils import REPO_ROOT

from analysis.char_classes import CharClass, create_char_classes
from analysis.utils import get_batch, load_model, to_numpy


def compute_pls_r2_curve(X: np.ndarray, y: np.ndarray, max_components: int) -> np.ndarray:
    """Compute PLS R² for varying numbers of components.

    Efficiently computes R² by fitting PLS once with max_components,
    then reconstructing predictions for each k ≤ max_components.

    Args:
        X: feature matrix of shape (n_samples, n_features)
        y: target vector of shape (n_samples,)
        max_components: maximum number of components to compute

    Returns:
        r2_scores: array of R² values for 1 to max_components
    """
    max_pls = min(max_components, X.shape[1])
    pls = PLSRegression(n_components=max_pls)
    pls.fit(X, y)

    r2_scores = []
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    X_centered = X - pls._x_mean

    for n_comp in range(1, max_pls + 1):
        # Reconstruct predictions using only first n_comp components
        # coef = (y_std / x_std) * (y_loadings @ x_rotations.T)
        coef_n = (pls._y_std / pls._x_std) * (pls.y_loadings_[:, :n_comp] @ pls.x_rotations_[:, :n_comp].T)
        y_pred = (X_centered @ coef_n.T).flatten() + pls._y_mean

        # Compute R² score
        ss_res = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        r2_scores.append(r2)

    return np.array(r2_scores)


def plot_variance_explained(
    output: Out,
    cache_key: CacheKey,
    char_classes: dict[str, CharClass],
    max_components: int = 25,
    save_path: str | None = None,
    figsize: tuple[int, int] = (6, 4),
    metric: str = "prob",
    show_pca: bool = True,
) -> dict[str, np.ndarray]:
    """Plot cumulative variance/R² explained by number of components.

    Args:
        output: model output with cache
        cache_key: which activation to analyze
        char_classes: dict of character classes to analyze
        max_components: maximum number of components to compute
        save_path: path to save plot (None to skip)
        figsize: figure size
        metric: which metric to use as target variable ('prob' or 'logit_diff')
        show_pca: whether to show PCA baselines (both centered and standardized)

    Returns:
        results: dict with per-class PLS R² curves
    """
    activations = to_numpy(output.cache.get_value(cache_key))
    activations_flat = activations.reshape(-1, activations.shape[-1])

    results = {}

    # Compute PLS R² for each character class
    for class_name, char_class in char_classes.items():
        value_fn = char_class.get_probabilities if metric == "prob" else char_class.get_logit_diff
        predictions = to_numpy(value_fn(output.logits))
        predictions_flat = predictions.reshape(-1)
        results[class_name] = compute_pls_r2_curve(activations_flat, predictions_flat, max_components)

    # Create plot
    plt.figure(figsize=figsize)

    # Plot PCA lines
    if show_pca:
        n_components_pca = min(max_components, activations_flat.shape[0], activations_flat.shape[1])

        # Regular PCA (centered only)
        pca = PCA(n_components=n_components_pca)
        pca.fit(activations_flat)
        pca_cumulative = np.cumsum(pca.explained_variance_ratio_)

        plt.plot(
            range(1, len(pca_cumulative) + 1),
            pca_cumulative * 100,
            linewidth=2,
            label="PCA (centered)",
            color="black",
        )

        # Standardized PCA (centered + scaled)
        activations_std = (activations_flat - activations_flat.mean(axis=0)) / (activations_flat.std(axis=0) + 1e-8)
        pca_std = PCA(n_components=n_components_pca)
        pca_std.fit(activations_std)
        pca_std_cumulative = np.cumsum(pca_std.explained_variance_ratio_)

        plt.plot(
            range(1, len(pca_std_cumulative) + 1),
            pca_std_cumulative * 100,
            linewidth=2,
            label="PCA (standardized)",
            color="black",
            linestyle="--",
            alpha=0.7,
        )

    # Plot PLS lines for each class
    colors = ["red", "green", "orange", "purple", "brown"]
    metric_label = "prob" if metric == "prob" else "logit_diff"
    for (class_name, r2_scores), color in zip([(k, v) for k, v in results.items() if k != "pca"], colors):
        plt.plot(
            range(1, len(r2_scores) + 1),
            r2_scores * 100,
            linewidth=2,
            label=f"PLS R² ({class_name}, {metric_label})",
            color=color,
            alpha=0.8,
        )

    # Add reference lines

    plt.xlabel("Number of Components")
    plt.ylabel("Variance / R² Explained (%)")
    title_metric = "Probability" if metric == "prob" else "Logit Diff"
    plt.title(f"Variance Analysis ({title_metric}): {cache_key.key} @ layer {cache_key.layer}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.2)
    plt.xlim(0, max_components)
    plt.ylim(None, 100)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved variance plot to: {save_path}")

    return results


def plot_residual_stream_r2(
    output: Out,
    cache_keys: list[CacheKey],
    char_class: CharClass,
    max_components: int = 25,
    save_path: str | None = None,
    figsize: tuple[int, int] = (8, 6),
    cmap: str = "viridis",
    metric: str = "prob",
) -> dict[str, np.ndarray]:
    """Plot PLS R² across residual stream positions.

    Shows how the number of components needed changes as we progress
    through the residual stream (resid_pre, resid_mid, resid_final).

    Args:
        output: model output with cache
        cache_keys: list of cache keys to analyze (in order)
        char_class: character class to predict
        class_name: name of the character class for plot title
        max_components: maximum number of components to compute
        save_path: path to save plot (None to skip)
        figsize: figure size
        cmap: matplotlib colormap name for sequential coloring
        metric: which metric to use as target variable ('prob' or 'logit_diff')

    Returns:
        results: dict mapping cache key string to R² curve
    """
    results = {}
    colormap = plt.get_cmap(cmap)
    n_keys = len(cache_keys)

    # Get predictions once (same for all layers)
    if metric == "prob":
        predictions = to_numpy(char_class.get_probabilities(output.logits))
    elif metric == "logit_diff":
        predictions = to_numpy(char_class.get_logit_diff(output.logits))
    else:
        raise ValueError(f"Unknown metric: {metric}. Must be 'prob' or 'logit_diff'")
    predictions_flat = predictions.reshape(-1)

    # Compute R² for each cache key
    for cache_key in cache_keys:
        activations = to_numpy(output.cache.get_value(cache_key))
        activations_flat = activations.reshape(-1, activations.shape[-1])
        r2_scores = compute_pls_r2_curve(activations_flat, predictions_flat, max_components)
        key_str = f"{cache_key.key}@L{cache_key.layer}" if cache_key.layer is not None else cache_key.key
        results[key_str] = r2_scores

    # Create plot
    plt.figure(figsize=figsize)

    for i, (key_str, r2_scores) in enumerate(results.items()):
        color = colormap(i / max(1, n_keys - 1))  # Spread colors across colormap
        plt.plot(
            range(1, len(r2_scores) + 1),
            r2_scores * 100,
            linewidth=2,
            label=key_str,
            color=color,
            alpha=0.9,
        )

    plt.xlabel("Number of Components")
    plt.ylabel("R² Explained (%)")
    title_metric = "Probability" if metric == "prob" else "Logit Diff"
    plt.title(f"PLS R² Across Residual Stream ({title_metric}): {char_class.name}")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.2)
    plt.xlim(0, max_components)
    plt.ylim(None, 100)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved residual stream plot to: {save_path}")

    return results


def compare_variance_with_filtering(
    data: np.memmap,
    model: GPT,
    tokenizer: CharTokenizer,
    cache_key: CacheKey,
    char_classes: dict[str, CharClass],
    batch_size: int = 64,
    max_components: int = 25,
    exclude_char: str = "|",
    metric: str = "prob",
    save_path: str | None = None,
    figsize: tuple[int, int] = (12, 5),
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compare variance analysis on filtered vs unfiltered data side-by-side.

    Args:
        data: memory-mapped data array
        model: GPT model
        tokenizer: tokenizer for decoding
        cache_key: which activation to analyze
        char_classes: dict of character classes to analyze
        batch_size: number of sequences per batch
        max_components: maximum number of components to compute
        exclude_char: character to exclude in filtered version
        metric: which metric to use ('prob' or 'logit_diff')
        save_path: path to save plot (None to skip)
        figsize: figure size for combined plot

    Returns:
        (results_unfiltered, results_filtered): tuple of results dicts
    """
    # Get unfiltered batch
    x_batch, y_batch = get_batch(data, batch_size, model.config.block_size)
    output = model(x_batch, targets=y_batch, enable_cache=True)

    # Get filtered batch
    x_batch_filtered, y_batch_filtered = get_batch(
        data, batch_size, model.config.block_size, tokenizer=tokenizer, exclude_char=exclude_char
    )
    output_filtered = model(x_batch_filtered, targets=y_batch_filtered, enable_cache=True)

    # Compute results for both
    activations = to_numpy(output.cache.get_value(cache_key))
    activations_flat = activations.reshape(-1, activations.shape[-1])
    activations_filt = to_numpy(output_filtered.cache.get_value(cache_key))
    activations_filt_flat = activations_filt.reshape(-1, activations_filt.shape[-1])

    results_unfiltered = {}
    results_filtered = {}

    for class_name, char_class in char_classes.items():
        value_fn = char_class.get_probabilities if metric == "prob" else char_class.get_logit_diff

        # Unfiltered
        predictions = to_numpy(value_fn(output.logits))
        predictions_flat = predictions.reshape(-1)
        results_unfiltered[class_name] = compute_pls_r2_curve(activations_flat, predictions_flat, max_components)

        # Filtered
        predictions_filt = to_numpy(value_fn(output_filtered.logits))
        predictions_filt_flat = predictions_filt.reshape(-1)
        results_filtered[class_name] = compute_pls_r2_curve(
            activations_filt_flat, predictions_filt_flat, max_components
        )

    # Create side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    colors = ["red", "green", "orange", "purple", "brown"]

    # Plot unfiltered
    for (class_name, r2_scores), color in zip(results_unfiltered.items(), colors, strict=False):
        ax1.plot(
            range(1, len(r2_scores) + 1),
            r2_scores * 100,
            linewidth=2,
            label=f"{class_name}",
            color=color,
            alpha=0.8,
        )

    ax1.set_xlabel("Number of Components")
    ax1.set_ylabel("R² Explained (%)")
    title_metric = "Probability" if metric == "prob" else "Logit Diff"
    ax1.set_title(f"All Data ({title_metric})")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(0, max_components)
    ax1.set_ylim(None, 100)

    # Plot filtered
    for (class_name, r2_scores), color in zip(results_filtered.items(), colors, strict=False):
        ax2.plot(
            range(1, len(r2_scores) + 1),
            r2_scores * 100,
            linewidth=2,
            label=f"{class_name}",
            color=color,
            alpha=0.8,
        )

    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("R² Explained (%)")
    ax2.set_title(f"Filtered (no '{exclude_char}')")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, alpha=0.2)
    ax2.set_xlim(0, max_components)
    ax2.set_ylim(None, 100)

    fig.suptitle(f"Variance Analysis: {cache_key.key} @ layer {cache_key.layer}", fontsize=12, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved comparison plot to: {save_path}")

    return results_unfiltered, results_filtered


def compare_residual_stream_with_filtering(
    data: np.memmap,
    model: GPT,
    tokenizer: CharTokenizer,
    cache_keys: list[CacheKey],
    char_class: CharClass,
    batch_size: int = 64,
    max_components: int = 25,
    exclude_char: str = "|",
    metric: str = "prob",
    save_path: str | None = None,
    figsize: tuple[int, int] = (14, 5),
    cmap: str = "viridis",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compare residual stream R² on filtered vs unfiltered data side-by-side.

    Args:
        data: memory-mapped data array
        model: GPT model
        tokenizer: tokenizer for decoding
        cache_keys: list of cache keys to analyze
        char_class: character class to predict
        batch_size: number of sequences per batch
        max_components: maximum number of components to compute
        exclude_char: character to exclude in filtered version
        metric: which metric to use ('prob' or 'logit_diff')
        save_path: path to save plot (None to skip)
        figsize: figure size for combined plot
        cmap: matplotlib colormap name

    Returns:
        (results_unfiltered, results_filtered): tuple of results dicts
    """
    # Get unfiltered batch
    x_batch, y_batch = get_batch(data, batch_size, model.config.block_size)
    output = model(x_batch, targets=y_batch, enable_cache=True)

    # Get filtered batch
    x_batch_filtered, y_batch_filtered = get_batch(
        data, batch_size, model.config.block_size, tokenizer=tokenizer, exclude_char=exclude_char
    )
    output_filtered = model(x_batch_filtered, targets=y_batch_filtered, enable_cache=True)

    # Get predictions
    value_fn = char_class.get_probabilities if metric == "prob" else char_class.get_logit_diff
    predictions = to_numpy(value_fn(output.logits))
    predictions_flat = predictions.reshape(-1)
    predictions_filt = to_numpy(value_fn(output_filtered.logits))
    predictions_filt_flat = predictions_filt.reshape(-1)

    # Compute R² for each cache key
    results_unfiltered = {}
    results_filtered = {}
    colormap = plt.get_cmap(cmap)
    n_keys = len(cache_keys)

    for cache_key in cache_keys:
        key_str = f"{cache_key.key}@L{cache_key.layer}" if cache_key.layer is not None else cache_key.key

        # Unfiltered
        activations = to_numpy(output.cache.get_value(cache_key))
        activations_flat = activations.reshape(-1, activations.shape[-1])
        results_unfiltered[key_str] = compute_pls_r2_curve(activations_flat, predictions_flat, max_components)

        # Filtered
        activations_filt = to_numpy(output_filtered.cache.get_value(cache_key))
        activations_filt_flat = activations_filt.reshape(-1, activations_filt.shape[-1])
        results_filtered[key_str] = compute_pls_r2_curve(activations_filt_flat, predictions_filt_flat, max_components)

    # Create side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # Plot unfiltered
    for i, (key_str, r2_scores) in enumerate(results_unfiltered.items()):
        color = colormap(i / max(1, n_keys - 1))
        ax1.plot(
            range(1, len(r2_scores) + 1),
            r2_scores * 100,
            linewidth=2,
            label=key_str,
            color=color,
            alpha=0.9,
        )

    ax1.set_xlabel("Number of Components")
    ax1.set_ylabel("R² Explained (%)")
    title_metric = "Probability" if metric == "prob" else "Logit Diff"
    ax1.set_title(f"All Data ({title_metric})")
    ax1.legend(loc="lower right", fontsize=6)
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(0, max_components)
    ax1.set_ylim(None, 100)

    # Plot filtered
    for i, (key_str, r2_scores) in enumerate(results_filtered.items()):
        color = colormap(i / max(1, n_keys - 1))
        ax2.plot(
            range(1, len(r2_scores) + 1),
            r2_scores * 100,
            linewidth=2,
            label=key_str,
            color=color,
            alpha=0.9,
        )

    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("R² Explained (%)")
    ax2.set_title(f"Filtered (no '{exclude_char}')")
    ax2.legend(loc="lower right", fontsize=6)
    ax2.grid(True, alpha=0.2)
    ax2.set_xlim(0, max_components)
    ax2.set_ylim(None, 100)

    fig.suptitle(f"Residual Stream R²: {char_class.name}", fontsize=12, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved comparison plot to: {save_path}")

    return results_unfiltered, results_filtered


# %%
# Load model
model_path = REPO_ROOT / "out-wiki-char/ckpt.pt"
model = load_model(str(model_path))
model.eval()
print(f"✓ Loaded model from {model_path}")

# Load tokenizer and data
tokenizer = CharTokenizer(vocab_path="data/wiki_char/vocab.json")
data_dir = REPO_ROOT / "data/wiki_char"
val_data = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
print(f"✓ Loaded validation dataset: {len(val_data):,} tokens")

# Create character classes
CHAR_CLASSES = create_char_classes(tokenizer)

# %%
# Get batch and run model
x_batch, y_batch = get_batch(
    val_data, batch_size=200, block_size=model.config.block_size, tokenizer=tokenizer, exclude_char=None
)
output = model(x_batch, targets=y_batch, enable_cache=True)
print(f"\n✓ Model output shape: {output.logits.shape}")


# %%
# Variance analysis
key = CacheKey("resid_pre", 2)
_ = plot_variance_explained(
    output,
    key,
    CHAR_CLASSES,
    max_components=25,
    save_path="variance_analysis.png",
    metric="logit_diff",
    show_pca=True,
)

# %%
# Residual stream analysis
# Build list of cache keys in order through residual stream
cache_keys = []
for layer in range(model.config.n_layer):
    cache_keys.append(CacheKey("resid_pre", layer))
    cache_keys.append(CacheKey("resid_mid", layer))
cache_keys.append(CacheKey("resid_final", None))  # Final residual

# Plot for a single character class
residual_results = plot_residual_stream_r2(
    output,
    cache_keys,
    CHAR_CLASSES["whitespace"],
    max_components=20,
    cmap="copper",
    metric="logit_diff",
)

# Print summary
print("\n✓ Residual Stream Analysis Summary:")
for key_str, r2_scores in residual_results.items():
    r2_90 = np.searchsorted(r2_scores, 0.9) + 1
    r2_max = r2_scores[-1]
    print(f"  {key_str}: {r2_90} components for 90% R², max R²={r2_max:.3f}")

# %%
# Comparison: Variance analysis with vs without filtering
key = CacheKey("resid_pre", 3)
unfiltered_results, filtered_results = compare_variance_with_filtering(
    val_data,
    model,
    tokenizer,
    key,
    CHAR_CLASSES,
    batch_size=200,
    max_components=25,
    exclude_char="|",
    metric="logit_diff",
    save_path="variance_comparison.png",
)

# Print comparison
print("\n✓ Variance Comparison Summary:")
for class_name in CHAR_CLASSES:
    r2_unfilt = unfiltered_results[class_name][-1]
    r2_filt = filtered_results[class_name][-1]
    print(f"  {class_name}: unfiltered max R²={r2_unfilt:.3f}, filtered max R²={r2_filt:.3f}")

# %%
# Comparison: Residual stream analysis with vs without filtering
cache_keys = []
for layer in range(model.config.n_layer):
    cache_keys.append(CacheKey("resid_pre", layer))
    cache_keys.append(CacheKey("resid_mid", layer))
cache_keys.append(CacheKey("resid_final", None))

unfiltered_resid, filtered_resid = compare_residual_stream_with_filtering(
    val_data,
    model,
    tokenizer,
    cache_keys,
    CHAR_CLASSES["whitespace"],
    batch_size=200,
    max_components=20,
    exclude_char="|",
    metric="prob",
    save_path="residual_stream_comparison.png",
    cmap="copper",
)

# %%
