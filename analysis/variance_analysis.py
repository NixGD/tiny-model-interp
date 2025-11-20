"""Variance/R² analysis showing components needed to explain variance."""

from collections.abc import Iterable

import matplotlib.pyplot as plt

from analysis.char_classes import CharClass, create_char_classes
from analysis.utils import (
    Metric,
    compute_pls_r2_curve,
    extract_activations,
    get_batch,
    load_model,
    save_fig,
    variance_curve_pca,
)
from tiny_model.model import GPT, CacheKey, ModelOut
from tiny_model.tokenizer.char_tokenizer import CharTokenizer
from tiny_model.utils import REPO_ROOT

OUTPATH = REPO_ROOT / "analysis/out/variance_analysis/"
OUTPATH.mkdir(parents=True, exist_ok=True)


def plot_variance_explained(
    output: ModelOut,
    cache_key: CacheKey,
    char_classes: Iterable[CharClass],
    metric: Metric = "prob",
    show_pca: bool = True,
    max_components: int = 25,
    figsize: tuple[int, int] = (6, 4),
    ax: plt.Axes | None = None,
) -> None:
    """Plot cumulative variance/R² explained by number of components."""
    activations = extract_activations(output, cache_key)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for char_class in char_classes:
        r2_scores = compute_pls_r2_curve(output, cache_key, char_class, metric, max_components)
        ax.plot(
            range(1, len(r2_scores) + 1),
            r2_scores * 100,
            label=f"PLS R² {char_class.name}",
        )

    if show_pca:
        variance_curve = variance_curve_pca(activations, max_components)
        ax.plot(
            range(1, len(variance_curve) + 1),
            variance_curve * 100,
            label="PCA (centered)",
            color="black",
        )

    title_metric = "Probability" if metric == "prob" else "Logit Diff"
    title = f"Variance Analysis ({title_metric}): {cache_key.key} @ layer {cache_key.layer}"
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Variance / R² Explained (%)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, max_components)
    ax.set_ylim(None, 100)
    plt.tight_layout()
    save_fig(OUTPATH / f"variance_explained_{cache_key.key}@L{cache_key.layer}.png")


def _get_resid_stream_cache_keys(model: GPT) -> list[CacheKey]:
    """Get list of cache keys in order through residual stream."""
    cache_keys = []
    for layer in range(model.config.n_layer):
        cache_keys.append(CacheKey("resid_pre", layer))
        cache_keys.append(CacheKey("resid_mid", layer))
    cache_keys.append(CacheKey("resid_final", None))
    return cache_keys


def plot_residual_stream_r2(
    output: ModelOut,
    char_class: CharClass,
    model: GPT,
    max_components: int = 25,
    figsize: tuple[int, int] = (8, 6),
    cmap: str = "viridis",
    metric: Metric = "prob",
    ax: plt.Axes | None = None,
) -> None:
    """Plot PLS R² across residual stream positions.

    Shows how the number of components needed changes as we progress
    through the residual stream (resid_pre, resid_mid, resid_final).
    """
    colormap = plt.get_cmap(cmap)
    cache_keys = _get_resid_stream_cache_keys(model)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Compute R² for each cache key
    for i, cache_key in enumerate(_get_resid_stream_cache_keys(model)):
        r2_scores = compute_pls_r2_curve(output, cache_key, char_class, metric, max_components)
        color = colormap(i / max(1, len(cache_keys) - 1))
        label = f"{cache_key.key}@L{cache_key.layer}" if cache_key.layer is not None else cache_key.key

        ax.plot(range(1, len(r2_scores) + 1), r2_scores, label=label, color=color)

    title_metric = "Probability" if metric == "prob" else "Logit Diff"
    title = f"PLS R² Across Residual Stream ({title_metric}): {char_class.name}"
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("R² Explained (%)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, max_components)
    ax.set_ylim(None, 1)
    plt.tight_layout()
    save_fig(OUTPATH / f"residual_stream_r2_{char_class.name}.png")


# %%
model = load_model()
tokenizer = CharTokenizer(vocab_path="data/wiki_char/vocab.json")
CHAR_CLASSES = create_char_classes(tokenizer)

xs, ys = get_batch(batch_size=500, block_size=model.config.block_size)
output = model(xs, targets=ys, cache_enabled=True, alphas_enabled=True)
print(f"\n✓ Model output shape: {output.logits.shape}")


for resid_key in _get_resid_stream_cache_keys(model):
    plot_variance_explained(
        output,
        resid_key,
        CHAR_CLASSES.values(),
        max_components=25,
        metric="logit_diff",
        show_pca=True,
    )

# %%
for char_class in CHAR_CLASSES.values():
    plot_residual_stream_r2(
        output,
        char_class,
        model,
        max_components=25,
        metric="logit_diff",
    )
