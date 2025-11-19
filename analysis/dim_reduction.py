"""Dimensionality reduction visualization of activations."""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from umap import UMAP

from analysis.char_classes import CharClass, create_char_classes
from analysis.utils import get_batch, load_model, to_numpy
from tiny_model.char_tokenizer import CharTokenizer
from tiny_model.model import CacheKey, Out
from tiny_model.utils import REPO_ROOT


def visualize_activations_2d(
    output: Out,
    cache_key: CacheKey,
    char_class: CharClass,
    method: str = "umap",
    title: str = "Activation Visualization",
    save_path: str | None = None,
    figsize: tuple[int, int] = (8, 8),
    **method_kwargs,
):
    """Visualize high-dimensional activations in 2D with color-coded labels.

    Args:
        output: model output with cache
        cache_key: which activation to visualize
        char_class: character class for coloring
        method: 'pca', 'umap', 'tsne', 'lda', or 'pls'
        title: plot title
        save_path: path to save plot (None to skip)
        figsize: figure size
        **method_kwargs: additional arguments for the reduction method

    Returns:
        embedding: 2D embedding of shape (n_samples, 2)
    """
    activations = to_numpy(output.cache.get_value(cache_key))
    activations = activations.reshape(-1, activations.shape[-1])
    predictions = to_numpy(char_class.get_probabilities(output.logits))
    predictions = predictions.reshape(-1)

    print(f"Activations shape: {activations.shape}")
    print(f"Predictions shape: {predictions.shape}")

    # Apply dimensionality reduction
    if method == "pca":
        reducer = PCA(n_components=2, **method_kwargs)
        embedding = reducer.fit_transform(activations)
        explained_var = reducer.explained_variance_ratio_
        method_label = f"PCA (var: {explained_var[0]:.2%}, {explained_var[1]:.2%})"
    elif method == "umap":
        reducer = UMAP(n_components=2, **method_kwargs)
        embedding = reducer.fit_transform(activations)
        method_label = "UMAP"
    elif method == "tsne":
        reducer = TSNE(n_components=2, **method_kwargs)
        embedding = reducer.fit_transform(activations)
        method_label = "t-SNE"
    elif method == "lda":
        reducer = LDA(n_components=1, **method_kwargs)
        embedding_1d = reducer.fit_transform(activations, predictions > 0.5)
        embedding = np.column_stack([embedding_1d, np.random.randn(len(embedding_1d)) * 0.1])
        method_label = "LDA (supervised)"
    elif method == "pls":
        reducer = PLSRegression(n_components=2, **method_kwargs)
        reducer.fit(activations, predictions)
        embedding = reducer.transform(activations)
        method_label = "PLS (supervised)"
    else:
        raise ValueError(f"Unknown method: {method}")

    # Create visualization
    plt.figure(figsize=figsize)

    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=predictions,
        alpha=0.5,
        s=10,
    )

    if method == "lda":
        xlabel, ylabel = "Discriminant 1", "Random jitter"
    elif method == "pls":
        xlabel, ylabel = "PLS Component 1", "PLS Component 2"
    else:
        xlabel, ylabel = "Component 1", "Component 2"

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title}\n{method_label}")
    plt.colorbar(label=f"Probability of {char_class.name}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved {method} visualization to: {save_path}")

    return embedding


def visualize_activations_2d_interactive(
    output: Out,
    input_tokens: torch.Tensor,
    cache_key: CacheKey,
    char_class: CharClass,
    tokenizer: CharTokenizer,
    method: str = "umap",
    title: str = "Activation Visualization",
    save_path: str | None = None,
    context_chars: int = 20,
    dims: tuple[int, int] = (0, 1),
    **method_kwargs,
):
    """Interactive Plotly visualization with hover text showing decoded inputs.

    Args:
        output: model output with cache
        input_tokens: input tokens of shape (batch_size, seq_len)
        cache_key: which activation to visualize
        char_class: character class for coloring
        tokenizer: for decoding tokens
        method: 'pca', 'umap', 'tsne', 'lda', or 'pls'
        title: plot title
        save_path: path to save HTML file (None to skip)
        context_chars: number of context characters to show in hover
        dims: which two dimensions to plot (for PCA/PLS with >2 components)
        **method_kwargs: additional arguments for the reduction method

    Returns:
        embedding: 2D embedding of shape (n_samples, 2)
    """
    activations = to_numpy(output.cache.get_value(cache_key))
    batch_size, seq_len, hidden_dim = activations.shape
    activations_flat = activations.reshape(-1, hidden_dim)

    predictions = to_numpy(char_class.get_probabilities(output.logits))
    predictions_flat = predictions.reshape(-1)

    n_components = max(2, *dims) + 1

    # Apply dimensionality reduction
    if method == "pca":
        reducer = PCA(n_components=n_components, **method_kwargs)
        embedding = reducer.fit_transform(activations_flat)
        explained_var = reducer.explained_variance_ratio_
        method_label = f"PCA (var: {explained_var[0]:.2%}, {explained_var[1]:.2%})"
    elif method == "umap":
        reducer = UMAP(n_components=n_components, **method_kwargs)
        embedding = reducer.fit_transform(activations_flat)
        method_label = "UMAP"
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, **method_kwargs)
        embedding = reducer.fit_transform(activations_flat)
        method_label = "t-SNE"
    elif method == "lda":
        reducer = LDA(n_components=1, **method_kwargs)
        embedding_1d = reducer.fit_transform(activations_flat, predictions_flat > 0.5)
        embedding = np.column_stack([embedding_1d, np.random.randn(len(embedding_1d)) * 0.1])
        method_label = "LDA (supervised)"
    elif method == "pls":
        reducer = PLSRegression(n_components=n_components, **method_kwargs)
        reducer.fit(activations_flat, predictions_flat)
        embedding = reducer.transform(activations_flat)
        method_label = "PLS (supervised)"
    else:
        raise ValueError(f"Unknown method: {method}")

    # Create hover text with decoded context
    hover_texts = []
    for batch_idx in range(batch_size):
        for pos_idx in range(seq_len):
            start_pos = max(0, pos_idx - context_chars)
            context_tokens = input_tokens[batch_idx, start_pos:pos_idx].tolist()
            context_text = tokenizer.decode(context_tokens)

            prob = predictions_flat[batch_idx * seq_len + pos_idx]
            hover_text = (
                f"Batch: {batch_idx}, Pos: {pos_idx}<br>"
                f"Context: {repr(context_text)}<br>"
                f"{char_class.name} prob: {prob:.3f}"
            )
            hover_texts.append(hover_text)

    # Create interactive scatter plot
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=embedding[:, dims[0]],
            y=embedding[:, dims[1]],
            mode="markers",
            marker=dict(
                size=4,
                color=predictions_flat,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=f"{char_class.name}<br>probability"),
                opacity=0.7,
            ),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"{title}<br>{method_label}",
        xaxis_title=f"Component {dims[0]}" if method != "lda" else f"Discriminant {dims[0]}",
        yaxis_title=f"Component {dims[1]}" if method != "lda" else "Random jitter",
        width=900,
        height=700,
        hovermode="closest",
    )

    if save_path:
        fig.write_html(save_path)
        print(f"✓ Saved interactive {method} visualization to: {save_path}")

    return embedding


if __name__ == "__main__":
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

    # Get batch and run model
    batch_size = 64
    x_batch, y_batch = get_batch(val_data, batch_size, model.config.block_size)
    output = model(x_batch, targets=y_batch, enable_cache=True)
    print(f"\n✓ Model output shape: {output.logits.shape}")

    # Static matplotlib visualization
    key = CacheKey("resid_pre", 2)
    embedding = visualize_activations_2d(
        output,
        key,
        CHAR_CLASSES["whitespace"],
        method="umap",
        title=f"Residual Stream Activations ({key.key} @ layer {key.layer})",
        save_path="activations_umap.png",
    )

    # Interactive Plotly visualization
    embedding_interactive = visualize_activations_2d_interactive(
        output,
        x_batch,
        key,
        CHAR_CLASSES["whitespace"],
        tokenizer,
        method="pls",
        title=f"Activations of {key.key} @ layer {key.layer}",
        save_path="activations_pls_interactive.html",
        context_chars=30,
        dims=(0, 1),
    )
