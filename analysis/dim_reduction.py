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
from analysis.utils import flatten_keep_last, get_batch, load_model, to_numpy
from tiny_model.model import CacheKey, ModelOut
from tiny_model.tokenizer.char_tokenizer import CharTokenizer

# %%


def visualize_activations_2d(
    output: ModelOut,
    cache_key: CacheKey,
    char_class: CharClass,
    method: str = "umap",
    title: str = "Activation Visualization",
    save_path: str | None = None,
    figsize: tuple[int, int] = (8, 8),
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
        reducer = PCA(n_components=2)
        embedding = reducer.fit_transform(activations)
        explained_var = reducer.explained_variance_ratio_
        method_label = f"PCA (var: {explained_var[0]:.2%}, {explained_var[1]:.2%})"
    elif method == "umap":
        reducer = UMAP(n_components=2)
        embedding = reducer.fit_transform(activations)
        method_label = "UMAP"
    elif method == "tsne":
        reducer = TSNE(n_components=2)
        embedding = reducer.fit_transform(activations)
        method_label = "t-SNE"
    elif method == "pls":
        reducer = PLSRegression(n_components=2)
        reducer.fit(activations, predictions)
        embedding = reducer.transform(activations)
        method_label = "PLS (supervised)"
    else:
        raise ValueError(f"Unknown method: {method}")

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
    output: ModelOut,
    input_tokens: torch.Tensor,
    cache_key: CacheKey,
    char_class: CharClass,
    tokenizer: CharTokenizer,
    method: str = "umap",
    title: str = "Activation Visualization",
    context_chars: int = 20,
    dims: tuple[int, int] = (0, 1),
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
    # activations = flatten_keep_last(to_numpy(output.cache.get_value(cache_key)))
    batch_size, seq_len, _ = output.logits.shape
    activations = flatten_keep_last(to_numpy(output.logits))
    predictions = to_numpy(char_class.get_probabilities(output.logits)).flatten()

    n_components = max(2, *dims) + 1

    # Apply dimensionality reduction
    if method == "pca":
        reducer = PCA(n_components=n_components)
        embedding = reducer.fit_transform(activations)
        explained_var = reducer.explained_variance_ratio_
        method_label = f"PCA (var: {explained_var[0]:.2%}, {explained_var[1]:.2%})"
    elif method == "umap":
        reducer = UMAP(n_components=n_components)
        embedding = reducer.fit_transform(activations)
        method_label = "UMAP"
    elif method == "tsne":
        reducer = TSNE(n_components=n_components)
        embedding = reducer.fit_transform(activations)
        method_label = "t-SNE"
    elif method == "pls":
        reducer = PLSRegression(n_components=n_components)
        reducer.fit(activations, predictions)
        embedding = reducer.transform(activations)
        method_label = "PLS (supervised)"
    else:
        raise ValueError(f"Unknown method: {method}")

    # Create hover text with decoded context

    next_token_probs = torch.softmax(output.logits, dim=-1)
    hover_texts = []
    for batch_idx in range(batch_size):
        for pos_idx in range(seq_len - 1):
            start_pos = max(0, pos_idx - context_chars)
            context_tokens = input_tokens[batch_idx, start_pos : pos_idx + 1].tolist()
            next_token = input_tokens[batch_idx, pos_idx + 1].item()
            next_token_text = tokenizer.decode_one(next_token)
            context_text = tokenizer.decode(context_tokens)

            prob = predictions[batch_idx * seq_len + pos_idx]
            hover_text = (
                f"Batch: {batch_idx}, Pos: {pos_idx}<br>"
                f"{repr(context_text)} -> {repr(next_token_text)}<br>"
                f"{char_class.name} prob: {prob:.3f}"
            )

            top3 = torch.topk(next_token_probs[batch_idx, pos_idx], k=3)
            for tok_id, prob in zip(top3.indices, top3.values, strict=True):
                hover_text += f"<br>{tokenizer.decode_one(tok_id.item())!r}\t{prob:.1%}"
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
                color=predictions,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=f"{char_class.name}<br>probability"),
                opacity=0.7,
                cmin=0,
                cmax=1,
            ),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"{title}<br>{method_label}",
        xaxis_title=f"Component {dims[0]}",
        yaxis_title=f"Component {dims[1]}",
        width=900,
        height=700,
        hovermode="closest",
    )

    fig.show()


# %%

if __name__ == "__main__":
    model = load_model()
    tokenizer = CharTokenizer()

    CHAR_CLASSES = create_char_classes(tokenizer)

    batch_size = 200
    x_batch, y_batch = get_batch(batch_size=batch_size)
    output = model(x_batch, targets=y_batch, cache_enabled=True)
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
        title=f"Activations of {key}",
        context_chars=20,
        dims=(0, 1),
    )
