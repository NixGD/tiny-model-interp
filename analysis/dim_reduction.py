"""Dimensionality reduction visualization of activations."""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from typing import cast
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from umap import UMAP

from analysis.char_classes import CharClass, create_char_classes
from analysis.utils import extract_activations, flatten_keep_last, get_batch, load_model, to_numpy
from tiny_model.model import CacheKey, ModelOut
from tiny_model.tokenizer.char_tokenizer import CharTokenizer
from analysis.utils import LogitLossFn

# %%


def get_embeddings(activations: np.ndarray, predictions: np.ndarray, method: str = "umap") -> np.ndarray:
    reduction_method = {"pca": PCA, "umap": UMAP, "pls": PLSRegression}[method]
    reducer = reduction_method(n_components=2)
    if method == "pls":
        reducer.fit(activations, predictions)
        return cast(np.ndarray, reducer.transform(activations))
    else:
        return cast(np.ndarray, reducer.fit_transform(activations))


def visualize_activations_2d(
    output: ModelOut,
    cache_key: CacheKey,
    loss_fn: LogitLossFn,
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
    activations = extract_activations(output, cache_key)
    predictions = to_numpy(loss_fn(output.logits)).flatten()
    embedding = get_embeddings(activations, predictions, method)
    plt.figure(figsize=figsize)

    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=predictions,
        alpha=0.5,
        s=10,
    )

    plt.xlabel(f"Component {dims[0]}")
    plt.ylabel(f"Component {dims[1]}")
    plt.title(f"{title}\n{method}")
    plt.colorbar(label=f"Loss")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved {method} visualization to: {save_path}")

    return embedding


def visualize_activations_2d_interactive(
    output: ModelOut,
    input_tokens: torch.Tensor,
    cache_key: CacheKey,
    loss_fn: LogitLossFn,
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
    batch_size, seq_len, _ = output.logits.shape
    activations = extract_activations(output, cache_key)
    predictions = to_numpy(loss_fn(output.logits)).flatten()
    embedding = get_embeddings(activations, predictions, method)

    next_token_probs = torch.softmax(output.logits, dim=-1)

    hover_texts = []
    for batch_idx in range(batch_size):
        for pos_idx in range(seq_len):
            start_pos = max(0, pos_idx - context_chars)
            context_tokens = input_tokens[batch_idx, start_pos : pos_idx + 1].tolist()
            if pos_idx == seq_len - 1:
                next_token_text = "[eos]"
            else:
                next_token = cast(int, input_tokens[batch_idx, pos_idx + 1].item())
                next_token_text = tokenizer.decode_one(next_token)
            context_text = tokenizer.decode(context_tokens)

            hover_text = ""
            hover_text += f"Batch: {batch_idx}, Pos: {pos_idx}<br>"
            hover_text += f"{repr(context_text)} -> {repr(next_token_text)}<br>"

            for tok_idx in next_token_probs[batch_idx, pos_idx].argsort(descending=True)[:3]:
                token = tokenizer.decode_one(tok_idx.item())
                prob = next_token_probs[batch_idx, pos_idx, tok_idx].item()
                hover_text += f"<br>{token!r}\t{prob:.1%}"

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
                colorbar=dict(title="Loss"),
                opacity=0.7,
                # cmin=0,
                # cmax=1,
            ),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"{title}<br>{method}",
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

    # %%
    loss_fn = CHAR_CLASSES["whitespace"].get_logit_diff
    key = CacheKey("resid_pre", 2)
    embedding = visualize_activations_2d(
        output,
        key,
        loss_fn,
        method="pca",
        title=f"Residual Stream Activations ({key.key} @ layer {key.layer})",
        save_path="activations_umap.png",
    )

    # %%
    target_token = "A"
    negative_token = "a"
    target_token_id = tokenizer.encode_one(target_token)
    negative_token_id = tokenizer.encode_one(negative_token)
    loss_fn = lambda logits: logits[..., target_token_id] - logits[..., negative_token_id]
    embedding_interactive = visualize_activations_2d_interactive(
        output,
        x_batch,
        key,
        loss_fn,
        tokenizer,
        method="pls",
        title=f"Activations of {key} (logit diff: {target_token} - {negative_token})",
        context_chars=20,
        dims=(0, 1),
    )
