"""Dimensionality reduction visualization of activations."""

import string
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from jaxtyping import Float
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from torch import Tensor
from umap import UMAP

from analysis.char_classes import LogitLossFn, LogitTensor, create_char_classes
from analysis.utils import extract_activations, load_model, run_batches, to_numpy
from tiny_model.model import CacheKey, ModelOut
from tiny_model.tokenizer.char_tokenizer import CharTokenizer

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
    dims: tuple[int, int] = (0, 1),
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
    plt.colorbar(label="Loss")
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
    mask: torch.Tensor | None = None,
    cmap: str = "Viridis",
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
    if mask is not None:
        predictions = predictions[mask.flatten()]
        activations = activations[mask.flatten()]
    embedding = get_embeddings(activations, predictions, method)

    next_token_probs = torch.softmax(output.logits, dim=-1)

    hover_texts = []
    if mask is None:
        mask = torch.ones(batch_size, seq_len).bool()

    assert mask is not None
    for batch_idx, pos_idx in mask.nonzero():
        batch_idx = cast(int, batch_idx.item())
        pos_idx = cast(int, pos_idx.item())
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
            token = tokenizer.decode_one(cast(int, tok_idx.item()))
            prob = next_token_probs[batch_idx, pos_idx, tok_idx].item()
            hover_text += f"<br>{token!r}\t{prob:.1%}"

        hover_texts.append(hover_text)

    # Create interactive scatter plot
    fig = go.Figure()

    vabs = np.abs(predictions).max()

    fig.add_trace(
        go.Scatter(
            x=embedding[:, dims[0]],
            y=embedding[:, dims[1]],
            mode="markers",
            marker=dict(
                size=4,
                color=predictions,
                colorscale=cmap,
                showscale=True,
                colorbar=dict(title="Loss"),
                opacity=0.7,
                cmin=-vabs,
                cmax=vabs,
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
        width=600,
        height=600,
        hovermode="closest",
    )

    fig.show()


# %%
def get_logit_diff_loss(positive_tokens: list[str], negative_tokens: list[str]) -> LogitLossFn:
    positive_token_ids = [tokenizer.encode_one(token) for token in positive_tokens]
    negative_token_ids = [tokenizer.encode_one(token) for token in negative_tokens]

    def loss_fn(logits: LogitTensor) -> Float[Tensor, "..."]:
        positive_logits = logits[..., positive_token_ids].mean(dim=-1)
        negative_logits = logits[..., negative_token_ids].mean(dim=-1)
        return positive_logits - negative_logits

    return loss_fn


def get_mask(y: torch.Tensor, allowed_tokens: list[str]) -> torch.Tensor:
    allowed_token_ids = [tokenizer.encode_one(token) for token in allowed_tokens]
    return torch.isin(y, torch.tensor(allowed_token_ids))


if __name__ == "__main__":
    model = load_model()
    tokenizer = CharTokenizer()

    CHAR_CLASSES = create_char_classes(tokenizer)
    key = CacheKey("resid_mid", 3)

    with torch.no_grad():
        output, x, y = run_batches(
            model,
            num_batches=2,
            batch_size=200,
            cache_enabled=[key],
        )

    # %%

    loss_fn = get_logit_diff_loss(list(string.ascii_uppercase), list(string.ascii_lowercase))
    # mask = get_mask(y, list(string.ascii_uppercase) + list(string.ascii_lowercase))
    mask = get_mask(x, ["t"])  # last token is t

    key = CacheKey("resid_mid", 3)
    embedding_interactive = visualize_activations_2d_interactive(
        output,
        x,
        key,
        loss_fn,
        tokenizer,
        method="pls",
        title=f"Activations of {key} (logit diff: uppercase - lowercase)",
        context_chars=20,
        dims=(0, 1),
        mask=mask,
        cmap="RdYlGn",
    )

# %%
