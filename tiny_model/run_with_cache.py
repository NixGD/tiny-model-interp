"""Load a trained GPT model from checkpoint and enable cache for analysis."""

import string
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from umap import UMAP

from tiny_model.char_tokenizer import CharTokenizer
from tiny_model.model import GPT, CacheKey, GPTConfig, Out
from tiny_model.utils import REPO_ROOT

# %%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_path: str | Path) -> GPT:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model_args = checkpoint["model_args"]
    model = GPT(GPTConfig(**model_args))
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    return model


model_path = REPO_ROOT / "out-wiki-char/ckpt.pt"
model = load_model(str(model_path))
model.eval()

# model.set_lxt_enabled(True)


# %%
tokenizer = CharTokenizer(vocab_path="data/wiki_char/vocab.json")


# %%
# Load validation dataset
data_dir = REPO_ROOT / "data/wiki_char"
val_data = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
print(f"✓ Loaded validation dataset: {len(val_data):,} tokens")


# %%
def get_batch(data: np.memmap, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Get a batch of data for training/evaluation.

    Returns:
        x: input tokens of shape (batch_size, block_size)
        y: target tokens of shape (batch_size, block_size)
    """
    block_size = model.config.block_size
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


# %%
# Get one batch from validation set

x_batch, y_batch = get_batch(val_data, batch_size=8)

print(f"Batch input shape: {x_batch.shape}")
print(f"First sequence: {repr(tokenizer.decode(x_batch[0].tolist()))}")

# %%
output = model(x_batch, targets=y_batch, enable_cache=True)
assert output.loss is not None
print(f"Loss: {output.loss.item():.4f}")
cache = output.cache

print("\nCached activations:")
for key in cache.keys():  # noqa: SIM118
    if key.layer == 0 or key.layer is None:
        print(f"  {key.key} @ {key.layer}:\t {list(cache.get_value(key).shape)}")


# %%
# Character classification utilities for eventual visualization
class CharClass:
    """A class representing a category of characters (e.g., numerals, punctuation)."""

    def __init__(self, name: str, chars: str, tokenizer: CharTokenizer):
        """Initialize a character class.

        Args:
            name: Name of this character class
            chars: String containing all characters in this class
            tokenizer: Tokenizer to use for getting indices
        """
        self.name = name
        self.chars = chars
        self.indices = self._get_indices(tokenizer)

    def _get_indices(self, tokenizer: CharTokenizer) -> list[int]:
        """Get token indices for characters in this class."""
        return [tokenizer.stoi[c] for c in self.chars if c in tokenizer.stoi]

    def get_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute probability mass on this character class.

        Args:
            logits: model logits of shape (..., vocab_size)
                    Supports vectorization over batch and sequence dimensions

        Returns:
            probabilities: sum of probabilities for this class, shape (...)
                          Preserves all dimensions except the vocab dimension
        """
        probs = torch.softmax(logits, dim=-1)
        char_mask = torch.zeros(probs.shape[-1], device=logits.device)
        char_mask[self.indices] = 1.0
        return (probs * char_mask).sum(dim=-1)

    def __repr__(self) -> str:
        return f"CharClass({self.name!r}, {len(self.indices)} chars)"


# Define character classes
CHAR_CLASSES = {
    "numeral": CharClass("numeral", string.digits, tokenizer),
    "eos_punct": CharClass("eos_punct", ".!?", tokenizer),
    "whitespace": CharClass("whitespace", " \t\n", tokenizer),
    "uppercase": CharClass("uppercase", string.ascii_uppercase, tokenizer),
    "lowercase": CharClass("lowercase", string.ascii_lowercase, tokenizer),
}


# %%
# Compute probabilities for each character class
print("\nProbabilities for character classes on first sequence, last position:")
for char_class in CHAR_CLASSES.values():
    prob = char_class.get_probabilities(output.logits[0, -1])
    print(f"  {char_class.name}: {prob.item():.4f}")


# %%
def visualize_char_class_probs(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    char_classes: Iterable[CharClass],
    tokenizer: CharTokenizer,
    num_tokens: int = 25,
    batch_index: int = 0,
    save_path: str | None = "char_class_probs.png",
    figsize: tuple[int, int] = (15, 3),
) -> np.ndarray:
    """Visualize character class probabilities across tokens as a heatmap.

    Args:
        logits: model output logits of shape (batch_size, seq_len, vocab_size)
        tokens: input tokens of shape (batch_size, seq_len)
        char_classes: list of CharClass instances to visualize
        tokenizer: tokenizer for decoding tokens
        num_tokens: number of tokens to visualize
        batch_index: which sequence in the batch to visualize
        save_path: path to save the plot (None to skip saving)
        figsize: figure size as (width, height)

    Returns:
        class_probs_matrix: numpy array of shape (num_classes, num_tokens)
    """
    # Compute probabilities for all classes (vectorized over sequence dimension)
    class_probs_matrix = []
    for char_class in char_classes:
        probs = char_class.get_probabilities(logits[batch_index, :num_tokens])
        class_probs_matrix.append(probs.detach().cpu().numpy())

    class_probs_matrix = np.array(class_probs_matrix)  # shape: (num_classes, num_tokens)
    class_names = [char_class.name for char_class in char_classes]

    # Get the actual characters for labeling
    token_list = tokens[batch_index, :num_tokens].tolist()
    chars = [tokenizer.decode_one(t) for t in token_list]

    # Create visualization
    plt.figure(figsize=figsize)
    im = plt.matshow(class_probs_matrix, cmap="Blues", aspect="auto", fignum=1, vmin=0, vmax=1)
    plt.colorbar(im, label="Probability")
    plt.yticks(range(len(class_names)), class_names)
    plt.xticks(range(num_tokens), [c for c in chars], rotation=0, fontsize=10)
    plt.xlabel("Token position")
    plt.ylabel("Character class")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n✓ Saved heatmap to: {save_path}")
        print(f"  Sequence: {repr(tokenizer.decode(token_list))}")

    return class_probs_matrix


# Example usage: visualize character class probabilities
class_probs = visualize_char_class_probs(
    output.logits,
    x_batch,
    CHAR_CLASSES.values(),
    tokenizer,
    num_tokens=25,
    batch_index=2,
)


# %%
# Example: compute gradient-based analysis for end-of-sentence punctuation
def get_loss_fn(tokenizer: CharTokenizer, pos_class_chars: str):
    """Create a loss function that encourages/discourages certain characters."""
    multiplier = torch.full((tokenizer.vocab_size,), -1.0)
    for char in pos_class_chars:
        if char in tokenizer.stoi:
            multiplier[tokenizer.stoi[char]] = 1.0

    def loss_fn(logits: torch.Tensor) -> torch.Tensor:
        return (logits * multiplier).sum(dim=-1)

    return loss_fn


# %%


def run_gradient_analysis():
    x_batch, y_batch = get_batch(val_data, batch_size=8)
    output = model(x_batch, enable_cache=True)
    cache = output.cache

    eos_punctuation_loss = get_loss_fn(tokenizer, ".!?")
    loss = eos_punctuation_loss(output.logits)
    loss.mean().backward()

    layer = 1
    mlp_acts = cache.get_value(CacheKey("mlp_pre_act", layer))
    mlp_grads = cache.get_grad(CacheKey("mlp_pre_act", layer))
    assert mlp_grads is not None

    print(f"\n✓ MLP activations shape: {mlp_acts.shape}")
    print(f"✓ MLP gradients shape: {mlp_grads.shape}")

    plt.figure(figsize=(8, 6))
    plt.scatter(mlp_acts[0, -1, :].cpu().numpy(), mlp_grads[0, -1, :].cpu().numpy(), alpha=0.3)
    plt.xlabel("MLP activations")
    plt.ylabel("MLP gradients")
    plt.title(f"MLP activations vs gradients (layer {layer}, batch 0, pos -1)")
    # plt.savefig("mlp_acts_vs_grads.png")  # Uncomment to save plot
    print("\n✓ Visualization ready (uncomment plt.savefig to save)")


# %%


x_batch, y_batch = get_batch(val_data, batch_size=64)
output = model(x_batch, targets=y_batch, enable_cache=True)

# %%


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


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
        # LDA gives k-1 dimensions for k classes, so 1D for binary
        reducer = LDA(n_components=1, **method_kwargs)
        embedding_1d = reducer.fit_transform(activations, predictions > 0.5)
        # Create 2D visualization by adding random y-jitter
        embedding = np.column_stack([embedding_1d, np.random.randn(len(embedding_1d)) * 0.1])
        method_label = "LDA (supervised)"
    elif method == "pls":
        # PLS regression with continuous target (probabilities)
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
            # Get context window around this position
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

    fig.show()
    return embedding


def plot_variance_explained(
    output: Out,
    cache_key: CacheKey,
    char_classes: dict[str, CharClass],
    max_components: int = 50,
    save_path: str | None = None,
    figsize: tuple[int, int] = (12, 8),
) -> dict:
    """Plot cumulative variance/R² explained by number of components.

    Args:
        output: model output with cache
        cache_key: which activation to analyze
        char_classes: dict of character classes to analyze
        max_components: maximum number of components to compute
        save_path: path to save plot (None to skip)
        figsize: figure size

    Returns:
        results: dict with 'pca' and per-class R² curves
    """
    activations = to_numpy(output.cache.get_value(cache_key))
    activations_flat = activations.reshape(-1, activations.shape[-1])

    results = {}

    # Compute PCA explained variance
    n_components_pca = min(max_components, activations_flat.shape[0], activations_flat.shape[1])
    pca = PCA(n_components=n_components_pca)
    pca.fit(activations_flat)
    pca_cumulative = np.cumsum(pca.explained_variance_ratio_)
    results['pca'] = pca_cumulative

    # Compute PLS R² for each character class
    for class_name, char_class in char_classes.items():
        predictions = to_numpy(char_class.get_probabilities(output.logits))
        predictions_flat = predictions.reshape(-1)

        r2_scores = []
        max_pls = min(max_components, activations_flat.shape[1])

        for n_comp in range(1, max_pls + 1):
            pls = PLSRegression(n_components=n_comp)
            pls.fit(activations_flat, predictions_flat)
            y_pred = pls.predict(activations_flat).flatten()

            # Compute R² score
            ss_res = np.sum((predictions_flat - y_pred) ** 2)
            ss_tot = np.sum((predictions_flat - np.mean(predictions_flat)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            r2_scores.append(r2)

        results[class_name] = np.array(r2_scores)

    # Create plot
    plt.figure(figsize=figsize)

    # Plot PCA line
    plt.plot(range(1, len(pca_cumulative) + 1), pca_cumulative * 100,
             linewidth=2, label='PCA (overall variance)', color='blue')

    # Plot PLS lines for each class
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    for (class_name, r2_scores), color in zip(
        [(k, v) for k, v in results.items() if k != 'pca'],
        colors
    ):
        plt.plot(range(1, len(r2_scores) + 1), r2_scores * 100,
                 linewidth=2, label=f'PLS R² ({class_name})', color=color, alpha=0.8)

    # Add reference lines
    for threshold in [50, 90, 95]:
        plt.axhline(y=threshold, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        plt.text(max_components * 0.95, threshold + 1, f'{threshold}%',
                 fontsize=8, color='gray', ha='right')

    plt.xlabel('Number of Components')
    plt.ylabel('Variance / R² Explained (%)')
    plt.title(f'Variance Analysis: {cache_key.key} @ layer {cache_key.layer}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.2)
    plt.xlim(1, max_components)
    plt.ylim(0, 105)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved variance plot to: {save_path}")

    plt.show()
    return results


# embedding = visualize_activations_2d(
#     output,
#     CacheKey("resid_mid", layer_to_viz),
#     CHAR_CLASSES["whitespace"],
#     method="umap",
#     title=f"Residual Stream Activations (Layer {layer_to_viz})\nColored by Whitespace Prediction",
#     save_path="activations_umap.png",
# )


# Interactive Plotly version with hover text
key = CacheKey("resid_pre", 2)
embedding_interactive = visualize_activations_2d_interactive(
    output,
    x_batch,
    key,
    CHAR_CLASSES["numeral"],
    tokenizer,
    method="pls",
    title=f"Activations of {key.key} @ layer {key.layer}",
    context_chars=30,
    dims=(2, 3),
)


# %%
# Variance analysis: how many components needed to explain variance/predict each class
variance_results = plot_variance_explained(
    output,
    key,
    CHAR_CLASSES,
    max_components=50,
    save_path="variance_analysis.png",
)

# Print summary
print("\n✓ Variance Analysis Summary:")
print(f"  PCA: {np.searchsorted(variance_results['pca'], 0.9) + 1} components for 90% variance")
for class_name in CHAR_CLASSES:
    if class_name in variance_results:
        r2_90 = np.searchsorted(variance_results[class_name], 0.9) + 1
        r2_max = variance_results[class_name][-1]
        print(f"  {class_name}: {r2_90} components for 90% R², max R²={r2_max:.3f}")


# %%
