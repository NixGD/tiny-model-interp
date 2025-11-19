"""Find task-relevant subspaces using LRP attributions and PCA.

Simplified version focusing on:
1. Computing LRP attribution vectors
2. Applying PCA transformations (centered vs standardized)
3. Scatter plots colored by prediction value
4. Variance explained curves

TODO: Try ICA later - it might find sparser task-relevant subspaces
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from tiny_model.char_tokenizer import CharTokenizer
from tiny_model.model import CacheKey, GPT
from tiny_model.utils import REPO_ROOT

from analysis.char_classes import CharClass, create_char_classes
from analysis.utils import get_batch, load_model, to_numpy

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
    model.eval()
    model.zero_grad()

    # Forward pass with cache
    output = model(x_batch, enable_cache=True)
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

    # Flatten batch and sequence dimensions
    _, _, hidden_dim = attributions.shape
    attributions_flat = to_numpy(attributions).reshape(-1, hidden_dim)
    predictions_flat = to_numpy(loss_value).reshape(-1)

    return attributions_flat, predictions_flat


def collect_attribution_dataset(
    model: GPT,
    data: np.memmap,
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
        x_batch, _ = get_batch(data, batch_size, block_size)
        attributions, predictions = compute_lrp_attributions(model, x_batch, cache_key, char_class)
        all_attributions.append(attributions)
        all_predictions.append(predictions)

        if (i + 1) % 5 == 0 or i == n_batches - 1:
            print(f"  {i + 1}/{n_batches} batches")

    all_attributions = np.vstack(all_attributions)[:n_samples]
    all_predictions = np.concatenate(all_predictions)[:n_samples]

    print(f"✓ Shape: {all_attributions.shape}")
    print(f"✓ Prediction range: [{all_predictions.min():.3f}, {all_predictions.max():.3f}]")

    return all_attributions, all_predictions


def plot_attribution_scatter(
    attributions_dict: dict[str, np.ndarray],
    predictions: np.ndarray,
    dims: tuple[int, int] = (0, 1),
    n_plot: int = 3000,
) -> None:
    """Plot scatter of attributions in different bases.

    Args:
        attributions_dict: dict mapping method name to transformed attributions
        predictions: prediction values for coloring
        dims: which two dimensions to plot
        n_plot: number of points to plot
    """
    methods = list(attributions_dict.keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4))
    if n_methods == 1:
        axes = [axes]

    # Subsample
    indices = np.random.choice(len(predictions), min(n_plot, len(predictions)), replace=False)
    preds_sub = predictions[indices]

    for ax, method in zip(axes, methods, strict=False):
        attrs_sub = attributions_dict[method][indices]

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

    fig.colorbar(scatter, ax=axes, label="Logit Diff (uppercase)", pad=0.02)
    plt.tight_layout()

    save_path = OUTPUT_DIR / "attribution_scatter.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved scatter plot to: {save_path}")
    plt.close()


def plot_variance_curves(
    attributions: np.ndarray,
    max_components: int = 50,
) -> None:
    """Plot variance explained by number of components.

    Args:
        attributions: attribution vectors of shape (n_samples, hidden_dim)
        max_components: maximum number of components
    """
    max_comp = min(max_components, attributions.shape[1])

    fig, ax = plt.subplots(figsize=(8, 6))

    # PCA (centered)
    pca = PCA(n_components=max_comp)
    pca.fit(attributions)
    pca_var = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(range(1, len(pca_var) + 1), pca_var * 100, "o-", label="PCA (centered)", color="blue", linewidth=2)

    # PCA (standardized)
    attrs_std = (attributions - attributions.mean(axis=0)) / (attributions.std(axis=0) + 1e-8)
    pca_std = PCA(n_components=max_comp)
    pca_std.fit(attrs_std)
    pca_std_var = np.cumsum(pca_std.explained_variance_ratio_)
    ax.plot(
        range(1, len(pca_std_var) + 1), pca_std_var * 100, "s-", label="PCA (standardized)", color="green", linewidth=2
    )

    # Reference lines
    for threshold in [50, 90, 95, 99]:
        ax.axhline(y=threshold, color="gray", linestyle="--", alpha=0.3, linewidth=1)
        ax.text(max_comp * 0.98, threshold + 0.5, f"{threshold}%", fontsize=8, color="gray", ha="right")

    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title("Variance Explained by Number of Components")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_comp)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    save_path = OUTPUT_DIR / "variance_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved variance curves to: {save_path}")
    plt.close()

    # Print summary
    print("\n✓ Variance summary:")
    print(f"  PCA (centered):     {pca_var[-1] * 100:.2f}% with {max_comp} components")
    print(f"  PCA (standardized): {pca_std_var[-1] * 100:.2f}% with {max_comp} components")

    # Print first few components
    print("\n✓ First 10 components:")
    for i in range(min(10, max_comp)):
        print(
            f"    {i}: PCA={pca.explained_variance_ratio_[i] * 100:.2f}%, PCA_std={pca_std.explained_variance_ratio_[i] * 100:.2f}%"
        )


def compute_prediction_r2(
    attributions: np.ndarray,
    predictions: np.ndarray,
    n_components: int = 10,
    method: str = "pca",
) -> dict[str, float]:
    """Compute R² for predicting logit_diff from PCA/PLS subspace.

    Args:
        attributions: attribution vectors of shape (n_samples, hidden_dim)
        predictions: target values of shape (n_samples,)
        n_components: number of components to use
        method: 'pca', 'pca_std', or 'pls'

    Returns:
        results: dict with R² scores for different numbers of components
    """
    r2_scores = {}

    for n in range(1, n_components + 1):
        if method == "pca":
            pca = PCA(n_components=n)
            features = pca.fit_transform(attributions)
        elif method == "pca_std":
            attrs_std = (attributions - attributions.mean(axis=0)) / (attributions.std(axis=0) + 1e-8)
            pca = PCA(n_components=n)
            features = pca.fit_transform(attrs_std)
        elif method == "pls":
            pls = PLSRegression(n_components=n)
            pls.fit(attributions, predictions)
            features = pls.transform(attributions)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Train linear regression
        lr = LinearRegression()
        lr.fit(features, predictions)
        preds = lr.predict(features)
        r2 = r2_score(predictions, preds)
        r2_scores[n] = r2

    return r2_scores


def compute_subspace_overlap(
    subspace1: np.ndarray,
    subspace2: np.ndarray,
) -> dict[str, float]:
    """Compute overlap between two subspaces using principal angles.

    Args:
        subspace1: orthonormal basis matrix of shape (n_features, n_dims1)
        subspace2: orthonormal basis matrix of shape (n_features, n_dims2)

    Returns:
        results: dict with overlap metrics
    """
    # Compute principal angles via SVD of subspace1.T @ subspace2
    # Principal angles θ satisfy: cos(θ) = singular values
    overlap_matrix = subspace1.T @ subspace2
    singular_values = np.linalg.svd(overlap_matrix, compute_uv=False)

    # Clip to [0, 1] to handle numerical errors
    cosines = np.clip(singular_values, 0, 1)
    angles_deg = np.rad2deg(np.arccos(cosines))

    # Compute overall overlap metrics
    # Mean cosine of principal angles (1 = perfect overlap, 0 = orthogonal)
    mean_cosine = cosines.mean()

    # Frobenius norm of projection: ||P1 @ P2||_F where P = U @ U.T
    proj1 = subspace1 @ subspace1.T
    proj2 = subspace2 @ subspace2.T
    proj_overlap = np.linalg.norm(proj1 @ proj2, "fro")

    return {
        "mean_cosine": mean_cosine,
        "max_cosine": cosines.max(),
        "min_cosine": cosines.min(),
        "mean_angle_deg": angles_deg.mean(),
        "max_angle_deg": angles_deg.max(),
        "min_angle_deg": angles_deg.min(),
        "projection_overlap": proj_overlap,
        "principal_cosines": cosines,
    }


def plot_prediction_r2(
    r2_dict: dict[str, dict[int, float]],
    save_path: str | None = None,
) -> None:
    """Plot R² curves for different methods.

    Args:
        r2_dict: dict mapping method name to {n_components: r2} dict
        save_path: path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {"pca": "blue", "pca_std": "green", "pls": "red"}
    labels = {"pca": "PCA (centered)", "pca_std": "PCA (standardized)", "pls": "PLS"}

    for method, r2_scores in r2_dict.items():
        ns = sorted(r2_scores.keys())
        r2s = [r2_scores[n] for n in ns]
        ax.plot(ns, np.array(r2s) * 100, "o-", label=labels.get(method, method),
                color=colors.get(method, "gray"), linewidth=2)

    ax.set_xlabel("Number of Components")
    ax.set_ylabel("R² (%)")
    ax.set_title("Prediction R² from LRP Attribution Subspace")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved R² plot to: {save_path}")

    plt.close()


if __name__ == "__main__":
    # Load model and enable LXT
    model_path = REPO_ROOT / "out-wiki-char/ckpt.pt"
    model = load_model(str(model_path))
    model.eval()
    model.set_lxt_enabled(True)
    print(f"✓ Loaded model from {model_path}")
    print("✓ LXT enabled")

    # Load tokenizer and data
    tokenizer = CharTokenizer(vocab_path="data/wiki_char/vocab.json")
    data_dir = REPO_ROOT / "data/wiki_char"
    val_data = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
    print(f"✓ Loaded validation dataset: {len(val_data):,} tokens")

    # Create character classes
    CHAR_CLASSES = create_char_classes(tokenizer)
    target_class = CHAR_CLASSES["uppercase"]
    print(f"✓ Target class: {target_class}")

    # Collect attribution dataset (SMALLER for debugging)
    cache_key = CacheKey("resid_mid", 2)
    print(f"\n{'=' * 60}")
    print(f"COLLECTING ATTRIBUTIONS: {cache_key}")
    print("=" * 60)

    attributions, predictions = collect_attribution_dataset(
        model,
        val_data,
        cache_key,
        target_class,
        n_samples=50_000,  # Keep it fast for debugging
        batch_size=128,
        block_size=256,
    )

    # Check for issues
    print(f"\n✓ Attribution statistics:")
    print(f"  Mean: {attributions.mean():.6f}")
    print(f"  Std: {attributions.std():.6f}")
    print(f"  Min: {attributions.min():.6f}")
    print(f"  Max: {attributions.max():.6f}")

    # Check if data is weirdly collinear
    U, S, Vt = np.linalg.svd(attributions, full_matrices=False)
    print(f"\n✓ Singular values (first 10):")
    for i in range(min(10, len(S))):
        print(f"    {i}: {S[i]:.4f}")

    # Apply transformations
    print(f"\n{'=' * 60}")
    print("APPLYING TRANSFORMATIONS")
    print("=" * 60)

    # Original space
    attrs_dict = {"original": attributions}

    # PCA (centered)
    pca = PCA(n_components=10)
    attrs_pca = pca.fit_transform(attributions)
    attrs_dict["pca"] = attrs_pca
    print(f"✓ PCA: {attrs_pca.shape}, variance explained: {pca.explained_variance_ratio_.sum() * 100:.2f}%")

    # PCA (standardized)
    attrs_std = (attributions - attributions.mean(axis=0)) / (attributions.std(axis=0) + 1e-8)
    pca_std = PCA(n_components=10)
    attrs_pca_std = pca_std.fit_transform(attrs_std)
    attrs_dict["pca_std"] = attrs_pca_std
    print(
        f"✓ PCA (std): {attrs_pca_std.shape}, variance explained: {pca_std.explained_variance_ratio_.sum() * 100:.2f}%"
    )

    # TODO: Try ICA - it should find sparser, more interpretable components
    # ica = FastICA(n_components=10, random_state=42, max_iter=200)
    # attrs_ica = ica.fit_transform(attributions)
    # attrs_dict["ica"] = attrs_ica

    # Plot scatter
    print(f"\n{'=' * 60}")
    print("SCATTER PLOTS")
    print("=" * 60)
    plot_attribution_scatter(attrs_dict, predictions, dims=(0, 1), n_plot=3000)

    # Plot variance curves
    print(f"\n{'=' * 60}")
    print("VARIANCE CURVES")
    print("=" * 60)
    plot_variance_curves(attributions, max_components=30)

    # Compute prediction R² from attribution subspaces
    print(f"\n{'=' * 60}")
    print("PREDICTION R² FROM ATTRIBUTION SUBSPACES")
    print("=" * 60)

    r2_dict = {}
    for method in ["pca", "pca_std", "pls"]:
        print(f"Computing {method.upper()} R²...")
        r2_dict[method] = compute_prediction_r2(attributions, predictions, n_components=20, method=method)

    plot_prediction_r2(r2_dict, save_path=str(OUTPUT_DIR / "prediction_r2.png"))

    # Print summary
    print("\n✓ R² at different component counts:")
    for n in [1, 5, 10, 20]:
        print(f"  {n} components:")
        for method in ["pca", "pca_std", "pls"]:
            if n in r2_dict[method]:
                print(f"    {method:8} R²={r2_dict[method][n]:.4f}")

    # Compare LRP subspace to PLS subspace on activations
    print(f"\n{'=' * 60}")
    print("COMPARING LRP SUBSPACE TO PLS SUBSPACE")
    print("=" * 60)

    # Get activations (not attributions) for the same data
    print("Collecting activations for PLS comparison...")
    model.eval()
    with torch.no_grad():
        x_batch, _ = get_batch(val_data, batch_size=128, block_size=256)
        output = model(x_batch, enable_cache=True)
        activations_tensor = output.cache.get_value(cache_key)
        activations_flat = to_numpy(activations_tensor).reshape(-1, activations_tensor.shape[-1])
        logit_diffs = to_numpy(target_class.get_logit_diff(output.logits)).reshape(-1)

    # Limit to same size as attributions for fair comparison
    n_samples_compare = min(len(attributions), len(activations_flat))
    activations_flat = activations_flat[:n_samples_compare]
    logit_diffs = logit_diffs[:n_samples_compare]
    attributions_compare = attributions[:n_samples_compare]

    # Compute PLS on activations
    n_dims = 5
    pls_acts = PLSRegression(n_components=n_dims)
    pls_acts.fit(activations_flat, logit_diffs)
    pls_acts_basis = pls_acts.x_rotations_  # (hidden_dim, n_dims)

    # Compute PCA on LRP attributions (standardized)
    attrs_std = (attributions_compare - attributions_compare.mean(axis=0)) / (
        attributions_compare.std(axis=0) + 1e-8
    )
    pca_attrs = PCA(n_components=n_dims)
    pca_attrs.fit(attrs_std)
    pca_attrs_basis = pca_attrs.components_.T  # (hidden_dim, n_dims)

    # Compute overlap
    overlap = compute_subspace_overlap(pls_acts_basis, pca_attrs_basis)

    print(f"\n✓ Subspace overlap ({n_dims}D subspaces):")
    print(f"  Mean cosine similarity:    {overlap['mean_cosine']:.4f}")
    print(f"  Max cosine similarity:     {overlap['max_cosine']:.4f}")
    print(f"  Min cosine similarity:     {overlap['min_cosine']:.4f}")
    print(f"  Mean principal angle:      {overlap['mean_angle_deg']:.2f}°")
    print(f"  Projection overlap (Frob): {overlap['projection_overlap']:.2f}")
    print(f"\n  Principal angle cosines: {overlap['principal_cosines']}")

    print(f"\n✓ All outputs saved to: {OUTPUT_DIR}")
