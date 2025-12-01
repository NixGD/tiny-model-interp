"""Piecewise linear analysis of MLP behavior.

This analysis models the MLP as a piecewise linear function, where the "pieces" are
defined by string-ending patterns.
"""

# %% Imports
import string
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console
from rich.progress import track
from rich.table import Table
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from analysis.common.loss import get_logit_diff_loss
from analysis.common.utils import flatten_keep_last, get_batch, load_model, to_numpy
from tiny_model.model import CacheKey
from tiny_model.tokenizer.char_tokenizer import CharTokenizer


# %% Data class
@dataclass
class LinearityData:
    """Holds all data for piecewise linearity analysis."""

    x: torch.Tensor  # (batch, seq_len) original tokens
    mlp_input: np.ndarray  # (n_samples, n_embd) flattened MLP inputs
    neuron_acts: np.ndarray  # (n_samples, n_neurons) post-ReLU activations
    contributions: np.ndarray  # (n_samples,) scalar contributions to logit diff
    decoded: list[str]  # decoded prefix strings for each position
    tokenizer: CharTokenizer
    direction: np.ndarray  # (n_neurons,) gradient direction for importance weighting

    @property
    def n_samples(self) -> int:
        return len(self.contributions)


# %% Data loading
def load_data(n_batches: int = 5, batch_size: int = 200) -> LinearityData:
    """Load model, run forward pass, compute MLP inputs and contributions."""
    model = load_model()
    tokenizer = CharTokenizer()

    # Loss function for gradient direction (space vs letters)
    loss_fn = get_logit_diff_loss(
        tokenizer,
        [" "],
        list(string.ascii_uppercase) + list(string.ascii_lowercase),
    )

    # Get gradient direction from a single batch
    torch.set_grad_enabled(True)
    model.zero_grad()
    x_grad, y_grad = get_batch(batch_size=1)
    output = model(x_grad, targets=y_grad, cache_enabled=True, alphas_enabled=True)
    loss = loss_fn(output.logits)
    loss.sum().backward()
    direction = to_numpy(output.cache.get_grad(CacheKey("mlp_post_act", 3))[0, 0])

    # Collect data from multiple batches
    all_x, all_mlp_input, all_neuron_act = [], [], []
    with torch.no_grad():
        for _ in range(n_batches):
            x, _ = get_batch(batch_size=batch_size)
            output = model(x, cache_enabled=True)
            all_x.append(x)
            all_mlp_input.append(flatten_keep_last(to_numpy(output.cache.get_value(CacheKey("resid_mid", 3)))))
            all_neuron_act.append(flatten_keep_last(to_numpy(output.cache.get_value(CacheKey("mlp_post_act", 3)))))

    x = torch.cat(all_x, dim=0)
    mlp_input = np.concatenate(all_mlp_input, axis=0)
    neuron_acts = np.concatenate(all_neuron_act, axis=0)
    contributions = neuron_acts @ direction

    # Pre-compute decoded prefixes for each position
    n_batch, seq_len = x.shape
    decoded = []
    for batch_idx in range(n_batch):
        full_text = tokenizer.decode(x[batch_idx].tolist())
        for pos in range(seq_len):
            decoded.append(full_text[: pos + 1])

    return LinearityData(x, mlp_input, neuron_acts, contributions, decoded, tokenizer, direction)


# %% Stats computation
def compute_stats(mlp_input: np.ndarray, contributions: np.ndarray) -> dict[str, float | int]:
    """Compute linearity statistics for a subset of data."""
    n = len(contributions)
    if n < 10:
        return {"count": n, "var": 0.0, "r2": 0.0, "resid_var": 0.0}

    var = float(np.var(contributions))

    reg = Ridge(alpha=0.01)
    reg.fit(mlp_input, contributions)
    pred = reg.predict(mlp_input)

    ss_res = np.sum((contributions - pred) ** 2)
    ss_tot = np.sum((contributions - contributions.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    resid_var = float(ss_res / n)

    return {"count": n, "var": var, "r2": max(0, r2), "resid_var": resid_var}


def get_ending_mask(data: LinearityData, suffix: str) -> np.ndarray:
    """Get boolean mask for positions where decoded string ends with suffix."""
    return np.array([d.endswith(suffix) for d in data.decoded])


# %% Diagnostic functions
def show_by_char(data: LinearityData, min_count: int = 10) -> None:
    """Display table with one row per ending character."""
    console = Console(force_jupyter=True)
    table = Table(title="MLP Linearity by Ending Character", show_lines=True)
    table.add_column("Char", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("%", justify="right")
    table.add_column("Var", justify="right")
    table.add_column("R²", justify="right")
    table.add_column("Resid Var", justify="right")

    results = []
    for char in data.tokenizer.chars:
        mask = get_ending_mask(data, char)
        if mask.sum() >= min_count:
            stats = compute_stats(data.mlp_input[mask], data.contributions[mask])
            stats["char"] = char
            stats["pct"] = 100 * stats["count"] / data.n_samples
            results.append(stats)

    # Sort by count descending
    results.sort(key=lambda x: x["count"], reverse=True)

    for r in results:
        r2_color = "green" if r["r2"] >= 0.9 else "yellow" if r["r2"] >= 0.7 else "red"
        table.add_row(
            repr(r["char"]),
            str(r["count"]),
            f"{r['pct']:.1f}",
            f"{r['var']:.3f}",
            f"[{r2_color}]{r['r2']:.3f}[/{r2_color}]",
            f"{r['resid_var']:.4f}",
        )

    console.print(table)


def drill_down(data: LinearityData, suffix: str, min_count: int = 5) -> None:
    """Show stats for all 'X{suffix}' patterns where X is any character.

    Example: drill_down(data, "t") shows stats for "at", "bt", " t", etc.
    """
    console = Console(force_jupyter=True)
    table = Table(title=f"MLP Linearity: X + {repr(suffix)}", show_lines=True)
    table.add_column("Ending", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("%", justify="right")
    table.add_column("Var", justify="right")
    table.add_column("R²", justify="right")
    table.add_column("Resid Var", justify="right")

    results = []
    for char in data.tokenizer.chars:
        ending = char + suffix
        mask = get_ending_mask(data, ending)
        if mask.sum() >= min_count:
            stats = compute_stats(data.mlp_input[mask], data.contributions[mask])
            stats["ending"] = ending
            stats["pct"] = 100 * stats["count"] / data.n_samples
            results.append(stats)

    # Sort by count descending
    results.sort(key=lambda x: x["count"], reverse=True)

    for r in results:
        r2_color = "green" if r["r2"] >= 0.95 else "yellow" if r["r2"] >= 0.9 else "red"
        table.add_row(
            repr(r["ending"]),
            str(r["count"]),
            f"{r['pct']:.1f}",
            f"{r['var']:.3f}",
            f"[{r2_color}]{r['r2']:.3f}[/{r2_color}]",
            f"{r['resid_var']:.4f}",
        )

    console.print(table)


# %% Importance-weighted embeddings for fast clustering
def get_importance_weighted_embedding(data: LinearityData, pattern: str, direction: np.ndarray) -> np.ndarray:
    """Get mean importance-weighted activation embedding for a pattern.

    Embedding = mean of (neuron_active * direction) across samples.
    This captures which important neurons are typically active for this pattern.
    """
    mask = get_ending_mask(data, pattern)
    if mask.sum() == 0:
        return np.zeros_like(direction)
    acts = data.neuron_acts[mask]
    weighted = (acts > 0).astype(float) * direction
    return weighted.mean(axis=0)


def compute_embedding_distance_matrix(
    data: LinearityData,
    direction: np.ndarray,
    patterns: list[str],
    min_count: int = 10,
) -> tuple[list[str], np.ndarray, dict[str, dict]]:
    """Compute cosine distance matrix based on importance-weighted embeddings."""
    # Get stats and embeddings for patterns with enough data
    stats: dict[str, dict] = {}
    embeddings: dict[str, np.ndarray] = {}
    for pattern in patterns:
        mask = get_ending_mask(data, pattern)
        if mask.sum() >= min_count:
            stats[pattern] = compute_stats(data.mlp_input[mask], data.contributions[mask])
            embeddings[pattern] = get_importance_weighted_embedding(data, pattern, direction)

    sorted_patterns = sorted(stats.keys(), key=lambda p: int(stats[p]["count"]), reverse=True)
    n = len(sorted_patterns)

    # Compute cosine distance matrix
    dist_matrix = np.zeros((n, n))
    for i, p1 in enumerate(sorted_patterns):
        for j, p2 in enumerate(sorted_patterns):
            e1, e2 = embeddings[p1], embeddings[p2]
            norm1, norm2 = np.linalg.norm(e1), np.linalg.norm(e2)
            if norm1 > 0 and norm2 > 0:
                dist_matrix[i, j] = 1 - np.dot(e1, e2) / (norm1 * norm2)
            else:
                dist_matrix[i, j] = 1.0

    return sorted_patterns, dist_matrix, stats


def cluster_with_r2_validation(
    data: LinearityData,
    direction: np.ndarray,
    patterns: list[str],
    min_count: int = 10,
    r2_threshold: float = 0.93,
) -> list[list[str]]:
    """Cluster using embedding distances but validate each merge with R².

    Uses hierarchical clustering tree from embeddings, but walks the tree
    and only permits merges where combined R² >= r2_threshold.
    """
    from scipy.cluster.hierarchy import linkage

    sorted_patterns, dist_matrix, _ = compute_embedding_distance_matrix(data, direction, patterns, min_count)

    if len(sorted_patterns) < 2:
        return [[p] for p in sorted_patterns]

    n = len(sorted_patterns)

    # Precompute masks for each pattern
    pattern_masks = {p: get_ending_mask(data, p) for p in sorted_patterns}

    def get_group_r2(group: set[str]) -> float:
        """Compute R² for a group of patterns."""
        mask = np.zeros(data.n_samples, dtype=bool)
        for p in group:
            mask |= pattern_masks[p]
        s = compute_stats(data.mlp_input[mask], data.contributions[mask])
        return float(s["r2"])

    # Convert to condensed form for linkage
    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed.append(dist_matrix[i, j])
    condensed = np.array(condensed)

    # Build hierarchical tree
    Z = linkage(condensed, method="average")

    # Walk the tree bottom-up, validating merges with R²
    clusters: dict[int, set[str]] = {i: {sorted_patterns[i]} for i in range(n)}

    for merge_idx, (idx1, idx2, _dist, _count) in enumerate(Z):
        idx1, idx2 = int(idx1), int(idx2)
        new_cluster_id = n + merge_idx

        cluster1 = clusters.get(idx1)
        cluster2 = clusters.get(idx2)

        if cluster1 is None or cluster2 is None:
            continue

        # Check if merge is permitted by R²
        merged = cluster1 | cluster2
        merged_r2 = get_group_r2(merged)

        if merged_r2 >= r2_threshold:
            clusters[new_cluster_id] = merged
            del clusters[idx1]
            del clusters[idx2]

    result = [sorted(c) for c in clusters.values()]
    result.sort(key=len, reverse=True)
    return result


# %% Iterative split-merge algorithm
def split_pattern(data: LinearityData, pattern: str, min_count: int = 10) -> list[str]:
    """Split a pattern into sub-patterns by prepending each character."""
    sub_patterns = []
    for char in data.tokenizer.chars:
        sub = char + pattern
        mask = get_ending_mask(data, sub)
        if mask.sum() >= min_count:
            sub_patterns.append(sub)
    return sub_patterns


def compute_piecewise_r2(
    train_data: LinearityData,
    groups: list[list[str]],
    eval_data: LinearityData | None = None,
) -> tuple[float, int, int]:
    """Compute overall R² for the composite piecewise linear function.

    Fits linear models on train_data, evaluates on eval_data (or train_data if not provided).
    Returns (overall_r2, n_covered, n_total).
    """
    if eval_data is None:
        eval_data = train_data

    # Fit a linear model for each group on training data
    group_models: list[Ridge | None] = []
    for group in groups:
        mask = np.zeros(train_data.n_samples, dtype=bool)
        for p in group:
            mask |= get_ending_mask(train_data, p)

        if mask.sum() >= 10:
            reg = Ridge(alpha=0.01)
            reg.fit(train_data.mlp_input[mask], train_data.contributions[mask])
            group_models.append(reg)
        else:
            group_models.append(None)

    # Evaluate on eval_data
    predictions = np.full(eval_data.n_samples, np.nan)
    covered_mask = np.zeros(eval_data.n_samples, dtype=bool)

    for group, model in track(zip(groups, group_models, strict=True), description="Computing piecewise R²"):
        if model is not None:
            mask = np.zeros(eval_data.n_samples, dtype=bool)
            for p in group:
                mask |= get_ending_mask(eval_data, p)
            predictions[mask] = model.predict(eval_data.mlp_input[mask])
            covered_mask |= mask

    n_covered = int(covered_mask.sum())
    if n_covered < 10:
        return 0.0, n_covered, eval_data.n_samples

    y_true = eval_data.contributions[covered_mask]
    y_pred = predictions[covered_mask]

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    overall_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return max(0, overall_r2), n_covered, eval_data.n_samples


def iterative_split_merge(
    data: LinearityData,
    direction: np.ndarray,
    min_count: int = 10,
    split_threshold: float = 0.90,
    merge_threshold: float = 0.93,
    max_iterations: int = 10,
    max_pattern_length: int = 5,
    verbose: bool = True,
) -> list[list[str]]:
    """Iteratively split non-linear patterns and merge similar linear ones.

    Algorithm:
    1. Start with single-character patterns
    2. Split patterns with R² < split_threshold into "Xt" sub-patterns
    3. Merge similar patterns (using R²-validated clustering) if R² stays high
    4. Repeat until convergence or max iterations

    Returns list of pattern groups, where each group can be modeled linearly.
    """
    # Start with single-char patterns that have enough data
    current_patterns: set[str] = set()
    for char in data.tokenizer.chars:
        mask = get_ending_mask(data, char)
        if mask.sum() >= min_count:
            current_patterns.add(char)

    if verbose:
        print(f"Starting with {len(current_patterns)} single-char patterns")

    for iteration in range(max_iterations):
        # Compute stats for all current patterns
        pattern_stats = {}
        for p in current_patterns:
            mask = get_ending_mask(data, p)
            count = int(mask.sum())
            if count >= 10:
                pattern_stats[p] = compute_stats(data.mlp_input[mask], data.contributions[mask])
            else:
                pattern_stats[p] = {"count": count, "r2": 0.0}

        # Find patterns that need splitting (low R²)
        to_split = [
            p
            for p, s in pattern_stats.items()
            if s["r2"] < split_threshold and len(p) < max_pattern_length and s["count"] >= min_count
        ]

        if verbose:
            n_high_r2 = sum(1 for s in pattern_stats.values() if s["r2"] >= split_threshold)
            valid_r2s = [s["r2"] for s in pattern_stats.values() if s["count"] >= min_count]
            avg_r2 = np.mean(valid_r2s) if valid_r2s else 0.0
            print(
                f"  Iter {iteration + 1}: {len(current_patterns)} patterns, "
                f"{n_high_r2} with R²>={split_threshold:.2f}, "
                f"avg R²={avg_r2:.3f}, splitting {len(to_split)}"
            )

        # Split low-R² patterns
        new_patterns = set()
        for pattern in to_split:
            subs = split_pattern(data, pattern, min_count)
            if subs:
                current_patterns.discard(pattern)
                new_patterns.update(subs)

        current_patterns.update(new_patterns)

        if not to_split:
            if verbose:
                print(f"  Converged after {iteration + 1} iterations")
            break

    # Filter to patterns with enough data
    valid_patterns = [p for p in sorted(current_patterns) if get_ending_mask(data, p).sum() >= min_count]

    if verbose:
        print(f"Merging {len(valid_patterns)} patterns (R² threshold={merge_threshold})...")

    if len(valid_patterns) < 2:
        return [[p] for p in valid_patterns]

    # Use R²-validated clustering to merge
    groups = cluster_with_r2_validation(
        data,
        direction,
        patterns=valid_patterns,
        min_count=min_count,
        r2_threshold=merge_threshold,
    )

    if verbose:
        print(f"Merged into {len(groups)} groups")

    return groups


def display_groups(
    train_data: LinearityData,
    groups: list[list[str]],
    eval_data: LinearityData | None = None,
) -> None:
    """Display results of split-merge algorithm.

    If eval_data is provided, shows both train and validation R².
    """
    console = Console(force_jupyter=True)
    console.print("\n[bold]Piecewise Linear Groups[/bold]")

    # Use eval_data for display stats if provided, otherwise train_data
    display_data = eval_data if eval_data is not None else train_data

    table = Table(show_lines=True)
    table.add_column("#", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Patterns")
    table.add_column("Total N", justify="right")
    table.add_column("% Data", justify="right")
    table.add_column("Group R²", justify="right")

    total_covered = 0
    for i, group in enumerate(groups):
        mask = np.zeros(display_data.n_samples, dtype=bool)
        for p in group:
            mask |= get_ending_mask(display_data, p)
        group_stats = compute_stats(display_data.mlp_input[mask], display_data.contributions[mask])
        total_covered += group_stats["count"]

        patterns_str = ", ".join(repr(p) for p in sorted(group))

        r2 = group_stats["r2"]
        r2_color = "green" if r2 >= 0.99 else "yellow" if r2 >= 0.95 else "red"
        pct = 100 * group_stats["count"] / display_data.n_samples

        table.add_row(
            str(i + 1),
            str(len(group)),
            patterns_str,
            str(group_stats["count"]),
            f"{pct:.1f}",
            f"[{r2_color}]{r2:.3f}[/{r2_color}]",
        )

    console.print(table)

    n_patterns = sum(len(g) for g in groups)
    n_groups = len(groups)
    pct_covered = 100 * total_covered / display_data.n_samples

    console.print(
        f"\n[bold]Summary:[/bold] {n_patterns} patterns → {n_groups} groups, covering {pct_covered:.1f}% of data"
    )

    # Compute overall piecewise R² (train)
    train_r2, n_covered, n_total = compute_piecewise_r2(train_data, groups)
    train_r2_color = "green" if train_r2 >= 0.95 else "yellow" if train_r2 >= 0.90 else "red"
    console.print(
        f"[bold]Train Piecewise R²:[/bold] [{train_r2_color}]{train_r2:.4f}[/{train_r2_color}] (on {n_covered}/{n_total} samples)"
    )

    # Compute validation R² if eval_data provided
    if eval_data is not None:
        val_r2, val_covered, val_total = compute_piecewise_r2(train_data, groups, eval_data)
        val_r2_color = "green" if val_r2 >= 0.95 else "yellow" if val_r2 >= 0.90 else "red"
        console.print(
            f"[bold]Val Piecewise R²:[/bold]   [{val_r2_color}]{val_r2:.4f}[/{val_r2_color}] (on {val_covered}/{val_total} samples)"
        )

    # Show uncovered data stats
    all_masks = np.zeros(display_data.n_samples, dtype=bool)
    for group in groups:
        for p in group:
            all_masks |= get_ending_mask(display_data, p)
    uncovered = ~all_masks
    if uncovered.sum() > 0:
        unc_pct = 100 * uncovered.sum() / display_data.n_samples
        unc_stats = compute_stats(display_data.mlp_input[uncovered], display_data.contributions[uncovered])
        console.print(f"[dim]Uncovered: {uncovered.sum()} samples ({unc_pct:.1f}%), R² = {unc_stats['r2']:.3f}[/dim]")


# %% Load data
data = load_data(n_batches=5, batch_size=200)
print(f"Loaded {data.n_samples} samples")

# %% Show by ending character
show_by_char(data)

# %% Drill down into a specific character
drill_down(data, "t")

# %% Drill down further
drill_down(data, " t")

# %% Iterative split-merge: automatically discover minimal pattern set
groups = iterative_split_merge(
    data,
    data.direction,
    split_threshold=0.95,
    merge_threshold=0.95,
)

# %% Evaluate on validation data
val_data = load_data(n_batches=5, batch_size=200)
display_groups(data, groups, eval_data=val_data)


# %%

weak_groups = iterative_split_merge(
    data,
    data.direction,
    split_threshold=0.90,
    merge_threshold=0.85,
    max_pattern_length=10,
)

# %% Evaluate on validation data
val_data = load_data(n_batches=5, batch_size=200)
display_groups(data, weak_groups, eval_data=val_data)


# %%


def get_group_mask(data: LinearityData, group: list[str]) -> np.ndarray:
    mask = np.zeros(data.n_samples, dtype=bool)
    for p in group:
        mask |= get_ending_mask(data, p)
    return mask


mask = get_group_mask(data, weak_groups[0])
acts = data.neuron_acts[mask]
weighted = (acts > 0).astype(float) * data.direction

# %%

pca = PCA(n_components=2)
pca_weighted_acts = pca.fit_transform(weighted)

# Fit linear model and compute residuals
reg = Ridge(alpha=0.01)
reg.fit(data.mlp_input[mask], data.contributions[mask])
pred = reg.predict(data.mlp_input[mask])
residuals = data.contributions[mask] - pred

plt.scatter(
    pca_weighted_acts[:, 0], pca_weighted_acts[:, 1], c=residuals, cmap="RdBu", marker=".", s=3, vmin=-1, vmax=1
)
plt.colorbar(label="Residual")
plt.show()


# %%
