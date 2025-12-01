"""Polytope clustering analysis of MLP activations.

This analysis clusters MLP activations based on which neurons are active (the "polytope"
the input lies in for a ReLU network). This is a bottom-up approach: cluster first,
then try to interpret what each cluster means.

## Approach

1. Compute importance-weighted embeddings: `(acts > 0) * direction` where direction
   is the gradient of logit_diff(space vs letters) w.r.t. MLP post-activations
2. Cluster using BisectingKMeans (100 clusters)
3. Manually inspect clusters and assign interpretations based on string patterns

"""

# %%
import json
import string
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import BisectingKMeans
from sklearn.metrics import silhouette_samples, silhouette_score

from analysis.cluster_analysis import mask_predicts_cluster
from analysis.common.loss import LogitLossFn, get_logit_diff_loss
from analysis.common.masks import ending_mask
from analysis.common.utils import flatten_keep_last, get_batch, load_model, to_numpy
from tiny_model.model import GPT, CacheKey
from tiny_model.tokenizer.char_tokenizer import CharTokenizer
from tiny_model.utils import REPO_ROOT

CACHE_PATH = REPO_ROOT / "analysis/out/polytopes/cluster_data_bisecting_kmeans.json"


# %%
@dataclass
class ClusterData:
    """Cached cluster data for analysis."""

    x: torch.Tensor  # (batch, seq_len) token ids
    clusters: np.ndarray  # (batch * seq_len,) cluster assignments
    sample_silhouettes: np.ndarray  # (batch * seq_len,) per-sample silhouette scores
    overall_silhouette: float
    n_clusters: int

    def save(self, path: Path = CACHE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "x": self.x.tolist(),
            "clusters": self.clusters.tolist(),
            "sample_silhouettes": self.sample_silhouettes.tolist(),
            "overall_silhouette": self.overall_silhouette,
            "n_clusters": self.n_clusters,
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Saved cluster data to {path}")

    @classmethod
    def load(cls, path: Path = CACHE_PATH) -> "ClusterData":
        with open(path) as f:
            data = json.load(f)
        clusters = np.array(data["clusters"])
        sample_silhouettes = np.array(data["sample_silhouettes"])
        return cls(
            x=torch.tensor(data["x"]),
            clusters=clusters,
            sample_silhouettes=sample_silhouettes,
            overall_silhouette=data.get("overall_silhouette", sample_silhouettes.mean()),
            n_clusters=data.get("n_clusters", int(clusters.max()) + 1),
        )


def get_direction(model: GPT, loss_fn: LogitLossFn) -> torch.Tensor:
    """Compute gradient direction (which neurons matter for this loss)."""
    model.zero_grad()
    x_grad, y_grad = get_batch(batch_size=1)
    output = model(x_grad, targets=y_grad, cache_enabled=True, alphas_enabled=True)
    loss = loss_fn(output.logits)
    loss.sum().backward()
    direction = output.cache.get_grad(CacheKey("mlp_post_act", 3))[0, 0]
    return direction


def get_acts_and_contributions(
    data: ClusterData,
    model: GPT | None = None,
    loss_fn: LogitLossFn | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute activations and contributions for linearity analysis.

    Args:
        data: ClusterData with x tokens
        model: GPT model (loaded if None)
        loss_fn: Loss function for direction (default: space vs letters)

    Returns:
        acts: Activations, shape (n_samples, n_features)
        contributions: Contribution to direction, shape (n_samples,)
    """
    if model is None:
        model = load_model()

    tokenizer = CharTokenizer()
    if loss_fn is None:
        loss_fn = get_logit_diff_loss(tokenizer, [" "], list(string.ascii_uppercase) + list(string.ascii_lowercase))

    # Get direction
    torch.set_grad_enabled(True)
    direction = get_direction(model, loss_fn)
    direction_np = to_numpy(direction)

    # Get activations
    with torch.no_grad():
        output = model(data.x, cache_enabled=True, alphas_enabled=True)
        neuron_act = flatten_keep_last(to_numpy(output.cache.get_value(CacheKey("mlp_post_act", 3))))
        input_acts = flatten_keep_last(to_numpy(output.cache.get_value(CacheKey("resid_mid", 3))))

    # Compute contributions
    contributions = neuron_act @ direction_np

    return input_acts, contributions


def compute_clusters(batch_size: int = 200, n_clusters: int = 100) -> ClusterData:
    """Compute polytope clusters from MLP activations. This is the expensive step."""
    model = load_model()
    tokenizer = CharTokenizer()
    torch.set_grad_enabled(True)

    # Get direction that increases P(space) - P(letters)
    loss_fn = get_logit_diff_loss(tokenizer, [" "], list(string.ascii_uppercase) + list(string.ascii_lowercase))
    direction = get_direction(model, loss_fn)

    # Get activations for clustering
    x, y = get_batch(batch_size=batch_size)
    with torch.no_grad():
        output = model(x, targets=y, cache_enabled=True, alphas_enabled=True)
    acts = output.cache.get_value(CacheKey("mlp_post_act", 3))

    # Embedding: which neurons are active, weighted by importance
    embedding = (acts > 0) * direction
    embedding = flatten_keep_last(to_numpy(embedding))

    # Cluster
    print(f"Clustering {embedding.shape[0]} samples into {n_clusters} clusters...")
    kmeans = BisectingKMeans(n_clusters=n_clusters, random_state=0, n_init=5)
    clusters = kmeans.fit_predict(embedding)

    # Compute silhouette scores (expensive)
    print("Computing silhouette scores...")
    overall_silhouette = silhouette_score(embedding, clusters)
    sample_silhouettes = silhouette_samples(embedding, clusters)

    print(f"Overall silhouette score: {overall_silhouette:.3f}")

    return ClusterData(
        x=x,
        clusters=clusters,
        sample_silhouettes=sample_silhouettes,
        overall_silhouette=overall_silhouette,
        n_clusters=n_clusters,
    )


def get_cluster_stats(data: ClusterData) -> dict:
    """Compute per-cluster statistics."""
    cluster_silhouettes = {}
    cluster_sizes = {}

    for cluster_id in range(data.n_clusters):
        mask = data.clusters == cluster_id
        cluster_silhouettes[cluster_id] = data.sample_silhouettes[mask].mean()
        cluster_sizes[cluster_id] = mask.sum()

    # Sort by silhouette (highest first)
    sorted_clusters = sorted(cluster_silhouettes.items(), key=lambda x: x[1], reverse=True)

    return {
        "cluster_silhouettes": cluster_silhouettes,
        "cluster_sizes": cluster_sizes,
        "sorted_clusters": sorted_clusters,
    }


def print_cluster_examples(
    data: ClusterData,
    cluster_id: int,
    tokenizer: CharTokenizer,
    n_examples: int = 10,
    context_chars: int = 40,
) -> None:
    """Print example sequences from a cluster."""
    _, seq_len = data.x.shape
    cluster_mask = data.clusters == cluster_id
    cluster_indices = np.where(cluster_mask)[0]

    n_examples = min(n_examples, len(cluster_indices))
    example_indices = np.random.choice(cluster_indices, size=n_examples, replace=False)

    sil = data.sample_silhouettes[cluster_mask].mean()
    print(f"\nCluster {cluster_id} (silhouette: {sil:.3f}, size: {cluster_mask.sum()}):")

    for flat_idx in example_indices:
        batch_idx = flat_idx // seq_len
        pos_idx = flat_idx % seq_len
        start_pos = max(0, pos_idx - context_chars)
        context_tokens = data.x[batch_idx, start_pos : pos_idx + 1].tolist()
        context_text = tokenizer.decode(context_tokens)
        print(f"  {repr(context_text)}")


def print_all_clusters(data: ClusterData, tokenizer: CharTokenizer, top_n: int | None = None) -> None:
    """Print examples from all clusters, sorted by silhouette score."""
    stats = get_cluster_stats(data)
    sorted_clusters = stats["sorted_clusters"]

    if top_n is not None:
        sorted_clusters = sorted_clusters[:top_n]

    print(f"\n{'ID':>3} {'Sil':>6} {'Size':>6}")
    for cluster_id, sil_score in sorted_clusters:
        size = stats["cluster_sizes"][cluster_id]
        print(f"{cluster_id:>3} {sil_score:>6.3f} {size:>6}")

    for cluster_id, _ in sorted_clusters:
        print_cluster_examples(data, cluster_id, tokenizer, n_examples=5)


# %% Cluster interpretations
# Interpretations for all clusters with identifiable patterns
# Format: cluster_id -> (pattern_tuple, description)
CLUSTER_INTERPRETATIONS = {
    # === High confidence (F1 >= 0.95) ===
    32: ((" and", "-and", "\nAnd", ". And"), "ends ' and'"),  # F1=0.980
    3: (
        (" tha", " Tha", "\nWha", " wha", "\nTha", ". Wha", ". Tha", "? Wha"),
        "ends ' tha'/wha (before 't')",
    ),  # F1=0.982
    22: ((" o",), "ends ' o' (before 'f'/'n'/'r')"),  # F1=0.964
    17: ((" an", " An", "-an", "\nAn"), "ends ' an'"),  # F1=0.961
    5: ((" th", " Th", "\nTh"), "ends ' th' (before 'e'/'at')"),  # F1=0.958
    16: ((" i",), "ends ' i' (lowercase)"),  # F1=0.971
    # === Good confidence (F1 0.90-0.95) ===
    10: ((" the", " The", "The", "\nThe"), "ends ' the'"),  # F1=0.940
    25: ((" to", " To", "\nTo"), "ends ' to'"),  # F1=0.938
    9: ((",", "."), "sentence/clause boundary (,/.)"),  # F1=0.895
    34: ((" yo", " y", "'y"), "ends ' yo'/y (before 'u')"),  # F1=0.897
    # === Medium confidence (F1 0.70-0.90) ===
    21: ((" t",), "ends ' t' (before various)"),  # F1=0.843
    12: (("ou",), "ends 'ou' (you/our/out)"),  # F1=0.768
    42: ((" re", " se", " ye", " te", " ne", " e"), "ends ' Xe' (re/se/te...)"),  # F1=0.742
    48: ((" a", "-a"), "ends ' a' (article position)"),  # F1=0.738
    29: ((" w", " v"), "ends ' w'/v"),  # F1=0.724
    45: ((". ", ".\n"), "end of sentence + space"),  # F1=0.722
    14: ((" s",), "ends ' s' (before vowels)"),  # F1=0.688
    # === Lower confidence (F1 0.50-0.70) ===
    35: ((" for", " of", " in"), "ends preposition (for/of/in)"),  # F1=0.630
    7: ((" wo", " co", " po", " ro", " mo"), "ends ' Xo' (wo/co/po...)"),  # F1=0.557
    46: ((" we", " wi", " wa", " wh"), "ends ' wX'"),  # F1=0.546
    33: ((" a",), "ends ' a' (lower precision)"),  # F1=0.534
    30: (("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"), "ends with digit"),  # F1=0.529
    # === Complex/residual patterns (no simple mask) ===
    41: (None, "space before content word"),
    8: (None, "space + determiner patterns"),
    37: (None, "space patterns"),
    1: (None, "large residual cluster"),
    11: (None, "mixed suffixes (-ing, -ed, -er, -ly)"),
    4: (None, "space + capital letter"),
    26: (None, "mixed consonant patterns"),
    24: (None, "space patterns"),
    49: (None, "space + clause patterns"),
    28: (None, "mixed patterns"),
    36: (None, "mixed patterns"),
    40: (None, "space + adjective patterns"),
    31: (None, "copula/auxiliary (is/was/has)"),
    2: (None, "newline patterns"),
    43: (None, "mixed patterns"),
    18: (None, "ends 't' (mixed)"),
    6: (None, "large residual cluster"),
    47: (None, "preposition patterns"),
    20: (None, "mixed patterns"),
    13: (None, "mixed patterns"),
    # Remaining clusters (0, 15, 19, 23, 27, 38, 39, 44) have negative silhouette - poorly defined
}


def test_interpretations(data: ClusterData, tokenizer: CharTokenizer) -> None:
    """Test all cluster interpretations and print results."""
    print("\n=== Cluster Interpretation Results ===")
    print(f"{'ID':>3} {'F1':>6} {'TPR':>6} {'Prec':>6} {'Description'}")
    print("-" * 60)

    for cluster_id, (pattern, desc) in sorted(CLUSTER_INTERPRETATIONS.items()):
        if pattern is None:
            print(f"{cluster_id:>3} {'N/A':>6} {'N/A':>6} {'N/A':>6} {desc}")
            continue
        mask = ending_mask(data.x, pattern, tokenizer)
        stats = mask_predicts_cluster(mask, data.clusters, cluster_id)
        print(f"{cluster_id:>3} {stats.f1:>6.3f} {stats.tpr:>6.3f} {stats.precision:>6.3f} {desc}")


def compute_linearity_stats(
    acts: np.ndarray,
    contributions: np.ndarray,
    clusters: np.ndarray,
    n_clusters: int,
) -> dict[int, dict[str, float]]:
    """Compute local linearity statistics for each cluster.

    For each cluster, fits a linear regression from input activations to output
    contribution and computes R² and residual variance.

    Args:
        acts: Input activations, shape (n_samples, n_features)
        contributions: Output contributions, shape (n_samples,)
        clusters: Cluster assignments, shape (n_samples,)
        n_clusters: Number of clusters

    Returns:
        Dict mapping cluster_id -> {var, r2, resid_var}
    """
    from sklearn.linear_model import Ridge

    results = {}
    for cluster_id in range(n_clusters):
        mask = clusters == cluster_id
        if mask.sum() < 10:  # Skip tiny clusters
            results[cluster_id] = {"var": 0, "r2": 0, "resid_var": 0}
            continue

        cluster_acts = acts[mask]
        cluster_contrib = contributions[mask]

        # Variance of contributions in this cluster
        var = float(np.var(cluster_contrib))

        # Linear regression: acts -> contribution
        # Use Ridge to handle potential collinearity
        reg = Ridge(alpha=0.01)
        reg.fit(cluster_acts, cluster_contrib)
        pred = reg.predict(cluster_acts)

        # R² and residual variance
        ss_res = np.sum((cluster_contrib - pred) ** 2)
        ss_tot = np.sum((cluster_contrib - cluster_contrib.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0
        resid_var = float(ss_res / len(cluster_contrib))

        results[cluster_id] = {"var": var, "r2": max(0, r2), "resid_var": resid_var}

    return results


def print_cluster_table(
    data: ClusterData,
    tokenizer: CharTokenizer,
    acts: np.ndarray | None = None,
    contributions: np.ndarray | None = None,
    interpretations: dict[int, tuple[tuple[str, ...] | None, str]] | None = None,
    show_errors: bool = False,
    n_error_examples: int = 3,
    context_chars: int = 25,
    sort_by: str = "silhouette",  # "silhouette", "r2", "size", "f1"
) -> None:
    """Print a rich table with cluster statistics and optional interpretations.

    Args:
        data: ClusterData with x, clusters, silhouettes
        tokenizer: CharTokenizer for decoding
        acts: Optional input activations for linearity analysis (n_samples, n_features)
        contributions: Optional output contributions for linearity analysis (n_samples,)
        interpretations: Optional dict mapping cluster_id -> (pattern_tuple, description).
            If None, only shows summary stats without F1/pattern columns.
        show_errors: Whether to show FP/FN example columns (requires interpretations)
        n_error_examples: Number of FP/FN examples to show
        context_chars: Characters of context for examples
        sort_by: Sort order - "silhouette", "r2", "size", or "f1"
    """
    from rich.console import Console
    from rich.table import Table

    console = Console(width=150)
    stats = get_cluster_stats(data)
    total_samples = len(data.clusters)
    _, seq_len = data.x.shape

    # Compute linearity stats if activations provided
    linearity_stats = None
    if acts is not None and contributions is not None:
        linearity_stats = compute_linearity_stats(acts, contributions, data.clusters, data.n_clusters)

    has_interpretations = interpretations is not None

    def decode_indices(indices: np.ndarray, n: int) -> list[str]:
        """Decode flat indices to text snippets."""
        examples = []
        for flat_idx in indices[:n]:
            batch_idx = flat_idx // seq_len
            pos_idx = flat_idx % seq_len
            start_pos = max(0, pos_idx - context_chars)
            tokens = data.x[batch_idx, start_pos : pos_idx + 1].tolist()
            text = tokenizer.decode(tokens)
            if len(text) > context_chars:
                text = "…" + text[-(context_chars - 1) :]
            examples.append(repr(text))
        return examples

    # Build table
    title = "Cluster Statistics" if not has_interpretations else "Cluster Interpretations"
    table = Table(title=title, show_lines=True, expand=True)
    table.add_column("ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("Sil", justify="right", no_wrap=True)
    table.add_column("%", justify="right", no_wrap=True)

    # Add interpretation columns only if interpretations provided
    if has_interpretations:
        table.add_column("F1", justify="right", no_wrap=True)
        table.add_column("TPR", justify="right", no_wrap=True)
        table.add_column("Prec", justify="right", no_wrap=True)
        table.add_column("Pattern", min_width=20)

    # Add linearity columns if data provided
    if linearity_stats is not None:
        table.add_column("Var", justify="right", no_wrap=True)
        table.add_column("R²", justify="right", no_wrap=True)
        table.add_column("ResVar", justify="right", no_wrap=True)

    # Add error columns if enabled (requires interpretations)
    if show_errors and has_interpretations:
        table.add_column("FP Examples", min_width=30)
        table.add_column("FN Examples", min_width=30)

    # Collect data for all clusters
    items = []

    # Determine which clusters to show
    if has_interpretations:
        cluster_ids = list(interpretations.keys())
    else:
        cluster_ids = list(range(data.n_clusters))

    for cluster_id in cluster_ids:
        sil = dict(stats["sorted_clusters"]).get(cluster_id, 0)
        size = stats["cluster_sizes"].get(cluster_id, 0)
        pct = 100 * size / total_samples
        lin = linearity_stats.get(cluster_id, {}) if linearity_stats else {}

        if has_interpretations:
            pattern, desc = interpretations[cluster_id]
            if pattern is None:
                items.append((cluster_id, sil, pct, None, None, desc, [], [], lin))
            else:
                mask = ending_mask(data.x, pattern, tokenizer)
                mask_flat = mask.flatten().numpy()
                in_cluster = data.clusters == cluster_id

                s = mask_predicts_cluster(mask, data.clusters, cluster_id)

                fp_indices = np.where(mask_flat & ~in_cluster)[0]
                fn_indices = np.where(~mask_flat & in_cluster)[0]

                fp_examples = decode_indices(fp_indices, n_error_examples) if show_errors else []
                fn_examples = decode_indices(fn_indices, n_error_examples) if show_errors else []

                items.append((cluster_id, sil, pct, s.f1, s, desc, fp_examples, fn_examples, lin))
        else:
            # No interpretations - just stats
            items.append((cluster_id, sil, pct, None, None, None, [], [], lin))

    # Sort based on sort_by parameter
    if sort_by == "silhouette":
        items.sort(key=lambda x: -x[1])  # sil descending
    elif sort_by == "r2":
        items.sort(key=lambda x: -x[8].get("r2", 0) if x[8] else 0)  # r2 descending
    elif sort_by == "size":
        items.sort(key=lambda x: -x[2])  # pct descending
    elif sort_by == "f1" and has_interpretations:
        items.sort(key=lambda x: (x[3] is None, -(x[3] or 0)))  # f1 descending, None at end

    for cluster_id, sil, pct, f1, s, desc, fp_ex, fn_ex, lin in items:
        row = [str(cluster_id), f"{sil:.2f}", f"{pct:.1f}"]

        # Add interpretation columns if present
        if has_interpretations:
            if f1 is None:
                row.extend(["[dim]—[/dim]", "[dim]—[/dim]", "[dim]—[/dim]", f"[dim]{desc or ''}[/dim]"])
            else:
                # Color F1 based on value
                if f1 >= 0.9:
                    f1_str = f"[green]{f1:.2f}[/green]"
                elif f1 >= 0.7:
                    f1_str = f"[yellow]{f1:.2f}[/yellow]"
                else:
                    f1_str = f"[red]{f1:.2f}[/red]"
                row.extend([f1_str, f"{s.tpr:.2f}", f"{s.precision:.2f}", desc])

        # Add linearity stats
        if linearity_stats is not None:
            if lin:
                # Color R² based on value (high = more linear)
                r2 = lin.get("r2", 0)
                if r2 >= 0.8:
                    r2_str = f"[green]{r2:.2f}[/green]"
                elif r2 >= 0.5:
                    r2_str = f"[yellow]{r2:.2f}[/yellow]"
                else:
                    r2_str = f"[red]{r2:.2f}[/red]"
                row.extend([f"{lin.get('var', 0):.3f}", r2_str, f"{lin.get('resid_var', 0):.3f}"])
            else:
                row.extend(["—", "—", "—"])

        # Add error examples
        if show_errors and has_interpretations:
            if f1 is None:
                row.extend(["", ""])
            else:
                fp_str = "\n".join(fp_ex) if fp_ex else "[dim]none[/dim]"
                fn_str = "\n".join(fn_ex) if fn_ex else "[dim]none[/dim]"
                row.extend([fp_str, fn_str])

        table.add_row(*row)

    console.print(table)


# %%
if __name__ == "__main__":
    # Compute and save (run once)
    # data = compute_clusters(batch_size=400, n_clusters=100)
    # data.save()

    # Load cached data
    data = ClusterData.load()
    tokenizer = CharTokenizer()

    print(f"Loaded {len(data.clusters)} samples, {data.n_clusters} clusters")
    print(f"Overall silhouette: {data.overall_silhouette:.3f}")

    # Get activations for linearity stats
    acts, contributions = get_acts_and_contributions(data)

    # Print cluster table (without interpretations - just stats)
    print_cluster_table(data, tokenizer, acts=acts, contributions=contributions, sort_by="r2")

    # To print with interpretations:
    # print_cluster_table(data, tokenizer, acts=acts, contributions=contributions,
    #                     interpretations=CLUSTER_INTERPRETATIONS, sort_by="f1")
