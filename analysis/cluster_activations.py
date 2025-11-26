"""Clustering analysis of activations at resid_pre layer 3."""

# %%

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

from analysis.utils import flatten_keep_last, load_model, run_batches, to_numpy
from tiny_model.model import CacheKey
from tiny_model.tokenizer.char_tokenizer import CharTokenizer

# %%

model = load_model()
tokenizer = CharTokenizer()

key = CacheKey("resid_pre", 3)

with torch.no_grad():
    output, x, y = run_batches(
        model,
        num_batches=1,
        batch_size=200,
        cache_enabled=[key],
    )

# %%

# Get activations and flatten to (n_samples, 128)
all_acts = to_numpy(output.cache.get_value(key))
acts_flat = flatten_keep_last(all_acts)  # Shape: (batch_size * seq_len, 128)

print(f"Activations shape: {acts_flat.shape}")
print(f"Number of samples: {acts_flat.shape[0]}")
print(f"Dimensionality: {acts_flat.shape[1]}")

# %%

# # Determine optimal number of clusters using silhouette score
# n_clusters_range = range(10, 25)
# silhouette_scores = []

# print("Finding optimal number of clusters...")
# for n_clusters in track(n_clusters_range, description="Testing cluster counts"):
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#     labels = kmeans.fit_predict(acts_flat)
#     score = silhouette_score(acts_flat, labels, sample_size=min(10000, len(acts_flat)))
#     silhouette_scores.append(score)
#     print(f"n_clusters={n_clusters}, silhouette_score={score:.4f}")

# # Plot silhouette scores
# plt.figure(figsize=(10, 5))
# plt.plot(list(n_clusters_range), silhouette_scores, marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Silhouette score')
# plt.title('Silhouette score vs number of clusters')
# plt.grid(True)
# plt.tight_layout()

# %%

# Use the best number of clusters
best_n_clusters = 20
print(f"\nBest number of clusters: {best_n_clusters}")

# Perform final clustering
kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(acts_flat)

print("\nCluster distribution:")
unique, counts = np.unique(cluster_labels, return_counts=True)
for cluster_id, count in zip(unique, counts):
    print(f"  Cluster {cluster_id}: {count} samples ({100 * count / len(cluster_labels):.2f}%)")

# %%

# Visualize clusters using PCA for dimensionality reduction
pca = PCA(n_components=2)
acts_2d = pca.fit_transform(acts_flat)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(acts_2d[:, 0], acts_2d[:, 1], c=cluster_labels, cmap="tab20", alpha=0.6, s=1)
plt.colorbar(scatter, label="Cluster ID")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
plt.title(f"Activations at {key} clustered into {best_n_clusters} groups (PCA visualization)")
plt.tight_layout()

# %%

# Plot cluster centers in PCA space
centers_2d = pca.transform(kmeans.cluster_centers_)
plt.figure(figsize=(12, 8))
scatter = plt.scatter(acts_2d[:, 0], acts_2d[:, 1], c=cluster_labels, cmap="tab20", alpha=0.3, s=1)
plt.scatter(
    centers_2d[:, 0],
    centers_2d[:, 1],
    c="red",
    marker="X",
    s=200,
    edgecolors="black",
    linewidths=2,
    label="Cluster centers",
)
for i, center in enumerate(centers_2d):
    plt.annotate(f"{i}", center, fontsize=10, ha="center", va="center")
plt.colorbar(scatter, label="Cluster ID")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
plt.title(f"K-means cluster centers at {key}")
plt.legend()
plt.tight_layout()

# %%

# Compute per-sample silhouette scores
print("\nComputing silhouette scores...")
sample_silhouettes = silhouette_samples(acts_flat, cluster_labels)

# Compute per-cluster silhouette scores and sort by them
cluster_silhouettes = {}
for cluster_id in range(best_n_clusters):
    mask = cluster_labels == cluster_id
    cluster_silhouettes[cluster_id] = sample_silhouettes[mask].mean()

# Sort clusters by silhouette score (highest first = most coherent)
sorted_clusters = sorted(cluster_silhouettes.items(), key=lambda x: x[1], reverse=True)

print("\nCluster silhouette scores (sorted):")
for cluster_id, sil_score in sorted_clusters:
    print(f"  Cluster {cluster_id}: {sil_score:.3f}")

# %%

# Analyze cluster characteristics with example sequences (sorted by silhouette)
print("\nCluster characteristics (sorted by silhouette score):")

batch_size, seq_len = x.shape
context_chars = 40  # How many characters of context to show

for cluster_id, sil_score in sorted_clusters:
    cluster_mask = cluster_labels == cluster_id
    cluster_acts = acts_flat[cluster_mask]

    # Compute statistics
    mean_norm = np.linalg.norm(cluster_acts, axis=1).mean()
    std_norm = np.linalg.norm(cluster_acts, axis=1).std()

    # Distance to cluster center
    center = kmeans.cluster_centers_[cluster_id]
    distances = np.linalg.norm(cluster_acts - center, axis=1)

    print(f"\nCluster {cluster_id} (silhouette: {sil_score:.3f}):")
    print(f"  Size: {cluster_acts.shape[0]}")
    print(f"  Mean activation norm: {mean_norm:.4f} ± {std_norm:.4f}")
    print(f"  Mean distance to center: {distances.mean():.4f}")
    print(f"  Max distance to center: {distances.max():.4f}")

    # Get example sequences
    cluster_indices = np.where(cluster_mask)[0]
    n_examples = min(10, len(cluster_indices))
    example_indices = np.random.choice(cluster_indices, size=n_examples, replace=False)

    print("\n  Example sequences:")
    for flat_idx in example_indices:
        batch_idx = flat_idx // seq_len
        pos_idx = flat_idx % seq_len

        # Get context tokens
        start_pos = max(0, pos_idx - context_chars)
        context_tokens = x[batch_idx, start_pos : pos_idx + 1].tolist()
        context_text = tokenizer.decode(context_tokens)

        # Get next token
        if pos_idx == seq_len - 1:
            next_token_text = "[eos]"
        else:
            next_token = x[batch_idx, pos_idx + 1].item()
            next_token_text = tokenizer.decode_one(next_token)

        print(f"    {repr(context_text)} -> {repr(next_token_text)}")

# %%


