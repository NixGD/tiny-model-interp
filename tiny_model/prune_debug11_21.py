"""Dimensionality reduction visualization of activations."""

# %%

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

from analysis.char_classes import LogitLossFn, create_char_classes
from analysis.utils import flatten_keep_last, get_batch, load_model, to_numpy
from tiny_model.model import CacheKey
from tiny_model.prune_config import PruningConfig
from tiny_model.tokenizer.char_tokenizer import CharTokenizer

# %%
# turn off grad
torch.set_grad_enabled(False)


model = load_model()
tokenizer = CharTokenizer()

CHAR_CLASSES = create_char_classes(tokenizer)


def get_pls_basis(loss_fn: LogitLossFn, key: CacheKey, n_components: int = 10, batch_size: int = 200) -> PLSRegression:
    x_batch, y_batch = get_batch(batch_size=batch_size)
    out = model(x_batch, targets=y_batch, cache_enabled=True)
    x = flatten_keep_last(to_numpy(out.cache.get_value(key)))
    y = to_numpy(loss_fn(out.logits)).flatten()
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(x, y)
    return pls


# %%

target_token = "."
negative_token = " "
target_token_id = tokenizer.encode_one(target_token)
negative_token_id = tokenizer.encode_one(negative_token)
loss_fn = lambda logits: logits[..., target_token_id] - logits[..., negative_token_id]

# %%

key = CacheKey("resid_final", None)
pls = get_pls_basis(loss_fn, key, n_components=10, batch_size=200)
# %%

batch_size = 100
x, _ = get_batch(batch_size=batch_size)
x_out = model(x, cache_enabled=True)
x_acts = x_out.cache.get_value(key)
x_acts_in_pls: np.ndarray = pls.transform(flatten_keep_last(to_numpy(x_acts)))  # type: ignore
x_logit_diff = loss_fn(x_out.logits)


# %%
def scatter(embedding: np.ndarray, predictions: np.ndarray, dims: tuple[int, int] = (0, 1)):
    plt.scatter(
        embedding[:, dims[0]],
        embedding[:, dims[1]],
        c=predictions,
        alpha=0.5,
        s=5,
        marker=".",
    )

    plt.xlabel(f"Component {dims[0]}")
    plt.ylabel(f"Component {dims[1]}")
    plt.colorbar(label="Loss")
    plt.tight_layout()


scatter(x_acts_in_pls, x_logit_diff.flatten(), dims=(0, 1))

# %%

x_prime, _ = get_batch(batch_size=batch_size)
x_prime_cache = model(x_prime, cache_enabled=True).cache

# %%

basis = torch.from_numpy(pls.x_weights_.T)
prune_config = PruningConfig(subspaces={key: basis}, seqpos=None)
hooks = prune_config.get_hook_dict(x_prime_cache)


# %%
transformed_logit_diff = loss_fn(model(x, hooks=hooks).logits)
x_prime_logit_diff = loss_fn(model(x_prime).logits)

plt.scatter(
    x_logit_diff.flatten(),
    transformed_logit_diff.flatten(),
    c=x_prime_logit_diff.flatten(),
    cmap="RdYlGn",
    s=5,
    marker=".",
)
plt.xlabel("Original Logit Diff")
plt.ylabel("Transformed Logit Diff")
plt.title("Logit Diff Comparison")
plt.tight_layout()

# %%

torch.set_grad_enabled(True)
model.zero_grad()

out = model(x, cache_enabled=True, alphas_enabled=True)
loss = loss_fn(out.logits)

mask = loss > 3
loss[mask].mean().backward()

acts = out.cache.get_value(key)
alphas = out.cache.get_grad(key)
attrs = alphas * acts

attrs_norm = torch.norm(attrs, dim=-1)


# %%
def cosine_sim_matrix(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Gets cosine similarity along the last axis of x1 and x2"""
    return np.dot(x1, x2.T) / (np.linalg.norm(x1, axis=-1)[:, None] * np.linalg.norm(x2, axis=-1))


def plot_cosine_sim(x: np.ndarray):
    matrix = cosine_sim_matrix(x, x)
    plt.imshow(matrix, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(label="Cosine Similarity")
    plt.tight_layout()


all_attrs_np = flatten_keep_last(to_numpy(attrs))
attrs_at_mask_np = flatten_keep_last(to_numpy(attrs[mask]))
plot_cosine_sim(attrs_at_mask_np[:20])

# %%

from analysis.utils import variance_curve_pca

n_components = 20
attrs_at_mask_variance = variance_curve_pca(attrs_at_mask_np, n_components)
all_attrs_variance = variance_curve_pca(all_attrs_np, n_components)

plt.plot(attrs_at_mask_variance, label="Attrs at Mask")
plt.plot(all_attrs_variance, label="All Attrs")
plt.legend()
plt.tight_layout()
plt.ylim(0, 1)


# %%
pca = PCA(n_components=10)
pca.fit(attrs_at_mask_np)
attrs_in_pca = pca.transform(attrs_at_mask_np)

# %%
attrs_in_pca


# %%


# problem -- seems somewhat confused where we are doing the attributions to.
