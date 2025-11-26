"""Dimensionality reduction visualization of activations."""

# %%

import string

import matplotlib.pyplot as plt
import torch
from jaxtyping import Float
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from torch import Tensor

from analysis.char_classes import LogitLossFn, LogitTensor
from analysis.masks import ending_mask, get_mask_y_token, regex_mask
from analysis.utils import flatten_keep_last, load_model, run_batches, to_numpy
from analysis.var_explained import get_permute_curve, pca_r2_of_x, pca_r2_of_y, pls_r2_of_y
from tiny_model.model import CacheKey
from tiny_model.tokenizer.char_tokenizer import CharTokenizer


# %%
def get_logit_diff_loss(positive_tokens: list[str], negative_tokens: list[str]) -> LogitLossFn:
    positive_token_ids = [tokenizer.encode_one(token) for token in positive_tokens]
    negative_token_ids = [tokenizer.encode_one(token) for token in negative_tokens]

    def loss_fn(logits: LogitTensor) -> Float[Tensor, "..."]:
        positive_logits = logits[..., positive_token_ids].mean(dim=-1)
        negative_logits = logits[..., negative_token_ids].mean(dim=-1)
        return positive_logits - negative_logits

    return loss_fn


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
masks = {
    "all": torch.ones_like(y).bool(),
    "next tok [a-zA-Z]": get_mask_y_token(y, list(string.ascii_uppercase) + list(string.ascii_lowercase), tokenizer),
    "next tok [A-Z]": get_mask_y_token(y, list(string.ascii_uppercase), tokenizer),
    "next tok [a-z]": get_mask_y_token(y, list(string.ascii_lowercase), tokenizer),
    "ends with '[.!?] '": ending_mask(x, (". ", "! ", "? "), tokenizer),
    "regex '[a-z] '": regex_mask(x, r"[a-z] ", tokenizer),
    "last tok [a-z]": ending_mask(x, tuple(string.ascii_lowercase), tokenizer),
    "last tok t": ending_mask(x, "t", tokenizer),
    "last tok b": ending_mask(x, "b", tokenizer),
    # "regex '[A-Z]{(2,)}'": regex_mask(x, r"[A-Z]{2,}", tokenizer),
}

loss_fn = get_logit_diff_loss(list(string.ascii_uppercase), list(string.ascii_lowercase))

all_acts = to_numpy(output.cache.get_value(key))
all_logit_diff = to_numpy(loss_fn(output.logits))

n_components = 100

fig, axes = plt.subplots(3, 3, figsize=(9, 8), sharey=True)

show_permute_curve = True
measure_permute_every = 20

for i, (mask_name, mask) in enumerate(masks.items()):
    ax = axes.flat[i]
    acts = all_acts[mask]
    logit_diff = all_logit_diff[mask]

    _n_components = min(n_components, len(acts))

    pca_var_exp = pca_r2_of_x(acts, _n_components)
    pls_results = pls_r2_of_y(acts, logit_diff, _n_components)
    pca_r2_of_y_results = pca_r2_of_y(acts, logit_diff, _n_components)

    ns = range(1, _n_components + 1)
    ax.plot(ns, pca_var_exp, label="PCA R² of X", linestyle="--", marker="o")
    ax.plot(ns, [r.r2 for r in pls_results], label="PLS R² of Y", marker="o")
    ax.plot(ns, [r.r2 for r in pca_r2_of_y_results], label="PCA R² of Y", marker="o")

    if show_permute_curve:
        permute_ns = range(1, _n_components + 1, measure_permute_every)
        permute_r2_scores = get_permute_curve(
            model=model, x=x, mask=mask, key=key, n_components=list(permute_ns), loss_fn=loss_fn
        )
        ax.plot(permute_ns, permute_r2_scores, label="Permute", marker="o")

    ax.set_title(f"{mask_name}\n(n={len(acts)})")

    if i == 0:
        # get handles
        handles = ax.get_lines()
        labels = [line.get_label() for line in handles]
        axes.flat[-1].legend(handles, labels)

plt.ylim(0, 1)
plt.tight_layout()
# %% [markdown]
"""
Do the masks correspond to clusters in the activations?
"""


def get_mask_prediction_r2(
    x,
    y,
):
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(x, y)
    acts_centered = x - pls._x_mean

    def get_r2_score(n_components: int) -> float:
        """Get the predictions from PLS components."""
        coef_n = (pls._y_std / pls._x_std) * (pls.y_loadings_[:, :n_components] @ pls.x_rotations_[:, :n_components].T)
        y_pred = (acts_centered @ coef_n.T).flatten() + pls._y_mean
        return r2_score(y, y_pred)

    return [get_r2_score(n_comp) for n_comp in range(1, n_components + 1)]


n_components = 25


ns = range(1, n_components + 1)
x = flatten_keep_last(all_acts)

for mask_name, mask in masks.items():
    r2 = [LogisticRegression()]

    results = pls_r2_of_y(x=flatten_keep_last(all_acts), y=to_numpy(mask.flatten()), n_components=n_components)
    plt.plot(ns, [r.r2 for r in results], label=mask_name, marker="o")

plt.legend()


# how easy it it to


# %%

fig, axes = plt.subplots(3, 3, figsize=(9, 8), sharey=True, sharex=True)

for i, (mask_name, mask) in enumerate(masks.items()):
    ax = axes.flat[i]
    acts = all_acts[mask]
    logit_diff = all_logit_diff[mask]

    ax.hist(logit_diff, bins=100, density=True)
    ax.set_title(f"{mask_name}\n(n={len(acts)})")

plt.ylim(0, 0.3)
plt.tight_layout()


# %%


fig, axes = plt.subplots(3, 3, figsize=(9, 8), sharey=True, sharex=True)

for i, (mask_name, mask) in enumerate(masks.items()):
    ax = axes.flat[i]
    acts = all_acts[mask]
    logit_diff = all_logit_diff[mask]

    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(acts, logit_diff)
    pred_logit_diff = pls.predict(acts)

    ax.scatter(logit_diff, pred_logit_diff, alpha=0.1, s=2, marker=".")
    ax.plot([-20, 20], [-20, 20], color="black")
    ax.set_title(f"{mask_name}\n(n={len(acts)})")

plt.tight_layout()
