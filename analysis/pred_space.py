"""Dimensionality reduction visualization of activations."""

# %%

import string

import matplotlib.pyplot as plt
import torch
from jaxtyping import Float
from rich.progress import track
from torch import Tensor

from analysis.char_classes import LogitLossFn, LogitTensor
from analysis.dim_reduction import visualize_activations_2d_interactive
from analysis.masks import ending_mask
from analysis.utils import get_most_common_words, load_model, run_batches, to_numpy
from analysis.var_explained import get_permute_curve, pca_r2_of_y, pls_r2_of_y
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

key = CacheKey("resid_mid", 3)

with torch.no_grad():
    output, x, y = run_batches(
        model,
        num_batches=1,
        batch_size=200,
        cache_enabled=[key],
    )

all_acts = to_numpy(output.cache.get_value(key))

space_tok_idx = tokenizer.encode_one(" ")
# loss_fn = lambda logits: logits[..., space_tok_idx]
loss_fn = get_logit_diff_loss([" "], list(string.ascii_uppercase) + list(string.ascii_lowercase))
all_loss = to_numpy(loss_fn(output.logits))


# %%
masks = {}
masks["all"] = torch.ones_like(y).bool()
masks["remainder"] = masks["all"].clone()

# Get most common words from the dataset
# Adjust n_words to scale (e.g., 1000 for more words)
words = get_most_common_words(n_words=1000, n_samples=10_000)
# words = [" " + word for word in words]

endings = words + [",", ".", " "] + [" " + letter for letter in string.ascii_lowercase + string.ascii_uppercase]

for ending in endings:
    mask_name = repr(ending)
    mask = ending_mask(x, ending, tokenizer)
    if mask.sum() > 5:
        masks[mask_name] = ending_mask(x, ending, tokenizer)
        masks["remainder"] = masks["remainder"] & ~masks[mask_name]

for mask_name, mask in masks.items():
    print(f"{mask_name}: {mask.sum()}/{mask.numel()}")

# %%

visualize_activations_2d_interactive(
    output,
    x,
    key,
    loss_fn,
    tokenizer,
    method="pls",
    title=f"Activations of {key} (logit diff: uppercase - lowercase)",
    context_chars=20,
    dims=(0, 1),
    mask=masks["remainder"],
    cmap="RdYlGn",
)


# %%
n_components = 15
show_permute_curve = False
measure_permute_every = 10

masks_to_plot = ["all", "remainder", repr(","), repr("."), repr(" "), repr("the"), repr(" W")]

n_cols = 3
n_rows = len(masks_to_plot) // n_cols + (len(masks_to_plot) % n_cols > 0)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 4 * n_rows), sharey=True)

for i, mask_name in track(enumerate(masks_to_plot), total=len(masks_to_plot), description="Plotting masks"):
    mask = masks[mask_name]
    ax = axes.flat[i]
    mask_np = to_numpy(mask)
    acts = all_acts[mask_np]
    logit_diff = all_loss[mask_np]

    assert mask.sum() > 0, f"Mask {mask_name} has no elements"

    _n_components = min(n_components, len(acts))

    # pca_var_exp = pca_r2_of_x(acts, _n_components)

    pls_results = pls_r2_of_y(acts, logit_diff, _n_components)
    pca_r2_of_y_results = pca_r2_of_y(acts, logit_diff, _n_components)

    # ax.plot(pca_var_exp, label="PCA R² of X", linestyle="--", marker="o")
    ax.plot([r.residuals.mean() for r in pls_results], label="PLS mean residual", marker="o")
    ax.plot([r.residuals.mean() for r in pca_r2_of_y_results], label="PCA mean residual", marker="o")

    if show_permute_curve:
        permute_ns = range(0, _n_components + 1, measure_permute_every)
        permute_r2_scores = get_permute_curve(
            model=model, x=x, mask=mask, key=key, n_components=list(permute_ns), loss_fn=loss_fn
        )
        ax.plot(
            permute_ns,
            [r.residuals.mean() for r in permute_r2_scores],
            label="Permute mean residual",
            marker="o",
            color="black",
        )

    ax.set_title(f"{mask_name}\n(n={len(acts)})")

    if i == 0:
        # get handles
        handles = ax.get_lines()
        labels = [line.get_label() for line in handles]
        axes.flat[-1].legend(handles, labels)

plt.ylim(0, None)
plt.tight_layout()

# %%
"""
So it kinda works.
But "ends with t, predict caps vs lowercase" is a bit weird.
Really it's measuring something like... confidence in the next token?
which is perfectly reasonable, in practice we do see that doing replacments within
this subclass is messy.

It might also be b/c the logit diff metric is just bad.
Like isn't well suited for 'capturing the behavior of interest'...

alas, task decomposition is hard.

I think it's basically correct to say, look there's a circuit in this model that predicts ' wit' -> 'h'. You can describe it if you like or you can ignore it.
"""



