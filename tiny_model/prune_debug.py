# %%
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA

from analysis.char_classes import LogitLossFn, create_char_classes
from analysis.utils import CacheKey, flatten_keep_last, get_batch, load_model, to_numpy
from tiny_model.centralizer import Centralizer
from tiny_model.lxt_attribution import get_resid_attribution_vectors
from tiny_model.model import GPT
from tiny_model.tokenizer import CharTokenizer


def get_pca(model: GPT, key: CacheKey, loss_fn: LogitLossFn, centralizer: Centralizer | None, batch_size: int = 50) -> PCA:
    attribution_batch_x, _ = get_batch(batch_size=batch_size)  # todo allow customization of data path
    attribution_vectors = get_resid_attribution_vectors(model, attribution_batch_x, key, loss_fn, centralizer)
    pca_input = flatten_keep_last(to_numpy(attribution_vectors))

    pca = PCA()
    pca.fit(pca_input)
    return pca


# %%
model = load_model()

# %%
centralizer = Centralizer()
x_fit, _ = get_batch(batch_size=100)
centralizer.fit(model(x_fit, cache_enabled=True).cache)

tokenizer = CharTokenizer()
char_class = create_char_classes(tokenizer)["uppercase"]
loss_fn = char_class.get_logit_diff

key = CacheKey("resid_mid", 3)

# %%
pca = get_pca(model, key, loss_fn, centralizer, batch_size=200)
pca_basis = torch.from_numpy(pca.components_)

# %%
x, _ = get_batch(batch_size=20)
x_out = model(x, cache_enabled=True)
x_cache = x_out.cache
x_loss = loss_fn(x_out.logits)

x_acts = x_cache.get_value(key)
x_acts_centered = centralizer.center(x_acts, key)
x_in_pca = x_acts_centered @ pca_basis.T

x_in_pca_np = flatten_keep_last(to_numpy(x_in_pca))
x_loss_np = to_numpy(x_loss).flatten()

# %%
plt.scatter(x_in_pca_np[:, 2], x_in_pca_np[:, 3], c=x_loss_np, cmap="RdYlGn")
plt.colorbar(label="Logit Diff")

# %%
x_flat_loss = x_loss.flatten()
shuffled_x_flat_loss = x_flat_loss[torch.randperm(x_flat_loss.shape[0])]
print((x_flat_loss - shuffled_x_flat_loss).abs().mean())
