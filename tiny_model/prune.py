import torch
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

from analysis.char_classes import LogitLossFn
from analysis.dim_reduction import PLSRegression
from analysis.utils import CacheKey, flatten_keep_last, get_batch, to_numpy
from tiny_model.centralizer import Centralizer
from tiny_model.model import GPT
from tiny_model.prune_config import PruningConfig, Subspace


def find_pca_basis(model: GPT, key: CacheKey, loss_fn: LogitLossFn, batch_size: int = 50) -> Subspace:
    attribution_batch_x, _ = get_batch(batch_size=batch_size)  # todo allow customization of data path
    # attribution_vectors = get_resid_attribution_vectors(model, attribution_batch_x, key, loss_fn)

    model.set_lxt_enabled(True)
    model.eval()
    model.zero_grad()

    out = model(attribution_batch_x, cache_enabled=True, alphas_enabled=True)
    loss = loss_fn(out.logits)
    mask = loss > 2
    loss[mask].mean().backward()

    grad = out.cache.get_grad(key)
    assert grad is not None
    acts = out.cache.get_value(key)
    attributions = grad * acts
    attribution_vectors = flatten_keep_last(to_numpy(attributions))

    pca = PCA(n_components=50)
    pca.fit(attribution_vectors)
    return torch.from_numpy(pca.components_)


def get_pls_basis(model: GPT, key: CacheKey, loss_fn: LogitLossFn, batch_size: int = 50) -> Subspace:
    print(f"Getting PLS basis for {key}")
    pls_batch_x, _ = get_batch(batch_size=batch_size)  # todo allow customization of data path
    loss = to_numpy(loss_fn(model(pls_batch_x).logits).flatten())
    acts = model(pls_batch_x, cache_enabled=True).cache.get_value(key)
    acts = flatten_keep_last(to_numpy(acts))
    pls = PLSRegression(n_components=10, scale=False)
    pls.fit(acts, loss)
    return torch.from_numpy(pls.x_weights_.T)


def find_subspace(model: GPT, key: CacheKey, loss_fn: LogitLossFn, target_r2: float = 0.9) -> Subspace:
    x_fit, _ = get_batch(batch_size=200)
    centralizer = Centralizer()
    centralizer.fit(model(x_fit, cache_enabled=True).cache)
    # basis = get_pls_basis(model, key, loss_fn, batch_size=500)
    basis = find_pca_basis(model, key, loss_fn, batch_size=500)
    x, _ = get_batch(batch_size=500)
    x_prime, _ = get_batch(batch_size=500)

    with torch.no_grad():  # we need grad to find the basis but not for the rest of it
        x_prime_cache = model(x_prime, cache_enabled=True).cache
        original_logit_diffs = to_numpy(loss_fn(model(x).logits)).flatten()
        mask = original_logit_diffs > 2

        for n_components in range(1, 128):
            subspace = basis[:n_components]
            prune_config = PruningConfig(subspaces={key: subspace}, seqpos=None)
            hooks = prune_config.get_mean_hook_dict(x_prime_cache, centralizer)
            pruned_logit_diffs = to_numpy(loss_fn(model(x, hooks=hooks).logits)).flatten()

            r2 = r2_score(original_logit_diffs[mask], pruned_logit_diffs[mask])
            print(f"{key=} {n_components=} {r2=:.3f}")

            if r2 > target_r2:
                return subspace

    raise ValueError(f"No subspace found for {key} with r2 greater than {target_r2}")


if __name__ == "__main__":
    from analysis.utils import load_model
    from tiny_model.tokenizer import CharTokenizer

    model = load_model()
    tokenizer = CharTokenizer()

    pos_token = "."
    neg_token = " "
    pos_token_id = tokenizer.encode_one(pos_token)
    neg_token_id = tokenizer.encode_one(neg_token)
    loss_fn = lambda logits: logits[..., pos_token_id] - logits[..., neg_token_id]

    key = CacheKey("resid_final", None)
    subspace = find_subspace(model, key, loss_fn)
