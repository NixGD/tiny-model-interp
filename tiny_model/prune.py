import torch
from sklearn.decomposition import PCA

from analysis.char_classes import CharClass, LogitLossFn
from analysis.dim_reduction import PLSRegression
from analysis.utils import CacheKey, flatten_keep_last, get_batch, to_numpy
from tiny_model.centralizer import Centralizer
from tiny_model.lxt_attribution import get_resid_attribution_vectors
from tiny_model.model import GPT
from tiny_model.prune_config import PruningConfig, Subspace


def find_pca_basis(
    model: GPT, key: CacheKey, loss_fn: LogitLossFn, centralizer: Centralizer | None, batch_size: int = 50
) -> Subspace:
    attribution_batch_x, _ = get_batch(batch_size=batch_size)  # todo allow customization of data path
    attribution_vectors = get_resid_attribution_vectors(model, attribution_batch_x, key, loss_fn, centralizer)
    pca_input = flatten_keep_last(to_numpy(attribution_vectors))

    pca = PCA(n_components=74)
    pca.fit(pca_input)
    return torch.from_numpy(pca.components_)


def get_pls_basis(
    model: GPT, key: CacheKey, loss_fn: LogitLossFn, centralizer: Centralizer, batch_size: int = 50
) -> Subspace:
    print(f"Getting PLS basis for {key}")
    pls_batch_x, _ = get_batch(batch_size=batch_size)  # todo allow customization of data path
    loss = to_numpy(loss_fn(model(pls_batch_x).logits).flatten())
    acts = model(pls_batch_x, cache_enabled=True).cache.get_value(key)
    acts = centralizer.center(acts, key)
    acts = flatten_keep_last(to_numpy(acts))
    pls = PLSRegression(n_components=20)
    pls.fit(acts, loss)
    return torch.from_numpy(pls.x_weights_.T)


def find_subspace(
    model: GPT, key: CacheKey, loss_fn: LogitLossFn, centralizer: Centralizer, target_delta: float = 0.1
) -> Subspace:
    basis = find_pca_basis(model, key, loss_fn, None, batch_size=500)
    # basis = get_pls_basis(model, key, loss_fn, centralizer, batch_size=500)
    x, _ = get_batch(batch_size=500)
    x_prime, _ = get_batch(batch_size=500)

    with torch.no_grad():  # we need grad to find the basis but not for the rest of it
        x_prime_cache = model(x_prime, cache_enabled=True).cache
        original_logit_diffs = loss_fn(model(x).logits)

        for n_components in range(1, 128):
            subspace = basis[:n_components]
            prune_config = PruningConfig(subspaces={key: subspace}, seqpos=None)
            hooks = prune_config.get_hook_dict(x_prime_cache, centralizer)
            pruned_logit_diffs = loss_fn(model(x, hooks=hooks).logits)

            mean_delta = (original_logit_diffs - pruned_logit_diffs).abs().mean()
            print(f"{key=} {n_components=} {mean_delta=:.3f}")

            if mean_delta < target_delta:
                return subspace

    raise ValueError(f"No subspace found for {key} with loss increment less than {target_delta}")


if __name__ == "__main__":
    from analysis.char_classes import create_char_classes
    from analysis.utils import load_model
    from tiny_model.tokenizer import CharTokenizer

    model = load_model()
    centralizer = Centralizer()

    x_fit, _ = get_batch(batch_size=500)
    centralizer.fit(model(x_fit, cache_enabled=True).cache)

    tokenizer = CharTokenizer()

    pos_token = "A"
    neg_token = "a"
    pos_token_id = tokenizer.encode_one(pos_token)
    neg_token_id = tokenizer.encode_one(neg_token)
    loss_fn = lambda logits: logits[..., pos_token_id] - logits[..., neg_token_id]

    subspace = find_subspace(model, CacheKey("resid_pre", 3), loss_fn, centralizer)
