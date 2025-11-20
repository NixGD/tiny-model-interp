from collections.abc import Callable

from jaxtyping import Float
from torch import Tensor

from tiny_model.centralizer import Centralizer
from tiny_model.model import GPT, CacheKey
from tiny_model.prune_config import TokenInputTensor

LogitLossFn = Callable[[Float[Tensor, "... vocab_size"]], Float[Tensor, "..."]]
AttributionVector = Float[Tensor, "... d_model"]


def get_resid_attribution_vectors(
    model: GPT, model_input: TokenInputTensor, key: CacheKey, loss_fn: LogitLossFn, centralizer: Centralizer | None
) -> AttributionVector:
    """For each batch/seqpos, returns a vector indiciating the direction of maxiumum attribution in the resid stream."""
    model.set_lxt_enabled(True)
    model.eval()
    model.zero_grad()

    out = model(model_input, cache_enabled=True, alphas_enabled=True)
    loss = loss_fn(out.logits).mean()
    loss.backward()

    grad = out.cache.get_grad(key)
    assert grad is not None
    acts = out.cache.get_value(key)
    if centralizer is not None:
        acts = centralizer.center(acts, key)
    attributions = grad * acts
    return attributions
