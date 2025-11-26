from jaxtyping import Float
from torch import Tensor

from tiny_model.model import AlphaCache, CacheKey


class Centralizer:
    _mean_activations: dict[CacheKey, Float[Tensor, " d_model"]] = {}

    def fit(self, cache: AlphaCache) -> None:
        for key in cache.keys():  # noqa: SIM118
            acts = cache.get_value(key)
            acts = acts.reshape(-1, acts.shape[-1])
            self._mean_activations[key] = acts.mean(dim=0)

    def center(self, x: Float[Tensor, "... d_model"], key: CacheKey) -> Float[Tensor, "... d_model"]:
        return x - self._mean_activations[key]

    def uncenter(self, x: Float[Tensor, "... d_model"], key: CacheKey) -> Float[Tensor, "... d_model"]:
        return x + self._mean_activations[key]

    def mean_activation(self, key: CacheKey):
        return self._mean_activations[key]
