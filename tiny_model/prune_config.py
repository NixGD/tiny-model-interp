import string
from abc import ABC, abstractmethod
from typing import cast

import torch
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel
from torch import Tensor

from tiny_model.centralizer import Centralizer
from tiny_model.model import AlphaCache, CacheKey, HookFn
from tiny_model.tokenizer import CharTokenizer

type TokenInputTensor = Int[Tensor, "batch seqpos"]
type SeqposMaskTensor = Bool[Tensor, "batch seqpos"]

type Subspace = Float[Tensor, "n_components d_model"]  # tensor must have orthonormal columns


class SeqposMask(ABC):
    @abstractmethod
    def __call__(self, input: TokenInputTensor, seqpos: int) -> SeqposMaskTensor: ...


class WordMask(SeqposMask):
    """The mask is true for all tokens in the same word (ascii letters) and the word-boundary token immediately before."""

    def __init__(self, tokenizer: CharTokenizer):
        self.ascii_letters_token_ids = tokenizer.encode(string.ascii_letters)

    def __call__(self, input: torch.Tensor, seqpos: int) -> torch.Tensor:
        mask = torch.zeros_like(input, dtype=torch.bool)
        is_letter_mask = torch.isin(input, torch.tensor(self.ascii_letters_token_ids))

        for batch_idx in range(input.shape[0]):
            for i in range(seqpos, 0, -1):
                mask[batch_idx, i] = True
                if not is_letter_mask[batch_idx, i]:
                    break
        return mask


def get_replace_hook(hook_key: CacheKey, subspace: Subspace, cache: AlphaCache, centralizer: Centralizer) -> HookFn:
    """Returns a hook which replaces the activation outside of the subspace with the value from x' (taken from the cache)."""

    x_prime = cache.get_value(hook_key)

    def project(x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        _subspace = subspace.to(x)
        return x @ _subspace.T @ _subspace

    def hook(x: torch.Tensor) -> torch.Tensor:
        """We want to replace the value of x with the value of x' outside of our subspace.

        This is equal to P(x) + (x' - P(x')) = P(x - x') + x'
        """
        return project(x - x_prime) + x_prime

    return hook


class Sentinel:
    pass


class PruningConfig(BaseModel, arbitrary_types_allowed=True):
    subspaces: dict[CacheKey, Subspace | None]
    seqpos: SeqposMask | None

    def __or__(self, other: "PruningConfig") -> "PruningConfig":
        """Merge two PruningConfig instances. The right operand overwrites the left for same keys and seqpos."""
        merged_subspaces = {**self.subspaces, **other.subspaces}
        merged_seqpos = other.seqpos or self.seqpos  # default to other.seqpos if non-None
        return PruningConfig(subspaces=merged_subspaces, seqpos=merged_seqpos)

    def with_prunes(
        self, subspaces: dict[CacheKey, Subspace] | None = None, seqpos: SeqposMask | None | type[Sentinel] = Sentinel
    ) -> "PruningConfig":
        """Create a new PruningConfig with updated subspaces and/or seqpos. Overwrites existing values."""
        updated_subspaces = {**self.subspaces}
        if subspaces is not None:
            updated_subspaces.update(subspaces)
        updated_seqpos = seqpos if seqpos is not Sentinel else self.seqpos
        return PruningConfig(subspaces=updated_subspaces, seqpos=cast(SeqposMask | None, updated_seqpos))

    def get_hook_dict(self, other_acts_cache: AlphaCache, centralizer: Centralizer) -> dict[CacheKey, HookFn]:
        if self.seqpos is not None:
            raise NotImplementedError("Seqpos masking not implemented yet")

        hooks = {
            key: get_replace_hook(key, subspace, other_acts_cache, centralizer)
            for key, subspace in self.subspaces.items()
            if subspace is not None
        }
        return hooks
