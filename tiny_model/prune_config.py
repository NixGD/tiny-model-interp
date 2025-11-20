import string
from abc import ABC, abstractmethod

import torch
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel
from torch import Tensor

from tiny_model.centralizer import Centralizer
from tiny_model.model import AlphaCache, CacheKey, HookFn
from tiny_model.tokenizer import CharTokenizer

type TokenInputTensor = Int[Tensor, "batch seqpos"]
type SeqposMaskTensor = Bool[Tensor, "batch seqpos"]

type Subspace = None | Float[Tensor, "n_components d_model"]  # tensor must have orthonormal columns


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


def project_to_subspace(x: Float[Tensor, "... d_model"], subspace: Subspace) -> Float[Tensor, "... d_model"]:
    """Return tensor of the same shape of x but with the activations outside the subspace set to 0."""
    return x if subspace is None else x @ subspace.T @ subspace


def get_replace_hook(hook_key: CacheKey, subspace: Subspace, cache: AlphaCache, centralizer: Centralizer) -> HookFn:
    """Returns a hook which replaces the activation outside of the subspace with the value from x' (taken from the cache)."""

    def hook(x: torch.Tensor) -> torch.Tensor:
        if subspace is None:
            return x

        x_centered = centralizer.center(x, hook_key)
        x_within_subspace = project_to_subspace(x_centered, subspace)
        other_act_centered = centralizer.center(cache.get_value(hook_key), hook_key)
        other_act_outside_subspace = other_act_centered - project_to_subspace(other_act_centered, subspace)
        recombined = x_within_subspace + other_act_outside_subspace
        return centralizer.uncenter(recombined, hook_key)

    return hook


class PruningConfig(BaseModel):
    subspaces: dict[CacheKey, Subspace]
    seqpos: SeqposMask | None

    def get_hook_dict(self, other_acts_cache: AlphaCache, centralizer: Centralizer) -> dict[CacheKey, HookFn]:
        if self.seqpos is not None:
            raise NotImplementedError("Seqpos masking not implemented yet")

        hooks = {
            key: get_replace_hook(key, subspace, other_acts_cache, centralizer)
            for key, subspace in self.subspaces.items()
        }
        return hooks
