"""Functions for creating masks over token sequences."""

import re

import torch

from tiny_model.tokenizer.char_tokenizer import CharTokenizer


def get_mask_y_token(y: torch.Tensor, allowed_tokens: list[str], tokenizer: CharTokenizer) -> torch.Tensor:
    allowed_token_ids = [tokenizer.encode_one(token) for token in allowed_tokens]
    return torch.isin(y, torch.tensor(allowed_token_ids))


def ending_mask(tokens: torch.Tensor, allowed_endings: str | tuple[str, ...], tokenizer: CharTokenizer) -> torch.Tensor:
    n_batch, n_seq = tokens.shape
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    for batch_idx in range(n_batch):
        decoded = tokenizer.decode(tokens[batch_idx].tolist())
        for seq_idx in range(n_seq):
            if decoded[: seq_idx + 1].endswith(allowed_endings):
                mask[batch_idx, seq_idx] = True
    return mask


def regex_mask(tokens: torch.Tensor, regex: str, tokenizer: CharTokenizer) -> torch.Tensor:
    n_batch, n_seq = tokens.shape
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    for batch_idx in range(n_batch):
        decoded = tokenizer.decode(tokens[batch_idx].tolist())
        for seq_idx in range(n_seq):
            if re.fullmatch(regex, decoded[: seq_idx + 1]):
                mask[batch_idx, seq_idx] = True
    return mask
