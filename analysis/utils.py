"""Common utilities for model analysis."""

from pathlib import Path

import numpy as np
import torch

from tiny_model.char_tokenizer import CharTokenizer
from tiny_model.model import GPT, GPTConfig
from tiny_model.utils import REPO_ROOT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_path: str | Path) -> GPT:
    """Load a trained GPT model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model_args = checkpoint["model_args"]
    model = GPT(GPTConfig(**model_args))
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    return model


def get_batch(
    data: np.memmap,
    batch_size: int,
    block_size: int,
    tokenizer: CharTokenizer | None = None,
    exclude_char: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get a batch of data for training/evaluation.

    Args:
        data: memory-mapped data array
        batch_size: number of sequences to return
        block_size: sequence length
        tokenizer: optional tokenizer for filtering (required if exclude_char is set)
        exclude_char: optional character to exclude (e.g., "|" to filter markdown tables)

    Returns:
        x: input tokens of shape (batch_size, block_size)
        y: target tokens of shape (batch_size, block_size)
    """
    if exclude_char is not None and tokenizer is None:
        raise ValueError("tokenizer must be provided when exclude_char is set")

    x_list = []
    y_list = []

    # Keep sampling until we have enough valid sequences
    max_attempts = batch_size * 10  # Safety limit to avoid infinite loops
    attempts = 0

    while len(x_list) < batch_size and attempts < max_attempts:
        # Sample more sequences than needed to account for filtering
        n_needed = batch_size - len(x_list)
        oversample = max(n_needed * 2, 10)  # Sample extra to reduce iterations

        ix = torch.randint(len(data) - block_size, (oversample,))

        for i in ix:
            if len(x_list) >= batch_size:
                break

            x_seq = torch.from_numpy((data[i : i + block_size]).astype(np.int64))
            y_seq = torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))

            # Filter if exclude_char is specified
            if exclude_char is not None:
                text = tokenizer.decode(x_seq.tolist())
                if exclude_char in text:
                    continue  # Skip this sequence

            x_list.append(x_seq)
            y_list.append(y_seq)

        attempts += 1

    if len(x_list) < batch_size:
        raise RuntimeError(
            f"Could not find {batch_size} valid sequences after {max_attempts} attempts. "
            f"Only found {len(x_list)}. Try relaxing the filter or using a different dataset."
        )

    x = torch.stack(x_list)
    y = torch.stack(y_list)
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to numpy array."""
    return tensor.detach().cpu().numpy()
