"""Prepare FineWeb dataset for character-level language modeling.

Downloads and processes a subset of the FineWeb dataset from HuggingFace.
"""

import json
import os
import string
from typing import cast

import numpy as np
from datasets import IterableDataset, load_dataset

from tiny_model.tokenizer.ascii_normalization import normalize_text

# Configuration
# Adjust these to control dataset size (aiming for ~100MB - 1GB total)
N_TRAIN = 100_000  # Number of training examples
N_TEST = 10_000  # Number of validation examples

# Fixed ASCII vocabulary
chars = sorted(set(string.printable))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
valid_chars = set(chars)

print(f"Vocab size: {len(chars)}")


def tokenize(text: str) -> list[int]:
    """Normalize text to ASCII, then filter and tokenize."""
    normalized = normalize_text(text)
    return [stoi[c] for c in normalized if c in valid_chars]


def get_dataset():
    """Load FineWeb dataset from HuggingFace."""
    print("Loading FineWeb dataset from HuggingFace (subset: sample-10BT)...")
    dataset = cast(IterableDataset, load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True))
    dataset = dataset.shuffle(seed=42, buffer_size=1000)  # type: ignore
    dataset = dataset.map(lambda x: {"ids": tokenize(x["text"])}, remove_columns=["text"])
    train = dataset.take(N_TRAIN)
    test = dataset.skip(N_TRAIN).take(N_TEST)
    return {"train": train, "val": test}


splits = get_dataset()
output_dir = os.path.dirname(__file__)
for split_name, examples in splits.items():
    print(f"Processing {split_name} split...")
    all_tokens = np.concatenate([np.array(ex["ids"], dtype=np.uint16) for ex in examples])
    filename = os.path.join(output_dir, f"{split_name}.bin")
    all_tokens.tofile(filename)
    size_mb = len(all_tokens) * 2 / (1024 * 1024)  # uint16 = 2 bytes
    print(f"{split_name}: {len(all_tokens):,} tokens ({size_mb:.1f} MB)")

meta = {"vocab_size": len(chars), "itos": itos, "stoi": stoi}
with open(os.path.join(output_dir, "meta.json"), "w") as f:
    json.dump(meta, f)
# Also create vocab.json for compatibility with analysis scripts
with open(os.path.join(output_dir, "vocab.json"), "w") as f:
    json.dump(meta, f)

print("Done!")
os._exit(0)
