"""Prepare FineWiki dataset for character-level language modeling.

script hangs due to https://github.com/huggingface/datasets/issues/7467"""

import json
import os
import string
from typing import cast

import numpy as np
from datasets import IterableDataset, load_dataset

# Configuration
N_TRAIN = 50_000
N_TEST = 5_000

# Fixed ASCII vocabulary
chars = sorted(set(string.printable))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
valid_chars = set(chars)

print(f"Vocab size: {len(chars)}")


def tokenize(text: str) -> list[int]:
    """Filter to ASCII and tokenize."""
    return [stoi[c] for c in text if c in valid_chars]


def get_dataset():
    dataset = cast(IterableDataset, load_dataset("HuggingFaceFW/finewiki", "en", split="train", streaming=True))
    dataset = dataset.shuffle(seed=42, buffer_size=500)  # type: ignore
    dataset = dataset.map(lambda x: {"ids": tokenize(x["text"])}, remove_columns=["text"])
    train = dataset.take(N_TRAIN)
    test = dataset.skip(N_TRAIN).take(N_TEST)
    return {"train": train, "val": test}


splits = get_dataset()
output_dir = os.path.dirname(__file__)
for split_name, examples in splits.items():
    all_tokens = np.concatenate([np.array(ex["ids"], dtype=np.uint16) for ex in examples])
    filename = os.path.join(output_dir, f"{split_name}.bin")
    all_tokens.tofile(filename)
    print(f"{split_name}: {len(all_tokens):,} tokens")

meta = {"vocab_size": len(chars), "itos": itos, "stoi": stoi}
with open(os.path.join(output_dir, "meta.json"), "w") as f:
    json.dump(meta, f)

print("Done!")
os._exit(0)
