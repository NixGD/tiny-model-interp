"""Simple character-level tokenizer."""

import json
import string
from pathlib import Path


class CharTokenizer:
    """Character-level tokenizer."""

    def __init__(self, vocab_path: str | None = None):
        if vocab_path and Path(vocab_path).exists():
            with open(vocab_path) as f:
                meta = json.load(f)
            self.stoi = meta["stoi"]
            self.itos = {int(k): v for k, v in meta["itos"].items()}
        else:
            chars = sorted(set(string.printable))
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        """Text to token IDs."""
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, tokens: list[int]) -> str:
        """Token IDs to text."""
        return "".join(self.itos.get(t, "") for t in tokens)

    def encode_one(self, text: str) -> int:
        assert len(text) == 1, "Only one character can be encoded at a time"
        return self.stoi[text]

    def decode_one(self, token: int) -> str:
        assert token < self.vocab_size, "Token out of range"
        return self.itos[token]

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)
