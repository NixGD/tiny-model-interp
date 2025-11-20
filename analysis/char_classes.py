"""Character class definitions for analysis."""

import string

import torch

from tiny_model.tokenizer.char_tokenizer import CharTokenizer


class CharClass:
    """A class representing a category of characters (e.g., numerals, punctuation)."""

    def __init__(self, name: str, chars: str, tokenizer: CharTokenizer):
        """Initialize a character class.

        Args:
            name: Name of this character class
            chars: String containing all characters in this class
            tokenizer: Tokenizer to use for getting indices
        """
        self.name = name
        self.chars = chars
        self.indices = self._get_indices(tokenizer)
        self.vocab_size = tokenizer.vocab_size

    def _get_indices(self, tokenizer: CharTokenizer) -> list[int]:
        """Get token indices for characters in this class."""
        return [tokenizer.stoi[c] for c in self.chars if c in tokenizer.stoi]

    def get_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute probability mass on this character class.

        Args:
            logits: model logits of shape (..., vocab_size)
                    Supports vectorization over batch and sequence dimensions

        Returns:
            probabilities: sum of probabilities for this class, shape (...)
                          Preserves all dimensions except the vocab dimension
        """
        probs = torch.softmax(logits, dim=-1)
        return probs[..., self.logit_mask].sum(dim=-1)

    def get_logit_diff(self, logits: torch.Tensor) -> torch.Tensor:
        """Difference between average logits of toks within class vs average logits of toks outside class."""
        mean_logits_within = logits[..., self.logit_mask].mean(dim=-1)
        mean_logits_outside = logits[..., ~self.logit_mask].mean(dim=-1)
        return mean_logits_within - mean_logits_outside

    @property
    def logit_mask(self) -> torch.Tensor:
        """Boolean mask of shape (..., vocab_size) indicating which tokens are in this class."""
        char_mask = torch.zeros(self.vocab_size).bool()
        char_mask[self.indices] = True
        return char_mask

    def __repr__(self) -> str:
        return f"CharClass({self.name!r}, {len(self.indices)} chars)"


def create_char_classes(tokenizer: CharTokenizer) -> dict[str, CharClass]:
    """Create standard character classes for analysis."""
    return {
        "numeral": CharClass("numeral", string.digits, tokenizer),
        "eos_punct": CharClass("eos_punct", ".!?", tokenizer),
        "whitespace": CharClass("whitespace", " \t\n", tokenizer),
        "uppercase": CharClass("uppercase", string.ascii_uppercase, tokenizer),
        "lowercase": CharClass("lowercase", string.ascii_lowercase, tokenizer),
    }
