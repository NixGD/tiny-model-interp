"""Character class probability visualization."""

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis.char_classes import CharClass, create_char_classes
from analysis.utils import get_batch, load_model
from tiny_model.char_tokenizer import CharTokenizer
from tiny_model.utils import REPO_ROOT


def visualize_char_class_probs(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    char_classes: Iterable[CharClass],
    tokenizer: CharTokenizer,
    num_tokens: int = 25,
    batch_index: int = 0,
    save_path: str | None = "char_class_probs.png",
    figsize: tuple[int, int] = (15, 3),
) -> np.ndarray:
    """Visualize character class probabilities across tokens as a heatmap.

    Args:
        logits: model output logits of shape (batch_size, seq_len, vocab_size)
        tokens: input tokens of shape (batch_size, seq_len)
        char_classes: iterable of CharClass instances to visualize
        tokenizer: tokenizer for decoding tokens
        num_tokens: number of tokens to visualize
        batch_index: which sequence in the batch to visualize
        save_path: path to save the plot (None to skip saving)
        figsize: figure size as (width, height)

    Returns:
        class_probs_matrix: numpy array of shape (num_classes, num_tokens)
    """
    # Compute probabilities for all classes (vectorized over sequence dimension)
    class_probs_matrix = []
    for char_class in char_classes:
        probs = char_class.get_probabilities(logits[batch_index, :num_tokens])
        class_probs_matrix.append(probs.detach().cpu().numpy())

    class_probs_matrix = np.array(class_probs_matrix)
    class_names = [char_class.name for char_class in char_classes]

    # Get the actual characters for labeling
    token_list = tokens[batch_index, :num_tokens].tolist()
    chars = [tokenizer.decode_one(t) for t in token_list]

    # Create visualization
    plt.figure(figsize=figsize)
    im = plt.matshow(class_probs_matrix, cmap="viridis", aspect="auto", fignum=1)
    plt.colorbar(im, label="Probability")
    plt.yticks(range(len(class_names)), class_names)
    plt.xticks(range(num_tokens), [repr(c) for c in chars], rotation=90, fontsize=8)
    plt.xlabel("Token position")
    plt.ylabel("Character class")
    plt.title(f"Character class probabilities across {num_tokens} tokens")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n✓ Saved heatmap to: {save_path}")
        print(f"  Sequence: {repr(tokenizer.decode(token_list))}")

    return class_probs_matrix


if __name__ == "__main__":
    # Load model
    model_path = REPO_ROOT / "out-wiki-char/ckpt.pt"
    model = load_model(str(model_path))
    model.eval()
    print(f"✓ Loaded model from {model_path}")

    # Load tokenizer and data
    tokenizer = CharTokenizer(vocab_path="data/wiki_char/vocab.json")
    data_dir = REPO_ROOT / "data/wiki_char"
    val_data = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
    print(f"✓ Loaded validation dataset: {len(val_data):,} tokens")

    # Create character classes
    CHAR_CLASSES = create_char_classes(tokenizer)
    print("\n✓ Character classes defined:")
    for char_class in CHAR_CLASSES.values():
        print(f"  {char_class}")

    # Get batch and run model
    batch_size = 8
    block_size = model.config.block_size
    x_batch, y_batch = get_batch(val_data, batch_size, block_size)

    output = model(x_batch, targets=y_batch, cache_enabled=True, alphas_enabled=True)
    print("\n✓ Model output:")
    print(f"  Loss: {output.loss.item():.4f}")
    print(f"  Output shape: {output.logits.shape}")

    # Visualize character class probabilities
    class_probs = visualize_char_class_probs(
        output.logits,
        x_batch,
        CHAR_CLASSES.values(),
        tokenizer,
        num_tokens=25,
        batch_index=0,
    )
