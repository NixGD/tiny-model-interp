"""Utilities for analyzing cluster quality and interpretability."""

from dataclasses import dataclass

import numpy as np
import torch

from tiny_model.tokenizer.char_tokenizer import CharTokenizer


@dataclass
class MaskClusterStats:
    """Stats for how well a mask predicts cluster membership."""
    tpr: float  # True Positive Rate (recall): P(mask=1 | in_cluster)
    tnr: float  # True Negative Rate (specificity): P(mask=0 | not_in_cluster)
    precision: float  # P(in_cluster | mask=1)
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int

    def __repr__(self) -> str:
        return (
            f"TPR={self.tpr:.3f} TNR={self.tnr:.3f} "
            f"Prec={self.precision:.3f} F1={self.f1:.3f} "
            f"(TP={self.tp}, FP={self.fp}, TN={self.tn}, FN={self.fn})"
        )


def mask_predicts_cluster(
    mask: torch.Tensor | np.ndarray,
    clusters: np.ndarray,
    cluster_idx: int,
) -> MaskClusterStats:
    """Evaluate how well a mask predicts membership in a specific cluster.

    Args:
        mask: Boolean mask, same shape as tokens or flattened (n_samples,)
        clusters: Cluster labels for each sample (flattened)
        cluster_idx: Which cluster to predict

    Returns:
        MaskClusterStats with TPR, TNR, precision, F1, and confusion matrix counts
    """
    # Flatten mask if needed
    if isinstance(mask, torch.Tensor):
        mask_flat = mask.flatten().numpy()
    else:
        mask_flat = mask.flatten()

    in_cluster = clusters == cluster_idx

    tp = int(np.sum(mask_flat & in_cluster))
    fp = int(np.sum(mask_flat & ~in_cluster))
    tn = int(np.sum(~mask_flat & ~in_cluster))
    fn = int(np.sum(~mask_flat & in_cluster))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    return MaskClusterStats(
        tpr=tpr, tnr=tnr, precision=precision, f1=f1,
        tp=tp, fp=fp, tn=tn, fn=fn
    )


def show_mask_errors(
    mask: torch.Tensor | np.ndarray,
    clusters: np.ndarray,
    cluster_idx: int,
    x: torch.Tensor,
    tokenizer: CharTokenizer,
    context_chars: int = 40,
    max_examples: int = 10,
) -> None:
    """Print false positives and false negatives for a mask predicting cluster membership.

    Args:
        mask: Boolean mask, same shape as x
        clusters: Cluster labels for each sample (flattened)
        cluster_idx: Which cluster to predict
        x: Token tensor (batch, seq_len)
        tokenizer: Tokenizer for decoding
        context_chars: How many chars of context to show
        max_examples: Max examples to show per category
    """
    if isinstance(mask, torch.Tensor):
        mask_flat = mask.flatten().numpy()
    else:
        mask_flat = mask.flatten()

    in_cluster = clusters == cluster_idx
    _, seq_len = x.shape

    fp_indices = np.where(mask_flat & ~in_cluster)[0]
    fn_indices = np.where(~mask_flat & in_cluster)[0]

    def decode_idx(flat_idx: int) -> str:
        batch_idx = flat_idx // seq_len
        pos_idx = flat_idx % seq_len
        start_pos = max(0, pos_idx - context_chars)
        context_tokens = x[batch_idx, start_pos : pos_idx + 1].tolist()
        return tokenizer.decode(context_tokens)

    print(f"False Positives (mask=1 but not in cluster {cluster_idx}): {len(fp_indices)}")
    for idx in fp_indices[:max_examples]:
        actual_cluster = clusters[idx]
        print(f"  [cluster {actual_cluster}] {repr(decode_idx(idx))}")
    if len(fp_indices) > max_examples:
        print(f"  ... and {len(fp_indices) - max_examples} more")

    print(f"\nFalse Negatives (mask=0 but in cluster {cluster_idx}): {len(fn_indices)}")
    for idx in fn_indices[:max_examples]:
        print(f"  {repr(decode_idx(idx))}")
    if len(fn_indices) > max_examples:
        print(f"  ... and {len(fn_indices) - max_examples} more")
