"""Variance explained metrics for dimensionality reduction."""

from typing import NamedTuple

import numpy as np
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from analysis.char_classes import LogitLossFn
from analysis.utils import to_numpy
from tiny_model.model import GPT, CacheKey
from tiny_model.prune_config import get_permute_hook


class R2Result(NamedTuple):
    r2: float
    residuals: np.ndarray  # per-datapoint absolute errors


def pca_r2_of_x(x: np.ndarray, n_components: int = 10) -> list[float]:
    pca = PCA(n_components=n_components)
    pca.fit(x)
    return [0] + np.cumsum(pca.explained_variance_ratio_).tolist()


def pls_r2_of_y(x: np.ndarray, y: np.ndarray, n_components: int = 10) -> list[R2Result]:
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(x, y)
    acts_centered = x - pls._x_mean
    y_flat = y.flatten()

    # Baseline (0 components): predict mean, r2=0
    baseline_residuals = np.abs(y_flat - np.mean(y_flat))
    results = [R2Result(r2=0.0, residuals=baseline_residuals)]

    def get_result(n_components: int) -> R2Result:
        """Get the predictions and residuals from PLS components."""
        coef_n = (pls._y_std / pls._x_std) * (pls.y_loadings_[:, :n_components] @ pls.x_rotations_[:, :n_components].T)
        y_pred = (acts_centered @ coef_n.T).flatten() + pls._y_mean
        residuals = np.abs(y_flat - y_pred)
        return R2Result(r2=r2_score(y, y_pred), residuals=residuals)

    results.extend([get_result(n_comp) for n_comp in range(1, n_components + 1)])
    return results


def pca_r2_of_y(x: np.ndarray, y: np.ndarray, n_components: int = 10) -> list[R2Result]:
    y_flat = y.flatten()

    # Baseline (0 components): predict mean, r2=0
    baseline_residuals = np.abs(y_flat - np.mean(y_flat))
    results = [R2Result(r2=0.0, residuals=baseline_residuals)]

    def _get_result(n_components: int) -> R2Result:
        pca = PCA(n_components=n_components)
        x_in_pca = pca.fit_transform(x)
        reg = LinearRegression()
        reg.fit(x_in_pca, y)
        y_pred = reg.predict(x_in_pca)
        residuals = np.abs(y_flat - y_pred.flatten())
        return R2Result(r2=reg.score(x_in_pca, y), residuals=residuals)

    results.extend([_get_result(n_comp) for n_comp in range(1, n_components + 1)])
    return results


@torch.no_grad()
def get_permute_curve(
    model: GPT,
    x: torch.Tensor,
    mask: torch.Tensor,
    key: CacheKey,
    n_components: int | list[int],
    loss_fn: LogitLossFn,
) -> list[R2Result]:
    # Run model with caching to get activations and logits
    output = model(x, cache_enabled=[key])
    acts = output.cache.get_value(key)[mask].cpu().numpy()
    logit_diff = loss_fn(output.logits)[mask].cpu().numpy()
    original_logit_diff = logit_diff

    # get pls subspace for pruning
    max_components = max(n_components) if isinstance(n_components, list) else n_components
    pls = PLSRegression(n_components=max_components)
    pls.fit(acts, logit_diff)
    subspace = torch.from_numpy(pls.x_weights_.T)

    # get permute-ablation pruning curve
    n_list = n_components if isinstance(n_components, list) else range(1, n_components + 1)
    results = []
    for n in n_list:
        if n == 0:
            y_pred = np.mean(logit_diff)
            residuals = np.abs(original_logit_diff - y_pred)
            results.append(R2Result(r2=0.0, residuals=residuals))
            continue

        subspace_n = subspace[:n]
        hook = get_permute_hook(key, subspace_n, mask)
        logits = model(x, hooks={key: hook}).logits
        loss = to_numpy(loss_fn(logits)[mask])
        r2 = r2_score(original_logit_diff, loss)
        residuals = np.abs(original_logit_diff - loss)
        results.append(R2Result(r2=r2, residuals=residuals))
    return results
