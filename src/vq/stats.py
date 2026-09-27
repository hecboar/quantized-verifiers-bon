"""Paired inference with the question as the unit (plan §6)."""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np


def paired_bootstrap(
    per_question: Mapping[str, np.ndarray],
    contrast: Callable[[Mapping[str, np.ndarray]], np.ndarray],
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Percentile bootstrap of mean(contrast), resampling questions jointly
    across every condition in `per_question`."""
    lengths = {len(v) for v in per_question.values()}
    if len(lengths) != 1:
        raise ValueError("all per-question arrays must have the same length")
    n = lengths.pop()
    diff = np.asarray(contrast(per_question), dtype=float)
    if n == 0 or diff.size == 0:
        raise ValueError("empty contrast")
    if not np.isfinite(diff).all():
        raise ValueError("non-finite values in contrast")
    estimate = float(diff.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    # two-sided bootstrap p-value for H0: mean = 0
    centred = boots - estimate
    p = float(np.mean(np.abs(centred) >= abs(estimate))) if estimate != 0 else 1.0
    return {"estimate": estimate, "ci_low": float(lo), "ci_high": float(hi), "p_value": p, "n": n}


def holm(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj_sorted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    adj = np.empty(m)
    adj[order] = np.minimum(adj_sorted, 1.0)
    return adj


def tost_equivalence(diff: np.ndarray, margin: float, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Equivalence iff the (1-2*alpha) bootstrap CI lies inside [-margin, margin].
    Otherwise the result is 'inconclusive', never 'preserved'."""
    diff = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha, 1 - alpha])
    return {
        "estimate": float(diff.mean()),
        "ci90_low": float(lo),
        "ci90_high": float(hi),
        "margin": margin,
        "equivalent": bool(lo > -margin and hi < margin),
    }
