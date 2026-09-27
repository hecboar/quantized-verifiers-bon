"""Assignment controls on stored scores (plan §4.6). They intervene on the
observed errors e = s_Q - s_BF16, not on weights: they test whether *which
candidate receives which error* matters, holding the error multiset fixed."""
from __future__ import annotations

import numpy as np


def score_strata(ref: np.ndarray, n_bins: int = 3) -> np.ndarray:
    """Within-question equal-count bins of the BF16 score (0 = lowest)."""
    ranks = np.argsort(np.argsort(ref, axis=1, kind="stable"), axis=1)
    return (ranks * n_bins) // ref.shape[1]


def permutation_control(ref: np.ndarray, q: np.ndarray, strata: np.ndarray | None, seed: int):
    """Permute errors within question (and within strata if given).
    Returns (controlled scores, fraction of errors that actually moved)."""
    rng = np.random.default_rng(seed)
    e = q - ref
    out = e.copy()
    moved = 0
    for i in range(e.shape[0]):
        groups = [np.arange(e.shape[1])] if strata is None else [np.flatnonzero(strata[i] == s) for s in np.unique(strata[i])]
        for idx in groups:
            if len(idx) < 2:
                continue
            perm = rng.permutation(idx)
            out[i, idx] = e[i, perm]
            moved += int(np.sum(perm != idx))
    return ref + out, moved / e.size


def gaussian_control(ref: np.ndarray, q: np.ndarray, seed: int) -> np.ndarray:
    """Weaker control: i.i.d. Gaussian errors with per-question mean/std of e."""
    rng = np.random.default_rng(seed)
    e = q - ref
    mu = e.mean(axis=1, keepdims=True)
    sd = e.std(axis=1, keepdims=True)
    return ref + mu + sd * rng.standard_normal(e.shape)
