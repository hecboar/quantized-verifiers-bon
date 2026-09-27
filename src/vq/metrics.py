"""Per-question metrics. Everything returns per-question vectors so the
paired bootstrap can resample questions."""
from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau


def _gather(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    q_idx = np.arange(values.shape[0]).reshape((-1,) + (1,) * (idx.ndim - 1))
    return values[q_idx, idx]


def accuracy_per_question(correct: np.ndarray, winners: np.ndarray) -> np.ndarray:
    return _gather(correct, winners).mean(axis=1)


def harm_benefit(correct: np.ndarray, w_ref: np.ndarray, w_new: np.ndarray):
    """H: ref correct & new incorrect; B: ref incorrect & new correct (per question)."""
    c_ref = _gather(correct, w_ref)
    c_new = _gather(correct, w_new)
    h = (c_ref & ~c_new).mean(axis=1)
    b = (~c_ref & c_new).mean(axis=1)
    return h, b


def winner_change_rate(w_ref: np.ndarray, w_new: np.ndarray) -> np.ndarray:
    return (w_ref != w_new).mean(axis=1)


def random_per_question(correct: np.ndarray, subsets: np.ndarray) -> np.ndarray:
    return _gather(correct, subsets).mean(axis=(1, 2))


def oracle_per_question(correct: np.ndarray, subsets: np.ndarray) -> np.ndarray:
    return _gather(correct, subsets).any(axis=2).mean(axis=1)


def selection_efficiency(acc, rand, oracle):
    """(A - A_rand) / (A_oracle - A_rand); undefined where there is no headroom."""
    head = oracle - rand
    defined = head > 1e-12
    eff = np.full_like(acc, np.nan, dtype=float)
    eff[defined] = (acc[defined] - rand[defined]) / head[defined]
    return eff, defined


def within_question_kendall(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty(a.shape[0])
    for q in range(a.shape[0]):
        out[q] = kendalltau(a[q], b[q]).statistic
    return out


def robust_scale(scores: np.ndarray) -> float:
    """MAD of within-question centred scores (fixed on dev, used to normalise
    margins and residuals across verifiers/precisions)."""
    centred = scores - np.median(scores, axis=1, keepdims=True)
    mad = np.median(np.abs(centred))
    return float(1.4826 * mad) if mad > 0 else float(np.std(centred) or 1.0)
