"""C3: sample-efficient prediction of selection instability (winner change).

Residual model fitted on development pools (within-question centred):
    e_i - mean(e) = slope * (r_i - mean(r)) + eps_i,  eps ~ N(0, sigma^2)
A common shift drops out; `slope` captures score compression/expansion and
`sigma` the candidate-specific error. The predicted winner-change rate is
obtained by simulating this model on the development BF16 scores. It is
compared against the empirical development rate (the strong baseline).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .selection import draw_subsets, select_bon, tiebreak_priority


@dataclass
class ResidualModel:
    slope: float
    sigma: float


def fit_residual_model(ref: np.ndarray, q: np.ndarray) -> ResidualModel:
    e = q - ref
    ec = e - e.mean(axis=1, keepdims=True)
    rc = ref - ref.mean(axis=1, keepdims=True)
    denom = float((rc**2).sum())
    slope = float((rc * ec).sum() / denom) if denom > 0 else 0.0
    resid = ec - slope * rc
    dof = max(resid.size - ref.shape[0] - 1, 1)
    sigma = float(np.sqrt((resid**2).sum() / dof))
    if sigma < 1e-9:
        sigma = 0.0
    if abs(slope) < 1e-12:
        slope = 0.0
    return ResidualModel(slope, sigma)


def empirical_flip_rate(ref: np.ndarray, q: np.ndarray, n: int, repeats: int, seed: int) -> float:
    Q, C = ref.shape
    subsets = draw_subsets(Q, C, n, repeats, seed)
    prio = tiebreak_priority(Q, C, seed)
    return float((select_bon(ref, subsets, prio) != select_bon(q, subsets, prio)).mean())


def predict_flip_rate(ref: np.ndarray, model: ResidualModel, n: int, repeats: int, seed: int) -> float:
    Q, C = ref.shape
    if model.sigma == 0.0 and model.slope >= -1.0 + 1e-12:
        # pure affine map with positive gain preserves the ranking
        return 0.0
    rng = np.random.default_rng(seed)
    rc = ref - ref.mean(axis=1, keepdims=True)
    subsets = draw_subsets(Q, C, n, repeats, seed)
    prio = tiebreak_priority(Q, C, seed)
    w_ref = select_bon(ref, subsets, prio)
    rates = []
    for _ in range(max(1, repeats if n == C else 1)):
        sim = ref + model.slope * rc + model.sigma * rng.standard_normal(ref.shape)
        rates.append((select_bon(sim, subsets, prio) != w_ref).mean())
    return float(np.mean(rates))
