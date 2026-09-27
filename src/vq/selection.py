"""Selection rules over frozen candidate pools.

Conventions: scores/keys/correct are [Q, C] arrays (questions x candidates);
subsets are [Q, R, N] candidate indices (R random draws of N candidates).
The same subsets and tie-break priorities are shared across all conditions,
so BF16 and quantized selections are paired draw by draw.
"""
from __future__ import annotations

import numpy as np


def draw_subsets(n_questions: int, n_cand: int, n: int, repeats: int, seed: int) -> np.ndarray:
    if n > n_cand:
        raise ValueError("n cannot exceed the pool size")
    if n == n_cand:
        base = np.arange(n_cand)
        return np.broadcast_to(base, (n_questions, 1, n_cand)).copy()
    rng = np.random.default_rng(seed)
    keys = rng.random((n_questions, repeats, n_cand))
    return np.argsort(keys, axis=-1)[..., :n]


def tiebreak_priority(n_questions: int, n_cand: int, seed: int) -> np.ndarray:
    """Fixed random priority per candidate; higher wins exact score ties."""
    return np.random.default_rng(seed).random((n_questions, n_cand))


def select_bon(scores: np.ndarray, subsets: np.ndarray, priority: np.ndarray) -> np.ndarray:
    q_idx = np.arange(scores.shape[0])[:, None, None]
    s = scores[q_idx, subsets]
    p = priority[q_idx, subsets]
    # lexicographic argmax on (score, priority)
    best = s.max(axis=-1, keepdims=True)
    masked = np.where(s == best, p, -np.inf)
    pos = masked.argmax(axis=-1)
    return np.take_along_axis(subsets, pos[..., None], axis=-1)[..., 0]


def select_vote(
    scores: np.ndarray,
    keys: np.ndarray,
    subsets: np.ndarray,
    priority: np.ndarray,
    temperature: float | None,
) -> np.ndarray:
    """Majority (temperature=None) or softmax(score/T)-weighted vote over answer keys.

    Unparseable answers (key -1) do not vote. The returned candidate is the
    best-scored member of the winning answer cluster. If no candidate in the
    subset is parseable, falls back to Best-of-N.
    """
    Q, R, N = subsets.shape
    out = np.empty((Q, R), dtype=np.int64)
    bon = select_bon(scores, subsets, priority)
    for q in range(Q):
        for r in range(R):
            idx = subsets[q, r]
            k = keys[q, idx]
            valid = k >= 0
            if not valid.any():
                out[q, r] = bon[q, r]
                continue
            idx, k = idx[valid], k[valid]
            s = scores[q, idx]
            if temperature is None:
                w = np.ones_like(s)
            else:
                z = (s - s.max()) / temperature
                w = np.exp(z)
            clusters = np.unique(k)
            totals = np.array([w[k == c].sum() for c in clusters])
            # tie between clusters: the one containing the best (score, priority) member
            top = np.flatnonzero(np.isclose(totals, totals.max()))
            cand_best = []
            for t in top:
                members = idx[k == clusters[t]]
                m = members[np.lexsort((priority[q, members], scores[q, members]))[-1]]
                cand_best.append(m)
            cand_best = np.array(cand_best)
            out[q, r] = cand_best[np.lexsort((priority[q, cand_best], scores[q, cand_best]))[-1]]
    return out
