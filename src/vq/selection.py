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
    best-scored (score, then priority) member of the winning answer cluster;
    ties between clusters go to the cluster holding that best member. If no
    candidate in the subset is parseable, falls back to Best-of-N.
    Vectorised; tests/vote_reference.py holds the loop specification.
    """
    Q, R, N = subsets.shape
    K = int(keys.max()) + 1
    out = select_bon(scores, subsets, priority)
    if K <= 0:
        return out
    chunk = max(1, int(4e7 // max(1, R * N * K)))
    for q0 in range(0, Q, chunk):
        q1 = min(Q, q0 + chunk)
        qi = np.arange(q0, q1)[:, None, None]
        idx = subsets[q0:q1]
        s, p, k = scores[qi, idx], priority[qi, idx], keys[qi, idx]
        valid = k >= 0
        if temperature is None:
            w = valid.astype(float)
        else:
            smax = np.where(valid, s, -np.inf).max(axis=-1, keepdims=True)
            w = np.where(valid, np.exp((s - np.where(np.isfinite(smax), smax, 0.0)) / temperature), 0.0)
        onehot = k[..., None] == np.arange(K)  # [q, R, N, K]
        tot = (w[..., None] * onehot).sum(axis=2)  # [q, R, K]
        present = onehot.any(axis=2)
        top = present & np.isclose(tot, tot.max(axis=-1, keepdims=True))
        eligible = valid & np.take_along_axis(top, np.where(valid, k, 0), axis=-1)
        s_el = np.where(eligible, s, -np.inf)
        best = s_el.max(axis=-1, keepdims=True)
        pos = np.where(eligible & (s_el == best), p, -np.inf).argmax(axis=-1)
        win = np.take_along_axis(idx, pos[..., None], axis=-1)[..., 0]
        any_valid = valid.any(axis=-1)
        out[q0:q1] = np.where(any_valid, win, out[q0:q1])
    return out
