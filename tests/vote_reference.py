"""Loop implementation kept as the executable specification of select_vote."""
import numpy as np

from vq.selection import select_bon


def select_vote_reference(
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
