"""Calibration recipes compared in the CH-GPTQ experiment (plan §5).

All recipes are label-free, draw from the same development pools and return
the same number of sequences; token totals are reported so budgets can be
checked. Recipes:
  generic      - generic text chunks (C4/WikiText), the standard GPTQ setup
  random       - random in-domain candidates (one per question)
  dpq          - DPQ adaptation: boundary candidates (smallest normalised
                 distance to their nearest competitor) + random anchors
  ch           - winner vs closest challenger pairs (smallest winner margins)
                 + random anchors; pairs feed the contrastive Hessian term
  ch_shuffled  - same sequences as `ch`, partners re-matched across questions
  generic_len  - generic text cut to the token lengths of the `random` recipe
                 (length control for the in-domain 3-bit calibration, revision 2)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .quant.apply import CalibrationSet

RECIPES = {"generic", "random", "dpq", "ch", "ch_shuffled", "generic_len"}


@dataclass
class Pool:
    scores: np.ndarray  # [Q, C] BF16 verifier scores on dev pools
    token_ids: list[list[list[int]]]  # [Q][C] verifier input ids


def _seq(ids: list[int]) -> torch.Tensor:
    return torch.tensor([ids], dtype=torch.long)


def _finish(seqs, pairs, meta) -> CalibrationSet:
    meta["n_tokens"] = int(sum(s.shape[1] for s in seqs))
    return CalibrationSet(sequences=seqs, pairs=pairs, meta=meta)


def _random_candidates(pool: Pool, n: int, rng, exclude_q=()):
    qs = [q for q in rng.permutation(len(pool.token_ids)) if q not in set(exclude_q)]
    if len(qs) < n:
        raise ValueError("not enough development questions for the requested budget")
    return [(int(q), int(rng.integers(len(pool.token_ids[q])))) for q in qs[:n]]


def _winner_challenger(scores: np.ndarray):
    order = np.argsort(-scores, axis=1, kind="stable")
    w, j = order[:, 0], order[:, 1]
    gap = scores[np.arange(len(scores)), w] - scores[np.arange(len(scores)), j]
    return w, j, gap


def build_recipe(
    recipe: str,
    pool: Pool | None,
    n_sequences: int,
    seed: int,
    anchor_frac: float = 0.25,
    boundary_frac: float = 0.75,
    generic_texts_ids: list[list[int]] | None = None,
    lengths: list[int] | None = None,
) -> CalibrationSet:
    if recipe not in RECIPES:
        raise ValueError(recipe)
    rng = np.random.default_rng(seed)

    if recipe == "generic":
        if not generic_texts_ids or len(generic_texts_ids) < n_sequences:
            raise ValueError("generic recipe needs >= n_sequences tokenised text chunks")
        idx = rng.choice(len(generic_texts_ids), n_sequences, replace=False)
        return _finish([_seq(generic_texts_ids[i]) for i in idx], [], {"recipe": recipe})

    if recipe == "generic_len":
        # generic chunks (shuffled with the seed) concatenated into one stream, then cut
        # into consecutive pieces with the given token lengths
        if not generic_texts_ids or not lengths or len(lengths) != n_sequences:
            raise ValueError("generic_len needs tokenised generic chunks and n_sequences lengths")
        order = rng.permutation(len(generic_texts_ids))
        stream = [t for i in order for t in generic_texts_ids[i]]
        if sum(lengths) > len(stream):
            raise ValueError(f"generic text too short: {len(stream)} tokens for {sum(lengths)}")
        seqs, pos = [], 0
        for n in lengths:
            seqs.append(_seq(stream[pos:pos + n]))
            pos += n
        return _finish(seqs, [], {"recipe": recipe})

    if recipe == "random":
        picks = _random_candidates(pool, n_sequences, rng)
        return _finish([_seq(pool.token_ids[q][c]) for q, c in picks], [], {"recipe": recipe, "sources": picks})

    if recipe == "dpq":
        # individual doubt: distance to the nearest competitor in the same pool
        s = pool.scores
        diff = np.abs(s[:, :, None] - s[:, None, :])
        diff[:, np.arange(s.shape[1]), np.arange(s.shape[1])] = np.inf
        nearest = diff.min(axis=2)
        best_c = nearest.argmin(axis=1)
        best_d = nearest[np.arange(len(s)), best_c]
        n_b = int(round(n_sequences * boundary_frac))
        qs_b = np.argsort(best_d, kind="stable")[:n_b]
        picks = [(int(q), int(best_c[q])) for q in qs_b]
        picks += _random_candidates(pool, n_sequences - n_b, rng, exclude_q=qs_b)
        return _finish(
            [_seq(pool.token_ids[q][c]) for q, c in picks], [],
            {"recipe": recipe, "n_boundary": n_b, "n_anchor": n_sequences - n_b, "sources": picks},
        )

    # ch / ch_shuffled
    n_pairs = int(n_sequences * (1 - anchor_frac)) // 2
    w, j, gap = _winner_challenger(pool.scores)
    qs = np.argsort(gap, kind="stable")[:n_pairs]
    seqs, pairs, sources = [], [], []
    for q in qs:
        seqs += [_seq(pool.token_ids[q][w[q]]), _seq(pool.token_ids[q][j[q]])]
        pairs.append((len(seqs) - 2, len(seqs) - 1))
        sources.append((int(q), int(w[q]), int(j[q])))
    anchors = _random_candidates(pool, n_sequences - len(seqs), rng, exclude_q=qs)
    seqs += [_seq(pool.token_ids[q][c]) for q, c in anchors]
    meta = {"recipe": recipe, "pair_sources": sources, "anchors": anchors}

    if recipe == "ch_shuffled":
        # derangement of challengers across questions (same sequences)
        perm = np.arange(n_pairs)
        while n_pairs > 1 and np.any(perm == np.arange(n_pairs)):
            perm = rng.permutation(n_pairs)
        pairs = [(pairs[k][0], pairs[perm[k]][1]) for k in range(n_pairs)]
        meta["pair_questions"] = [(sources[k][0], sources[perm[k]][0]) for k in range(n_pairs)]
    return _finish(seqs, pairs, meta)
