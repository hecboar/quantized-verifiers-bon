import numpy as np
import pytest

from vq.calibration import Pool, build_recipe, RECIPES


def _pool(Q=40, C=8, seed=0):
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(Q, C))
    ids = [[list(rng.integers(1, 100, size=rng.integers(10, 20))) for _ in range(C)] for _ in range(Q)]
    return Pool(scores=scores, token_ids=ids)


@pytest.mark.parametrize("recipe", sorted(RECIPES - {"generic", "generic_len"}))
def test_same_sequence_budget(recipe):
    cs = build_recipe(recipe, _pool(), n_sequences=32, seed=0)
    assert len(cs.sequences) == 32


def test_random_indomain_has_no_pairs():
    cs = build_recipe("random", _pool(), 32, seed=0)
    assert cs.pairs == []


def test_ch_pairs_are_winner_and_closest_challenger():
    pool = _pool()
    cs = build_recipe("ch", pool, 32, seed=0, anchor_frac=0.25)
    meta = cs.meta
    assert len(cs.pairs) == 12  # (32 * 0.75) / 2
    for (a, b), (q, w, j) in zip(cs.pairs, meta["pair_sources"]):
        s = pool.scores[q]
        assert w == int(np.argmax(s))
        others = [k for k in range(len(s)) if k != w]
        assert j == min(others, key=lambda k: s[w] - s[k])
        assert cs.sequences[a].tolist()[0] == pool.token_ids[q][w]
        assert cs.sequences[b].tolist()[0] == pool.token_ids[q][j]


def test_ch_pairs_one_per_question_smallest_margins():
    pool = _pool()
    cs = build_recipe("ch", pool, 32, seed=0, anchor_frac=0.25)
    qs = [q for q, _, _ in cs.meta["pair_sources"]]
    assert len(set(qs)) == len(qs)
    margins = np.sort(pool.scores, axis=1)
    gap = margins[:, -1] - margins[:, -2]
    chosen = set(qs)
    assert max(gap[list(chosen)]) <= min(gap[[q for q in range(40) if q not in chosen]]) + 1e-12


def test_shuffled_uses_same_sequences_but_cross_question_partners():
    pool = _pool()
    ch = build_recipe("ch", pool, 32, seed=0)
    sh = build_recipe("ch_shuffled", pool, 32, seed=0)
    as_list = lambda cs: sorted(tuple(s.tolist()[0]) for s in cs.sequences)
    assert as_list(ch) == as_list(sh)
    src = sh.meta["pair_questions"]
    assert all(qa != qb for qa, qb in src)


def test_dpq_adapted_mixes_boundary_and_anchors():
    cs = build_recipe("dpq", _pool(), 32, seed=0, boundary_frac=0.75)
    assert cs.meta["n_boundary"] == 24 and cs.meta["n_anchor"] == 8
    assert cs.pairs == []


def test_recipes_are_label_free():
    # Pool has no correctness field at all: building must still work
    for r in RECIPES - {"generic", "generic_len"}:
        build_recipe(r, _pool(), 16, seed=1)


def test_token_totals_reported():
    cs = build_recipe("random", _pool(), 16, seed=0)
    assert cs.meta["n_tokens"] == sum(s.shape[1] for s in cs.sequences)


def test_generic_len_matches_random_lengths():
    pool = _pool()
    ref = build_recipe("random", pool, 16, seed=0)
    lengths = [s.shape[1] for s in ref.sequences]
    chunks = [list(range(i * 50, i * 50 + 50)) for i in range(40)]
    cs = build_recipe("generic_len", None, 16, seed=0, generic_texts_ids=chunks, lengths=lengths)
    assert [s.shape[1] for s in cs.sequences] == lengths
    flat = [t for s in cs.sequences for t in s[0].tolist()]
    assert len(set(flat)) == len(flat)  # consecutive, non-overlapping pieces


def test_generic_len_rejects_short_text():
    with pytest.raises(ValueError):
        build_recipe("generic_len", None, 2, seed=0, generic_texts_ids=[[1, 2, 3]], lengths=[5, 5])
