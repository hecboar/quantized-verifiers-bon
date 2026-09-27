import numpy as np

from vq.controls import gaussian_control, permutation_control, score_strata


def _data(Q=50, C=32, seed=0):
    rng = np.random.default_rng(seed)
    ref = rng.normal(size=(Q, C))
    q = ref + 0.3 * ref + rng.normal(scale=0.1, size=(Q, C))  # structured error
    return ref, q


def test_free_permutation_preserves_error_multiset_per_question():
    ref, q = _data()
    out, moved = permutation_control(ref, q, strata=None, seed=0)
    e, e_p = q - ref, out - ref
    for i in range(ref.shape[0]):
        np.testing.assert_allclose(np.sort(e[i]), np.sort(e_p[i]))
    assert moved > 0.9


def test_stratified_permutation_stays_within_strata():
    ref, q = _data()
    strata = score_strata(ref, n_bins=3)
    out, _ = permutation_control(ref, q, strata=strata, seed=0)
    e, e_p = q - ref, out - ref
    for i in range(ref.shape[0]):
        for s in np.unique(strata[i]):
            m = strata[i] == s
            np.testing.assert_allclose(np.sort(e[i, m]), np.sort(e_p[i, m]))


def test_singleton_strata_cannot_move():
    ref, q = _data(Q=2, C=4)
    strata = np.arange(4)[None, :].repeat(2, axis=0)  # every candidate alone
    out, moved = permutation_control(ref, q, strata=strata, seed=0)
    np.testing.assert_allclose(out, q)
    assert moved == 0.0


def test_score_strata_balanced_bins():
    ref, _ = _data(C=30)
    s = score_strata(ref, n_bins=3)
    assert set(np.unique(s)) == {0, 1, 2}
    assert all(np.bincount(row, minlength=3).min() == 10 for row in s)
    # bin increases with score
    i = 0
    assert ref[i, s[i] == 2].min() > ref[i, s[i] == 0].max()


def test_gaussian_control_matches_mean_and_std_per_question():
    ref, q = _data(C=2000, Q=3)
    out = gaussian_control(ref, q, seed=0)
    e, g = q - ref, out - ref
    np.testing.assert_allclose(g.mean(1), e.mean(1), atol=0.05)
    np.testing.assert_allclose(g.std(1), e.std(1), rtol=0.1)


def test_controls_deterministic():
    ref, q = _data()
    a, _ = permutation_control(ref, q, None, seed=3)
    b, _ = permutation_control(ref, q, None, seed=3)
    np.testing.assert_array_equal(a, b)
