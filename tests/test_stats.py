import numpy as np
import pytest

from vq.stats import holm, paired_bootstrap, tost_equivalence


def test_bootstrap_zero_difference_ci_contains_zero():
    rng = np.random.default_rng(0)
    a = rng.random(500)
    res = paired_bootstrap({"a": a, "b": a.copy()}, contrast=lambda d: d["a"] - d["b"], n_boot=2000, seed=0)
    assert res["estimate"] == 0.0
    assert res["ci_low"] <= 0.0 <= res["ci_high"]


def test_bootstrap_detects_clear_difference():
    rng = np.random.default_rng(1)
    a = rng.random(800)
    b = a - 0.1
    res = paired_bootstrap({"a": a, "b": b}, contrast=lambda d: d["a"] - d["b"], n_boot=2000, seed=0)
    assert res["estimate"] == pytest.approx(0.1)
    assert res["ci_low"] > 0.05


def test_bootstrap_resamples_questions_jointly():
    # a and b perfectly correlated noise: paired CI must be tiny
    rng = np.random.default_rng(2)
    noise = rng.normal(size=1000)
    res = paired_bootstrap({"a": noise + 0.01, "b": noise}, contrast=lambda d: d["a"] - d["b"], n_boot=1000, seed=0)
    assert res["ci_high"] - res["ci_low"] < 1e-9


def test_bootstrap_length_mismatch_raises():
    with pytest.raises(ValueError):
        paired_bootstrap({"a": np.zeros(3), "b": np.zeros(4)}, contrast=lambda d: d["a"], n_boot=10, seed=0)


def test_holm_adjusts_and_preserves_order():
    p = np.array([0.01, 0.04, 0.03])
    adj = holm(p)
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06])


def test_holm_caps_at_one():
    assert holm(np.array([0.6, 0.9])).max() <= 1.0


def test_tost_equivalent_when_ci_inside_margin():
    rng = np.random.default_rng(3)
    d = rng.normal(0.0, 0.05, size=2000)
    res = tost_equivalence(d, margin=0.03, n_boot=2000, seed=0)
    assert res["equivalent"] is True


def test_tost_inconclusive_when_ci_too_wide():
    rng = np.random.default_rng(4)
    d = rng.normal(0.0, 1.0, size=50)
    res = tost_equivalence(d, margin=0.03, n_boot=2000, seed=0)
    assert res["equivalent"] is False
