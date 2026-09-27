import numpy as np
import pytest

from vq.diagnostic import (
    empirical_flip_rate,
    fit_residual_model,
    predict_flip_rate,
)


def _simulate(Q, C=32, slope=0.2, sigma=0.3, seed=0):
    rng = np.random.default_rng(seed)
    ref = rng.normal(size=(Q, C))
    c = ref - ref.mean(1, keepdims=True)
    q = ref + 1.5 + slope * c + sigma * rng.normal(size=(Q, C))
    return ref, q


def test_fit_recovers_slope_and_sigma():
    ref, q = _simulate(400)
    m = fit_residual_model(ref, q)
    assert m.slope == pytest.approx(0.2, abs=0.02)
    assert m.sigma == pytest.approx(0.3, abs=0.02)


def test_common_shift_has_zero_flip_rate():
    ref, _ = _simulate(50)
    assert empirical_flip_rate(ref, ref + 3.0, n=8, repeats=50, seed=0) == 0.0
    m = fit_residual_model(ref, ref + 3.0)
    assert predict_flip_rate(ref, m, n=8, repeats=50, seed=0) == 0.0


def test_prediction_matches_empirical_under_correct_model():
    ref, q = _simulate(600, seed=1)
    m = fit_residual_model(ref, q)
    emp = empirical_flip_rate(ref, q, n=16, repeats=50, seed=0)
    pred = predict_flip_rate(ref, m, n=16, repeats=50, seed=0)
    assert pred == pytest.approx(emp, abs=0.03)


def test_flip_rate_increases_with_n_under_noise():
    ref, q = _simulate(300, seed=2)
    rates = [empirical_flip_rate(ref, q, n, repeats=50, seed=0) for n in (2, 8, 32)]
    assert rates[0] < rates[1] < rates[2]


def test_model_based_estimate_less_variable_than_empirical_on_small_dev():
    truth_ref, truth_q = _simulate(3000, seed=9)
    truth = empirical_flip_rate(truth_ref, truth_q, n=32, repeats=1, seed=0)
    err_emp, err_mod = [], []
    for s in range(30):
        ref, q = _simulate(20, seed=100 + s)
        err_emp.append(abs(empirical_flip_rate(ref, q, 32, 1, seed=0) - truth))
        err_mod.append(abs(predict_flip_rate(ref, fit_residual_model(ref, q), 32, 20, seed=s) - truth))
    assert np.mean(err_mod) <= np.mean(err_emp)
