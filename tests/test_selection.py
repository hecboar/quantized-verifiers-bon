import numpy as np
import pytest

from vq.selection import (
    draw_subsets,
    select_bon,
    select_vote,
    tiebreak_priority,
)
from vq.metrics import (
    accuracy_per_question,
    harm_benefit,
    oracle_per_question,
    random_per_question,
    selection_efficiency,
    winner_change_rate,
    within_question_kendall,
    robust_scale,
)


def test_draw_subsets_shape_and_uniqueness():
    s = draw_subsets(n_questions=5, n_cand=32, n=8, repeats=10, seed=0)
    assert s.shape == (5, 10, 8)
    for q in range(5):
        for r in range(10):
            assert len(set(s[q, r])) == 8
    assert s.min() >= 0 and s.max() < 32


def test_draw_subsets_deterministic():
    a = draw_subsets(3, 32, 4, 5, seed=1)
    b = draw_subsets(3, 32, 4, 5, seed=1)
    np.testing.assert_array_equal(a, b)


def test_draw_subsets_full_n_is_single_identity_draw():
    s = draw_subsets(2, 32, 32, 10, seed=0)
    assert s.shape == (2, 1, 32)
    np.testing.assert_array_equal(np.sort(s[0, 0]), np.arange(32))


def test_select_bon_picks_max():
    scores = np.array([[0.1, 0.9, 0.5]])
    subsets = np.array([[[0, 1, 2], [0, 2, 0]]])[:, :, :2]  # [[0,1],[0,2]]
    prio = tiebreak_priority(1, 3, seed=0)
    w = select_bon(scores, subsets, prio)
    np.testing.assert_array_equal(w, [[1, 2]])


def test_select_bon_ties_use_shared_priority():
    scores = np.array([[1.0, 1.0, 0.0]])
    prio = np.array([[0.2, 0.7, 0.1]])
    subsets = np.array([[[0, 1, 2]]])
    assert select_bon(scores, subsets, prio)[0, 0] == 1


def test_common_shift_does_not_change_bon():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(20, 32))
    subsets = draw_subsets(20, 32, 8, 50, seed=0)
    prio = tiebreak_priority(20, 32, seed=0)
    a = select_bon(scores, subsets, prio)
    b = select_bon(scores + 3.7, subsets, prio)
    np.testing.assert_array_equal(a, b)


def test_select_vote_majority():
    # answers: cluster ids; candidates 0,1 -> answer 5; candidate 2 -> answer 9 (high score)
    scores = np.array([[0.0, 0.0, 10.0]])
    keys = np.array([[5, 5, 9]])
    subsets = np.array([[[0, 1, 2]]])
    prio = tiebreak_priority(1, 3, seed=0)
    maj = select_vote(scores, keys, subsets, prio, temperature=None)
    assert keys[0, maj[0, 0]] == 5
    weighted = select_vote(scores, keys, subsets, prio, temperature=1.0)
    assert keys[0, weighted[0, 0]] == 9


def test_select_vote_ignores_unparseable():
    scores = np.array([[0.0, 0.0, 0.0]])
    keys = np.array([[-1, -1, 3]])
    subsets = np.array([[[0, 1, 2]]])
    prio = tiebreak_priority(1, 3, seed=0)
    assert select_vote(scores, keys, subsets, prio, temperature=None)[0, 0] == 2


def test_select_vote_all_unparseable_falls_back_to_bon():
    scores = np.array([[0.0, 5.0]])
    keys = np.array([[-1, -1]])
    subsets = np.array([[[0, 1]]])
    prio = tiebreak_priority(1, 2, seed=0)
    assert select_vote(scores, keys, subsets, prio, temperature=None)[0, 0] == 1


def test_accuracy_and_harm_benefit_identity():
    correct = np.array([[1, 0], [0, 1], [1, 1]], dtype=bool)
    w_ref = np.array([[0], [0], [0]])
    w_q = np.array([[1], [1], [1]])
    a_ref = accuracy_per_question(correct, w_ref)
    a_q = accuracy_per_question(correct, w_q)
    h, b = harm_benefit(correct, w_ref, w_q)
    np.testing.assert_allclose(a_ref, [1, 0, 1])
    np.testing.assert_allclose(a_q, [0, 1, 1])
    np.testing.assert_allclose(h, [1, 0, 0])
    np.testing.assert_allclose(b, [0, 1, 0])
    np.testing.assert_allclose(a_q - a_ref, b - h)


def test_winner_change_rate():
    assert winner_change_rate(np.array([[0, 1]]), np.array([[0, 2]]))[0] == 0.5


def test_random_and_oracle():
    correct = np.array([[1, 0, 0, 0]], dtype=bool)
    subsets = np.array([[[0, 1], [2, 3]]])
    np.testing.assert_allclose(random_per_question(correct, subsets), [0.25])
    np.testing.assert_allclose(oracle_per_question(correct, subsets), [0.5])


def test_selection_efficiency_undefined_when_no_headroom():
    eff, defined = selection_efficiency(
        acc=np.array([0.5, 1.0]), rand=np.array([0.25, 1.0]), oracle=np.array([0.75, 1.0])
    )
    assert defined.tolist() == [True, False]
    assert eff[0] == pytest.approx(0.5)


def test_within_question_kendall_perfect_and_reversed():
    a = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    b = np.array([[10.0, 20.0, 30.0], [3.0, 2.0, 1.0]])
    np.testing.assert_allclose(within_question_kendall(a, b), [1.0, -1.0])


def test_robust_scale_is_mad_based():
    x = np.array([[0.0, 1.0, 2.0, 3.0, 100.0]])
    s = robust_scale(x)
    assert 0.5 < s < 3.0  # outlier does not blow up the scale
