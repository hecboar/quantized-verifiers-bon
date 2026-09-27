import numpy as np
import pandas as pd
import pytest

from vq.analysis import condition_metrics, confirmatory, evaluate_gate, score_matrix
from vq.store import CampaignStore


def _setup(tmp_path, Q=60, C=8, noise=0.5):
    rng = np.random.default_rng(0)
    store = CampaignStore(tmp_path)
    rows, s_ref, s_q = [], [], []
    for q in range(Q):
        corr = rng.random(C) < 0.5
        base = rng.normal(size=C) + 1.0 * corr  # useful verifier
        for c in range(C):
            rows.append({"question_id": f"e{q}", "cand": c, "correct": bool(corr[c]), "answer_key": int(c % 3)})
            s_ref.append({"split": "eval", "question_id": f"e{q}", "cand": c, "score": base[c]})
            s_q.append({"split": "eval", "question_id": f"e{q}", "cand": c, "score": base[c] + noise * rng.normal()})
    store.write("candidates", "G__eval", pd.DataFrame(rows))
    store.write("scores", "G__v__bf16", pd.DataFrame(s_ref))
    store.write("scores", "G__v__q", pd.DataFrame(s_q))
    return store


def test_score_matrix_aligned_with_candidates(tmp_path):
    store = _setup(tmp_path)
    S, correct, keys, qids = score_matrix(store, "G", "v", "bf16", "eval")
    assert S.shape == correct.shape == keys.shape == (60, 8)
    assert qids[0] == "e0"


def test_condition_metrics_identity_and_ranges(tmp_path):
    store = _setup(tmp_path)
    m = condition_metrics(store, "G", "v", "q", budgets=[1, 4, 8], repeats=30, seed=0)
    for n in (1, 4, 8):
        r = m[n]
        np.testing.assert_allclose(r["acc_q"] - r["acc_ref"], r["B"] - r["H"], atol=1e-12)
        assert 0 <= r["winner_change"].mean() <= 1
    # N=1: every verifier selects the same candidate
    assert m[1]["winner_change"].max() == 0.0
    assert m[8]["acc_ref"].mean() > m[8]["random"].mean()  # BF16 useful


def test_self_condition_has_zero_effect(tmp_path):
    store = _setup(tmp_path)
    m = condition_metrics(store, "G", "v", "bf16", budgets=[8], repeats=10, seed=0)
    assert m[8]["winner_change"].max() == 0.0 and m[8]["H"].max() == 0.0


def test_confirmatory_returns_holm_adjusted(tmp_path):
    rng = np.random.default_rng(1)
    per_q = {"c1": rng.normal(0.05, 0.1, 500), "c2": rng.normal(0.0, 0.1, 500), "c3": rng.normal(-0.02, 0.1, 500)}
    res = confirmatory(per_q, n_boot=2000, seed=0)
    assert set(res) == {"c1", "c2", "c3"}
    assert all(res[k]["p_holm"] >= res[k]["p_value"] for k in res)
    assert res["c1"]["ci_low"] > 0


def test_gate_pass_and_fail():
    ok = evaluate_gate(acc_bf16_32=0.60, acc_random_32=0.50, mixed_frac=0.40, projected_eur=60, implementation_ok=True,
                       ceiling=71)
    assert ok["go"] is True
    bad = evaluate_gate(acc_bf16_32=0.52, acc_random_32=0.50, mixed_frac=0.40, projected_eur=60, implementation_ok=True,
                        ceiling=71)
    assert bad["go"] is False and bad["criteria"]["bf16_useful"] is False
