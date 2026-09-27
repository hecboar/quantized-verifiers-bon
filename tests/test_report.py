import json

import numpy as np
import pandas as pd

from vq.report import write_report
from vq.store import CampaignStore


def _mk(tmp_path, Q=40, C=8):
    rng = np.random.default_rng(0)
    st = CampaignStore(tmp_path / "store")
    cands, base = [], {}
    for split, pre in (("eval", "e"), ("dev", "d")):
        for q in range(Q):
            corr = rng.random(C) < 0.5
            base[(split, q)] = rng.normal(size=C) + corr
            for c in range(C):
                cands.append({"split": split, "question_id": f"{pre}{q}", "cand": c, "correct": bool(corr[c]),
                              "answer_key": c % 3})
    cdf = pd.DataFrame(cands)
    for split in ("eval", "dev"):
        st.write("candidates", f"G__{split}", cdf[cdf.split == split].drop(columns="split"))

    def scores(noise, seed):
        r = np.random.default_rng(seed)
        rows = []
        for (split, q), b in base.items():
            pre = "e" if split == "eval" else "d"
            for c in range(C):
                rows.append({"split": split, "question_id": f"{pre}{q}", "cand": c, "score": b[c] + noise * r.normal()})
        return pd.DataFrame(rows)

    for v in ("sky-4b", "sky-8b"):
        st.write("scores", f"G__{v}__bf16", scores(0.0, 1))
        st.write("scores", f"G__{v}__gptq-w4", scores(0.4, 2))
        for rec in ("random", "dpq", "ch", "ch_shuffled"):
            for s in (0, 1):
                st.write("scores", f"G__{v}__gptq-w4-{rec}-s{s}", scores(0.3 if rec == "ch" else 0.5, 10 + s))
    return st


def test_report_writes_tables(tmp_path):
    st = _mk(tmp_path)
    out = tmp_path / "tables"
    write_report(st, "G", out, budgets=[1, 8], repeats=20, n_boot=500)
    main = pd.read_csv(out / "main.csv")
    assert {"verifier", "condition", "N", "acc_ref", "acc_q", "delta", "delta_lo", "delta_hi", "H", "B",
            "winner_change"} <= set(main.columns)
    assert (main.loc[main.N == 1, "winner_change"] == 0).all()
    conf = json.loads((out / "confirmatory.json").read_text())
    assert "c1_delta_8b_gptq_w4" in conf and "c3_ch_vs_dpq" in conf
    assert conf["c2_memory_frontier"]["status"].startswith("missing")
    controls = pd.read_csv(out / "controls.csv")
    assert {"real", "perm_free", "perm_strat", "gaussian", "frac_moved_strat"} <= set(controls.columns)
    c3 = pd.read_csv(out / "c3.csv")
    assert {"predicted_flip", "dev_empirical_flip", "eval_observed_flip"} <= set(c3.columns)


def test_memory_frontier_selects_on_dev_and_evaluates_on_eval(tmp_path):
    st = _mk(tmp_path)
    for v in ("sky-4b", "sky-8b"):  # deployment (packed) scores
        st.write("scores", f"G__{v}__gptq-w4-packed", st.read("scores", f"G__{v}__gptq-w4"))
    mem = {
        "sky-4b__bf16": {"peak_gb": 9.0}, "sky-4b__gptq-w4": {"peak_gb": 3.5},
        "sky-8b__bf16": {"peak_gb": 17.0}, "sky-8b__gptq-w4": {"peak_gb": 5.5},
    }
    mp = tmp_path / "memory.json"
    mp.write_text(json.dumps(mem))
    out = tmp_path / "tables"
    write_report(st, "G", out, budgets=[8], repeats=20, n_boot=500, memory_path=mp, ceilings_gb=(6, 10, 20),
                 primary_ceiling_gb=10)
    fr = pd.read_csv(out / "frontier.csv")
    row6 = fr[fr.ceiling_gb == 6].iloc[0]
    assert row6["best_bf16"] == "none"  # no BF16 model fits in 6 GB
    assert row6["best_any"] in ("sky-4b__gptq-w4-packed", "sky-8b__gptq-w4-packed")
    assert row6["selected_on"] == "dev"
    conf = json.loads((out / "confirmatory.json").read_text())
    assert conf["c2_memory_frontier"]["status"].startswith("ok (primary ceiling 10 GB")  # sky-4b bf16 fits
    assert {"delta_eval", "delta_lo", "delta_hi"} <= set(fr.columns)


def test_ch_advantage_detected_in_synthetic_data(tmp_path):
    st = _mk(tmp_path, Q=150)
    out = tmp_path / "tables"
    write_report(st, "G", out, budgets=[8], repeats=20, n_boot=500)
    conf = json.loads((out / "confirmatory.json").read_text())
    assert conf["c3_ch_vs_dpq"]["estimate"] > 0


def test_confirmatory_family_is_fixed_even_when_runs_missing(tmp_path):
    st = _mk(tmp_path)
    # remove the 4B CH seed 1 -> CH/DPQ seed sets no longer match for 4B
    import os
    os.remove(st._path("scores", "G__sky-4b__gptq-w4-ch-s1"))
    out = tmp_path / "tables"
    write_report(st, "G", out, budgets=[8], repeats=10, n_boot=200)
    conf = json.loads((out / "confirmatory.json").read_text())
    assert set(conf) >= {"c1_delta_8b_gptq_w4", "c2_memory_frontier", "c3_ch_vs_dpq"}
    assert conf["c3_ch_vs_dpq"]["status"].startswith("missing")
    assert conf["c3_ch_vs_dpq"]["p_holm"] == 1.0
    assert conf["c2_memory_frontier"]["p_holm"] == 1.0  # no memory.json -> conservative


def test_frontier_uses_deployment_scores_and_skips_failed_measurements(tmp_path):
    st = _mk(tmp_path)
    # deployment (packed) scores exist for the 8B only
    st.write("scores", "G__sky-8b__gptq-w4-packed", st.read("scores", "G__sky-8b__gptq-w4"))
    mem = {"sky-4b__bf16": {"peak_gb": 9.0}, "sky-8b__bf16": {"peak_gb": 17.0},
           "sky-8b__gptq-w4": {"peak_gb": 5.5}, "sky-4b__gptq-w4": {"status": "missing packed checkpoint"}}
    mp = tmp_path / "memory.json"
    mp.write_text(json.dumps(mem))
    out = tmp_path / "tables"
    write_report(st, "G", out, budgets=[8], repeats=10, n_boot=200, memory_path=mp, ceilings_gb=(6, 10))
    fr = pd.read_csv(out / "frontier.csv")
    assert fr.loc[fr.ceiling_gb == 6, "best_any"].item() == "sky-8b__gptq-w4-packed"
    assert "sky-4b__gptq-w4" not in set(fr["best_any"])
