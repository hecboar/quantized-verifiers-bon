"""Analyses requested by the manuscript review (2026-10-14). CPU only, on stored scores.

1. majority.csv        verifier-independent majority vote@N (H6 rule: canonical clusters,
                       unparseable candidates do not vote, ties -> cluster appearing first
                       in candidate order, no parseable candidate -> 0), per population.
2. transitions.csv     N = 32 answer transitions per verifier x condition (main, H6, H7):
                       candidate change, canonical-answer change, correct->wrong (harm),
                       wrong->correct (benefit); Delta = benefit - harm.
3. controls.csv        exploratory selection-rule controls on the main pools: PRM last-step
                       aggregation; reward-weighted vote (registered dev-fitted temperature);
                       parseable-only Best-of-N.
4. confirmatory_all.csv every confirmatory contrast (c1-c5, H6, H7) with population, n,
                       absolute accuracies, estimate, CI, raw and adjusted p, bootstrap
                       exceedance counts and decision.
5. populations.csv     composition of each population.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vq.analysis import score_matrix  # noqa: E402
from vq.data import load_manifest  # noqa: E402
from vq.report import _acc_bon, domain_of  # noqa: E402
from vq.selection import draw_subsets, select_vote, tiebreak_priority  # noqa: E402
from vq.stats import paired_bootstrap  # noqa: E402
from vq.store import CampaignStore  # noqa: E402

OUT = Path("outputs/tables/revision")
ST = CampaignStore("outputs/store")
MAIN = "qwen2.5-1.5b"
STUDIES = {"main": MAIN, "H6": "qwen2.5-math-7b", "H7": "qwen2.5-1.5b-h7"}


def majority_correct(K, C, idx=None):
    """Per-problem correctness of majority vote over candidates idx (default: all)."""
    out = np.zeros(len(K))
    for i in range(len(K)):
        k = K[i] if idx is None else K[i][idx[i]]
        c = C[i] if idx is None else C[i][idx[i]]
        v = k[k >= 0]
        if len(v):
            ids, first, counts = np.unique(v, return_index=True, return_counts=True)
            tied = counts == counts.max()
            w = ids[tied][np.argmin(first[tied])]
            out[i] = float(c[k == w].any())
    return out


def majority_at_n(K, C, n, repeats=200, seed=7):
    if n == K.shape[1]:
        return majority_correct(K, C)
    sub = draw_subsets(K.shape[0], K.shape[1], n, repeats, seed)  # (Q, R, n), same subsets as Best-of-N
    acc = np.zeros(K.shape[0])
    for r in range(repeats):
        acc += majority_correct(K, C, sub[:, r, :])
    return acc / repeats


def acc32(S, C):
    return _acc_bon(S, C, 32, 200, 7, 11)


def winners(S):
    prio = tiebreak_priority(S.shape[0], S.shape[1], 11)
    # argmax with the registered tie-break (higher priority wins among equal scores)
    best = S.max(1, keepdims=True)
    return np.where(S == best, prio, -np.inf).argmax(1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------------- 1. majority vote
    rows = []
    _, C, K, q = score_matrix(ST, MAIN, "sky-8b", "bf16", "eval")
    c1_pop = set(load_manifest("data/manifest_eval.json")["ids"])
    dom = np.array([domain_of(x) for x in q])
    pops = {"main_all_2000": np.ones(len(q), bool), "main_c1_1200": np.array([x in c1_pop for x in q]),
            "main_gsm8k_800": dom == "gsm8k", "main_math_1200": dom == "math"}
    for n in (1, 2, 4, 8, 16, 32):
        m = majority_at_n(K, C, n)
        for pname, msk in pops.items():
            rows.append({"population": pname, "N": n, "majority": m[msk].mean(), "n": int(msk.sum())})
    for study in ("H6", "H7"):
        _, Cs, Ks, qs = score_matrix(ST, STUDIES[study], "sky-8b", "bf16", "eval")
        hard = set(load_manifest(f"data/manifest_{'h6_hard' if study == 'H6' else 'h7'}.json")["ids"])
        msk = np.array([x in hard for x in qs])
        rows.append({"population": f"{study}_hard", "N": 32, "majority": majority_correct(Ks, Cs)[msk].mean(),
                     "n": int(msk.sum())})
    maj = pd.DataFrame(rows)
    maj.to_csv(OUT / "majority.csv", index=False)
    print(maj[maj.N == 32].round(4).to_string(index=False))

    # ---------------- 2. transitions
    rows = []
    plan = [("main", v, c) for v in ("sky-1.7b", "sky-4b", "sky-8b", "prm-7b")
            for c in ("bf16-null", "nf4", "gptq-w4", "gptq-w4-packed", "awq-w4", "rtn-w4", "rtn-w8", "gptq-w3")]
    plan += [(s, v, c) for s in ("H6", "H7") for v in ("sky-8b", "prm-7b") for c in ("bf16-null", "gptq-w4", "gptq-w3")]
    for study, v, c in plan:
        try:
            S1, C1, K1, q1 = score_matrix(ST, STUDIES[study], v, c, "eval")
        except (KeyError, ValueError, FileNotFoundError):
            continue
        S0, _, _, _ = score_matrix(ST, STUDIES[study], v, "bf16", "eval")
        if study != "main":
            hard = set(load_manifest(f"data/manifest_{'h6_hard' if study == 'H6' else 'h7'}.json")["ids"])
            msk = np.array([x in hard for x in q1])
        else:
            msk = np.ones(len(q1), bool)
        r = np.arange(len(q1))
        w0, w1 = winners(S0), winners(S1)
        c0, c1 = C1[r, w0], C1[r, w1]
        changed = w0 != w1
        # two different winners without an extractable answer are not 'the same answer'
        ans_changed = (K1[r, w0] != K1[r, w1]) | (changed & (K1[r, w0] < 0))
        d = {"study": study, "verifier": v, "condition": c, "n": int(msk.sum()),
             "candidate_change": changed[msk].mean(), "answer_change": ans_changed[msk].mean(),
             "same_answer_share_of_changes": (changed & ~ans_changed)[msk].sum() / max(1, changed[msk].sum()),
             "harm": (c0 & ~c1)[msk].mean(), "benefit": (~c0 & c1)[msk].mean()}
        d["delta"] = d["benefit"] - d["harm"]
        rows.append(d)
    tr = pd.DataFrame(rows)
    tr.to_csv(OUT / "transitions.csv", index=False)
    print(tr.round(4).to_string(index=False))

    # ---------------- 3. selection-rule controls (main pools, N = 32)
    temps = json.loads(Path("outputs/tables/vote_temperatures.json").read_text())
    rows = []
    for v in ("sky-1.7b", "sky-4b", "sky-8b", "prm-7b"):
        S0, C, K, q = score_matrix(ST, MAIN, v, "bf16", "eval")
        sub = draw_subsets(len(q), 32, 32, 1, 7)
        prio = tiebreak_priority(len(q), 32, 11)
        T = temps[v]["temperature"]
        for c in ("nf4", "gptq-w4", "gptq-w3"):
            S1, _, _, _ = score_matrix(ST, MAIN, v, c, "eval")
            r = np.arange(len(q))
            for rule in ("weighted_vote", "parseable_only"):
                if rule == "weighted_vote":
                    a0 = C[r, select_vote(S0, K, sub, prio, T)[:, 0]].astype(float)
                    a1 = C[r, select_vote(S1, K, sub, prio, T)[:, 0]].astype(float)
                else:
                    a0 = acc32(np.where(K >= 0, S0, -np.inf), C)
                    a1 = acc32(np.where(K >= 0, S1, -np.inf), C)
                b = paired_bootstrap({"d": a1 - a0}, lambda x: x["d"], 10_000, 7)
                rows.append({"verifier": v, "condition": c, "rule": rule, "acc_bf16": a0.mean(), "acc_q": a1.mean(),
                             "delta": b["estimate"], "lo": b["ci_low"], "hi": b["ci_high"]})
            if v == "prm-7b":
                L0, _, _, _ = score_matrix(ST, MAIN, v, "bf16", "eval", score_col="score_last")
                L1, _, _, _ = score_matrix(ST, MAIN, v, c, "eval", score_col="score_last")
                a0, a1 = acc32(L0, C), acc32(L1, C)
                b = paired_bootstrap({"d": a1 - a0}, lambda x: x["d"], 10_000, 7)
                rows.append({"verifier": v, "condition": c, "rule": "prm_last_step", "acc_bf16": a0.mean(),
                             "acc_q": a1.mean(), "delta": b["estimate"], "lo": b["ci_low"], "hi": b["ci_high"]})
    ctl = pd.DataFrame(rows)
    ctl.to_csv(OUT / "controls.csv", index=False)
    print(ctl.round(4).to_string(index=False))

    # ---------------- 4./5. confirmatory table and populations
    reg = json.loads(Path("outputs/tables/confirmatory.json").read_text())
    tost = pd.read_csv("outputs/tables/review3/c1_tost.csv").iloc[0]
    rows = [{"test": "c1 (8B ORM, GPTQ-W4, equivalence +-3 pp)", "population": "main_c1_1200", "n": 1200,
             "estimate": reg["c1_delta_8b_gptq_w4"]["estimate"], "ci": "90%",
             "lo": tost.ci90_low, "hi": tost.ci90_high, "p_raw": tost.p_tost,
             "p_note": "TOST, 0 of 10,000 resamples beyond either margin", "decision": "equivalent"}]
    for k, name in (("c2_memory_frontier", "c2 (6 GB ceiling: 4B GPTQ-W4 packed vs 1.7B BF16)"),
                    ("c3_ch_vs_dpq", "c3 (CH-GPTQ vs DPQ; not run)"),
                    ("c4_math_delta_gptq_w4_n32", "c4 (MATH Delta, 4B+8B GPTQ-W4, N=32)"),
                    ("c5_math_growth_with_n_gptq_w4", "c5 (MATH Delta growth N=4 -> 32)")):
        r = reg[k]
        rows.append({"test": name, "population": "main_all_2000" if k == "c2_memory_frontier" else "main_math_1200",
                     "n": r.get("n"), "estimate": r.get("estimate"), "ci": "95%", "lo": r.get("ci_low"),
                     "hi": r.get("ci_high"), "p_raw": r.get("p_value"), "p_holm": r.get("p_holm"),
                     "decision": "supported" if r.get("p_holm", 1) < 0.05 else "not supported / not tested"})
    for study, tag in (("H6", "h6"), ("H7", "h7")):
        rep = json.loads(Path(f"outputs/tables/{tag}/{tag}_confirmatory.json").read_text())
        for k, cr in rep["confirmatory"].items():
            v = k.split("-", 1)[1]
            pv = rep["per_verifier"][v]
            is_a = k[2] == "a"
            rows.append({"test": k, "population": f"{study}_hard", "n": pv["n"], "acc_bf16": pv["acc_bf16"],
                         "acc_q": pv["acc_gptq_w4"], "majority": pv["majority"], "estimate": cr["estimate"],
                         "ci": "95%", "lo": (pv["delta_ci95"] if is_a else pv["R_ci95"])[0],
                         "hi": (pv["delta_ci95"] if is_a else pv["R_ci95"])[1], "p_raw": cr["p"],
                         "p_holm": cr["p_holm"], "exceedances_of_10000": round(cr["p"] * 10_000),
                         "decision": "supported" if cr["supported"] else cr.get("reading", "not supported")})
    pd.DataFrame(rows).to_csv(OUT / "confirmatory_all.csv", index=False)
    pop = [{"population": "main_all_2000", "composition": "800 GSM8K test + 400 MATH500 + 800 MATH test"},
           {"population": "main_c1_1200", "composition": "800 GSM8K test + 400 MATH500 (original manifest)"},
           {"population": "main_math_1200", "composition": "400 MATH500 + 800 MATH test"},
           {"population": "H6_hard", "composition": "800 MATH test Level 5 (new)"},
           {"population": "H7_hard", "composition": "905 MATH test Level 4 + 155 Level 5 (new)"}]
    pd.DataFrame(pop).to_csv(OUT / "populations.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
