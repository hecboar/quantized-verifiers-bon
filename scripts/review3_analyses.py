"""Analyses requested by the third external review (all on existing scores, no GPU).

1. c1 equivalence: TOST at the registered +-3 pp margin (90% bootstrap CI), with
   the per-question vector used for c1 in the registered report.
2. Selection-free baselines: oracle pass@32 and (unweighted) majority vote @32.
3. Packed GPTQ-W4 (fp16 Marlin) vs fake-quant GPTQ-W4: paired answer and
   correctness transitions.
4. 'Contested' questions: harm/benefit by BF16 answer-redundancy bin with
   denominators and bootstrap CIs, strict and lenient graders (bins fixed after
   first inspection: exploratory).
5. Confusion matrix of the 50-item blinded format audit.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vq.analysis import score_matrix  # noqa: E402
from vq.data import load_manifest  # noqa: E402
from vq.report import _acc_bon, domain_of  # noqa: E402
from vq.stats import tost_equivalence  # noqa: E402
from vq.store import CampaignStore  # noqa: E402

G = "qwen2.5-1.5b"
OUT = Path("outputs/tables/review3")


def boot_ci(x, n=10_000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), *map(float, np.percentile(b, [2.5, 97.5]))


def lenient_matrix(cand, q):
    return cand.set_index(["question_id", "cand"]).correct_lenient.unstack("cand").loc[q].to_numpy(bool)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    st = CampaignStore("outputs/store")
    lz = pd.read_parquet(f"outputs/tables/exploratory/lenient_{G}.parquet")
    cand = st.read("candidates", f"{G}__eval").merge(lz, on=["question_id", "cand"])

    # 1. c1 TOST (same subsets/tie-break as the registered report: repeats 200, seed 7, tiebreak 11)
    S0, C0, K0, q = score_matrix(st, G, "sky-8b", "bf16", "eval")
    S1, C1, _, _ = score_matrix(st, G, "sky-8b", "gptq-w4", "eval")
    keep = np.array([x in set(load_manifest("data/manifest_eval.json")["ids"]) for x in q])
    d1 = (_acc_bon(S1, C1, 32, 200, 7, 11) - _acc_bon(S0, C0, 32, 200, 7, 11))[keep]
    t = tost_equivalence(d1, margin=0.03, n_boot=10_000, seed=7)
    # bootstrap TOST p: larger of the two one-sided p-values
    rng = np.random.default_rng(7)
    b = d1[rng.integers(0, len(d1), (10_000, len(d1)))].mean(1)
    t["p_tost"] = float(max(np.mean(b <= -0.03), np.mean(b >= 0.03)))
    t["errors_increase_at_margin"] = 0.03 / (1 - float(_acc_bon(S0, C0, 32, 200, 7, 11)[keep].mean()))
    pd.DataFrame([t]).to_csv(OUT / "c1_tost.csv", index=False)
    print("c1 TOST:", {k: round(v, 4) if isinstance(v, float) else v for k, v in t.items()})

    # 2. baselines per domain
    rows = []
    for v in ("sky-1.7b", "sky-4b", "sky-8b", "prm-7b"):
        S, C, K, qq = score_matrix(st, G, v, "bf16", "eval")
        dom = np.array([domain_of(x) for x in qq])
        L = lenient_matrix(cand, qq)
        maj = np.array([np.bincount(k[k >= 0]).argmax() if (k >= 0).any() else -1 for k in K])
        maj_ok = np.array([C[i][K[i] == maj[i]].any() if maj[i] >= 0 else False for i in range(len(qq))])
        for key in ("all", "gsm8k", "math"):
            m = np.ones(len(qq), bool) if key == "all" else dom == key
            rows.append({"verifier": v, "subset": key, "bon32_bf16": _acc_bon(S, C, 32, 200, 7, 11)[m].mean(),
                         "bon32_bf16_lenient": _acc_bon(S, L, 32, 200, 7, 11)[m].mean(),
                         "majority32": maj_ok[m].mean(), "oracle_pass32": C[m].any(1).mean(),
                         "oracle_pass32_lenient": L[m].any(1).mean(), "random_candidate": C[m].mean()})
    base = pd.DataFrame(rows)
    base.to_csv(OUT / "baselines.csv", index=False)
    print(base.round(4).to_string(index=False))

    # 3. packed vs fake transitions
    rows = []
    for v in ("sky-1.7b", "sky-4b", "sky-8b"):
        Sf, Cf, Kf, qq = score_matrix(st, G, v, "gptq-w4", "eval")
        Sp, Cp, Kp, _ = score_matrix(st, G, v, "gptq-w4-packed", "eval")
        r = np.arange(len(qq))
        wf, wp = Sf.argmax(1), Sp.argmax(1)
        cf, cp = Cf[r, wf], Cp[r, wp]
        rows.append({"verifier": v, "winner_same": np.mean(wf == wp), "answer_same": np.mean(Kf[r, wf] == Kp[r, wp]),
                     "both_correct": np.mean(cf & cp), "fake_only": np.mean(cf & ~cp), "packed_only": np.mean(~cf & cp),
                     "both_wrong": np.mean(~cf & ~cp), "acc_fake": cf.mean(), "acc_packed": cp.mean()})
    tr = pd.DataFrame(rows)
    tr.to_csv(OUT / "packed_vs_fake.csv", index=False)
    print(tr.round(4).to_string(index=False))

    # 4. contested questions
    rows = []
    for v in ("sky-1.7b", "sky-4b", "sky-8b"):
        S0, C0, K0, qq = score_matrix(st, G, v, "bf16", "eval")
        L = lenient_matrix(cand, qq)
        r = np.arange(len(qq))
        w0 = S0.argmax(1)
        red = (K0 == K0[r, w0][:, None]).mean(1)
        bins = pd.cut(red, [0, .2, .5, 1.0], include_lowest=True)
        for cond in ("bf16-null", "gptq-w4", "gptq-w3"):
            S, *_ = score_matrix(st, G, v, cond, "eval")
            w = S.argmax(1)
            for grader, M in (("strict", C0), ("lenient", L)):
                net = M[r, w].astype(float) - M[r, w0].astype(float)
                for b in bins.categories:
                    m = np.asarray(bins == b)
                    est, lo, hi = boot_ci(net[m])
                    rows.append({"verifier": v, "condition": cond, "grader": grader, "redundancy_bin": str(b),
                                 "n": int(m.sum()), "net": est, "lo": lo, "hi": hi,
                                 "winner_change": float(np.mean(w[m] != w0[m]))})
    ct = pd.DataFrame(rows)
    ct.to_csv(OUT / "contested.csv", index=False)
    print(ct[ct.grader == "lenient"].round(4).to_string(index=False))

    # 5. format-audit confusion matrix
    key = pd.read_csv("outputs/tables/exploratory/audit/audit_key_DO_NOT_OPEN.csv")
    ver = pd.read_csv("outputs/tables/exploratory/audit/audit_verdicts_blind.csv")
    a = key.merge(ver, on="audit_id")
    cm = pd.crosstab([a.dom, a.correct_lenient.map({True: "lenient: correct", False: "lenient: wrong"})],
                     a.blind_correct.map({1: "audit: correct", 0: "audit: wrong"}))
    cm.to_csv(OUT / "format_audit_confusion.csv")
    print(cm)


if __name__ == "__main__":
    main()
