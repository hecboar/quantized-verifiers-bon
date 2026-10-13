"""H6 analysis, frozen with docs/preregistration_h6.md (§5-§7) before any H6 data exists.

Confirmatory family (Holm over four, alpha 0.05), stratum H:
  H6a-v: TOST, mean Delta_v within +-3 pp     p = max(P*(m <= -.03), P*(m >= .03))
  H6b-v: R_v = mean[(A_q - Maj) - 0.5 (A_bf16 - Maj)] > 0   p = P*(m <= 0)
for v in {sky-8b, prm-7b}; paired percentile bootstrap over questions, 10,000, seed 7.
Everything else is exploratory. `--dry-run` runs the same code on an existing pool
(e.g. the registered generator's Level-5 questions) to test the script only.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vq.analysis import score_matrix  # noqa: E402
from vq.data import load_items, load_manifest  # noqa: E402
from vq.grading import grade_lenient  # noqa: E402
from vq.report import _acc_bon  # noqa: E402
from vq.stats import holm  # noqa: E402
from vq.store import CampaignStore  # noqa: E402

VERIFIERS = ("sky-8b", "prm-7b")
MARGIN, R_FRAC, B, SEED = 0.03, 0.5, 10_000, 7
REPEATS, SUBSET_SEED, TIEBREAK = 200, 7, 11
_LV = Path("data/math_levels.json")
LEVELS = json.loads(_LV.read_text()) if _LV.exists() else {}  # MATH difficulty level per item id


def boot(x, seed=SEED):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    return x[rng.integers(0, len(x), (B, len(x)))].mean(1)


def ci(b, level=0.95):
    a = (1 - level) / 2
    return [float(np.quantile(b, a)), float(np.quantile(b, 1 - a))]


def majority(K, C):
    """Registered canonical clusters (vq.generate.answer_keys); key -1 does not vote;
    ties -> the tied cluster that appears first in candidate order; no parseable -> 0.
    (On full pools this equals the lowest key id, since answer_keys numbers clusters by
    first appearance; with candidates removed it need not, hence the explicit rule.)"""
    out = np.zeros(len(K))
    for i, (k, c) in enumerate(zip(K, C)):
        v = k[k >= 0]
        if len(v):
            ids, first, counts = np.unique(v, return_index=True, return_counts=True)
            tied = counts == counts.max()
            w = ids[tied][np.argmin(first[tied])]
            out[i] = float(c[k == w].any())
    return out


def acc(S, C, n=32):
    return _acc_bon(S, C, n, REPEATS, SUBSET_SEED, TIEBREAK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="qwen2.5-math-7b")
    ap.add_argument("--hard", default="data/manifest_h6_hard.json")
    ap.add_argument("--easy", default="data/manifest_h6_easy.json")
    ap.add_argument("--out", default="outputs/tables/h6")
    ap.add_argument("--lenient-cache", default=None, help="parquet from scripts/lenient_cache.py")
    ap.add_argument("--prefix", default="H6", help="hypothesis label prefix (H7 reuses this frozen analysis)")
    ap.add_argument("--dry-run", action="store_true", help="test on an existing pool; results are not H6")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    st = CampaignStore("outputs/store")
    items = {it.id: it for it in load_items("data/items.jsonl")}
    hard = set(load_manifest(a.hard)["ids"])
    easy = set() if a.easy.lower() == "none" else set(load_manifest(a.easy)["ids"])

    cand = st.read("candidates", f"{a.generator}__eval").sort_values(["question_id", "cand"], kind="stable")
    qids = sorted(cand.question_id.unique())
    Cn = cand.groupby("question_id").size().iloc[0]
    trunc = cand.truncated.to_numpy(bool).reshape(len(qids), Cn)
    if a.lenient_cache:  # scripts/lenient_cache.py: same verdicts, safe on Windows (math_verify timeouts)
        lz = pd.read_parquet(a.lenient_cache).set_index(["question_id", "cand"]).correct_lenient
        L = lz.loc[list(zip(cand.question_id, cand.cand))].to_numpy(bool)
    else:
        L = np.array([grade_lenient(t, items[q].gold, items[q].kind) for q, t in zip(cand.question_id, cand.text)])
    L = L.reshape(len(qids), Cn)
    mH = np.array([q in hard for q in qids])
    mE = np.array([q in easy for q in qids])
    # input invariants: pool = manifest(s) exactly, equal pools of 32, no duplicates
    assert set(qids) == hard | easy, "pool questions differ from the manifest(s)"
    assert Cn == 32 and (cand.groupby("question_id").size() == 32).all(), "pool sizes are not all 32"
    assert not cand.duplicated(["question_id", "cand"]).any(), "duplicated candidates"
    assert mH.sum() == len(hard), "pool does not cover stratum H"

    def load(v, c):
        S, C, K, q = score_matrix(st, a.generator, v, c, "eval")
        assert q == qids
        return S, C, K

    res, expl = {}, []
    pvals = {}
    for v in VERIFIERS:
        S0, C, K = load(v, "bf16")
        S1, _, _ = load(v, "gptq-w4")
        A0, A1, Mj = acc(S0, C), acc(S1, C), majority(K, C)
        d, g = (A1 - A0)[mH], (A0 - Mj)[mH]
        r = ((A1 - Mj) - R_FRAC * (A0 - Mj))[mH]
        bd, br, bg = boot(d), boot(r), boot(g)
        pvals[f"{a.prefix}a-{v}"] = float(max(np.mean(bd <= -MARGIN), np.mean(bd >= MARGIN)))
        pvals[f"{a.prefix}b-{v}"] = float(np.mean(br <= 0))
        res[v] = {
            "n": int(mH.sum()), "acc_bf16": A0[mH].mean(), "acc_gptq_w4": A1[mH].mean(), "majority": Mj[mH].mean(),
            "oracle_pass32": C[mH].any(1).mean(), "random_candidate": C[mH].mean(),
            "extractable": float((K[mH] >= 0).mean()), "truncated": float(trunc[mH].mean()),
            "delta": d.mean(), "delta_ci95": ci(bd), "delta_ci90": ci(bd, 0.90),
            "R": r.mean(), "R_ci95": ci(br), "gain_M": g.mean(), "gain_M_ci97.5": ci(bg, 0.975),
            "winner_change": float(np.mean(S0.argmax(1)[mH] != S1.argmax(1)[mH])),
        }
        # --- exploratory
        if ci(bg)[0] > 0:
            num, den = boot((A1 - Mj)[mH]), boot((A0 - Mj)[mH])
            res[v]["retention_fraction"] = float((A1 - Mj)[mH].mean() / (A0 - Mj)[mH].mean())
            ratio = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)  # paired resamples
            res[v]["retention_fraction_ci95"] = ci(ratio[np.isfinite(ratio)])
        rr = np.arange(len(qids))
        w0 = S0.argmax(1)
        majk = np.array([np.bincount(k[k >= 0]).argmax() if (k >= 0).any() else -2 for k in K])
        contested = K[rr, w0] != majk
        for name, m in (("contested", mH & contested), ("agree", mH & ~contested)):
            bb = boot((A1 - A0)[m])
            expl.append({"verifier": v, "analysis": f"delta_{name}", "n": int(m.sum()),
                         "estimate": (A1 - A0)[m].mean(), "ci95": ci(bb)})
        for grader, CC in (("lenient", L),):
            a0, a1, mj = acc(S0, CC), acc(S1, CC), majority(K, CC)
            for k_, x in (("delta", a1 - a0), ("R", (a1 - mj) - R_FRAC * (a0 - mj)), ("gain_M", a0 - mj)):
                expl.append({"verifier": v, "analysis": f"{grader}:{k_}", "n": int(mH.sum()),
                             "estimate": x[mH].mean(), "ci95": ci(boot(x[mH]))})
        Kt = np.where(trunc, -1, K)
        a0, a1 = acc(np.where(trunc, -np.inf, S0), C), acc(np.where(trunc, -np.inf, S1), C)
        allt = trunc.all(1)  # nothing left to select once truncated candidates are removed
        a0, a1 = np.where(allt, 0.0, a0), np.where(allt, 0.0, a1)
        mj = majority(Kt, C)
        for k_, x in (("delta", a1 - a0), ("R", (a1 - mj) - R_FRAC * (a0 - mj)), ("gain_M", a0 - mj)):
            expl.append({"verifier": v, "analysis": f"no_truncated:{k_}", "n": int(mH.sum()),
                         "estimate": x[mH].mean(), "ci95": ci(boot(x[mH]))})
        for cond in ("bf16-null", "gptq-w3"):
            try:
                Sx, _, _ = load(v, cond)
            except (KeyError, ValueError, FileNotFoundError):
                expl.append({"verifier": v, "analysis": f"{cond}", "n": 0, "estimate": None, "ci95": None,
                             "note": "not run"})
                continue
            ax = acc(Sx, C)
            for k_, x in (("delta", ax - A0), ("R", (ax - Mj) - R_FRAC * (A0 - Mj))):
                expl.append({"verifier": v, "analysis": f"{cond}:{k_}", "n": int(mH.sum()),
                             "estimate": x[mH].mean(), "ci95": ci(boot(x[mH])),
                             "winner_change": float(np.mean(Sx.argmax(1)[mH] != w0[mH]))})
        if mE.any():
            for k_, x in (("delta", A1 - A0), ("R", (A1 - Mj) - R_FRAC * (A0 - Mj)), ("gain_M", A0 - Mj)):
                expl.append({"verifier": v, "analysis": f"stratumE:{k_}", "n": int(mE.sum()),
                             "estimate": x[mE].mean(), "ci95": ci(boot(x[mE]))})
            hd = (A1 - A0)
            bH, bE = boot(hd[mH], 1), boot(hd[mE], 2)
            expl.append({"verifier": v, "analysis": "delta_H_minus_E", "n": int(mH.sum() + mE.sum()),
                         "estimate": hd[mH].mean() - hd[mE].mean(), "ci95": ci(bH - bE)})
        for n in (1, 2, 4, 8, 16, 32):
            x = (acc(S1, C, n) - acc(S0, C, n))[mH]
            expl.append({"verifier": v, "analysis": f"delta_N{n}", "n": int(mH.sum()),
                         "estimate": x.mean(), "ci95": ci(boot(x))})
        # per-level estimates and (if several levels) the equal-level-weight aggregate
        lv = np.array([LEVELS.get(q, 0) for q in qids])
        for lvl in sorted(set(lv[mH])):
            mL = mH & (lv == lvl)
            for k_, x in (("delta", A1 - A0), ("R", (A1 - Mj) - R_FRAC * (A0 - Mj)), ("gain_M", A0 - Mj)):
                expl.append({"verifier": v, "analysis": f"level{lvl}:{k_}", "n": int(mL.sum()),
                             "estimate": x[mL].mean(), "ci95": ci(boot(x[mL]))})
            expl.append({"verifier": v, "analysis": f"level{lvl}:truncated_rate", "n": int(mL.sum()),
                         "estimate": float(trunc[mL].mean()), "ci95": None})
        if len(set(lv[mH])) > 1:
            for k_, x in (("delta", A1 - A0), ("R", (A1 - Mj) - R_FRAC * (A0 - Mj))):
                expl.append({"verifier": v, "analysis": f"equal_level_weight:{k_}", "n": int(mH.sum()),
                             "estimate": float(np.mean([x[mH & (lv == u)].mean() for u in set(lv[mH])])),
                             "ci95": None})
        # subject-stratified bootstrap of the confirmatory estimates
        subj = np.array(["-".join(q.split("-")[1:-1]) for q in qids])[mH]
        rng = np.random.default_rng(SEED)
        for k_, x in (("delta", d), ("R", r)):
            bs = []
            for _ in range(2000):
                bs.append(np.mean(np.concatenate([rng.choice(x[subj == s], (subj == s).sum()) for s in np.unique(subj)])))
            expl.append({"verifier": v, "analysis": f"subject_stratified:{k_}", "n": int(mH.sum()),
                         "estimate": x.mean(), "ci95": ci(np.array(bs))})

    keys = [f"{a.prefix}a-{v}" for v in VERIFIERS] + [f"{a.prefix}b-{v}" for v in VERIFIERS]
    adj = holm(np.array([pvals[k] for k in keys]))
    conf = {}
    for k, pa in zip(keys, adj):
        v = k.split("-", 1)[1]
        is_a = k.startswith(f"{a.prefix}a-")
        est = res[v]["delta"] if is_a else res[v]["R"]
        conf[k] = {"estimate": est, "p": pvals[k], "p_holm": float(pa), "supported": bool(pa < 0.05)}
        if is_a and pa >= 0.05:
            lo, hi = res[v]["delta_ci95"]
            conf[k]["reading"] = "detected change" if (lo > 0 or hi < 0) else "inconclusive"
    subjects = pd.Series(["-".join(q.split("-")[1:-1]) for q in qids])[mH].value_counts().to_dict()
    report = {"dry_run": a.dry_run, "generator": a.generator, "confirmatory": conf, "per_verifier": res,
              "stratum_H_subjects": subjects}
    tag = "dryrun_" if a.dry_run else ""
    (out / f"{tag}{a.prefix.lower()}_confirmatory.json").write_text(json.dumps(report, indent=1, default=float))
    pd.DataFrame(expl).to_csv(out / f"{tag}{a.prefix.lower()}_exploratory.csv", index=False)
    print(json.dumps(conf, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
