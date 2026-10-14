"""Exploratory revision experiments (docs/preregistration_rev.md), CPU only.

A  skyllama-8b / tulu-8b: Delta_32 on the main 2,000 and on H7 (95% CI, 90% CI vs +-3 pp),
   H7 gain over majority vote and R = 0.5 gain + Delta, answer transitions, GPTQ-W3 Delta.
B  gptq-w3-math vs BF16 and vs generic gptq-w3 (paired difference), 1.7B and 8B.
C  outputs/memory_revision.json summary (memory and throughput).
Writes outputs/tables/revision/runs_A.csv, runs_B.csv, memory_C.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vq.analysis import score_matrix  # noqa: E402
from vq.data import load_manifest  # noqa: E402
from vq.report import _acc_bon  # noqa: E402
from vq.stats import paired_bootstrap  # noqa: E402
from vq.store import CampaignStore  # noqa: E402

OUT = Path("outputs/tables/revision")
ST = CampaignStore("outputs/store")


def boot(x):
    b = paired_bootstrap({"d": np.asarray(x, float)}, lambda d: d["d"], 10_000, 7)
    rng = np.random.default_rng(7)
    bs = x[rng.integers(0, len(x), (10_000, len(x)))].mean(1)
    b["ci90"] = [float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))]
    b["p_le0"] = float(np.mean(bs <= 0))
    return b


def majority(K, C):
    out = np.zeros(len(K))
    for i, (k, c) in enumerate(zip(K, C)):
        v = k[k >= 0]
        if len(v):
            ids, first, counts = np.unique(v, return_index=True, return_counts=True)
            tied = counts == counts.max()
            out[i] = float(c[k == ids[tied][np.argmin(first[tied])]].any())
    return out


def acc(S, C):
    return _acc_bon(S, C, 32, 200, 7, 11)


def load(gen, v, c):
    try:
        return score_matrix(ST, gen, v, c, "eval")
    except (KeyError, ValueError, FileNotFoundError):
        return None


def transitions(S0, S1, C, K):
    r = np.arange(len(C))
    w0, w1 = S0.argmax(1), S1.argmax(1)
    c0, c1 = C[r, w0], C[r, w1]
    ch = w0 != w1
    ans = (K[r, w0] != K[r, w1]) | (ch & (K[r, w0] < 0))
    return {"cand_change": float(np.mean(ch)), "answer_change": float(np.mean(ans)),
            "harm": float(np.mean(c0 & ~c1)), "benefit": float(np.mean(~c0 & c1))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    h7 = set(load_manifest("data/manifest_h7.json")["ids"])
    for v in ("skyllama-8b", "tulu-8b"):
        for study, gen in (("main", "qwen2.5-1.5b"), ("H7", "qwen2.5-1.5b-h7")):
            base = load(gen, v, "bf16")
            if base is None:
                continue
            S0, C, K, q = base
            msk = np.ones(len(q), bool) if study == "main" else np.array([x in h7 for x in q])
            a0, mj = acc(S0, C), majority(K, C)
            g = boot(((a0 - mj))[msk])
            for cond in ("gptq-w4", "gptq-w3"):
                r = load(gen, v, cond)
                if r is None:
                    continue
                S1 = r[0]
                a1 = acc(S1, C)
                d = boot((a1 - a0)[msk])
                row = {"verifier": v, "study": study, "condition": cond, "n": int(msk.sum()),
                       "acc_bf16": a0[msk].mean(), "acc_q": a1[msk].mean(), "majority": mj[msk].mean(),
                       "gain": g["estimate"], "gain_lo": g["ci_low"], "gain_hi": g["ci_high"],
                       "delta": d["estimate"], "delta_lo": d["ci_low"], "delta_hi": d["ci_high"],
                       "delta_ci90": d["ci90"], "within_3pp_90": bool(-0.03 < d["ci90"][0] and d["ci90"][1] < 0.03)}
                if study == "H7":
                    R = boot(((a1 - mj) - 0.5 * (a0 - mj))[msk])
                    row.update({"R": R["estimate"], "R_lo": R["ci_low"], "R_hi": R["ci_high"], "R_p_le0": R["p_le0"]})
                row.update(transitions(S0[msk], S1[msk], C[msk], K[msk]))
                rows.append(row)
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "runs_A.csv", index=False)
    print(A.round(4).to_string(index=False))

    rows = []
    for v in ("sky-1.7b", "sky-8b"):
        S0, C, K, q = score_matrix(ST, "qwen2.5-1.5b", v, "bf16", "eval")
        g = load("qwen2.5-1.5b", v, "gptq-w3")
        m = load("qwen2.5-1.5b", v, "gptq-w3-math")
        if m is None:
            continue
        a0, ag, am = acc(S0, C), acc(g[0], C), acc(m[0], C)
        dm, dd = boot(am - a0), boot(am - ag)
        rows.append({"verifier": v, "acc_bf16": a0.mean(), "acc_w3_generic": ag.mean(), "acc_w3_math": am.mean(),
                     "delta_math": dm["estimate"], "delta_math_lo": dm["ci_low"], "delta_math_hi": dm["ci_high"],
                     "math_minus_generic": dd["estimate"], "mmg_lo": dd["ci_low"], "mmg_hi": dd["ci_high"],
                     **transitions(S0, m[0], C, K)})
    B = pd.DataFrame(rows)
    B.to_csv(OUT / "runs_B.csv", index=False)
    print(B.round(4).to_string(index=False))

    mp = Path("outputs/memory_revision.json")
    if mp.exists():
        mem = json.loads(mp.read_text())
        Cdf = pd.DataFrame([{"config": k, "peak_gib": x.get("peak_gb"), "weights_gib": x.get("weights_gb"),
                             "seqs_per_s": (x.get("speed") or {}).get("seqs_per_s"),
                             "tokens_per_s": (x.get("speed") or {}).get("tokens_per_s"),
                             "gpu": (x.get("speed") or {}).get("gpu"), "status": x.get("status", "ok")}
                            for k, x in mem.items()])
        Cdf.to_csv(OUT / "memory_C.csv", index=False)
        print(Cdf.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
