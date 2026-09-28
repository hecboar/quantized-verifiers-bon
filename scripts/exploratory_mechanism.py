"""Exploratory (not registered): why quantization changes the Best-of-32 winner
often but accuracy rarely. Per verifier x condition, by domain and MATH level:
winner-change rate, share of changes that keep the final answer, harm H and
benefit B (flips of correctness), BF16 answer redundancy and score margins."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vq.analysis import score_matrix  # noqa: E402
from vq.report import domain_of  # noqa: E402
from vq.store import CampaignStore  # noqa: E402

G = "qwen2.5-1.5b"
OUT = Path("outputs/tables/exploratory")
LEVELS = json.loads(Path("data/math_levels.json").read_text())


def boot(x, rng, n=2000):
    x = np.asarray(x, float)
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), *np.percentile(b, [2.5, 97.5])


def per_question(store, v, cond):
    S0, C0, K0, q = score_matrix(store, G, v, "bf16", "eval")
    S, C, K, q2 = score_matrix(store, G, v, cond, "eval")
    assert q == q2
    r = np.arange(len(q))
    w0, w = S0.argmax(1), S.argmax(1)
    srt = np.sort(S0, 1)
    pert = np.abs(S - S0)
    same_ans = K0 == K0[r, w0][:, None]  # candidates sharing the BF16 winner's answer
    df = pd.DataFrame({
        "question_id": q,
        "domain": [domain_of(x) for x in q],
        "level": [LEVELS.get(x, 0) for x in q],
        "changed": w != w0,
        "same_answer": K[r, w] == K0[r, w0],
        "harm": C0[r, w0] & ~C[r, w],
        "benefit": ~C0[r, w0] & C[r, w],
        "redundancy": same_ans.mean(1),                       # answer mass of the BF16 winner
        "margin_rel": (srt[:, -1] - srt[:, -2]) / (pert.mean(1) + 1e-9),  # top-2 gap / mean perturbation
        "pool_acc": C0.mean(1),
    })
    df.loc[df.question_id.str.startswith("gsm8k"), "level"] = 0
    return df


def summarise(df, rng):
    ch = df[df.changed]
    d = (df.benefit.astype(float) - df.harm.astype(float)).to_numpy()
    m, lo, hi = boot(d, rng)
    return {"n": len(df), "winner_change": df.changed.mean(),
            "same_answer_given_change": ch.same_answer.mean() if len(ch) else np.nan,
            "H": df.harm.mean(), "B": df.benefit.mean(), "dA": m, "dA_lo": lo, "dA_hi": hi,
            "redundancy": df.redundancy.mean()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    store, rng, rows, allq = CampaignStore("outputs/store"), np.random.default_rng(0), [], []
    for v in ("sky-1.7b", "sky-4b", "sky-8b"):
        for cond in ("nf4", "gptq-w4", "awq-w4", "rtn-w4", "gptq-w3", "rtn-w8", "bf16-null"):
            try:
                df = per_question(store, v, cond)
            except (FileNotFoundError, KeyError, ValueError):
                continue
            df["verifier"], df["condition"] = v, cond
            allq.append(df)
            for key, g in [("all", df), ("gsm8k", df[df.domain == "gsm8k"]), ("math", df[df.domain == "math"])] + \
                          [(f"math-L{l}", df[df.level == l]) for l in range(1, 6)]:
                rows.append({"verifier": v, "condition": cond, "subset": key, **summarise(g, rng)})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "mechanism.csv", index=False)
    pq = pd.concat(allq, ignore_index=True)
    pq.to_parquet(OUT / "mechanism_per_question.parquet")
    # harm/benefit vs BF16 answer redundancy (all verifiers, 4-bit GPTQ + W3)
    pq["red_bin"] = pd.cut(pq.redundancy, [0, .1, .2, .35, .5, .75, 1.0], include_lowest=True)
    red = pq[pq.condition.isin(["gptq-w4", "gptq-w3"])].groupby(["condition", "red_bin"], observed=True).agg(
        n=("harm", "size"), change=("changed", "mean"), H=("harm", "mean"), B=("benefit", "mean"))
    red["net"] = red.B - red.H
    red.to_csv(OUT / "mechanism_by_redundancy.csv")
    pd.set_option("display.width", 220)
    print(tab.round(4).to_string())
    print(red.round(4).to_string())


if __name__ == "__main__":
    main()
