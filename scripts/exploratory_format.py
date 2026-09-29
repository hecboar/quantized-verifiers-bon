"""Exploratory (preregistration §7.6): answer-format effects.

1. Re-grade candidates without an extractable final answer with a lenient
   grader (vq.grading.grade_lenient) -> how many of them are right in prose.
2. For every verifier x condition: winners' unparseable rate and the
   decomposition of the Best-of-32 change,
       dA = dp * a0|p  (format: more/fewer parseable winners)
          + p1 * (a1|p - a0|p)  (content: accuracy among parseable winners),
   under the registered (strict) grader, plus dA under the lenient grader.
3. Lenient-grader sensitivity of the confirmatory estimands c1, c4, c5.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vq.analysis import confirmatory, score_matrix  # noqa: E402
from vq.data import load_items, load_manifest  # noqa: E402
from vq.grading import grade_lenient  # noqa: E402
from vq.report import _acc_bon, domain_of  # noqa: E402
from vq.store import CampaignStore  # noqa: E402

OUT = Path("outputs/tables/exploratory")
GENS = {"qwen2.5-1.5b": ("sky-1.7b", "sky-4b", "sky-8b", "prm-7b"), "llama3.2-3b": ("sky-4b", "prm-7b")}
CONDS = ("bf16-null", "nf4", "rtn-w8", "rtn-w4", "awq-w4", "gptq-w4", "gptq-w3")


def lenient_pool(store, gen, items):
    path = OUT / f"lenient_{gen}.parquet"
    c = store.read("candidates", f"{gen}__eval")
    if path.exists():
        lz = pd.read_parquet(path)
    else:
        lz = c[["question_id", "cand"]].copy()
        lz["correct_lenient"] = [
            bool(ok) if p else grade_lenient(t, items[q].gold, items[q].kind)
            for t, q, p, ok in zip(c.text, c.question_id, c.parsed, c.correct)]
        lz.to_parquet(path)
    return c.merge(lz, on=["question_id", "cand"])


def matrix(df, col, qids):
    d = df.set_index(["question_id", "cand"])[col].unstack("cand").loc[qids]
    return d.to_numpy(bool)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    store, items = CampaignStore("outputs/store"), {it.id: it for it in load_items("data/items.jsonl")}
    pool_rows, rows, sens = [], [], {}
    for gen, verifiers in GENS.items():
        c = lenient_pool(store, gen, items)
        c["domain"] = [domain_of(q) for q in c.question_id]
        for dom, g in [("all", c)] + list(c.groupby("domain")):
            u = g[~g.parsed]
            pool_rows.append({"generator": gen, "domain": dom, "unparseable": (~g.parsed).mean(),
                              "unparseable_right_lenient": u.correct_lenient.mean() if len(u) else np.nan,
                              "acc_strict": g.correct.mean(), "acc_lenient": g.correct_lenient.mean()})
        for v in verifiers:
            try:
                S0, C0, _, q = score_matrix(store, gen, v, "bf16", "eval")
            except FileNotFoundError:
                continue
            P = matrix(c, "parsed", q)
            L = matrix(c, "correct_lenient", q)
            dom = np.array([domain_of(x) for x in q])
            w0 = S0.argmax(1)
            r = np.arange(len(q))
            a0s, a0l = _acc_bon(S0, C0, 32, 200, 0, 11), _acc_bon(S0, L, 32, 200, 0, 11)
            for cond in CONDS:
                try:
                    S, C, _, q2 = score_matrix(store, gen, v, cond, "eval")
                except (FileNotFoundError, KeyError, ValueError):
                    continue
                assert q2 == q
                w = S.argmax(1)
                a1s, a1l = _acc_bon(S, C, 32, 200, 0, 11), _acc_bon(S, L, 32, 200, 0, 11)
                sens[(gen, v, cond)] = (a1s - a0s, a1l - a0l, dom, q)
                for key in ("all", "gsm8k", "math"):
                    m = np.ones(len(q), bool) if key == "all" else dom == key
                    p0, p1 = P[r, w0][m].mean(), P[r, w][m].mean()
                    c0p = C0[r, w0][m][P[r, w0][m]].mean()
                    c1p = C[r, w][m][P[r, w][m]].mean()
                    rows.append({"generator": gen, "verifier": v, "condition": cond, "subset": key, "n": int(m.sum()),
                                 "winner_parsed_bf16": p0, "winner_parsed_q": p1,
                                 "dA_strict": (a1s - a0s)[m].mean(), "dA_lenient": (a1l - a0l)[m].mean(),
                                 "dA_format": (p1 - p0) * c0p, "dA_content": p1 * (c1p - c0p),
                                 "acc_parsed_bf16": c0p, "acc_parsed_q": c1p})
    pd.DataFrame(pool_rows).to_csv(OUT / "format_pool.csv", index=False)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "format_decomposition.csv", index=False)

    # lenient-grader sensitivity of the confirmatory estimands (exploratory)
    orig = set(load_manifest("data/manifest_eval.json")["ids"])
    g = "qwen2.5-1.5b"
    contrasts = {}
    for label, idx in (("strict", 0), ("lenient", 1)):
        d8, dom, q = sens[(g, "sky-8b", "gptq-w4")][idx], sens[(g, "sky-8b", "gptq-w4")][2], sens[(g, "sky-8b", "gptq-w4")][3]
        keep = np.array([x in orig for x in q])
        contrasts[f"c1_{label}"] = d8[keep]
        m = dom == "math"
        hi = np.mean([sens[(g, v, "gptq-w4")][idx] for v in ("sky-4b", "sky-8b")], axis=0)[m]
        contrasts[f"c4_{label}"] = hi
    out = confirmatory(contrasts, n_boot=10_000, seed=0)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(pool_rows).round(4).to_string())
    show = tab[tab.condition.isin(["gptq-w4", "gptq-w3", "awq-w4", "rtn-w4", "nf4"])]
    print(show.round(4).to_string())
    for k, v in out.items():
        print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    pd.DataFrame(out).T.to_csv(OUT / "format_confirmatory_sensitivity.csv")


if __name__ == "__main__":
    main()
