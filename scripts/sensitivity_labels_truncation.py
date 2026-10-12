"""Post-hoc sensitivity of the confirmatory contrasts (exploratory; registered results stand).

1. Labels: GSM8K questions re-graded with GSM8K-Platinum (Vendrow et al., 2025,
   arXiv:2502.03461; madrylab/gsm8k-platinum, MIT): questions it rejects as poorly
   written are dropped and its revised labels replace the originals. Applied to every
   question, independently of our audit. MATH labels are unchanged (no equivalent
   revision exists).
2. Truncation: candidates that hit the generator's 1,024-token limit are removed from
   the pool before selection (score -inf), i.e. a deployment that discards unfinished
   generations. The pool stays fixed in size; a subset with only truncated candidates
   falls back to the registered tie-break.
3. Both.

Contrasts exactly as registered (N=32 Best-of-N, repeats 200, subset seed 7, tie-break
seed 11, bootstrap 10,000 / seed 7); c2 keeps the registered dev-selected pair."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vq.analysis import score_matrix  # noqa: E402
from vq.data import load_items, load_manifest, norm_text  # noqa: E402
from vq.grading import grade, gsm8k_gold  # noqa: E402
from vq.report import _acc_bon, domain_of  # noqa: E402
from vq.stats import holm, paired_bootstrap  # noqa: E402
from vq.store import CampaignStore  # noqa: E402

G = "qwen2.5-1.5b"
PLAT = Path("data/external/gsm8k_platinum/main/test-00000-of-00001.parquet")
OUT = Path("outputs/tables/exploratory/sensitivity")
REG = Path("outputs/tables/confirmatory.json")
RUNS = {"c1": [("sky-8b", "gptq-w4"), ("sky-8b", "bf16")],
        "c2": [("sky-4b", "gptq-w4-packed"), ("sky-1.7b", "bf16")],
        "c45": [(v, c) for v in ("sky-4b", "sky-8b") for c in ("gptq-w4", "bf16")]}
KEYS = ("c1_delta_8b_gptq_w4", "c2_memory_frontier", "c3_ch_vs_dpq",
        "c4_math_delta_gptq_w4_n32", "c5_math_growth_with_n_gptq_w4")


def platinum_labels(items):
    p = pd.read_parquet(PLAT)
    lab = {norm_text(q): gsm8k_gold(a) for q, a in zip(p.question, p.answer)}
    out = {}
    for it in items.values():
        if it.id.startswith("gsm8k-"):
            out[it.id] = lab.get(norm_text(it.question))  # None = rejected by Platinum
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    st = CampaignStore("outputs/store")
    items = {it.id: it for it in load_items("data/items.jsonl")}
    cand = st.read("candidates", f"{G}__eval").sort_values(["question_id", "cand"], kind="stable")
    qids = sorted(cand.question_id.unique())
    Cn = cand.groupby("question_id").size().iloc[0]
    trunc = cand.truncated.to_numpy(bool).reshape(len(qids), Cn)

    plat = platinum_labels(items)
    keep_plat = np.array([plat.get(q, "math") is not None for q in qids])
    relabel = {q: g for q, g in plat.items() if g is not None and g != items[q].gold}
    C_plat = cand.correct.to_numpy(bool).reshape(len(qids), Cn).copy()
    for q, g in relabel.items():
        i = qids.index(q)
        C_plat[i] = [grade(t, g, items[q].kind) for t in cand[cand.question_id == q].text]
    print(f"GSM8K-Platinum: {(~keep_plat).sum()} eval questions dropped, {len(relabel)} relabeled "
          f"({sum(C_plat[qids.index(q)].sum() for q in relabel)} candidates correct under new labels)")

    c1_pop = set(load_manifest("data/manifest_eval.json")["ids"])
    is_math = np.array([domain_of(q) == "math" for q in qids])
    in_c1 = np.array([q in c1_pop for q in qids])

    def per_q(v, cond, n, labels, drop_trunc):
        S, C, _, q = score_matrix(st, G, v, cond, "eval")
        assert q == qids
        if labels == "platinum":
            C = C_plat
        if drop_trunc:
            S = np.where(trunc, -np.inf, S)
        return _acc_bon(S, C, n, 200, 7, 11)

    rows, reg = [], json.loads(REG.read_text())
    for labels in ("original", "platinum"):
        for drop in (False, True):
            m = keep_plat if labels == "platinum" else np.ones(len(qids), bool)
            a = {(v, c, n): per_q(v, c, n, labels, drop) for v, c in sum(RUNS.values(), []) for n in (4, 32)}
            con = {
                "c1_delta_8b_gptq_w4": (a[("sky-8b", "gptq-w4", 32)] - a[("sky-8b", "bf16", 32)])[in_c1 & m],
                "c2_memory_frontier": (a[("sky-4b", "gptq-w4-packed", 32)] - a[("sky-1.7b", "bf16", 32)])[m],
            }
            hi = np.mean([a[(v, "gptq-w4", 32)] - a[(v, "bf16", 32)] for v in ("sky-4b", "sky-8b")], axis=0)
            lo = np.mean([a[(v, "gptq-w4", 4)] - a[(v, "bf16", 4)] for v in ("sky-4b", "sky-8b")], axis=0)
            con["c4_math_delta_gptq_w4_n32"] = hi[is_math & m]
            con["c5_math_growth_with_n_gptq_w4"] = (hi - lo)[is_math & m]
            res = {k: paired_bootstrap({"d": d}, lambda x: x["d"], 10_000, 7) for k, d in con.items()}
            ps = [res[k]["p_value"] if k in res else 1.0 for k in KEYS]  # c3 not tested: p = 1
            for k, ph in zip(KEYS, holm(np.array(ps))):
                if k in res:
                    rows.append({"labels": labels, "truncated_excluded": drop, "contrast": k, **res[k], "p_holm": ph,
                                 "registered_estimate": reg[k]["estimate"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "labels_truncation.csv", index=False)
    with pd.option_context("display.width", 200):
        print(df.drop(columns="registered_estimate").round(4).to_string(index=False))

    # truncation rates
    dom = np.array([domain_of(q) for q in qids])
    rates = {"all": trunc.mean(), "gsm8k": trunc[dom == "gsm8k"].mean(), "math": trunc[dom == "math"].mean(),
             "questions_all_truncated": float(trunc.all(1).mean())}
    for v, c in RUNS["c2"] + RUNS["c1"]:
        S, *_ = score_matrix(st, G, v, c, "eval")
        rates[f"winner_truncated:{v}:{c}"] = float(trunc[np.arange(len(qids)), S.argmax(1)].mean())
    (OUT / "truncation_rates.json").write_text(json.dumps(rates, indent=1))
    print(json.dumps({k: round(x, 4) for k, x in rates.items()}, indent=1))


if __name__ == "__main__":
    main()
