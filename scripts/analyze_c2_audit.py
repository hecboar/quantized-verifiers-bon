"""Unblinding of the exhaustive c2 audit (exploratory validation of the outcome measure).

Inputs: the human rater's verdicts as first entered (`verdicts_final/`), the same
documents after a blind second pass on every 'unsure'/'incorrect' item
(`verdicts_reviewed/`: 'unsure' resolved, procedure tagged, label issues flagged),
and the key. Outputs: agreement of each human verdict set with the strict and the
lenient grader, and c2 recomputed with human verdicts replacing the strict grader
on every audited winner (all winners without an extractable answer are audited)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vq.analysis import score_matrix  # noqa: E402
from vq.stats import paired_bootstrap  # noqa: E402
from vq.store import CampaignStore  # noqa: E402

G = "qwen2.5-1.5b"
A = Path("outputs/tables/exploratory/audit_c2")
ARMS = (("sky-4b", "gptq-w4-packed"), ("sky-1.7b", "bf16"))


def load(d):
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((A / d / "verdicts").glob("*.json"))}


def main():
    first, rev = load("verdicts_final"), load("verdicts_reviewed")
    key = pd.read_csv(A / "key_DO_NOT_OPEN.csv")
    blind = {b["id"]: b for b in json.loads((A / "items_blind.json").read_text(encoding="utf-8"))}
    key["human_first"] = key.item_id.map(lambda i: first[i]["verdict"])
    key["human_final"] = key.item_id.map(lambda i: rev[i]["verdict"])
    key["procedure"] = key.item_id.map(lambda i: rev[i].get("procedure", ""))
    key["gold_issue"] = key.item_id.map(lambda i: bool(rev[i].get("gold_issue")))
    key["dom"] = key.question_id.str.split("-").str[0]
    assert (key.truncated == key.item_id.map(lambda i: blind[i]["ended_by_length"])).all()
    key.to_csv(A / "audit_c2_unblinded.csv", index=False)

    rows = []
    for col in ("human_first", "human_final"):
        for stratum, g in key.groupby("stratum"):
            for grader in ("correct", "correct_lenient"):
                ct = pd.crosstab(g[grader].map({True: "grader correct", False: "grader wrong"}), g[col])
                print(f"\n[{col}] {stratum} vs {grader}\n{ct}")
                for gv, r in ct.iterrows():
                    rows.append({"verdicts": col, "stratum": stratum, "grader": grader, "grader_says": gv, **r.to_dict()})
    pd.DataFrame(rows).fillna(0).to_csv(A / "audit_c2_confusion.csv", index=False)

    # c2 with human verdicts on audited winners
    st = CampaignStore("outputs/store")
    lut = {(r.question_id, r.cand): r for r in key.itertuples()}
    res, per_q = [], {}
    for v, cond in ARMS:
        S, C, _, q = score_matrix(st, G, v, cond, "eval")
        w = S.argmax(1)
        strict = C[np.arange(len(q)), w].astype(float)
        out = {"strict": strict}
        for col in ("human_first", "human_final"):
            x = strict.copy()
            for i, (qi, wi) in enumerate(zip(q, w)):
                r = lut.get((qi, int(wi)))
                if r is not None and not r.parsed:  # audited no-extractable-answer winner
                    x[i] = {"correct": 1.0, "incorrect": 0.0}.get(getattr(r, col), np.nan)
            out[col] = x
        per_q[(v, cond)] = (q, out)
    (q4, o4), (q1, o1) = per_q[ARMS[0]], per_q[ARMS[1]]
    assert list(q4) == list(q1)
    for col in ("strict", "human_first", "human_final"):
        d = o4[col] - o1[col]
        m = ~np.isnan(d)
        b = paired_bootstrap({"d": d[m]}, lambda x: x["d"], n_boot=10_000, seed=0)
        res.append({"grading": col, "acc_4b_gptq_w4_packed": np.nanmean(o4[col]), "acc_1.7b_bf16": np.nanmean(o1[col]), **b})
    res = pd.DataFrame(res)
    res.to_csv(A / "c2_human_graded.csv", index=False)
    print("\n", res.round(4).to_string(index=False))
    t = key[key.truncated]
    print("\ntruncated audited items:", len(t), "| procedure:", t.procedure.value_counts().to_dict(),
          "| final verdicts:", t.human_final.value_counts().to_dict())
    print("label issues flagged:", key[key.gold_issue][["item_id", "question_id"]].to_dict("records"))


if __name__ == "__main__":
    main()
