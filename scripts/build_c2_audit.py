"""Exhaustive blinded audit for c2 (post-review, exploratory validation of the
outcome measure): every unique Best-of-32 winner WITHOUT an extractable final
answer in either c2 arm (Skywork-4B packed GPTQ-W4, Skywork-1.7B BF16), plus 30
random winners WITH one as controls. The blind file shows question, gold answer
and response only; generator/verifier/condition/grader verdicts stay in the key."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vq.analysis import score_matrix  # noqa: E402
from vq.data import load_items  # noqa: E402
from vq.store import CampaignStore  # noqa: E402

G = "qwen2.5-1.5b"
ARMS = (("sky-4b", "gptq-w4-packed"), ("sky-1.7b", "bf16"))
OUT = Path("outputs/tables/exploratory/audit_c2")


def main(n_controls=30, seed=20260929):
    OUT.mkdir(parents=True, exist_ok=True)
    store, items = CampaignStore("outputs/store"), {it.id: it for it in load_items("data/items.jsonl")}
    lz = pd.read_parquet(f"outputs/tables/exploratory/lenient_{G}.parquet")
    cand = store.read("candidates", f"{G}__eval").merge(lz, on=["question_id", "cand"]).set_index(["question_id", "cand"])
    arms = {}
    for v, cond in ARMS:
        S, _, _, q = score_matrix(store, G, v, cond, "eval")
        for qi, wi in zip(q, S.argmax(1)):
            arms.setdefault((qi, int(wi)), []).append(f"{v}:{cond}")
    sel = pd.DataFrame([{"question_id": k[0], "cand": k[1], "arms": "|".join(a)} for k, a in arms.items()])
    sel = sel.join(cand[["text", "parsed", "truncated", "correct", "correct_lenient"]], on=["question_id", "cand"])
    rng = np.random.default_rng(seed)
    prose = sel[~sel.parsed]
    parsed = sel[sel.parsed]
    ctrl = parsed.iloc[rng.choice(len(parsed), n_controls, replace=False)]
    s = pd.concat([prose.assign(stratum="no_extractable_answer"), ctrl.assign(stratum="control_extractable")])
    s = s.sample(frac=1, random_state=seed).reset_index(drop=True)
    s["item_id"] = [f"R{i + 1:03d}" for i in range(len(s))]
    s.drop(columns="text").to_csv(OUT / "key_DO_NOT_OPEN.csv", index=False)
    blind = [{"id": r.item_id, "question": items[r.question_id].question, "gold": items[r.question_id].gold,
              "response": r.text} for r in s.itertuples()]
    (OUT / "items_blind.json").write_text(json.dumps(blind, ensure_ascii=False), encoding="utf-8")
    print(len(blind), "items:", s.stratum.value_counts().to_dict())


if __name__ == "__main__":
    main()
