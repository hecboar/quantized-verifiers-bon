"""Lenient-grader verdicts for one candidate pool, computed safely on Windows.

`vq.grading` runs math_verify with 5 s timeouts on Linux but without any timeout on
Windows, where a pathological expression can hang and grow without bound (H7: one
process reached 18 GB). Semantics are kept identical to `grade_lenient` as run on
Linux:
- candidates with an extractable final answer: grade_lenient == grade, so the
  annotated `correct` (graded on the pod, Linux, with timeouts) is reused;
- the others: grade_lenient in a worker process with a 5 s limit per candidate; a
  timeout counts as not correct, as a math_verify timeout does on Linux.
Output: <out>/lenient_<pool>.parquet with question_id, cand, correct_lenient, timed_out.
"""
import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

TIMEOUT_S = 5


def _grade(args):
    from vq.grading import grade_lenient

    text, gold, kind = args
    return bool(grade_lenient(text, gold, kind))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="candidates store name, e.g. qwen2.5-1.5b-h7__eval")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from vq.data import load_items
    from vq.store import CampaignStore

    items = {it.id: it for it in load_items("data/items.jsonl")}
    cand = CampaignStore("outputs/store").read("candidates", a.pool)
    res = cand[["question_id", "cand"]].copy()
    res["correct_lenient"] = cand["correct"].astype(bool)
    res["timed_out"] = False
    todo = cand.index[~cand["parsed"]].tolist()
    print(f"{len(cand)} candidates; {len(todo)} without an extractable answer to grade", flush=True)
    pool = mp.Pool(4, maxtasksperchild=200)
    jobs = {i: pool.apply_async(_grade, ((cand.at[i, "text"], items[cand.at[i, "question_id"]].gold,
                                          items[cand.at[i, "question_id"]].kind),)) for i in todo}
    n_to = 0
    for k, (i, job) in enumerate(jobs.items()):
        try:
            res.at[i, "correct_lenient"] = job.get(timeout=TIMEOUT_S)
        except mp.TimeoutError:
            # a hung worker: kill the pool, resubmit what is left, count this one as wrong
            n_to += 1
            res.at[i, "correct_lenient"] = False
            res.at[i, "timed_out"] = True
            pool.terminate()
            pool = mp.Pool(4, maxtasksperchild=200)
            rest = list(jobs)[k + 1:]
            for j in rest:
                if jobs[j].ready():
                    continue
                jobs[j] = pool.apply_async(_grade, ((cand.at[j, "text"], items[cand.at[j, "question_id"]].gold,
                                                     items[cand.at[j, "question_id"]].kind),))
        if k % 500 == 0:
            print(f"{k}/{len(todo)} graded, {n_to} timeouts", flush=True)
    pool.terminate()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out / f"lenient_{a.pool}.parquet", index=False)
    print(f"done: {n_to} timeouts; lenient correct {res.correct_lenient.mean():.4f} vs strict {cand.correct.mean():.4f}")


if __name__ == "__main__":
    main()
