"""CPU analysis over stored scores: per-condition metrics, confirmatory
contrasts (Holm), and the phase-1 gate (plan §6, §8)."""
from __future__ import annotations

import numpy as np

from .metrics import (
    accuracy_per_question,
    harm_benefit,
    oracle_per_question,
    random_per_question,
    winner_change_rate,
)
from .selection import draw_subsets, select_bon, select_vote, tiebreak_priority
from .stats import holm, paired_bootstrap
from .store import CampaignStore


def _check_alignment(cands, s, score_col: str, label: str) -> None:
    """Scores must be unique, finite and question-complete: every scored
    question has a score for every one of its candidates and nothing else."""
    if s.duplicated(["question_id", "cand"]).any():
        raise ValueError(f"{label}: duplicate score rows")
    if not np.isfinite(s[score_col].to_numpy(float)).all():
        raise ValueError(f"{label}: non-finite scores")
    scored_q = set(s["question_id"])
    expected = set(zip(cands.loc[cands["question_id"].isin(scored_q), "question_id"],
                       cands.loc[cands["question_id"].isin(scored_q), "cand"]))
    got = set(zip(s["question_id"], s["cand"]))
    if expected != got:
        raise ValueError(f"{label}: coverage mismatch ({len(expected - got)} candidates unscored, "
                         f"{len(got - expected)} scores without a candidate)")


def score_matrix(store: CampaignStore, generator: str, verifier: str, condition: str, split: str,
                 score_col: str = "score", questions: list[str] | None = None):
    """Returns (scores [Q,C], correct [Q,C], answer keys [Q,C], question ids).
    `questions` restricts to a frozen subset (e.g. the gate manifest)."""
    cands = store.read("candidates", f"{generator}__{split}")
    s = store.read("scores", f"{generator}__{verifier}__{condition}")
    s = s[s["split"] == split]
    if questions is not None:
        missing = set(questions) - set(s["question_id"])
        if missing:
            raise ValueError(f"{len(missing)} requested questions were not scored under {condition}")
        s = s[s["question_id"].isin(set(questions))]
    _check_alignment(cands, s, score_col, f"{generator}__{verifier}__{condition}/{split}")
    df = cands.merge(s[["question_id", "cand", score_col]], on=["question_id", "cand"], how="inner")
    df = df.sort_values(["question_id", "cand"], kind="stable")
    qids = sorted(df["question_id"].unique())
    C = df.groupby("question_id").size()
    if C.nunique() != 1:
        raise ValueError("unequal pool sizes across questions")
    shape = (len(qids), int(C.iloc[0]))
    return (
        df[score_col].to_numpy(float).reshape(shape),
        df["correct"].to_numpy(bool).reshape(shape),
        df["answer_key"].to_numpy(int).reshape(shape),
        qids,
    )


def condition_metrics(
    store: CampaignStore, generator: str, verifier: str, condition: str,
    budgets=(1, 2, 4, 8, 16, 32), repeats: int = 200, seed: int = 7, split: str = "eval",
    vote_temperature: float | None = 1.0, reference: str = "bf16", questions: list[str] | None = None,
    tiebreak_seed: int = 11,
) -> dict:
    ref, correct, keys, qids_ref = score_matrix(store, generator, verifier, reference, split, questions=questions)
    q, _, _, qids_q = score_matrix(store, generator, verifier, condition, split, questions=questions)
    if qids_ref != qids_q:
        raise ValueError(f"{condition} and {reference} cover different questions; refusing to pair")
    Q, C = ref.shape
    prio = tiebreak_priority(Q, C, tiebreak_seed)  # configs/experiment.yaml seeds.tiebreak
    out = {}
    for n in budgets:
        if n > C:
            continue
        subsets = draw_subsets(Q, C, n, repeats, seed)
        w_ref, w_q = select_bon(ref, subsets, prio), select_bon(q, subsets, prio)
        h, b = harm_benefit(correct, w_ref, w_q)
        r = {
            "acc_ref": accuracy_per_question(correct, w_ref),
            "acc_q": accuracy_per_question(correct, w_q),
            "H": h,
            "B": b,
            "winner_change": winner_change_rate(w_ref, w_q),
            "random": random_per_question(correct, subsets),
            "oracle": oracle_per_question(correct, subsets),
        }
        if vote_temperature is not None:
            r["vote_q"] = accuracy_per_question(correct, select_vote(q, keys, subsets, prio, vote_temperature))
            r["majority"] = accuracy_per_question(correct, select_vote(q, keys, subsets, prio, None))
        out[n] = r
    return out


def confirmatory(per_question_contrasts: dict[str, np.ndarray], n_boot: int = 10_000, seed: int = 0) -> dict:
    """The three pre-registered contrasts (plan §6); Holm across them."""
    res = {k: paired_bootstrap({"d": v}, contrast=lambda d: d["d"], n_boot=n_boot, seed=seed)
           for k, v in per_question_contrasts.items()}
    adj = holm(np.array([res[k]["p_value"] for k in res]))
    for k, p in zip(res, adj):
        res[k]["p_holm"] = float(p)
    return res


def mixed_fraction(correct: np.ndarray) -> float:
    return float((correct.any(axis=1) & ~correct.all(axis=1)).mean())


def evaluate_gate(acc_bf16_32: float, acc_random_32: float, mixed_frac: float, projected_eur: float,
                  implementation_ok: bool, ceiling: float = 71.0) -> dict:
    """Phase-1 gate on development data only (plan §8). No damage required."""
    crit = {
        "implementation_ok": bool(implementation_ok),
        "bf16_useful": acc_bf16_32 - acc_random_32 >= 0.05,
        "mixed_pools": mixed_frac >= 0.30,
        "budget_ok": projected_eur <= ceiling,
    }
    return {"go": all(crit.values()), "criteria": crit}


def compare_conditions(store: CampaignStore, generator: str, verifier: str, a: str, b: str, split: str = "eval") -> dict:
    """Numerical parity between two conditions on the same candidates (e.g.
    fake-quant vs packed deployment): score differences and winner agreement."""
    sa, _, _, qa = score_matrix(store, generator, verifier, a, split)
    sb, _, _, qb = score_matrix(store, generator, verifier, b, split)
    if qa != qb:
        raise ValueError("conditions cover different questions")
    Q, C = sa.shape
    subsets = draw_subsets(Q, C, C, 1, 0)
    prio = tiebreak_priority(Q, C, 11)
    d = np.abs(sa - sb)
    return {"n_questions": Q, "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
            "winner_agreement_full_pool": float((select_bon(sa, subsets, prio) == select_bon(sb, subsets, prio)).mean())}


def early_signal(store: CampaignStore, generator: str, verifiers: list[str], conditions: list[str],
                 split: str = "dev", reference: str = "bf16", n_boot: int = 10_000, seed: int = 7,
                 null: str = "bf16-null") -> dict:
    """Phase-1 early signal (development data only): for each quantized
    condition, the change in Best-of-N accuracy at the full pool, harm/benefit,
    and the winner-change rate next to the BF16 null-control noise floor."""
    out = {}
    for v in verifiers:
        res = {}
        for c in conditions:
            _, corr, _, _ = score_matrix(store, generator, v, reference, split)
            n = corr.shape[1]
            m = condition_metrics(store, generator, v, c, budgets=[n], split=split, reference=reference,
                                  vote_temperature=None, seed=seed)[n]
            bs = paired_bootstrap({"a": m["acc_q"], "b": m["acc_ref"]}, lambda d: d["a"] - d["b"], n_boot, seed)
            res[c] = {"delta_acc": bs["estimate"], "delta_lo": bs["ci_low"], "delta_hi": bs["ci_high"],
                      "H": float(m["H"].mean()), "B": float(m["B"].mean()),
                      "winner_change": float(m["winner_change"].mean()), "acc_ref": float(m["acc_ref"].mean()),
                      "n_questions": int(len(m["acc_q"]))}
        floor = next((res[c]["winner_change"] for c in res if c.endswith(null)), None)
        res["noise_floor_winner_change"] = floor
        for c, r in list(res.items()):
            if isinstance(r, dict) and floor is not None:
                r["above_noise_floor"] = bool(r["winner_change"] > 2 * floor + 0.01) and not c.endswith(null)
        out[v] = res
    return out
