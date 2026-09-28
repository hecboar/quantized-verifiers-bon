"""Paper tables from stored scores (CPU only).

main.csv               per verifier x condition x N: accuracy, delta with paired CI, H, B, winner change, votes
controls.csv           real errors vs permutation (free / stratified) vs Gaussian, at the largest N
c3.csv                 residual-model prediction of winner change vs dev empirical vs eval observed
frontier.csv           best deployed configuration per memory ceiling (dev-selected, eval-reported)
vote_temperatures.json weighted-vote temperature fitted on dev per verifier
confirmatory.json      the three pre-registered contrasts; fixed Holm family
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import condition_metrics, score_matrix
from .controls import gaussian_control, permutation_control, score_strata
from .diagnostic import empirical_flip_rate, fit_residual_model, predict_flip_rate
from .metrics import accuracy_per_question
from .selection import draw_subsets, fit_vote_temperature, select_bon, tiebreak_priority
from .stats import holm, paired_bootstrap
from .store import CampaignStore

SKIP_PREFIXES = ("dev-full", "lamsel-", "devsub-")
# plan §6 + docs/preregistration.md (c4, c5 added 2026-09-28, before eval generation):
# fixed family for Holm
CONFIRMATORY = ("c1_delta_8b_gptq_w4", "c2_memory_frontier", "c3_ch_vs_dpq",
                "c4_math_delta_gptq_w4_n32", "c5_math_growth_with_n_gptq_w4")
PREREG_VERIFIERS = ("sky-4b", "sky-8b")
PREREG_LOW_N, PREREG_HIGH_N = 4, 32


def domain_of(question_id: str) -> str:
    """Registered domains only (preregistration §3); unknown ids are an error."""
    if question_id.startswith("gsm8k-"):
        return "gsm8k"
    if question_id.startswith(("math500-", "mathtest-", "mathtrain-")):
        return "math"
    raise ValueError(f"unregistered question id {question_id!r}")
# memory measurement -> score condition of the *same* deployment
DEPLOYMENT = {"bf16": "bf16", "gptq-w4": "gptq-w4-packed", "nf4": "nf4-bnb"}


def _conditions(store: CampaignStore, generator: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name in store.list("scores"):
        parts = name.split("__")
        if len(parts) != 3 or parts[0] != generator or parts[2].startswith(SKIP_PREFIXES):
            continue
        out.setdefault(parts[1], []).append(parts[2])
    return out


def _acc_bon(scores, correct, n, repeats, seed, tiebreak_seed):
    Q, C = scores.shape
    subsets = draw_subsets(Q, C, n, repeats, seed)
    prio = tiebreak_priority(Q, C, tiebreak_seed)
    return accuracy_per_question(correct, select_bon(scores, subsets, prio))


def _vote_temperature(store, generator, v, tiebreak_seed):
    """Fitted on development BF16 scores, frozen for eval (config: fit_on_dev)."""
    try:
        s, corr, keys, _ = score_matrix(store, generator, v, "bf16", "dev")
    except (KeyError, ValueError, FileNotFoundError):
        return 1.0, "default 1.0 (no dev scores)"
    return fit_vote_temperature(s, corr, keys, seed=tiebreak_seed)


def write_report(store: CampaignStore, generator: str, outdir: Path, budgets=(1, 2, 4, 8, 16, 32),
                 repeats: int = 200, n_boot: int = 10_000, seed: int = 7,
                 memory_path: Path = Path("outputs/memory.json"), ceilings_gb=(4, 6, 8, 12, 16),
                 primary_ceiling_gb: float = 6, tiebreak_seed: int = 11,
                 expected_ch_seeds: dict[str, list[str]] | None = None,
                 c1_questions: set[str] | None = None) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    conds = _conditions(store, generator)
    main_rows, ctrl_rows, c3_rows = [], [], []
    per_q_top: dict[tuple[str, str], np.ndarray] = {}
    per_q_delta: dict[tuple[str, str, int], np.ndarray] = {}  # acc_q - acc_ref per question
    qids_by_v: dict[str, list[str]] = {}
    dom_rows = []
    temps = {}

    for v, cs in sorted(conds.items()):
        if "bf16" not in cs:
            continue
        ref, correct, _, qids = score_matrix(store, generator, v, "bf16", "eval")
        qids_by_v[v] = qids
        doms = np.array([domain_of(q) for q in qids])
        C = ref.shape[1]
        n_top = max(n for n in budgets if n <= C)
        T, table = _vote_temperature(store, generator, v, tiebreak_seed)
        temps[v] = {"temperature": T, "dev_table": table}
        for c in sorted(cs):
            m = condition_metrics(store, generator, v, c, budgets=budgets, repeats=repeats, seed=seed,
                                  vote_temperature=T, tiebreak_seed=tiebreak_seed)
            for n, r in m.items():
                per_q_delta[(v, c, n)] = r["acc_q"] - r["acc_ref"]
                if c != "bf16":
                    for d in ("gsm8k", "math"):
                        sel = doms == d
                        if sel.any():
                            db = paired_bootstrap({"x": per_q_delta[(v, c, n)][sel]}, lambda x: x["x"], n_boot, seed)
                            dom_rows.append({"verifier": v, "condition": c, "domain": d, "N": n,
                                             "n_questions": int(sel.sum()), "acc_ref": r["acc_ref"][sel].mean(),
                                             "delta": db["estimate"], "delta_lo": db["ci_low"], "delta_hi": db["ci_high"]})
                bs = paired_bootstrap({"a": r["acc_q"], "b": r["acc_ref"]}, lambda d: d["a"] - d["b"], n_boot, seed)
                main_rows.append({
                    "verifier": v, "condition": c, "N": n, "acc_ref": r["acc_ref"].mean(), "acc_q": r["acc_q"].mean(),
                    "delta": bs["estimate"], "delta_lo": bs["ci_low"], "delta_hi": bs["ci_high"],
                    "H": r["H"].mean(), "B": r["B"].mean(), "winner_change": r["winner_change"].mean(),
                    "random": r["random"].mean(), "oracle": r["oracle"].mean(),
                    "vote_q": r["vote_q"].mean(), "majority": r["majority"].mean(), "vote_T": T,
                })
            per_q_top[(v, c)] = m[n_top]["acc_q"]
            if c == "bf16":
                continue
            q, _, _, _ = score_matrix(store, generator, v, c, "eval")
            pf, _ = permutation_control(ref, q, None, seed)
            ps, moved = permutation_control(ref, q, score_strata(ref, 3), seed)
            ga = gaussian_control(ref, q, seed)
            acc = lambda x: _acc_bon(x, correct, n_top, repeats, seed, tiebreak_seed).mean()
            ctrl_rows.append({"verifier": v, "condition": c, "N": n_top, "real": acc(q), "perm_free": acc(pf),
                              "perm_strat": acc(ps), "gaussian": acc(ga), "frac_moved_strat": moved})
            try:
                dref, _, _, _ = score_matrix(store, generator, v, "bf16", "dev")
                dq, _, _, _ = score_matrix(store, generator, v, c, "dev")
            except (KeyError, ValueError, FileNotFoundError):
                continue
            model = fit_residual_model(dref, dq)
            c3_rows.append({
                "verifier": v, "condition": c, "N": n_top, "slope": model.slope, "sigma": model.sigma,
                "predicted_flip": predict_flip_rate(dref, model, n_top, repeats, seed),
                "dev_empirical_flip": empirical_flip_rate(dref, dq, n_top, repeats, seed),
                "eval_observed_flip": empirical_flip_rate(ref, q, n_top, repeats, seed),
            })

    pd.DataFrame(main_rows).to_csv(outdir / "main.csv", index=False)
    pd.DataFrame(ctrl_rows).to_csv(outdir / "controls.csv", index=False)
    pd.DataFrame(c3_rows).to_csv(outdir / "c3.csv", index=False)
    pd.DataFrame(dom_rows).to_csv(outdir / "by_domain.csv", index=False)
    (outdir / "vote_temperatures.json").write_text(json.dumps(temps, indent=1))

    # ------------------------------- confirmatory family: always three tests
    contrasts: dict[str, np.ndarray] = {}
    missing: dict[str, str] = {}
    if ("sky-8b", "gptq-w4") in per_q_top and ("sky-8b", "bf16") in per_q_top:
        d1 = per_q_top[("sky-8b", "gptq-w4")] - per_q_top[("sky-8b", "bf16")]
        keep = None
        if c1_questions is not None:  # registered population (original eval manifest)
            keep = np.array([q in c1_questions for q in qids_by_v["sky-8b"]])
        if keep is not None and keep.sum() != len(c1_questions):
            missing["c1_delta_8b_gptq_w4"] = "missing: registered c1 population not fully scored"
        else:
            contrasts["c1_delta_8b_gptq_w4"] = d1 if keep is None else d1[keep]
    else:
        missing["c1_delta_8b_gptq_w4"] = "missing: sky-8b bf16 / gptq-w4 runs"

    ch_diffs, why = [], []
    for v in ("sky-4b", "sky-8b"):
        def by_seed(rec):
            return {c.rsplit("-s", 1)[1]: a for (vv, c), a in per_q_top.items()
                    if vv == v and c.startswith(f"gptq-w4-{rec}-s")}
        ch, dpq = by_seed("ch"), by_seed("dpq")
        want = set(expected_ch_seeds[v]) if expected_ch_seeds and v in expected_ch_seeds else set(ch)
        if not ch or set(ch) != set(dpq) or set(ch) != want:
            why.append(f"{v}: CH seeds {sorted(ch)} vs DPQ seeds {sorted(dpq)} (required {sorted(want)})")
            continue
        ch_diffs.append(np.mean([ch[k] for k in sorted(ch)], axis=0) - np.mean([dpq[k] for k in sorted(dpq)], axis=0))
    if len(ch_diffs) == 2 and len({tuple(qids_by_v.get(v, ())) for v in ("sky-4b", "sky-8b")}) == 1:
        contrasts["c3_ch_vs_dpq"] = np.mean(ch_diffs, axis=0)
    elif len(ch_diffs) == 2:
        why.append("sky-4b and sky-8b were scored on different questions")
    else:
        missing["c3_ch_vs_dpq"] = "missing: matched CH/DPQ seed sets required for both ORMs (" + "; ".join(why) + ")"

    # c4 / c5 (preregistered): MATH questions, gptq-w4, averaged over sky-4b and sky-8b
    have = all((v, "gptq-w4", n) in per_q_delta for v in PREREG_VERIFIERS for n in (PREREG_LOW_N, PREREG_HIGH_N))
    same_q = have and len({tuple(qids_by_v[v]) for v in PREREG_VERIFIERS}) == 1
    if have and same_q:
        math = np.array([domain_of(q) == "math" for q in qids_by_v[PREREG_VERIFIERS[0]]])
        hi = np.mean([per_q_delta[(v, "gptq-w4", PREREG_HIGH_N)][math] for v in PREREG_VERIFIERS], axis=0)
        lo = np.mean([per_q_delta[(v, "gptq-w4", PREREG_LOW_N)][math] for v in PREREG_VERIFIERS], axis=0)
        contrasts["c4_math_delta_gptq_w4_n32"] = hi
        contrasts["c5_math_growth_with_n_gptq_w4"] = hi - lo
    else:
        why = "missing: gptq-w4 at N=4 and N=32 for sky-4b and sky-8b on the same questions"
        missing["c4_math_delta_gptq_w4_n32"] = why
        missing["c5_math_growth_with_n_gptq_w4"] = why

    c2_note = "outputs/memory.json not found"
    if Path(memory_path).exists():
        mem = json.loads(Path(memory_path).read_text())
        frontier, c2, c2_note = _frontier(store, generator, mem, per_q_top, budgets, repeats, n_boot, seed,
                                          ceilings_gb, primary_ceiling_gb, tiebreak_seed)
        frontier.to_csv(outdir / "frontier.csv", index=False)
        if c2 is not None:
            contrasts["c2_memory_frontier"] = c2
    if "c2_memory_frontier" not in contrasts:
        missing["c2_memory_frontier"] = "missing: " + c2_note

    result = {}
    for k in CONFIRMATORY:
        if k in contrasts:
            result[k] = paired_bootstrap({"d": contrasts[k]}, lambda d: d["d"], n_boot, seed)
            result[k]["status"] = f"ok ({c2_note})" if k == "c2_memory_frontier" else "ok"
        else:  # a missing test stays in the family with p = 1 (conservative)
            result[k] = {"status": missing[k], "estimate": None, "p_value": 1.0}
    adj = holm(np.array([result[k]["p_value"] for k in CONFIRMATORY]))
    for k, p in zip(CONFIRMATORY, adj):
        result[k]["p_holm"] = float(p)
    (outdir / "confirmatory.json").write_text(json.dumps(result, indent=1))


def _frontier(store, generator, mem, per_q_eval, budgets, repeats, n_boot, seed, ceilings, primary, tiebreak_seed):
    """Best *deployed* configuration under each memory ceiling. Memory and
    accuracy come from the same deployment (packed GPTQ, bitsandbytes NF4,
    BF16); failed measurements are skipped. Chosen by dev accuracy, reported
    on eval against the best BF16 deployment fitting the same ceiling."""
    peak, dev_acc = {}, {}
    for key, entry in mem.items():
        v, scheme = key.split("__", 1)
        cond = DEPLOYMENT.get(scheme)
        if cond is None or "peak_gb" not in entry or (v, cond) not in per_q_eval:
            continue
        try:
            s, corr, _, _ = score_matrix(store, generator, v, cond, "dev")
        except (KeyError, ValueError, FileNotFoundError):
            continue
        n = max(b for b in budgets if b <= s.shape[1])
        dev_acc[(v, cond)] = _acc_bon(s, corr, n, repeats, seed, tiebreak_seed).mean()
        peak[(v, cond)] = entry["peak_gb"]
    rows, c2, note = [], None, f"no deployment pair fits the primary {primary:g} GB ceiling"
    for ceil in ceilings:
        fits = [k for k in dev_acc if peak[k] <= ceil]
        best_any = max(fits, key=dev_acc.get, default=None)
        best_bf = max([k for k in fits if k[1] == "bf16"], key=dev_acc.get, default=None)
        row = {"ceiling_gb": ceil, "selected_on": "dev",
               "best_any": "__".join(best_any) if best_any else "none",
               "best_bf16": "__".join(best_bf) if best_bf else "none",
               "delta_eval": np.nan, "delta_lo": np.nan, "delta_hi": np.nan}
        if best_any and best_bf:
            d = per_q_eval[best_any] - per_q_eval[best_bf]
            if best_any != best_bf:
                bs = paired_bootstrap({"d": d}, lambda x: x["d"], n_boot, seed)
                row.update(delta_eval=bs["estimate"], delta_lo=bs["ci_low"], delta_hi=bs["ci_high"])
            else:
                row.update(delta_eval=0.0, delta_lo=0.0, delta_hi=0.0)
            if ceil == primary:
                c2 = d
                note = f"primary ceiling {primary:g} GB: {row['best_any']} vs {row['best_bf16']}"
        rows.append(row)
    return pd.DataFrame(rows), c2, note
