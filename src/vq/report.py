"""Paper tables from stored scores (CPU only).

main.csv          per verifier x condition x N: accuracy, delta with paired CI, H, B, winner change, votes
controls.csv      real errors vs permutation (free / stratified) vs Gaussian, at the largest N
c3.csv            residual-model prediction of winner change vs dev empirical vs eval observed
confirmatory.json the three pre-registered contrasts with Holm adjustment
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import condition_metrics, confirmatory, score_matrix
from .controls import gaussian_control, permutation_control, score_strata
from .diagnostic import empirical_flip_rate, fit_residual_model, predict_flip_rate
from .metrics import accuracy_per_question
from .selection import draw_subsets, select_bon, tiebreak_priority
from .stats import paired_bootstrap
from .store import CampaignStore

SKIP_PREFIXES = ("dev-full", "lamsel-")


def _conditions(store: CampaignStore, generator: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name in store.list("scores"):
        parts = name.split("__")
        if len(parts) != 3 or parts[0] != generator or parts[2].startswith(SKIP_PREFIXES):
            continue
        out.setdefault(parts[1], []).append(parts[2])
    return out


def _acc_bon(scores, correct, n, repeats, seed):
    Q, C = scores.shape
    subsets = draw_subsets(Q, C, n, repeats, seed)
    prio = tiebreak_priority(Q, C, seed + 1)
    return accuracy_per_question(correct, select_bon(scores, subsets, prio))


def write_report(store: CampaignStore, generator: str, outdir: Path, budgets=(1, 2, 4, 8, 16, 32),
                 repeats: int = 200, n_boot: int = 10_000, seed: int = 7,
                 memory_path: Path = Path("outputs/memory.json"), ceilings_gb=(4, 6, 8, 12, 16),
                 primary_ceiling_gb: float = 6) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    conds = _conditions(store, generator)
    main_rows, ctrl_rows, c3_rows = [], [], []
    per_q_top: dict[tuple[str, str], np.ndarray] = {}

    for v, cs in sorted(conds.items()):
        if "bf16" not in cs:
            continue
        ref, correct, _, _ = score_matrix(store, generator, v, "bf16", "eval")
        C = ref.shape[1]
        n_top = max(n for n in budgets if n <= C)
        for c in sorted(cs):
            m = condition_metrics(store, generator, v, c, budgets=budgets, repeats=repeats, seed=seed)
            for n, r in m.items():
                bs = paired_bootstrap({"a": r["acc_q"], "b": r["acc_ref"]}, lambda d: d["a"] - d["b"], n_boot, seed)
                main_rows.append({
                    "verifier": v, "condition": c, "N": n, "acc_ref": r["acc_ref"].mean(), "acc_q": r["acc_q"].mean(),
                    "delta": bs["estimate"], "delta_lo": bs["ci_low"], "delta_hi": bs["ci_high"],
                    "H": r["H"].mean(), "B": r["B"].mean(), "winner_change": r["winner_change"].mean(),
                    "random": r["random"].mean(), "oracle": r["oracle"].mean(),
                    "vote_q": r.get("vote_q", np.array([np.nan])).mean(),
                    "majority": r.get("majority", np.array([np.nan])).mean(),
                })
            per_q_top[(v, c)] = m[n_top]["acc_q"]
            if c == "bf16":
                continue
            q, _, _, _ = score_matrix(store, generator, v, c, "eval")
            strata = score_strata(ref, 3)
            pf, _ = permutation_control(ref, q, None, seed)
            ps, moved = permutation_control(ref, q, strata, seed)
            ga = gaussian_control(ref, q, seed)
            ctrl_rows.append({
                "verifier": v, "condition": c, "N": n_top,
                "real": _acc_bon(q, correct, n_top, repeats, seed).mean(),
                "perm_free": _acc_bon(pf, correct, n_top, repeats, seed).mean(),
                "perm_strat": _acc_bon(ps, correct, n_top, repeats, seed).mean(),
                "gaussian": _acc_bon(ga, correct, n_top, repeats, seed).mean(),
                "frac_moved_strat": moved,
            })
            try:
                dref, _, _, _ = score_matrix(store, generator, v, "bf16", "dev")
                dq, _, _, _ = score_matrix(store, generator, v, c, "dev")
                model = fit_residual_model(dref, dq)
                c3_rows.append({
                    "verifier": v, "condition": c, "N": n_top, "slope": model.slope, "sigma": model.sigma,
                    "predicted_flip": predict_flip_rate(dref, model, n_top, repeats, seed),
                    "dev_empirical_flip": empirical_flip_rate(dref, dq, n_top, repeats, seed),
                    "eval_observed_flip": empirical_flip_rate(ref, q, n_top, repeats, seed),
                })
            except (KeyError, ValueError):
                pass

    pd.DataFrame(main_rows).to_csv(outdir / "main.csv", index=False)
    pd.DataFrame(ctrl_rows).to_csv(outdir / "controls.csv", index=False)
    pd.DataFrame(c3_rows).to_csv(outdir / "c3.csv", index=False)

    # ------------------------------------------------ confirmatory contrasts
    contrasts: dict[str, np.ndarray] = {}
    if ("sky-8b", "gptq-w4") in per_q_top and ("sky-8b", "bf16") in per_q_top:
        contrasts["c1_delta_8b_gptq_w4"] = per_q_top[("sky-8b", "gptq-w4")] - per_q_top[("sky-8b", "bf16")]
    ch_diffs = []
    for v in ("sky-4b", "sky-8b"):
        pick = lambda rec: [a for (vv, c), a in per_q_top.items() if vv == v and c.startswith("gptq-w4-" + rec + "-s")]
        ch, dpq = pick("ch"), pick("dpq")
        if ch and dpq:
            ch_diffs.append(np.mean(ch, axis=0) - np.mean(dpq, axis=0))
    if ch_diffs:
        contrasts["c3_ch_vs_dpq"] = np.mean(ch_diffs, axis=0)
    c2_status = "skipped: outputs/memory.json not found"
    if Path(memory_path).exists():
        mem = json.loads(Path(memory_path).read_text())
        frontier, c2 = _frontier(store, generator, conds, mem, per_q_top, budgets, repeats, n_boot, seed,
                                 ceilings_gb, primary_ceiling_gb)
        frontier.to_csv(outdir / "frontier.csv", index=False)
        contrasts.update(c2)
        c2_status = f"primary ceiling {primary_ceiling_gb} GB; selection on dev, evaluation on eval"
    result = confirmatory(contrasts, n_boot=n_boot, seed=seed) if contrasts else {}
    result["c2_memory_frontier"] = {"status": c2_status}
    (outdir / "confirmatory.json").write_text(json.dumps(result, indent=1))


def _frontier(store, generator, conds, mem, per_q_eval, budgets, repeats, n_boot, seed, ceilings, primary):
    """Best configuration under each memory ceiling, chosen by *dev* accuracy
    (avoids selecting on the test set), reported on eval against the best BF16
    configuration that fits the same ceiling."""
    dev_acc = {}
    for v, cs in conds.items():
        for c in cs:
            key = f"{v}__{c}"
            if key not in mem or (v, c) not in per_q_eval:
                continue
            try:
                s, corr, _, _ = score_matrix(store, generator, v, c, "dev")
            except (KeyError, ValueError, FileNotFoundError):
                continue
            n = max(b for b in budgets if b <= s.shape[1])
            dev_acc[(v, c)] = _acc_bon(s, corr, n, repeats, seed).mean()
    rows, contrasts = [], {}
    for ceil in ceilings:
        fits = [k for k in dev_acc if mem[f"{k[0]}__{k[1]}"]["peak_gb"] <= ceil]
        best_any = max(fits, key=dev_acc.get, default=None)
        best_bf = max([k for k in fits if k[1] == "bf16"], key=dev_acc.get, default=None)
        row = {"ceiling_gb": ceil, "selected_on": "dev",
               "best_any": "__".join(best_any) if best_any else "none",
               "best_bf16": "__".join(best_bf) if best_bf else "none",
               "delta_eval": np.nan, "delta_lo": np.nan, "delta_hi": np.nan}
        if best_any and best_bf:
            d = per_q_eval[best_any] - per_q_eval[best_bf]
            bs = paired_bootstrap({"d": d}, lambda x: x["d"], n_boot, seed)
            row.update(delta_eval=bs["estimate"], delta_lo=bs["ci_low"], delta_hi=bs["ci_high"])
            if ceil == primary and best_any != best_bf:
                contrasts[f"c2_frontier_{ceil:g}gb"] = d
        rows.append(row)
    return pd.DataFrame(rows), contrasts
