"""End-to-end execution of one scoring run (plan §4-§5):
candidates -> (calibration) -> fake-quant -> audit -> sharded scoring -> store.

Every run scores the full evaluation pools plus the first `dev_scored`
development questions (needed by the gate and by C3). The special
`dev_full=True` BF16 run scores all development pools; its scores feed the
label-free calibration recipes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import torch

from .calibration import Pool, build_recipe
from .experiment import FROZEN_LAMBDA, ScoreRun
from .quant.apply import quantize_model, target_linears
from .selection import draw_subsets, select_bon, tiebreak_priority
from .store import CampaignStore
from .verifiers import Scorer, VerifierSpec, load_verifier


@dataclass
class RunContext:
    store: CampaignStore
    questions: dict[str, str]
    device: str = "cuda"
    loader: Callable = load_verifier
    verifier_specs: dict[str, VerifierSpec] = field(default_factory=dict)
    n_calib: int = 128
    dev_scored: int = 100
    shard_size: int = 2000
    generic_ids: list[list[int]] | None = None
    frozen_lambda: dict[str, float] = field(default_factory=dict)
    max_tokens: int = 16_384
    dtype: torch.dtype = torch.bfloat16


def dev_full_name(generator: str, verifier: str) -> str:
    return f"{generator}__{verifier}__dev-full-bf16"


def _fingerprints(model) -> dict[str, tuple[float, float]]:
    return {n: (float(p.detach().double().sum()), float(p.detach().double().abs().sum()))
            for n, p in model.named_parameters()}


def _work_rows(ctx: RunContext, generator: str, dev_full: bool) -> pd.DataFrame:
    dev = ctx.store.read("candidates", f"{generator}__dev")
    if dev_full:
        return dev.assign(split="dev")
    ev = ctx.store.read("candidates", f"{generator}__eval").assign(split="eval")
    keep = sorted(dev["question_id"].unique())[: ctx.dev_scored]
    dev = dev[dev["question_id"].isin(keep)].assign(split="dev")
    df = pd.concat([ev, dev], ignore_index=True)
    return df.sort_values(["split", "question_id", "cand"], kind="stable").reset_index(drop=True)


def _pool_from_dev(ctx: RunContext, run: ScoreRun, scorer: Scorer) -> Pool:
    s = ctx.store.read("scores", dev_full_name(run.generator, run.verifier))
    cands = ctx.store.read("candidates", f"{run.generator}__dev")
    s = s.merge(cands[["question_id", "cand", "text"]], on=["question_id", "cand"])
    qids = sorted(s["question_id"].unique())
    scores, ids = [], []
    for q in qids:
        g = s[s["question_id"] == q].sort_values("cand")
        scores.append(g["score"].to_numpy())
        ids.append([scorer.encode(ctx.questions[q], t) for t in g["text"]])
    return Pool(scores=np.stack(scores), token_ids=ids)


def _resolve_lambda(ctx: RunContext, run: ScoreRun) -> float:
    if run.lam is None:
        return 0.0
    if run.lam == FROZEN_LAMBDA:
        if run.verifier not in ctx.frozen_lambda:
            raise RuntimeError(f"lambda for {run.verifier} not frozen yet (run lambda selection first)")
        return float(ctx.frozen_lambda[run.verifier])
    return float(run.lam)


def execute_run(run: ScoreRun, ctx: RunContext, dev_full: bool = False, stop_after_shards: int | None = None) -> str:
    name = dev_full_name(run.generator, run.verifier) if dev_full else run.name
    if ctx.store.done("scores", name):
        return "skipped"

    spec = ctx.verifier_specs[run.verifier]
    tok, model = ctx.loader(spec, device=ctx.device, dtype=ctx.dtype)
    scorer = Scorer(spec, tok, model, max_tokens=ctx.max_tokens)
    meta: dict = {"run": run.__dict__, "name": name, "torch": torch.__version__}
    try:
        import transformers

        meta["transformers"] = transformers.__version__
    except ImportError:
        pass

    if run.method != "none":
        before = _fingerprints(model)
        calib = None
        if run.method in ("gptq", "awq"):
            if run.recipe == "generic":
                calib = build_recipe("generic", None, ctx.n_calib, run.seed, generic_texts_ids=ctx.generic_ids)
            else:
                calib = build_recipe(run.recipe, _pool_from_dev(ctx, run, scorer), ctx.n_calib, run.seed)
            meta["calibration"] = {k: v for k, v in calib.meta.items() if k in ("recipe", "n_tokens", "n_boundary", "n_anchor")}
            meta["calibration"]["n_sequences"] = len(calib.sequences)
            meta["calibration"]["n_pairs"] = len(calib.pairs)
        t0 = time.perf_counter()
        meta["quant"] = quantize_model(model, run.method, run.bits or 4, run.group_size, calib=calib,
                                       lam=_resolve_lambda(ctx, run))
        meta["quant"]["seconds"] = time.perf_counter() - t0
        after = _fingerprints(model)
        changed = [n for n in before if before[n] != after[n]]
        targets = {f"{n}.weight" for n, _ in target_linears(model)}
        meta["audit"] = {
            "n_changed": len(changed),
            "n_targets": len(targets),
            "non_target_changed": sorted(set(changed) - targets),
            "head_or_embed_changed": any(("embed" in n or "norm" in n or n.startswith("score")) for n in changed),
        }

    rows = _work_rows(ctx, run.generator, dev_full)
    done = set(ctx.store.shard_ids("scores", name))
    n_tokens, seconds, written = 0, 0.0, 0
    for sh, start in enumerate(range(0, len(rows), ctx.shard_size)):
        if sh in done:
            continue
        part = rows.iloc[start : start + ctx.shard_size]
        t0 = time.perf_counter()
        res = scorer.score([ctx.questions[q] for q in part["question_id"]], part["text"].tolist())
        seconds += time.perf_counter() - t0
        n_tokens += sum(res["n_tokens"])
        out = part[["split", "question_id", "cand"]].copy()
        for k in ("score", "score_last", "n_steps", "n_tokens", "truncated"):
            if k in res:
                out[k] = res[k]
        if "step_rewards" in res:
            out["step_rewards"] = [list(map(float, r)) for r in res["step_rewards"]]
        ctx.store.write_shard("scores", name, sh, out)
        written += 1
        if stop_after_shards is not None and written >= stop_after_shards:
            return "partial"

    meta["throughput"] = {"tokens": n_tokens, "seconds": seconds, "tok_per_s": n_tokens / seconds if seconds else None}
    ctx.store.merge_shards("scores", name, meta=meta)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return "done"


def select_lambda(ref_dev: np.ndarray, candidates: dict[float, np.ndarray]) -> float:
    """Pick the CH lambda with the lowest dev winner-change rate vs BF16 at
    full pool size (label-free); ties -> smaller lambda."""
    Q, C = ref_dev.shape
    subsets = draw_subsets(Q, C, C, 1, seed=0)
    prio = tiebreak_priority(Q, C, seed=0)
    w_ref = select_bon(ref_dev, subsets, prio)
    rates = {lam: float((select_bon(s, subsets, prio) != w_ref).mean()) for lam, s in candidates.items()}
    return min(sorted(rates), key=lambda lam: rates[lam])
