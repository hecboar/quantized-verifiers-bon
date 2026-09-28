"""End-to-end execution of one scoring run (plan §4-§5):
candidates -> (calibration) -> fake-quant -> audit -> sharded scoring -> store.

Every run scores the full evaluation pools plus the frozen dev-scored set (`dev_scored_ids`)
development questions (needed by the gate and by C3). The special
`dev_full=True` BF16 run scores all development pools; its scores feed the
label-free calibration recipes.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import json
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
    dev_scored_ids: list[str] | None = None  # frozen, stratified (manifest_gate); preferred
    shard_size: int = 2000
    generic_ids: list[list[int]] | None = None
    generic_texts: list[str] | None = None  # tokenised per verifier (preferred)
    generic_chunk_tokens: int = 512
    frozen_lambda: dict[str, float] = field(default_factory=dict)
    # generator -> question ids its evaluation pool must contain exactly (manifest)
    expected_eval_ids: dict[str, set] | None = None
    max_tokens: int = 16_384
    dtype: torch.dtype = torch.bfloat16


def dev_full_name(generator: str, verifier: str, condition: str = "bf16") -> str:
    return f"{generator}__{verifier}__dev-full-{condition}"


def generic_chunks(tokenizer, texts: list[str], chunk_tokens: int, n: int) -> list[list[int]]:
    """Concatenate generic texts and cut `n` chunks of exactly `chunk_tokens`."""
    ids: list[int] = []
    for t in texts:
        ids += tokenizer(t, add_special_tokens=False)["input_ids"] if _accepts_kw(tokenizer) else tokenizer(t)["input_ids"]
        if len(ids) >= chunk_tokens * n:
            break
    chunks = [ids[i : i + chunk_tokens] for i in range(0, len(ids) - chunk_tokens + 1, chunk_tokens)]
    if len(chunks) < n:
        raise ValueError(f"generic texts yield {len(chunks)} chunks, need {n}")
    return chunks[:n]


def _accepts_kw(tokenizer) -> bool:
    import inspect

    try:
        return "add_special_tokens" in inspect.signature(tokenizer.__call__).parameters
    except (TypeError, ValueError):
        return True


def _fingerprints(model, chunk: int = 1 << 24) -> dict[str, tuple[float, float]]:
    """(sum, abs-sum) per parameter, accumulated in float64 block by block:
    never materialises a float64 copy of a whole tensor (a full .double() of the
    8B embedding table needs 4.6 GB of GPU memory)."""
    out = {}
    for n, p in model.named_parameters():
        flat = p.detach().reshape(-1)
        s = a = 0.0
        for block in flat.split(chunk):
            b = block.to(torch.float64)
            s += float(b.sum())
            a += float(b.abs().sum())
            del b
        out[n] = (s, a)
    return out


def _dev_scored_rows(ctx: RunContext, dev: pd.DataFrame) -> pd.DataFrame:
    if ctx.dev_scored_ids is not None:
        keep = list(ctx.dev_scored_ids)
    else:  # seeded sample, never "first N sorted" (that would be one domain only)
        qs = sorted(dev["question_id"].unique())
        keep = list(np.random.default_rng(0).permutation(qs)[: ctx.dev_scored])
    return dev[dev["question_id"].isin(keep)].assign(split="dev")


def _work_rows(ctx: RunContext, generator: str, dev_full: bool, dev_scored_only: bool = False) -> pd.DataFrame:
    has_dev = ctx.store.done("candidates", f"{generator}__dev")
    if dev_scored_only:  # lambda selection: no evaluation pools needed
        if not has_dev:
            raise FileNotFoundError(f"candidates {generator}__dev not generated")
        df = _dev_scored_rows(ctx, ctx.store.read("candidates", f"{generator}__dev"))
        return df.sort_values(["split", "question_id", "cand"], kind="stable").reset_index(drop=True)
    if dev_full:
        if not has_dev:
            raise FileNotFoundError(f"candidates {generator}__dev not generated")
        return ctx.store.read("candidates", f"{generator}__dev").assign(split="dev")
    if not ctx.store.done("candidates", f"{generator}__eval"):
        raise FileNotFoundError(f"candidates {generator}__eval not generated (use dev_full for the phase-1 gate)")
    ev = ctx.store.read("candidates", f"{generator}__eval")
    if ctx.expected_eval_ids is not None and generator in ctx.expected_eval_ids:
        want, got = set(ctx.expected_eval_ids[generator]), set(ev["question_id"])
        if want != got:
            raise ValueError(f"{generator}__eval does not match its manifest "
                             f"({len(want - got)} missing, {len(got - want)} unexpected questions)")
        sizes = ev.groupby("question_id").size()
        if sizes.nunique() != 1 or ev.duplicated(["question_id", "cand"]).any():
            raise ValueError(f"{generator}__eval: unequal or duplicated candidate pools")
    parts = [ev.assign(split="eval")]
    if has_dev:  # the secondary generator has no dev pools
        parts.append(_dev_scored_rows(ctx, ctx.store.read("candidates", f"{generator}__dev")))
    df = pd.concat(parts, ignore_index=True)
    return df.sort_values(["split", "question_id", "cand"], kind="stable").reset_index(drop=True)


def _pool_from_dev(ctx: RunContext, run: ScoreRun, scorer: Scorer) -> Pool:
    s = ctx.store.read("scores", dev_full_name(run.calib_gen, run.verifier))
    cands = ctx.store.read("candidates", f"{run.calib_gen}__dev")
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


def _run_fingerprint(run: ScoreRun, ctx: RunContext, rows: pd.DataFrame, lam: float, mode: str) -> str:
    """Identity of a run: resuming or reusing it with any different input or
    setting must fail loudly instead of mixing results."""
    spec = ctx.verifier_specs[run.verifier]
    h = hashlib.sha256()
    h.update(json.dumps({
        "run": {k: v for k, v in run.__dict__.items()}, "lam": lam, "mode": mode, "hf_id": spec.hf_id,
        "shard_size": ctx.shard_size, "dtype": str(ctx.dtype), "n_calib": ctx.n_calib,
        "dev_scored_ids": list(ctx.dev_scored_ids) if ctx.dev_scored_ids is not None else ctx.dev_scored,
    }, sort_keys=True, default=str).encode())
    h.update(pd.util.hash_pandas_object(rows[["split", "question_id", "cand", "text"]], index=False).values.tobytes())
    return h.hexdigest()


def _file_digest(path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return "absent"


def _run_fingerprint_v2(run: ScoreRun, ctx: RunContext, rows: pd.DataFrame, lam: float, mode: str) -> str:
    """v1 plus everything else that determines the scores:
    the question texts actually scored, the effective scoring batch budget and the
    calibration inputs (generic texts, or the BF16 dev scores that feed CH/DPQ pairs)."""
    h = hashlib.sha256(_run_fingerprint(run, ctx, rows, lam, mode).encode())
    qtext = [ctx.questions[q] for q in rows["question_id"].unique()]
    calib = None
    if run.method in ("gptq", "awq"):
        if run.recipe == "generic":
            src = ctx.generic_ids if ctx.generic_ids is not None else ctx.generic_texts
            calib = hashlib.sha256(json.dumps(src, default=str).encode()).hexdigest()
        else:
            calib = _file_digest(ctx.store._path("scores", dev_full_name(run.calib_gen, run.verifier)))
    h.update(json.dumps({"questions": qtext, "batch_tokens": run.batch_tokens or ctx.max_tokens,
                         "calibration": calib, "chunk": ctx.generic_chunk_tokens}, sort_keys=True).encode())
    return h.hexdigest()


def execute_run(run: ScoreRun, ctx: RunContext, dev_full: bool = False, stop_after_shards: int | None = None,
                dev_scored_only: bool = False) -> str:
    """dev_full: all development pools (calibration source / gate).
    dev_scored_only: only the frozen dev-scored subset (lambda selection)."""
    if dev_full:
        name, mode = dev_full_name(run.generator, run.verifier, run.condition), "dev_full"
    elif dev_scored_only:
        name, mode = f"{run.generator}__{run.verifier}__devsub-{run.condition}", "dev_scored"
    else:
        name, mode = run.name, "default"
    rows = _work_rows(ctx, run.generator, dev_full, dev_scored_only)
    lam = _resolve_lambda(ctx, run) if run.method == "gptq" else 0.0
    fp = _run_fingerprint(run, ctx, rows, lam, mode)
    fp2 = _run_fingerprint_v2(run, ctx, rows, lam, mode)
    if ctx.store.done("scores", name):
        stored = ctx.store.meta("scores", name)
        # v2 when recorded; campaigns from before v2 are accepted only if their v1 matches
        ok = stored.get("fingerprint_v2") == fp2 if "fingerprint_v2" in stored else stored.get("fingerprint") == fp
        if not ok:
            raise RuntimeError(f"{name} exists with a different fingerprint (inputs or settings changed); "
                               "delete it explicitly to recompute")
        return "skipped"
    ctx.store.check_shard_fingerprint("scores", name, fp2)

    spec = ctx.verifier_specs[run.verifier]
    tok, model = ctx.loader(spec, device=ctx.device, dtype=ctx.dtype)
    batch_tokens = run.batch_tokens or ctx.max_tokens
    scorer = Scorer(spec, tok, model, max_tokens=batch_tokens)
    meta: dict = {"run": run.__dict__, "name": name, "torch": torch.__version__, "fingerprint": fp, "fingerprint_v2": fp2, "mode": mode,
                  "batch_tokens": batch_tokens}
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
                gids = ctx.generic_ids
                if gids is None:
                    # extra chunks so the seeded choice of n_calib is non-trivial
                    gids = generic_chunks(tok, ctx.generic_texts, ctx.generic_chunk_tokens, 2 * ctx.n_calib)
                calib = build_recipe("generic", None, ctx.n_calib, run.seed, generic_texts_ids=gids)
            else:
                calib = build_recipe(run.recipe, _pool_from_dev(ctx, run, scorer), ctx.n_calib, run.seed)
            meta["calibration"] = {k: v for k, v in calib.meta.items() if k in ("recipe", "n_tokens", "n_boundary", "n_anchor")}
            meta["calibration"]["n_sequences"] = len(calib.sequences)
            meta["calibration"]["n_pairs"] = len(calib.pairs)
            meta["calibration"]["calib_generator"] = run.calib_gen
        t0 = time.perf_counter()
        meta["quant"] = quantize_model(model, run.method, run.bits or 4, run.group_size, calib=calib, lam=lam)
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
        a = meta["audit"]
        if a["n_targets"] == 0 or a["n_changed"] != a["n_targets"] or a["non_target_changed"]:
            raise RuntimeError(f"{name}: quantization audit failed ({a}); nothing scored")

    done = set(ctx.store.shard_ids("scores", name))
    n_shards = -(-len(rows) // ctx.shard_size)
    if done - set(range(n_shards)):
        raise RuntimeError(f"{name}: unexpected shard ids {sorted(done - set(range(n_shards)))}; refusing to merge")
    n_tokens, seconds, written = 0, 0.0, 0
    for sh, start in enumerate(range(0, len(rows), ctx.shard_size)):
        if sh in done:
            continue
        part = rows.iloc[start : start + ctx.shard_size]
        t0 = time.perf_counter()
        res = scorer.score([ctx.questions[q] for q in part["question_id"]], part["text"].tolist())
        seconds += time.perf_counter() - t0
        n_tokens += sum(res["n_tokens"])
        if not np.isfinite(np.asarray(res["score"], dtype=float)).all():
            raise ValueError(f"{name}: non-finite scores in shard {sh}; aborting (nothing written)")
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
    merged = ctx.store.merge_shards("scores", name, meta=meta)
    df = ctx.store.read("scores", name)
    if len(df) != len(rows) or df.duplicated(["split", "question_id", "cand"]).any():
        merged.unlink()
        raise RuntimeError(f"{name}: merged shards do not match the expected rows")
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
