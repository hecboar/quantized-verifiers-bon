"""Phase 0 on GPU: measure throughput and run every integrity check that must
pass before money is spent on the grid (plan §7, §8.1)."""
from __future__ import annotations

import time

import torch

from .calibration import build_recipe
from .generate import annotate
from .quant.apply import quantize_model, target_linears

PARITY_REL_TOL = 0.25  # of the within-question score scale: catches gross bugs, not BF16 noise


def parity_ok(max_abs_diff: float, score_scale: float) -> bool:
    """Batched-vs-single scoring agreement. BF16 on GPU legitimately differs by a
    few % of the score scale (measured separately by the BF16 null control);
    a gross bug (e.g. pooling the wrong token) differs by a large fraction."""
    if score_scale <= 0:
        return max_abs_diff == 0
    return max_abs_diff <= PARITY_REL_TOL * score_scale


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def _free(device):
    import gc

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def _score_check(rep, v, spec, loader, qs, texts, device, dtype):
    from .verifiers import Scorer

    tok, model = loader(spec, device=device, dtype=dtype)
    sc = Scorer(spec, tok, model)
    _sync(device)
    t0 = time.perf_counter()
    res = sc.score(qs, texts)
    _sync(device)
    n_tok = sum(res["n_tokens"])
    rep["score"][v] = float(n_tok / max(time.perf_counter() - t0, 1e-9))
    rep["avg_score_tokens_by_verifier"][v] = n_tok / len(res["n_tokens"])
    k = min(16, len(qs))
    sc.max_tokens = 1
    single = sc.score(qs[:k], texts[:k])["score"]
    rep["parity"][v] = float(max(abs(a - b) for a, b in zip(res["score"][:k], single)))
    import numpy as np

    s = np.asarray(res["score"], dtype=float)
    rep.setdefault("score_scale", {})[v] = float(1.4826 * np.median(np.abs(s - np.median(s))))
    sc.max_tokens = 16_384
    # the model must survive fake-quant + scoring (custom-code PRM included);
    # quantized in place: this model is discarded afterwards
    try:
        quantize_model(model, "rtn", 4, 128 if spec.kind == "prm" else 32)
        sc.score(qs[:4], texts[:4])
        rep["rtn_runs"][v] = "ok"
    except Exception as e:  # reported, not raised: the pilot must finish
        rep["rtn_runs"][v] = f"FAILED: {type(e).__name__}: {e}"[:300]


def _gptq_check(rep, v, spec, loader, n_calib, device, dtype, generic_chunk):
    tok, model = loader(spec, device=device, dtype=dtype)
    before = {n: float(p.detach().double().sum()) for n, p in model.named_parameters()}
    calib = build_recipe("generic", None, n_calib, 0, generic_texts_ids=generic_chunk(tok, 2 * n_calib))
    _sync(device)
    t0 = time.perf_counter()
    quantize_model(model, "gptq", 4, 32 if device == "cpu" else 128, calib=calib)
    _sync(device)
    secs = time.perf_counter() - t0
    changed = {n for n, p in model.named_parameters() if float(p.detach().double().sum()) != before[n]}
    targets = {f"{n}.weight" for n, _ in target_linears(model)}
    rep["quant_check"][v] = {"gptq_seconds": secs, "n_calib": n_calib, "audit_ok": changed == targets,
                             "per_calib_seq_seconds": secs / n_calib}


def run_pilot(items: dict, ids: list[str], generate_fn, loader, specs: dict, quant_verifier: str,
              n_calib: int = 16, device: str = "cuda", generic_chunk=None, dtype=torch.bfloat16) -> dict:
    """generate_fn(list[Item]) -> raw candidate DataFrame; loader(spec, device, dtype) -> (tok, model);
    generic_chunk(tokenizer, n) -> list of token-id lists for generic calibration."""
    rep: dict = {"score": {}, "avg_score_tokens_by_verifier": {}, "parity": {}, "quant_check": {}, "rtn_runs": {},
                 "errors": {}}
    t0 = time.perf_counter()
    out = generate_fn([items[i] for i in ids])
    if isinstance(out, tuple):  # (candidates, throughput measured by the generation itself)
        raw, measured = out
        rep["generate"], rep["generate_source"] = float(measured), "measured_by_generation"
    else:
        raw = out
        rep["generate"] = float(raw["n_tokens"].sum() / max(time.perf_counter() - t0, 1e-9))
        rep["generate_source"] = "wall_clock_including_load"
    df = annotate(raw, items)
    rep["avg_gen_tokens"] = float(df["n_tokens"].mean())
    rep["truncated_frac"] = float(df["truncated"].mean())
    rep["parsed_frac"] = float(df["parsed"].mean())
    rep["accuracy"] = float(df["correct"].mean())
    qs = [items[q].question for q in df["question_id"]]
    texts = df["text"].tolist()

    for v, spec in specs.items():
        # each check runs in its own function: on return no reference to the
        # model survives, so the next (possibly 8B) model fits on the GPU
        # one failing verifier (download, disk, remote code) must not hide the others
        try:
            _score_check(rep, v, spec, loader, qs, texts, device, dtype)
            if v == quant_verifier:
                _free(device)
                _gptq_check(rep, v, spec, loader, n_calib, device, dtype, generic_chunk)
        except Exception as e:
            rep["errors"][v] = f"{type(e).__name__}: {e}"[:500]
        _free(device)

    done = rep["avg_score_tokens_by_verifier"]
    rep["avg_score_tokens"] = float(sum(done.values()) / len(done)) if done else 0.0
    rep["ok"] = (not rep["errors"] and len(rep["score"]) == len(specs)
                 and all(parity_ok(p, rep["score_scale"].get(v, 0.0)) for v, p in rep["parity"].items())
                 and all(s == "ok" for s in rep["rtn_runs"].values())
                 and all(q["audit_ok"] for q in rep["quant_check"].values()))
    return rep
