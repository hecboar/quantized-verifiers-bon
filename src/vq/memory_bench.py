"""GPU-only: real memory of packed deployments for the C2 frontier (phase 6).

For each verifier: BF16, bitsandbytes NF4 (reward head skipped), and our
GPTQ-W4 exported as a packed checkpoint (loaded with the GPTQ kernels
available to transformers). Peak memory is measured while scoring a fixed
workload so every configuration sees the same batch/token budget. Score
parity (packed vs fake-quant) is recorded as the numerical audit.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

WORKLOAD_SEQS = 64
WORKLOAD_TOKENS = 900


def _workload(tok, n=WORKLOAD_SEQS, L=WORKLOAD_TOKENS):
    g = torch.Generator().manual_seed(0)
    vocab = min(tok.vocab_size, 30_000)
    return [torch.randint(100, vocab, (L,), generator=g).tolist() for _ in range(n)]


def _forward_all(model, seqs, pad_id):
    """Plain forward passes over the workload (works for ORM heads and the PRM)."""
    from .verifiers import _pad_right, length_batches

    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in length_batches([len(s) for s in seqs], 16_384):
            ids, mask = _pad_right([seqs[i] for i in batch], pad_id, device)
            model(input_ids=ids, attention_mask=mask, use_cache=False)


def _peak_gb(model, tok, pad_id):
    import time

    seqs = _workload(tok)
    torch.cuda.reset_peak_memory_stats()
    resident = torch.cuda.memory_allocated() / 2**30
    _forward_all(model, seqs, pad_id)  # also the warm-up for the timing below
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**30
    t0 = time.perf_counter()
    for _ in range(3):
        _forward_all(model, seqs, pad_id)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 3
    return resident, peak, {"seconds_per_workload": dt, "seqs_per_s": len(seqs) / dt,
                            "tokens_per_s": len(seqs) * WORKLOAD_TOKENS / dt,
                            "gpu": torch.cuda.get_device_name(0)}


def measure_config(spec, scheme: str, packed_dir: Path | None = None) -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

    from .verifiers import from_pretrained_compat

    token = os.environ.get("HF_TOKEN")
    tok = AutoTokenizer.from_pretrained(spec.hf_id, token=token)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    kw = dict(num_labels=1, token=token, device_map="cuda")
    if scheme == "bf16":
        model = from_pretrained_compat(AutoModelForSequenceClassification, spec.hf_id, dtype=torch.bfloat16, **kw)
    elif scheme == "nf4":
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                               llm_int8_skip_modules=["score"])
        model = AutoModelForSequenceClassification.from_pretrained(spec.hf_id, quantization_config=q, **kw)
    elif scheme == "gptq-w4":
        model = from_pretrained_compat(AutoModelForSequenceClassification, str(packed_dir), dtype=torch.float16, **kw)
    else:
        raise ValueError(scheme)
    model.eval()
    weights_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**30
    weights_gb += sum(b.numel() * b.element_size() for b in model.buffers()) / 2**30
    resident, peak, speed = _peak_gb(model, tok, pad)
    del model
    torch.cuda.empty_cache()
    return {"weights_gb": weights_gb, "resident_gb": resident, "peak_gb": peak, "speed": speed,
            "workload": {"seqs": WORKLOAD_SEQS, "tokens": WORKLOAD_TOKENS}}


def measure_prm(spec, scheme: str) -> dict:
    """PRM (remote code): BF16 or deployed bitsandbytes NF4; packed GPTQ export does not
    support its remote modelling code."""
    from .verifiers import load_verifier, load_verifier_deployed

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    if scheme == "bf16":
        tok, model = load_verifier(spec)
    else:
        tok, model = load_verifier_deployed(spec, scheme)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    weights_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**30
    weights_gb += sum(b.numel() * b.element_size() for b in model.buffers()) / 2**30
    resident, peak, speed = _peak_gb(model, tok, pad)
    del model
    torch.cuda.empty_cache()
    return {"weights_gb": weights_gb, "resident_gb": resident, "peak_gb": peak, "speed": speed,
            "workload": {"seqs": WORKLOAD_SEQS, "tokens": WORKLOAD_TOKENS}}


def measure_all(verifier_keys: list[str], store_root: Path, packed_root: Path) -> dict:
    from .verifiers import VERIFIERS

    out = {}
    for v in verifier_keys:
        spec = VERIFIERS[v]
        if spec.kind != "orm":
            for scheme in ("bf16", "nf4"):
                out[f"{v}__{scheme}"] = measure_prm(spec, scheme)
                print(v, scheme, json.dumps(out[f"{v}__{scheme}"]), flush=True)
            continue
        for scheme in ("bf16", "nf4", "gptq-w4"):
            pdir = Path(packed_root) / f"{v}__gptq-w4"
            if scheme == "gptq-w4" and not (pdir / "config.json").exists():
                out[f"{v}__{scheme}"] = {"status": "missing packed checkpoint (run export first)"}
                continue
            out[f"{v}__{scheme}"] = measure_config(spec, scheme, pdir)
            print(v, scheme, json.dumps(out[f"{v}__{scheme}"]), flush=True)
    return out
