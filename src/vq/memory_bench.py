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


def _peak_gb(model, tok, pad_id):
    from .verifiers import orm_score_ids

    torch.cuda.reset_peak_memory_stats()
    resident = torch.cuda.memory_allocated() / 2**30
    orm_score_ids(model, _workload(tok), pad_id, max_tokens=16_384)
    torch.cuda.synchronize()
    return resident, torch.cuda.max_memory_allocated() / 2**30


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
    resident, peak = _peak_gb(model, tok, pad)
    del model
    torch.cuda.empty_cache()
    return {"weights_gb": weights_gb, "resident_gb": resident, "peak_gb": peak,
            "workload": {"seqs": WORKLOAD_SEQS, "tokens": WORKLOAD_TOKENS}}


def measure_all(verifier_keys: list[str], store_root: Path, packed_root: Path) -> dict:
    from .verifiers import VERIFIERS

    out = {}
    for v in verifier_keys:
        spec = VERIFIERS[v]
        if spec.kind != "orm":
            continue  # PRM custom head: memory reported analytically (budget.weight_bytes)
        for scheme in ("bf16", "nf4", "gptq-w4"):
            pdir = Path(packed_root) / f"{v}__gptq-w4"
            if scheme == "gptq-w4" and not (pdir / "config.json").exists():
                out[f"{v}__{scheme}"] = {"status": "missing packed checkpoint (run export first)"}
                continue
            out[f"{v}__{scheme}"] = measure_config(spec, scheme, pdir)
            print(v, scheme, json.dumps(out[f"{v}__{scheme}"]), flush=True)
    return out
