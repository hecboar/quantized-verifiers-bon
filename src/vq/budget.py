"""Analytic weight-memory accounting (C2) and GPU cost forecasting (plan §7)."""
from __future__ import annotations

from dataclasses import dataclass


def weight_bytes(target_params: int, other_params: int, scheme: str, bits: int = 4, group_size: int = 128) -> float:
    """Stored weight bytes. Non-target params (embeddings, norms, head) stay BF16.
    gptq/awq/rtn: packed ints + fp16 scale + packed zero per group.
    nf4: bitsandbytes blockwise (64) with fp32 absmax, no double quantization."""
    other = other_params * 2.0
    if scheme == "bf16":
        return target_params * 2.0 + other
    if scheme in ("gptq", "awq", "rtn"):
        groups = target_params / group_size
        return target_params * bits / 8 + groups * (2.0 + bits / 8) + other
    if scheme == "nf4":
        return target_params * 0.5 + (target_params / 64) * 4.0 + other
    if scheme == "int8":
        return target_params * 1.0 + other  # per-row scales negligible
    raise ValueError(f"unknown scheme {scheme!r}")


@dataclass
class Campaign:
    name: str
    tokens: int  # total tokens processed (scoring: input tokens; generation: output tokens)
    kind: str  # "score" | "generate" | "quantize"


def forecast(
    campaigns: list[Campaign],
    tok_per_s: dict[str, float],
    eur_per_hour: float = 0.69,
    overhead_frac: float = 0.15,
    ceiling_eur: float | None = None,
) -> dict:
    """Hours/euros from *measured* throughput (phase 0). overhead_frac covers
    model loading, quantization and idle time."""
    by = {}
    total_h = 0.0
    for c in campaigns:
        h = c.tokens / tok_per_s[c.kind] / 3600 * (1 + overhead_frac)
        by[c.name] = {"hours": h, "eur": h * eur_per_hour}
        total_h += h
    eur = total_h * eur_per_hour
    return {
        "hours": total_h,
        "eur": eur,
        "by_campaign": by,
        "over_ceiling": ceiling_eur is not None and eur > ceiling_eur,
    }
