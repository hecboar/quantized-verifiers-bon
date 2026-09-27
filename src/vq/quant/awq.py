"""AWQ-style activation-aware scaling (Lin et al., 2024), per linear layer.

Simplification vs. the reference: scales are searched per linear (not shared
across q/k/v) and there is no clipping search. For fake-quant the scaled
quantization Q(W diag(s)) diag(s)^-1 is output-equivalent to folding 1/s into
the previous op. Parity against AutoAWQ is checked on GPU in phase 0.
"""
from __future__ import annotations

import torch

from .grid import quantize_rtn


@torch.no_grad()
def awq_quantize(
    w: torch.Tensor,
    x: torch.Tensor,
    bits: int,
    group_size: int | None = 128,
    n_grid: int = 21,
) -> torch.Tensor:
    dtype = w.dtype
    W = w.float()
    X = x.reshape(-1, x.shape[-1]).float()
    act = X.abs().mean(dim=0).clamp_min(1e-8)
    ref = X @ W.T
    best_err, best_q = None, None
    for alpha in torch.linspace(0, 1, n_grid):
        s = act.pow(alpha)
        s = s / torch.sqrt(s.max() * s.min())
        q = quantize_rtn(W * s, bits, group_size) / s
        err = ((X @ q.T - ref) ** 2).mean()
        if best_err is None or err < best_err:
            best_err, best_q = err, q
    return best_q.to(dtype)
