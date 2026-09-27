"""GPTQ (Frantar et al., 2023), following the reference implementation, plus
the contrastive Hessian term used by CH-GPTQ (plan §5):

    H' = 2/n X^T X + lam * 2/m D^T D,   D = rows of (h_w - h_j)
"""
from __future__ import annotations

import torch

from .grid import quantize_with, sym_params


def build_hessian(x: torch.Tensor, diffs: torch.Tensor | None = None, lam: float = 0.0) -> torch.Tensor:
    x = x.float()
    h = 2.0 * x.T @ x / x.shape[0]
    if diffs is not None and lam > 0:
        d = diffs.float()
        h = h + lam * 2.0 * d.T @ d / d.shape[0]
    return h


class HessianAccumulator:
    """Streaming version of build_hessian for layer hooks."""

    def __init__(self, dim: int, device="cpu"):
        self.h = torch.zeros(dim, dim, device=device, dtype=torch.float32)
        self.hd = torch.zeros(dim, dim, device=device, dtype=torch.float32)
        self.n = 0
        self.m = 0

    def add(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).float()
        self.h += x.T @ x
        self.n += x.shape[0]

    def add_diffs(self, d: torch.Tensor) -> None:
        d = d.reshape(-1, d.shape[-1]).float()
        self.hd += d.T @ d
        self.m += d.shape[0]

    def finalize(self, lam: float = 0.0) -> torch.Tensor:
        if self.n == 0:
            raise RuntimeError("no calibration activations collected")
        h = 2.0 * self.h / self.n
        if lam > 0 and self.m > 0:
            h = h + lam * 2.0 * self.hd / self.m
        return h


@torch.no_grad()
def gptq_quantize(
    w: torch.Tensor,
    h: torch.Tensor,
    bits: int,
    group_size: int | None = 128,
    damp: float = 0.01,
    block_size: int = 128,
    return_params: bool = False,
):
    """Returns dequantized weights (same shape/dtype as w); with
    return_params also the per-group scales and zeros [out, n_groups].
    No act-order (static groups, required for packed-kernel export)."""
    dtype = w.dtype
    W = w.clone().float()
    H = h.clone().float().to(W.device)
    cols = W.shape[1]

    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0

    if damp > 0:
        H += damp * torch.mean(torch.diag(H)) * torch.eye(cols, device=W.device)
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    Q = torch.zeros_like(W)
    scale = zero = None
    all_scales, all_zeros = [], []
    if group_size is None:
        scale, zero = sym_params(W, bits)
        all_scales.append(scale)
        all_zeros.append(zero)

    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(count):
            col = i1 + i
            if group_size is not None and col % group_size == 0:
                # reference quirk: params from W (not W1) for the group
                scale, zero = sym_params(W[:, col : col + group_size], bits)
                all_scales.append(scale)
                all_zeros.append(zero)
            w_col = W1[:, i : i + 1]
            d = Hinv1[i, i]
            q = quantize_with(w_col, scale, zero, bits)
            Q1[:, i : i + 1] = q
            err = (w_col - q) / d
            W1[:, i:] -= err @ Hinv1[i : i + 1, i:]
            Err1[:, i : i + 1] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    if return_params:
        return Q.to(dtype), torch.cat(all_scales, dim=1), torch.cat(all_zeros, dim=1)
    return Q.to(dtype)
