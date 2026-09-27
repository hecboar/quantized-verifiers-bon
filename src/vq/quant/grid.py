"""Quantization grids for fake-quant (quantize -> dequantize, same dtype).

Symmetric integer grid follows the GPTQ reference convention (sym=True,
zero point at (maxq+1)/2), which is what Marlin/exllama kernels expect.
"""
from __future__ import annotations

import torch

# bitsandbytes NF4 code values (Dettmers et al., QLoRA)
NF4_CODEBOOK = torch.tensor(
    [
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
        0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
        0.7229568362236023, 1.0,
    ]
)


def sym_params(x: torch.Tensor, bits: int, fp16_scales: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric scale/zero for a [rows, cols] block."""
    maxq = 2**bits - 1
    xmax = x.abs().amax(dim=1, keepdim=True)
    xmax = torch.where(xmax == 0, torch.ones_like(xmax), xmax)
    # fp16-representable scales, as stored by packed kernels (the dequantized
    # weight is then cast to the model dtype, so fake-quant (BF16) and packed
    # (FP16) execution are close but NOT bit-identical: parity is measured).
    # Unlike the reference GPTQ quantizer, groups with no negative values are
    # also mirrored (the reference clips them at ~0.47 xmax).
    scale = 2 * xmax / maxq
    if fp16_scales:
        scale = scale.half().float()
    zero = torch.full_like(scale, (maxq + 1) / 2)
    return scale, zero


def quantize_with(x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, bits: int) -> torch.Tensor:
    maxq = 2**bits - 1
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


def quantize_rtn(w: torch.Tensor, bits: int, group_size: int | None, return_params: bool = False,
                 fp16_scales: bool = True):
    dtype = w.dtype
    x = w.float()
    out = torch.empty_like(x)
    step = group_size or x.shape[1]
    scales, zeros = [], []
    for g in range(0, x.shape[1], step):
        blk = x[:, g : g + step]
        scale, zero = sym_params(blk, bits, fp16_scales)
        out[:, g : g + step] = quantize_with(blk, scale, zero, bits)
        scales.append(scale)
        zeros.append(zero)
    if return_params:
        return out.to(dtype), torch.cat(scales, dim=1), torch.cat(zeros, dim=1)
    return out.to(dtype)


def quantize_nf4(w: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """Blockwise absmax NF4 (bitsandbytes layout: contiguous blocks of the
    flattened tensor). Double quantization of absmax is not simulated."""
    dtype, shape = w.dtype, w.shape
    x = w.float().reshape(-1)
    if x.numel() % block_size:
        raise ValueError("numel must be divisible by block_size")
    blocks = x.reshape(-1, block_size)
    absmax = blocks.abs().amax(dim=1, keepdim=True)
    safe = torch.where(absmax == 0, torch.ones_like(absmax), absmax)
    code = NF4_CODEBOOK.to(x.device)
    idx = (blocks / safe).unsqueeze(-1).sub(code).abs().argmin(dim=-1)
    deq = code[idx] * absmax
    return deq.reshape(shape).to(dtype)
