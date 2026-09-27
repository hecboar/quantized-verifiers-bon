"""Integer codes + int4 packing for exporting fake-quant weights to a packed
GPTQ-style checkpoint (qweight packs 8 int4 along the input dimension into
int32, low bits first). Loading with a real kernel is validated on GPU in
phase 0 (score parity fake-quant vs packed)."""
from __future__ import annotations

import torch


def codes_from_params(wq: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group_size: int) -> torch.Tensor:
    """wq [out, in] dequantized weights -> int codes [out, in]."""
    s = scales.repeat_interleave(group_size, dim=1)[:, : wq.shape[1]]
    z = zeros.repeat_interleave(group_size, dim=1)[:, : wq.shape[1]]
    return torch.round(wq.float() / s + z).to(torch.int32)


def dequant_codes(codes: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group_size: int) -> torch.Tensor:
    s = scales.repeat_interleave(group_size, dim=1)[:, : codes.shape[1]]
    z = zeros.repeat_interleave(group_size, dim=1)[:, : codes.shape[1]]
    return s * (codes.float() - z)


def pack_int4_cols(q: torch.Tensor) -> torch.Tensor:
    """q [in, out] ints in [0, 15] -> [in/8, out] int32."""
    if q.shape[0] % 8:
        raise ValueError("input dim must be divisible by 8")
    q = q.to(torch.int64).reshape(q.shape[0] // 8, 8, q.shape[1])
    shifts = (torch.arange(8, dtype=torch.int64) * 4).view(1, 8, 1)
    packed = (q << shifts).sum(dim=1)
    packed = torch.where(packed >= 2**31, packed - 2**32, packed)
    return packed.to(torch.int32)


def unpack_int4_cols(packed: torch.Tensor) -> torch.Tensor:
    p = packed.to(torch.int64) & 0xFFFFFFFF
    shifts = (torch.arange(8, dtype=torch.int64) * 4).view(1, 8, 1)
    q = (p.unsqueeze(1) >> shifts) & 0xF
    return q.reshape(-1, packed.shape[1]).to(torch.int32)
