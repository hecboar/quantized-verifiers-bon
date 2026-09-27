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


def export_gptq_checkpoint(model, params: dict, outdir, bits: int = 4, group_size: int = 128) -> None:
    """Write a GPTQ (checkpoint_format="gptq", v1: qzeros store zero-1)
    safetensors checkpoint from a fake-quantized model and its per-linear
    (scales, zeros). Non-target tensors are kept as-is."""
    import json
    from pathlib import Path

    from safetensors.torch import save_file

    from .apply import target_linears

    if bits != 4:
        raise NotImplementedError("only int4 packing is supported")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    for name, lin in target_linears(model):
        scales, zeros = params[name]  # [out, groups]
        w = lin.weight.detach().float().cpu()
        codes = codes_from_params(w, scales, zeros, group_size)  # [out, in]
        del sd[f"{name}.weight"]
        sd[f"{name}.qweight"] = pack_int4_cols(codes.T.contiguous())
        sd[f"{name}.qzeros"] = pack_int4_cols((zeros.round().to(torch.int32) - 1)).T.contiguous()
        sd[f"{name}.scales"] = scales.T.contiguous().half()
        sd[f"{name}.g_idx"] = (torch.arange(w.shape[1]) // group_size).to(torch.int32)
    save_file(sd, str(outdir / "model.safetensors"))
    cfg = model.config.to_dict()
    cfg["quantization_config"] = {
        "quant_method": "gptq", "bits": bits, "group_size": group_size, "desc_act": False,
        "sym": True, "checkpoint_format": "gptq",
    }
    (outdir / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
