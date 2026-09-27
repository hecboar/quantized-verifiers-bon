"""Apply fake-quant to a verifier, layer by layer (GPTQ-style sequential
calibration: layer i is calibrated on the outputs of already-quantized layers
0..i-1). Only nn.Linear modules inside decoder layers are touched; embeddings,
norms and the reward head stay in their original precision (plan §4.3).
Within a layer all linears see the same (unquantized-layer) inputs, i.e. no
"true-sequential" ordering; this is documented in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .awq import awq_quantize
from .gptq import HessianAccumulator, gptq_quantize
from .grid import quantize_nf4, quantize_rtn

CALIBRATED = {"gptq", "awq"}
METHODS = {"rtn", "nf4", "gptq", "awq"}


@dataclass
class CalibrationSet:
    """Token-id tensors [1, T] (batch 1, no padding). `pairs` are (winner,
    challenger) sequence indices; their decisive position is the last token."""

    sequences: list[torch.Tensor]
    pairs: list[tuple[int, int]] = field(default_factory=list)
    awq_max_rows: int = 4096


class _Stop(Exception):
    pass


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    for path in ("model.layers", "model.model.layers", "transformer.h"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList):
            return obj
    raise ValueError("could not locate decoder layers")


def target_linears(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    layers = decoder_layers(model)
    layer_ids = {id(m) for m in layers.modules()}
    return [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and id(m) in layer_ids]


def audit_changes(ref: dict, new: dict) -> list[str]:
    return [k for k in ref if not torch.equal(ref[k], new[k])]


def _set_weight(lin: nn.Linear, w: torch.Tensor) -> None:
    lin.weight.data.copy_(w.to(lin.weight.dtype))


@torch.no_grad()
def _capture_layer0_inputs(model, layers, seqs):
    captured = []

    def hook(_mod, args, kwargs):
        captured.append((args, kwargs))
        raise _Stop

    h = layers[0].register_forward_pre_hook(hook, with_kwargs=True)
    device = next(model.parameters()).device
    try:
        for ids in seqs:
            try:
                model(input_ids=ids.to(device), use_cache=False)
            except _Stop:
                pass
    finally:
        h.remove()
    return captured


def _run_layer(layer, args, kwargs):
    out = layer(*args, **kwargs)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def quantize_model(
    model: nn.Module,
    method: str,
    bits: int = 4,
    group_size: int | None = 128,
    calib: CalibrationSet | None = None,
    lam: float = 0.0,
    damp: float = 0.01,
) -> dict:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    if method in CALIBRATED and calib is None:
        raise ValueError(f"{method} requires a CalibrationSet")

    if method in ("rtn", "nf4"):
        for _, lin in target_linears(model):
            w = lin.weight.data
            q = quantize_rtn(w, bits, group_size) if method == "rtn" else quantize_nf4(w)
            _set_weight(lin, q)
        return {"method": method, "n_linears": len(target_linears(model))}

    layers = decoder_layers(model)
    inputs = _capture_layer0_inputs(model, layers, calib.sequences)
    diff_rows: dict[str, int] = {}

    for li, layer in enumerate(layers):
        lins = {n: m for n, m in layer.named_modules() if isinstance(m, nn.Linear)}
        acc = {n: HessianAccumulator(m.in_features, device=m.weight.device) for n, m in lins.items()}
        last: dict[str, dict[int, torch.Tensor]] = {n: {} for n in lins}
        rows: dict[str, list[torch.Tensor]] = {n: [] for n in lins}
        state = {"seq": 0}

        def make_hook(name):
            def hook(_m, inp, _out):
                x = inp[0].detach()
                acc[name].add(x)
                last[name][state["seq"]] = x[0, -1].float()
                if method == "awq":
                    rows[name].append(x.reshape(-1, x.shape[-1]).float().cpu())
            return hook

        handles = [m.register_forward_hook(make_hook(n)) for n, m in lins.items()]
        for si, (args, kwargs) in enumerate(inputs):
            state["seq"] = si
            _run_layer(layer, args, kwargs)
        for h in handles:
            h.remove()

        for name, lin in lins.items():
            if method == "gptq":
                if calib.pairs and lam > 0:
                    d = torch.stack([last[name][w] - last[name][j] for w, j in calib.pairs])
                    acc[name].add_diffs(d)
                    diff_rows[f"{li}.{name}"] = d.shape[0]
                elif calib.pairs:
                    diff_rows[f"{li}.{name}"] = len(calib.pairs)
                H = acc[name].finalize(lam=lam)
                q = gptq_quantize(lin.weight.data, H, bits, group_size, damp=damp)
            else:  # awq
                X = torch.cat(rows[name])
                if X.shape[0] > calib.awq_max_rows:
                    g = torch.Generator().manual_seed(li)
                    X = X[torch.randperm(X.shape[0], generator=g)[: calib.awq_max_rows]]
                q = awq_quantize(lin.weight.data, X.to(lin.weight.device), bits, group_size)
            _set_weight(lin, q)

        # propagate through the quantized layer
        new_inputs = []
        for args, kwargs in inputs:
            hs = _run_layer(layer, args, kwargs)
            if args:
                new_inputs.append(((hs,) + tuple(args[1:]), kwargs))
            else:  # hidden_states passed by keyword
                new_inputs.append(((), {**kwargs, "hidden_states": hs}))
        inputs = new_inputs

    return {
        "method": method,
        "n_pairs": len(calib.pairs),
        "lam": lam,
        "diff_rows_per_linear": diff_rows,
    }
