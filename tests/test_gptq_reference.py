"""Our GPTQ vs the official reference implementation (IST-DASLab/gptq,
Apache-2.0, vendored unmodified in tests/reference/)."""
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from vq.quant.gptq import build_hessian, gptq_quantize

REF = Path(__file__).parent / "reference"


@pytest.fixture(scope="module")
def reference(monkey_module=None):
    sys.path.insert(0, str(REF))
    real_sync = torch.cuda.synchronize
    torch.cuda.synchronize = lambda *a, **k: None  # reference calls it unconditionally
    try:
        import gptq as ref_gptq  # noqa: E402
        import quant as ref_quant  # noqa: E402
        yield ref_gptq, ref_quant
    finally:
        torch.cuda.synchronize = real_sync
        sys.path.remove(str(REF))


def _reference_quantize(ref, w, h, bits, group):
    ref_gptq, ref_quant = ref
    layer = nn.Linear(w.shape[1], w.shape[0], bias=False)
    layer.weight.data = w.clone()
    g = ref_gptq.GPTQ(layer)
    g.quantizer = ref_quant.Quantizer()
    g.quantizer.configure(bits, perchannel=True, sym=True, mse=False)
    g.H = h.clone()
    g.fasterquant(blocksize=128, percdamp=0.01, groupsize=group, actorder=False)
    return layer.weight.data.clone()


@pytest.mark.parametrize("bits,group", [(4, 128), (3, 128), (4, -1)])
def test_matches_reference_exactly(reference, bits, group):
    torch.manual_seed(bits * 10 + (group > 0))
    x = torch.randn(512, 256) @ (torch.eye(256) + 0.3 * torch.randn(256, 256))
    w = torch.randn(48, 256)
    h = build_hessian(x)
    ours = gptq_quantize(w, h, bits=bits, group_size=None if group == -1 else group, damp=0.01,
                         fp16_scales=False)
    theirs = _reference_quantize(reference, w, h, bits, group)
    torch.testing.assert_close(ours, theirs, atol=1e-5, rtol=0)


def test_all_positive_group_is_where_we_intentionally_differ(reference):
    """Reference sym quantizer does not mirror xmin for all-positive groups
    (scale = xmax/maxq with zero 8 -> clips at ~0.47 xmax). Ours is symmetric."""
    torch.manual_seed(0)
    w = torch.rand(8, 128) + 0.1  # every group all-positive
    h = torch.eye(128)
    ours = gptq_quantize(w, h, bits=4, group_size=128, damp=0.0, fp16_scales=False)
    theirs = _reference_quantize(reference, w, h, 4, 128)
    assert (w - ours).abs().max() < (w - theirs).abs().max()
    assert ours.max() > 0.9 * w.max() and theirs.max() < 0.5 * w.max()


@pytest.mark.parametrize("bits,group", [(4, 128), (3, 128), (4, None)])
def test_memory_lean_gptq_is_bit_identical_to_previous_version(bits, group):
    """The PRM-7B (down_proj: 18944 inputs) OOMed on 24 GB inside the Hessian
    inversion. The lean version (no Hessian clone, diagonal damping instead of an
    identity matrix, eager frees) must give bit-identical results, so runs made
    before the change stay valid."""
    import importlib.util
    from pathlib import Path

    import torch

    from vq.quant.gptq import gptq_quantize

    spec = importlib.util.spec_from_file_location("gptq_old", Path(__file__).parent / "reference" / "gptq_before_memfix.py")
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    g = torch.Generator().manual_seed(bits * 7 + (group or 0))
    w = torch.randn(48, 384, generator=g).to(torch.bfloat16)
    x = torch.randn(1000, 384, generator=g)
    x[:, 5] = 0.0  # a dead input column
    h = 2.0 * x.T @ x / x.shape[0]
    ref = old.gptq_quantize(w, h.clone(), bits, group, return_params=True)
    h_keep = h.clone()
    new = gptq_quantize(w, h, bits, group, return_params=True)
    assert all(torch.equal(a, b) for a, b in zip(ref, new))
    assert torch.equal(h, h_keep)  # caller's Hessian untouched by default
    new2 = gptq_quantize(w, h.clone(), bits, group, return_params=True, consume_h=True)
    assert all(torch.equal(a, b) for a, b in zip(ref, new2))


def test_boxed_hessian_is_consumed_and_bit_identical():
    import torch

    from vq.quant.gptq import gptq_quantize

    g = torch.Generator().manual_seed(3)
    w = torch.randn(16, 256, generator=g).to(torch.bfloat16)
    x = torch.randn(500, 256, generator=g)
    h = 2.0 * x.T @ x / 500
    ref = gptq_quantize(w, h.clone(), 4, 128, return_params=True)
    box = [h.clone()]
    out = gptq_quantize(w, box, 4, 128, return_params=True)
    assert box == [] and all(torch.equal(a, b) for a, b in zip(ref, out))
