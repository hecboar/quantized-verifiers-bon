import torch

from vq.quant.export import codes_from_params, dequant_codes, pack_int4_cols, unpack_int4_cols
from vq.quant.gptq import build_hessian, gptq_quantize
from vq.quant.grid import quantize_rtn


def test_pack_unpack_roundtrip():
    q = torch.randint(0, 16, (64, 24), dtype=torch.int32)  # [in, out]
    packed = pack_int4_cols(q)
    assert packed.shape == (8, 24) and packed.dtype == torch.int32
    torch.testing.assert_close(unpack_int4_cols(packed), q)


def test_scales_are_fp16_representable():
    w = torch.randn(8, 256)
    _, scales, _ = quantize_rtn(w, 4, 128, return_params=True)
    torch.testing.assert_close(scales, scales.half().float(), atol=0, rtol=0)


def test_gptq_codes_reproduce_fakequant_exactly():
    torch.manual_seed(0)
    x = torch.randn(256, 256)
    w = torch.randn(32, 256)
    wq, scales, zeros = gptq_quantize(w, build_hessian(x), bits=4, group_size=128, return_params=True)
    assert scales.shape == (32, 2)
    codes = codes_from_params(wq, scales, zeros, group_size=128)
    assert codes.min() >= 0 and codes.max() <= 15
    torch.testing.assert_close(dequant_codes(codes, scales, zeros, group_size=128), wq, atol=0, rtol=0)


def test_return_params_default_off():
    w = torch.randn(4, 128)
    assert isinstance(quantize_rtn(w, 4, 128), torch.Tensor)
