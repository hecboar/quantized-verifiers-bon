import json

import torch
from safetensors.torch import load_file

from conftest import make_tiny_rm
from vq.quant.apply import CalibrationSet, quantize_model, target_linears
from vq.quant.export import dequant_codes, export_gptq_checkpoint, unpack_int4_cols


def test_export_roundtrip_matches_fakequant(tmp_path, calib_ids):
    model = make_tiny_rm(seed=6)
    stats = quantize_model(model, "gptq", bits=4, group_size=32, calib=CalibrationSet(calib_ids), keep_params=True)
    export_gptq_checkpoint(model, stats["params"], tmp_path, bits=4, group_size=32)

    cfg = json.loads((tmp_path / "config.json").read_text())
    qc = cfg["quantization_config"]
    assert qc["quant_method"] == "gptq" and qc["bits"] == 4 and qc["group_size"] == 32
    assert qc["sym"] is True and qc["desc_act"] is False

    sd = load_file(tmp_path / "model.safetensors")
    for name, lin in target_linears(model):
        qweight = sd[f"{name}.qweight"]            # [in/8, out]
        qzeros = sd[f"{name}.qzeros"]              # [groups, out/8]
        scales = sd[f"{name}.scales"].float()      # [groups, out]
        codes = unpack_int4_cols(qweight).T        # [out, in]
        zeros = unpack_int4_cols(qzeros.T.contiguous()).T.float() + 1  # v1 format stores zero - 1
        w = dequant_codes(codes, scales.T, zeros.T, group_size=32)
        torch.testing.assert_close(w, lin.weight.float(), atol=0, rtol=0)
        assert f"{name}.weight" not in sd
    # non-target tensors (embeddings, norms, head) kept in full precision
    assert any(k.startswith("score") for k in sd)


def test_keep_params_off_by_default(calib_ids):
    model = make_tiny_rm(seed=6)
    stats = quantize_model(model, "gptq", bits=4, group_size=32, calib=CalibrationSet(calib_ids))
    assert "params" not in stats
