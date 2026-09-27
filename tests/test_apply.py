import copy

import torch

from vq.quant.apply import (
    CalibrationSet,
    audit_changes,
    quantize_model,
    target_linears,
)
from vq.quant.grid import quantize_rtn


def _score(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits.float().squeeze()


def test_target_linears_exclude_head_and_embeddings(tiny_rm):
    names = [n for n, _ in target_linears(tiny_rm)]
    assert names, "no targets found"
    assert all(".layers." in n for n in names)
    assert not any(n.startswith("score") for n in names)
    # 7 projections per decoder layer (q,k,v,o,gate,up,down) x 2 layers
    assert len(names) == 14


def test_rtn_changes_only_layer_linears(tiny_rm):
    ref = copy.deepcopy(tiny_rm.state_dict())
    quantize_model(tiny_rm, method="rtn", bits=4, group_size=32)
    changed = audit_changes(ref, tiny_rm.state_dict())
    assert changed and all(".layers." in n and n.endswith(".weight") for n in changed)
    assert not any("norm" in n or "embed" in n or n.startswith("score") for n in changed)


def test_rtn_matches_direct_quantization(tiny_rm):
    name, lin = target_linears(tiny_rm)[0]
    w = lin.weight.detach().clone()
    quantize_model(tiny_rm, method="rtn", bits=4, group_size=32)
    torch.testing.assert_close(lin.weight.detach(), quantize_rtn(w, 4, 32))


def test_nf4_runs_and_changes_weights(tiny_rm):
    ref = copy.deepcopy(tiny_rm.state_dict())
    quantize_model(tiny_rm, method="nf4")
    assert len(audit_changes(ref, tiny_rm.state_dict())) == 14


def test_gptq_needs_calibration(tiny_rm):
    import pytest

    with pytest.raises(ValueError):
        quantize_model(tiny_rm, method="gptq", bits=4, group_size=32)


def test_gptq_sequential_beats_rtn_on_scores(calib_ids):
    from conftest import make_tiny_rm

    base = make_tiny_rm(seed=3)
    held = [torch.randint(1, 256, (1, 30), generator=torch.Generator().manual_seed(i)) for i in range(24)]
    s0 = torch.stack([_score(base, x) for x in held])

    m_rtn = copy.deepcopy(base)
    quantize_model(m_rtn, method="rtn", bits=3, group_size=32)
    m_gptq = copy.deepcopy(base)
    quantize_model(m_gptq, method="gptq", bits=3, group_size=32, calib=CalibrationSet(calib_ids))

    err = lambda m: ((torch.stack([_score(m, x) for x in held]) - s0) ** 2).mean()
    assert err(m_gptq) < err(m_rtn)


def test_contrastive_collects_pair_differences(tiny_rm, calib_ids):
    pairs = [(0, 1), (2, 3), (4, 5)]
    calib = CalibrationSet(calib_ids, pairs=pairs)
    stats = quantize_model(tiny_rm, method="gptq", bits=4, group_size=32, calib=calib, lam=1.0)
    assert stats["n_pairs"] == 3
    assert all(v == 3 for v in stats["diff_rows_per_linear"].values())


def test_contrastive_lambda_changes_result(calib_ids):
    from conftest import make_tiny_rm

    a, b = make_tiny_rm(seed=4), make_tiny_rm(seed=4)
    pairs = [(i, i + 1) for i in range(0, 16, 2)]
    quantize_model(a, method="gptq", bits=3, group_size=32, calib=CalibrationSet(calib_ids, pairs=pairs), lam=0.0)
    quantize_model(b, method="gptq", bits=3, group_size=32, calib=CalibrationSet(calib_ids, pairs=pairs), lam=5.0)
    diffs = audit_changes(a.state_dict(), b.state_dict())
    assert len(diffs) > 0


def test_awq_runs(tiny_rm, calib_ids):
    ref = copy.deepcopy(tiny_rm.state_dict())
    quantize_model(tiny_rm, method="awq", bits=4, group_size=32, calib=CalibrationSet(calib_ids))
    assert len(audit_changes(ref, tiny_rm.state_dict())) == 14


def test_calibration_capture_works_when_forward_rejects_use_cache(calib_ids):
    """Custom-code PRMs may not accept `use_cache`; capture must still work."""
    from conftest import make_tiny_rm

    base = make_tiny_rm(seed=8)

    class NoCacheKw(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.model = inner.model
            self.score = inner.score
            self._inner = inner

        def forward(self, input_ids, attention_mask=None):
            return self._inner(input_ids=input_ids, attention_mask=attention_mask)

    wrapped = NoCacheKw(base)
    stats = quantize_model(wrapped, method="gptq", bits=4, group_size=32, calib=CalibrationSet(calib_ids))
    assert stats["method"] == "gptq"
