"""Qwen2.5-Math-PRM-7B ships its own 1.6k-line modeling code (trust_remote_code).
We instantiate a TINY random model from its real config + code and run every
pipeline operation on it. Run with: pytest -m network tests/test_prm_architecture.py"""
import copy
import os

import pytest
import torch

pytestmark = pytest.mark.network
PRM_ID = "Qwen/Qwen2.5-Math-PRM-7B"


@pytest.fixture(scope="module")
def tiny_prm():
    from dotenv import load_dotenv
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    load_dotenv(".env")
    tok = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(PRM_ID, trust_remote_code=True, token=tok)
    cfg = AutoConfig.from_pretrained(PRM_ID, trust_remote_code=True, token=tok)
    from vq.verifiers import patch_prm_config

    patch_prm_config(cfg, tokenizer)
    cfg.update(dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                    num_key_value_heads=2, max_window_layers=2))
    torch.manual_seed(0)
    model = AutoModel.from_config(cfg, trust_remote_code=True).eval()
    return tokenizer, model


QUESTION = "What is 2 + 3?"
RESP = ["We add 2 and 3.\n\nThat gives 5.\n\nThe answer is \\boxed{5}.",
        "Two plus three.\n\nIt is 6.\n\nSo \\boxed{6}.",
        "\\boxed{5}"]


def _seqs(tokenizer):
    from vq.verifiers import format_prm

    return [format_prm(tokenizer, QUESTION, r) for r in RESP]


def test_head_is_outside_decoder_layers(tiny_prm):
    from vq.quant.apply import target_linears

    _, model = tiny_prm
    names = [n for n, _ in target_linears(model)]
    assert len(names) == 14 and not any(n.startswith("score") for n in names)
    assert any(n.startswith("score.") for n, _ in model.named_parameters())


def test_step_rewards_one_per_separator_and_batch_parity(tiny_prm):
    from vq.verifiers import PRM_STEP_TOKEN, prm_score_ids

    tokenizer, model = tiny_prm
    sep = tokenizer.encode(PRM_STEP_TOKEN, add_special_tokens=False)[0]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    seqs = _seqs(tokenizer)
    batched = prm_score_ids(model, seqs, sep, pad, max_tokens=100_000)
    single = prm_score_ids(model, seqs, sep, pad, max_tokens=1)
    assert [len(r) for r in batched] == [3, 3, 1]
    for a, b in zip(batched, single):
        assert a == pytest.approx(b, abs=1e-4)
    assert all(0.0 <= x <= 1.0 for r in batched for x in r)


@pytest.mark.parametrize("method", ["rtn", "nf4", "gptq"])
def test_quantization_runs_and_leaves_head_untouched(tiny_prm, method):
    from vq.quant.apply import CalibrationSet, audit_changes, quantize_model
    from vq.verifiers import PRM_STEP_TOKEN, prm_score_ids

    tokenizer, base = tiny_prm
    model = copy.deepcopy(base)
    ref = copy.deepcopy(model.state_dict())
    seqs = _seqs(tokenizer)
    calib = CalibrationSet([torch.tensor([s]) for s in seqs]) if method == "gptq" else None
    quantize_model(model, method, bits=4, group_size=32, calib=calib)
    changed = audit_changes(ref, model.state_dict())
    assert len(changed) == 14
    assert not any(n.startswith("score") or "embed" in n or "norm" in n for n in changed)
    sep = tokenizer.encode(PRM_STEP_TOKEN, add_special_tokens=False)[0]
    out = prm_score_ids(model, seqs, sep, tokenizer.eos_token_id, max_tokens=100_000)
    assert [len(r) for r in out] == [3, 3, 1]


def test_from_pretrained_round_trip_after_rotary_repair(tiny_prm, tmp_path):
    """transformers 5 + the PRM's remote code: from_pretrained left the rotary
    buffers uninitialised (garbage positions). After repair the reloaded model
    must reproduce the original logits."""
    import glob
    import shutil

    from transformers import AutoConfig, AutoModel

    from vq.verifiers import check_rotary, from_pretrained_compat, patch_prm_config, repair_legacy_rotary

    tokenizer, model = tiny_prm
    model.save_pretrained(tmp_path)
    for f in glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-PRM-7B/snapshots/*/*.py")):
        shutil.copy(f, tmp_path)
    cfg = patch_prm_config(AutoConfig.from_pretrained(tmp_path, trust_remote_code=True), tokenizer)
    m1 = from_pretrained_compat(AutoModel, str(tmp_path), dtype=torch.float32, config=cfg, trust_remote_code=True,
                                device_map="cpu").eval()
    assert repair_legacy_rotary(m1) == 2
    check_rotary(m1)
    ids = torch.tensor([_seqs(tokenizer)[0]])
    with torch.no_grad():
        a = model(input_ids=ids, use_cache=False)[0].float()
        b = m1(input_ids=ids, use_cache=False)[0].float()
    assert (a - b).abs().max().item() < 1e-3
