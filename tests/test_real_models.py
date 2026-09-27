"""Phase-0 checks on real checkpoints (CPU). Run with: pytest -m network"""
import copy
import os

import pytest
import torch

pytestmark = pytest.mark.network

Q = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
GOOD = "In April she sold 48 clips.\n\nIn May she sold 48 / 2 = 24 clips.\n\nAltogether she sold 48 + 24 = 72 clips.\n\nThe answer is \\boxed{72}."
BAD = "In April she sold 48 clips.\n\nIn May she sold 48 * 2 = 96 clips.\n\nAltogether she sold 48 + 96 = 144 clips.\n\nThe answer is \\boxed{144}."


@pytest.fixture(scope="module")
def sky06():
    from dotenv import load_dotenv
    from vq.verifiers import VERIFIERS, load_verifier, Scorer

    load_dotenv()
    spec = VERIFIERS["sky-0.6b"]
    tok, model = load_verifier(spec, device="cpu", dtype=torch.float32)
    return Scorer(spec, tok, model, max_tokens=4096)


def test_template_has_no_system_prompt_and_single_bos(sky06):
    ids = sky06.encode(Q, GOOD)
    text = sky06.tok.decode(ids)
    assert "system" not in text.split("user")[0].lower()
    if sky06.tok.bos_token:
        assert text.count(sky06.tok.bos_token) <= 1


def test_batched_equals_single_real(sky06):
    qs = [Q] * 4
    rs = [GOOD, BAD, GOOD[:60], "The answer is \\boxed{72}."]
    batched = sky06.score(qs, rs)["score"]
    sky06.max_tokens = 1
    single = sky06.score(qs, rs)["score"]
    sky06.max_tokens = 4096
    assert batched == pytest.approx(single, abs=2e-3)


def test_correct_beats_wrong(sky06):
    s = sky06.score([Q, Q], [GOOD, BAD])["score"]
    assert s[0] > s[1]


def test_quantization_runs_on_real_architecture(sky06):
    from vq.quant.apply import CalibrationSet, audit_changes, quantize_model, target_linears

    model = copy.deepcopy(sky06.model)
    ref = copy.deepcopy(model.state_dict())
    calib = CalibrationSet([torch.tensor([sky06.encode(Q, r)]) for r in (GOOD, BAD)], pairs=[(0, 1)])
    quantize_model(model, method="gptq", bits=4, group_size=128, calib=calib, lam=1.0)
    changed = audit_changes(ref, model.state_dict())
    assert len(changed) == len(target_linears(model))
    assert not any(n.startswith("score") or "embed" in n or "norm" in n for n in changed)
    from vq.verifiers import Scorer

    s = Scorer(sky06.spec, sky06.tok, model).score([Q, Q], [GOOD, BAD])["score"]
    assert all(torch.isfinite(torch.tensor(s)))


def test_prm_tokenizer_format():
    from dotenv import load_dotenv
    from transformers import AutoTokenizer
    from vq.verifiers import PRM_STEP_TOKEN, format_prm

    load_dotenv()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-PRM-7B", token=os.environ.get("HF_TOKEN"), trust_remote_code=True)
    ids = format_prm(tok, Q, GOOD)
    sep = tok.encode(PRM_STEP_TOKEN, add_special_tokens=False)
    assert len(sep) == 1, "step separator must be a single token"
    assert ids.count(sep[0]) == 4  # four steps in GOOD
