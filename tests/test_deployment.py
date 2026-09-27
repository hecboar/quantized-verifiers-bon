"""Deployment parity: fake-quant vs the real packed/bitsandbytes execution."""
import copy

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import make_tiny_rm
from vq.quant.apply import quantize_model


def test_fake_nf4_matches_bitsandbytes_deployment_scores(tmp_path):
    pytest.importorskip("bitsandbytes")
    from vq.verifiers import VerifierSpec, load_verifier_deployed

    base = make_tiny_rm(seed=1)
    base.save_pretrained(tmp_path / "m")
    fake = copy.deepcopy(base)
    quantize_model(fake, "nf4")
    _, dep = load_verifier_deployed(VerifierSpec("t", str(tmp_path / "m"), "orm"), "nf4", device="cpu",
                                    dtype=torch.float32, tokenizer=False)
    assert type(dep.model.layers[0].self_attn.q_proj).__name__ == "Linear4bit"
    assert type(dep.score).__name__ == "Linear"  # reward head kept in full precision
    ids = torch.randint(1, 256, (6, 25), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        a = fake(input_ids=ids).logits.squeeze()
        b = dep(input_ids=ids).logits.squeeze()
    torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)


def test_compare_conditions_reports_parity(tmp_path):
    from vq.analysis import compare_conditions
    from vq.store import CampaignStore

    st = CampaignStore(tmp_path)
    rows = [{"question_id": f"q{q}", "cand": c, "correct": c == 0, "answer_key": c} for q in range(5) for c in range(4)]
    st.write("candidates", "G__eval", pd.DataFrame(rows))
    base = pd.DataFrame([{"split": "eval", "question_id": r["question_id"], "cand": r["cand"], "score": float(r["cand"])}
                         for r in rows])
    st.write("scores", "G__v__gptq-w4", base)
    other = base.copy()
    other.loc[other.cand == 3, "score"] += 0.01
    st.write("scores", "G__v__gptq-w4-packed", other)
    rep = compare_conditions(st, "G", "v", "gptq-w4", "gptq-w4-packed", "eval")
    assert rep["max_abs_diff"] == pytest.approx(0.01)
    assert rep["winner_agreement_full_pool"] == 1.0 and rep["n_questions"] == 5
