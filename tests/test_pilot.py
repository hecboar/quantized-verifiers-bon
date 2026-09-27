import pandas as pd

from test_runner import _candidates, fake_loader
from vq.data import Item
from vq.pilot import run_pilot
from vq.verifiers import VerifierSpec


def fake_generate(items):
    df = _candidates(n_q=len(items), C=4, prefix="x")
    df["question_id"] = df["question_id"].map({f"x{i}": it.id for i, it in enumerate(items)})
    df["finish_reason"] = "stop"
    df["text"] = [f"so the answer is {i % 7}." for i in range(len(df))]
    return df.drop(columns=["correct", "answer_key", "parsed", "truncated"])


def test_pilot_report_has_every_phase0_check():
    items = {f"q{i}": Item(f"q{i}", f"question {i}?", str(i % 7), "numeric", "gsm8k-train") for i in range(6)}
    specs = {"a": VerifierSpec("a", "none", "orm"), "b": VerifierSpec("b", "none", "orm")}
    rep = run_pilot(items, list(items), fake_generate, fake_loader, specs, quant_verifier="a", n_calib=4,
                    device="cpu", generic_chunk=lambda tok, n: [[5 + i] * 12 for i in range(n)])
    assert rep["generate"] > 0 and 0 <= rep["truncated_frac"] <= 1 and rep["parsed_frac"] == 1.0
    assert set(rep["score"]) == {"a", "b"} and set(rep["avg_score_tokens_by_verifier"]) == {"a", "b"}
    assert all(v < 1e-3 for v in rep["parity"].values())
    assert rep["quant_check"]["a"]["gptq_seconds"] > 0
    assert rep["quant_check"]["a"]["audit_ok"] is True
    assert all(rep["rtn_runs"][v] == "ok" for v in specs)
    assert rep["ok"] is True


def test_previous_model_is_freed_before_next_load():
    """on a 24 GB GPU the 8B must be gone before the PRM-7B loads."""
    import gc
    import weakref

    items = {f"q{i}": Item(f"q{i}", f"question {i}?", str(i % 7), "numeric", "gsm8k-train") for i in range(4)}
    specs = {"a": VerifierSpec("a", "none", "orm"), "b": VerifierSpec("b", "none", "orm"),
             "c": VerifierSpec("c", "none", "orm")}
    alive = []

    def tracking_loader(spec, device="cpu", dtype=None):
        gc.collect()
        leaked = [r for r in alive if r() is not None]
        assert not leaked, f"{len(leaked)} earlier model(s) still referenced when loading {spec.name}"
        tok, model = fake_loader(spec, device, dtype)
        alive.append(weakref.ref(model))
        return tok, model

    rep = run_pilot(items, list(items), fake_generate, tracking_loader, specs, quant_verifier="b", n_calib=4,
                    device="cpu", generic_chunk=lambda tok, n: [[5 + i] * 12 for i in range(n)])
    assert rep["ok"] is True
