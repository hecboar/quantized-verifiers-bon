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


def test_one_failing_verifier_does_not_abort_the_pilot():
    items = {f"q{i}": Item(f"q{i}", f"question {i}?", str(i % 7), "numeric", "gsm8k-train") for i in range(4)}
    specs = {"a": VerifierSpec("a", "none", "orm"), "broken": VerifierSpec("broken", "none", "orm"),
             "c": VerifierSpec("c", "none", "orm")}

    def loader(spec, device="cpu", dtype=None):
        if spec.name == "broken":
            raise OSError("No space left on device")
        return fake_loader(spec, device, dtype)

    rep = run_pilot(items, list(items), fake_generate, loader, specs, quant_verifier="a", n_calib=4,
                    device="cpu", generic_chunk=lambda tok, n: [[5 + i] * 12 for i in range(n)])
    assert set(rep["score"]) == {"a", "c"}
    assert "No space left" in rep["errors"]["broken"]
    assert rep["ok"] is False


def test_parity_is_judged_relative_to_score_scale():
    from vq.pilot import parity_ok

    assert parity_ok(max_abs_diff=0.25, score_scale=4.0) is True    # BF16 noise, measured by the null control
    assert parity_ok(max_abs_diff=3.0, score_scale=4.0) is False    # gross bug (e.g. wrong pooled token)
    assert parity_ok(max_abs_diff=0.0, score_scale=0.0) is True
