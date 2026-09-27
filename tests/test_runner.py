import numpy as np
import pandas as pd
import pytest

from conftest import make_tiny_rm
from vq.experiment import ScoreRun
from vq.runner import RunContext, execute_run, select_lambda
from vq.store import CampaignStore
from vq.verifiers import VerifierSpec


class FakeTok:
    bos_token = None
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        return " ".join(m["content"] for m in msgs)

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [(ord(c) % 250) + 1 for c in text][:48]}

    def encode(self, text, add_special_tokens=False):
        return self(text)["input_ids"]


def fake_loader(spec, device="cpu", dtype=None):
    return FakeTok(), make_tiny_rm(seed=5)


def _candidates(n_q=12, C=6, prefix="q"):
    rng = np.random.default_rng(0)
    rows = []
    for q in range(n_q):
        for c in range(C):
            rows.append({"question_id": f"{prefix}{q}", "cand": c, "text": f"answer {rng.integers(0, 99)} step {c}",
                         "correct": bool(rng.random() < 0.5), "answer_key": int(c % 3), "parsed": True,
                         "truncated": False, "n_tokens": 10})
    return pd.DataFrame(rows)


@pytest.fixture
def ctx(tmp_path):
    store = CampaignStore(tmp_path)
    store.write("candidates", "G__eval", _candidates(prefix="e"))
    store.write("candidates", "G__dev", _candidates(prefix="d"))
    questions = {f"{p}{q}": f"question {p}{q}?" for p in "ed" for q in range(12)}
    return RunContext(
        store=store, questions=questions, device="cpu", loader=fake_loader,
        verifier_specs={"tiny": VerifierSpec("tiny", "none", "orm")},
        n_calib=8, dev_scored=6, shard_size=20, generic_ids=[[i % 200 + 1 for i in range(k, k + 30)] for k in range(20)],
    )


def test_bf16_run_scores_eval_and_dev_subset(ctx):
    run = ScoreRun("G", "tiny", "bf16", "none")
    execute_run(run, ctx)
    df = ctx.store.read("scores", run.name)
    assert set(df["split"]) == {"eval", "dev"}
    assert (df["split"] == "eval").sum() == 12 * 6
    assert (df["split"] == "dev").sum() == 6 * 6
    assert df["score"].notna().all()


def test_quantized_run_is_audited_and_head_untouched(ctx):
    execute_run(ScoreRun("G", "tiny", "bf16", "none"), ctx)
    execute_run(ScoreRun("G", "tiny", "bf16", "none"), ctx, dev_full=True)
    run = ScoreRun("G", "tiny", "gptq-w4-ch-s0", "gptq", 4, 32, "ch", 0, 1.0)
    execute_run(run, ctx)
    meta = ctx.store.meta("scores", run.name)
    assert meta["audit"]["n_changed"] == 14
    assert meta["audit"]["head_or_embed_changed"] is False
    assert meta["quant"]["n_pairs"] > 0
    a = ctx.store.read("scores", "G__tiny__bf16")["score"].to_numpy()
    b = ctx.store.read("scores", run.name)["score"].to_numpy()
    assert not np.allclose(a, b)


def test_generic_calibration_run(ctx):
    run = ScoreRun("G", "tiny", "gptq-w4", "gptq", 4, 32, "generic", 0, None)
    execute_run(run, ctx)
    assert ctx.store.meta("scores", run.name)["calibration"]["recipe"] == "generic"


def test_generic_texts_tokenised_with_verifier_tokenizer(ctx):
    from vq.runner import generic_chunks

    # FakeTok caps each text at 48 tokens -> 3 texts = 144 tokens = 7 chunks of 20
    ids = generic_chunks(FakeTok(), ["x" * 500, "y" * 500, "z" * 500], chunk_tokens=20, n=5)
    assert len(ids) == 5 and all(len(c) == 20 for c in ids)
    ctx.generic_ids = None
    ctx.generic_texts = ["hello world " * 50] * 40
    ctx.generic_chunk_tokens = 16
    run = ScoreRun("G", "tiny", "gptq-w4", "gptq", 4, 32, "generic", 0, None)
    execute_run(run, ctx)
    assert ctx.store.meta("scores", run.name)["calibration"]["n_sequences"] == 8


def test_skip_when_done(ctx):
    run = ScoreRun("G", "tiny", "bf16", "none")
    execute_run(run, ctx)
    assert execute_run(run, ctx) == "skipped"


def test_resume_from_shards(ctx):
    run = ScoreRun("G", "tiny", "bf16", "none")
    execute_run(run, ctx, stop_after_shards=2)
    assert not ctx.store.done("scores", run.name)
    assert ctx.store.shard_ids("scores", run.name) == [0, 1]
    execute_run(run, ctx)
    df = ctx.store.read("scores", run.name)
    assert len(df) == 12 * 6 + 6 * 6 and not df.duplicated(["split", "question_id", "cand"]).any()


def test_select_lambda_minimises_dev_winner_change():
    ref = np.random.default_rng(0).normal(size=(30, 8))
    cands = {0.5: ref + 0.5 * np.random.default_rng(1).normal(size=ref.shape),
             2.0: ref + 0.05 * np.random.default_rng(2).normal(size=ref.shape),
             8.0: ref + 1.0 * np.random.default_rng(3).normal(size=ref.shape)}
    assert select_lambda(ref, cands) == 2.0


def test_dev_full_quantized_condition_scores_only_dev(ctx):
    # phase 1 gate: eval pools do not exist yet
    execute_run(ScoreRun("G", "tiny", "bf16", "none"), ctx, dev_full=True)
    run = ScoreRun("G", "tiny", "nf4", "nf4")
    execute_run(run, ctx, dev_full=True)
    df = ctx.store.read("scores", "G__tiny__dev-full-nf4")
    assert set(df["split"]) == {"dev"} and len(df) == 12 * 6


def test_calibration_from_primary_generator_evaluated_on_secondary(tmp_path):
    store = CampaignStore(tmp_path)
    store.write("candidates", "G__dev", _candidates(prefix="d"))
    store.write("candidates", "H__eval", _candidates(n_q=5, prefix="h"))  # no H__dev (gen2 subset)
    questions = {f"d{q}": f"q d{q}" for q in range(12)} | {f"h{q}": f"q h{q}" for q in range(5)}
    c = RunContext(store=store, questions=questions, device="cpu", loader=fake_loader,
                   verifier_specs={"tiny": VerifierSpec("tiny", "none", "orm")}, n_calib=8, dev_scored=6, shard_size=20)
    execute_run(ScoreRun("G", "tiny", "bf16", "none"), c, dev_full=True)
    run = ScoreRun("H", "tiny", "ch-gptq-w4", "gptq", 4, 32, "ch", 0, 1.0, calib_generator="G")
    execute_run(run, c)
    df = store.read("scores", run.name)
    assert set(df["split"]) == {"eval"} and len(df) == 5 * 6
    assert store.meta("scores", run.name)["calibration"]["calib_generator"] == "G"


def test_missing_eval_pools_error_is_explicit(ctx):
    ctx.store = CampaignStore(ctx.store.root / "empty")
    ctx.store.write("candidates", "G__dev", _candidates(prefix="d"))
    with pytest.raises(FileNotFoundError, match="G__eval"):
        execute_run(ScoreRun("G", "tiny", "bf16", "none"), ctx)
