"""The real CLI commands, end to end, with a tiny model instead of the GPU verifiers."""
import json

import pandas as pd
import pytest

import vq.cli as cli
from test_cli import fake_loaders
from test_runner import FakeTok, _candidates, fake_loader
from vq.experiment import load_config
from vq.store import CampaignStore
from vq.verifiers import VerifierSpec


@pytest.fixture
def env(tmp_path, monkeypatch):
    data, out = tmp_path / "data", tmp_path / "out"
    monkeypatch.setattr(cli, "DATA", data)
    monkeypatch.setattr(cli, "OUT", out)
    cli.prepare_data(load_config(cli.DEFAULT_CFG), fake_loaders(), data)
    ev = json.loads((data / "manifest_eval.json").read_text())["ids"][:10]
    dev = json.loads((data / "manifest_gate.json").read_text())["ids"]  # gate subset must be fully pooled

    def pools(ids):
        df = _candidates(n_q=len(ids), C=6, prefix="x")
        df["question_id"] = df["question_id"].map({f"x{i}": q for i, q in enumerate(ids)})
        return df

    store = CampaignStore(out / "store")
    store.write("candidates", "G__dev", pools(dev))
    real_context = cli._context

    def tiny_context(args):
        monkeypatch.setattr(cli, "_generic_texts", lambda: ["generic calibration text " * 30] * 50)
        ctx = real_context(args)
        ctx.device, ctx.loader, ctx.n_calib, ctx.dev_scored, ctx.shard_size = "cpu", fake_loader, 8, 6, 40
        ctx.verifier_specs = {"sky-4b": VerifierSpec("sky-4b", "none", "orm")}
        ctx.dtype = None
        ctx.generic_chunk_tokens = 16
        return ctx

    monkeypatch.setattr(cli, "_context", tiny_context)
    return store, ev, pools


def test_gate_then_main_phase_then_analyze(env, capsys):
    store, ev, pools = env
    # phase 1: dev only (evaluation pools do not exist yet)
    cli.main(["score", "--phase", "phase3_main", "--generator", "G", "--only", "sky-4b",
              "--conditions", "bf16,nf4,gptq-w4", "--dev-only"])
    assert store.done("scores", "G__sky-4b__dev-full-nf4")
    # phase 3 on evaluation pools
    store.write("candidates", "G__eval", pools(ev))
    cli.main(["score", "--phase", "phase3_main", "--generator", "G", "--only", "sky-4b"])
    assert store.done("scores", "G__sky-4b__gptq-w4") and store.done("scores", "G__sky-4b__awq-w4")
    meta = store.meta("scores", "G__sky-4b__gptq-w4")
    assert meta["audit"]["head_or_embed_changed"] is False
    cli.main(["analyze", "--generator", "G"])
    main = pd.read_csv(cli.OUT / "tables" / "main.csv")
    assert {"nf4", "gptq-w4", "awq-w4"} <= set(main["condition"])
    assert (main.loc[main.N == 1, "winner_change"] == 0).all()


def test_gate_output_is_json(env, capsys):
    cli.main(["score", "--phase", "phase3_main", "--generator", "G", "--only", "sky-4b",
              "--conditions", "bf16", "--dev-only"])
    capsys.readouterr()
    cli.main(["gate", "--generator", "G", "--verifier", "sky-4b", "--projected-eur", "60", "--implementation-ok"])
    res = json.loads(capsys.readouterr().out)
    assert set(res["criteria"]) == {"implementation_ok", "bf16_useful", "mixed_pools", "budget_ok"}


def test_select_lambda_scores_dev_subset_only(env, monkeypatch):
    store, ev, pools = env
    cli.main(["select-lambda", "--generator", "G", "--verifier", "sky-4b"])
    names = store.list("scores")
    assert any(n.startswith("G__sky-4b__devsub-lamsel-") for n in names)
    assert not store.done("candidates", "G__eval")  # worked before eval pools even exist
    for n in names:
        if "lamsel" in n:
            assert set(store.read("scores", n)["split"]) == {"dev"}
    frozen = json.loads((cli.OUT / "frozen_lambda.json").read_text())
    assert frozen["sky-4b"] == frozen["sky-8b"]


def test_pilot_command_end_to_end_on_cpu(env, monkeypatch, capsys):
    """The exact `pilot` command, with vLLM (child process) and the verifier
    downloads replaced by fakes. Catches wiring errors before paying for a pod."""
    import subprocess

    import vq.verifiers
    from test_pilot import fake_generate

    store, ev, pools = env
    items = cli._items()

    def fake_run(argv, check=True):
        assert argv[1:4] == ["-m", "vq.cli", "generate"] and "--split" in argv and "gate" in argv
        n = int(argv[argv.index("--limit") + 1])
        ids = cli._manifest("gate")[:n]
        df = fake_generate([items[i] for i in ids])
        store.write("raw_candidates", f"G__gate-pilot{n}", df)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    from conftest import make_tiny_prm
    from test_runner import FakeTok

    def by_kind(spec, device="cpu", dtype=None):
        return (FakeTok(), make_tiny_prm(seed=2)) if spec.kind == "prm" else fake_loader(spec, device, dtype)

    monkeypatch.setattr(vq.verifiers, "load_verifier", by_kind)
    monkeypatch.setattr(cli, "_generic_texts", lambda: ["generic calibration text " * 400] * 20)
    cli.main(["pilot", "--generator", "G", "--questions", "4", "--device", "cpu", "--chunk-tokens", "16"])
    out = capsys.readouterr().out
    assert "PILOT OK" in out
    tp = json.loads((cli.OUT / "throughput.json").read_text())
    assert set(tp["score"]) == {"sky-1.7b", "sky-4b", "sky-8b", "prm-7b"}
    assert tp["quant_hours"]["gptq"] > 0
