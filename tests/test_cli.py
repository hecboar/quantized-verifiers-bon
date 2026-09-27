from pathlib import Path

import pytest

from vq.cli import main, prepare_data
from vq.data import Item, load_items, load_manifest
from vq.experiment import load_config

CFG = Path(__file__).parents[1] / "configs" / "experiment.yaml"


def fake_loaders():
    mk = lambda p, n, kind: [Item(f"{p}-{i}", f"{p} question {i}", str(i), kind, p) for i in range(n)]
    return {
        "gsm8k_test": lambda: mk("gsm8k-test", 1319, "numeric"),
        "gsm8k_train": lambda: mk("gsm8k-train", 7473, "numeric"),
        "math500": lambda: mk("math500", 500, "math"),
        "math_train": lambda: mk("mathtrain", 7500, "math"),
    }


def test_prepare_data_sizes_and_disjointness(tmp_path):
    prepare_data(load_config(CFG), fake_loaders(), tmp_path)
    ev = load_manifest(tmp_path / "manifest_eval.json")["ids"]
    dev = load_manifest(tmp_path / "manifest_dev.json")["ids"]
    gen2 = load_manifest(tmp_path / "manifest_gen2.json")["ids"]
    gate = load_manifest(tmp_path / "manifest_gate.json")["ids"]
    assert len(ev) == 1200 and len(dev) == 400
    assert sum(i.startswith("gsm8k-test") for i in ev) == 800
    assert sum(i.startswith("math500") for i in ev) == 400
    assert not set(ev) & set(dev)
    assert set(gen2) <= set(ev) and len(gen2) == 300
    assert set(gate) <= set(dev) and len(gate) == 100
    items = load_items(tmp_path / "items.jsonl")
    assert {it.id for it in items} >= set(ev) | set(dev)


def test_prepare_data_refuses_to_overwrite_manifests(tmp_path):
    prepare_data(load_config(CFG), fake_loaders(), tmp_path)
    with pytest.raises(FileExistsError):
        prepare_data(load_config(CFG), fake_loaders(), tmp_path)


def test_plan_command_lists_runs(capsys):
    main(["plan", "--config", str(CFG), "--generator", "qwen2.5-1.5b"])
    out = capsys.readouterr().out
    assert "phase3_main: 20 runs" in out  # 16 + 4 BF16 null controls
    assert "phase5_ch: 20 runs" in out
    assert "EUR" in out


def test_forecast_includes_every_paid_step():
    from vq.cli import DEFAULT_THROUGHPUT, plan_campaigns
    from vq.experiment import load_config

    plan = plan_campaigns(load_config(CFG), "G", DEFAULT_THROUGHPUT)
    for step in ("phase0_pilot", "phase1_gate", "prereq_dev_full_bf16", "lambda_selection", "phase2_generation",
                 "phase3_main", "phase4_gen2", "phase5_ch", "phase6_deployment", "pod_overhead"):
        assert step in plan, step
    names = [c.name for c in plan["prereq_dev_full_bf16"][1]]
    assert len(names) == 4  # one BF16 dev-full pass per verifier (calibration source)
    assert len(plan["phase6_deployment"][1]) >= 5  # NF4-bnb x3 ORMs + packed GPTQ x2 + exports


def test_plan_total_is_reported_against_ceiling(capsys):
    main(["plan", "--config", str(CFG), "--generator", "G"])
    out = capsys.readouterr().out
    assert "pod_overhead" in out and "TOTAL" in out and "submission ceiling" in out
