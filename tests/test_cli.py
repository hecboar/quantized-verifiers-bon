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


def test_extend_math_eval_is_disjoint_and_logged(tmp_path):
    """Preregistered extension (2026-09-28): +800 MATH test problems outside MATH500,
    added before any evaluation candidate exists. The frozen eval manifest is untouched."""
    from vq.cli import extend_math_eval
    from vq.data import Item, load_items, load_manifest

    cfg = load_config(CFG)
    prepare_data(cfg, fake_loaders(), tmp_path)
    frozen = (tmp_path / "manifest_eval.json").read_text()
    math500 = [Item(f"math500-{i}", f"math problem {i}", str(i), "math", "math500") for i in range(500)]
    # math test = the 500 MATH500 problems (same text, other ids) + 4500 others
    math_test = [Item(f"mathtest-{i}", f"math problem {i}", str(i), "math", "math-test") for i in range(5000)]
    extend_math_eval(cfg, {"math_test": lambda: math_test, "math500": lambda: math500}, tmp_path)

    assert (tmp_path / "manifest_eval.json").read_text() == frozen
    ext = load_manifest(tmp_path / "manifest_eval_mathext.json")["ids"]
    assert len(ext) == 800
    texts = {it.id: it.question for it in math_test}
    assert not {texts[i] for i in ext} & {it.question for it in math500}  # never a MATH500 problem
    full = load_manifest(tmp_path / "manifest_eval_full.json")["ids"]
    assert len(full) == 2000 and set(load_manifest(tmp_path / "manifest_eval.json")["ids"]) < set(full)
    items = {it.id for it in load_items(tmp_path / "items.jsonl")}
    assert set(ext) <= items
    with pytest.raises(FileExistsError):
        extend_math_eval(cfg, {"math_test": lambda: math_test, "math500": lambda: math500}, tmp_path)


def test_eval_generation_uses_the_configured_full_manifest():
    from vq.cli import eval_manifest_for

    cfg = load_config(CFG)
    assert eval_manifest_for(cfg, "eval") == "eval_full"
    assert eval_manifest_for(cfg, "dev") == "dev" and eval_manifest_for(cfg, "gen2") == "gen2"


def test_plan_counts_the_math_extension():
    from vq.cli import DEFAULT_THROUGHPUT, plan_campaigns

    cfg = load_config(CFG)
    gen = plan_campaigns(cfg, "G", DEFAULT_THROUGHPUT)["phase2_generation"][1][0]
    n_eval = sum(cfg["data"]["eval"].values()) + cfg["data"]["eval_extension"]["math_test"]
    expected = (n_eval + sum(cfg["data"]["dev"].values()) + cfg["data"]["gen2_subset"]) * 32 * DEFAULT_THROUGHPUT["avg_gen_tokens"]
    assert gen.tokens == expected
