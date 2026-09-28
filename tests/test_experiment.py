from pathlib import Path

import pytest

from vq.experiment import ScoreRun, load_config, phase_runs

CFG = Path(__file__).parents[1] / "configs" / "experiment.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CFG)


def test_phase3_has_22_scoring_campaigns_including_null_and_repeat_controls(cfg):
    runs = phase_runs(cfg, "phase3_main", generator="G")
    assert len(runs) == 22
    assert len({r.name for r in runs}) == 22
    reps = [r for r in runs if r.condition == "bf16-repeat"]
    assert {r.verifier for r in reps} == {"sky-4b", "sky-8b"} and all(r.batch_tokens is None for r in reps)
    nulls = [r for r in runs if r.condition == "bf16-null"]
    assert {r.verifier for r in nulls} == {"sky-1.7b", "sky-4b", "sky-8b", "prm-7b"}
    assert all(r.method == "none" and r.batch_tokens and r.batch_tokens != 16_384 for r in nulls)


def test_phase5_counts_match_plan(cfg):
    runs = phase_runs(cfg, "phase5_ch", generator="G")
    by_v = {}
    for r in runs:
        by_v[r.verifier] = by_v.get(r.verifier, 0) + 1
    assert by_v == {"sky-8b": 12, "sky-4b": 8}
    assert all(r.method == "gptq" and r.bits == 4 for r in runs)
    assert {r.recipe for r in runs} == {"random", "dpq", "ch", "ch_shuffled"}


def test_ch_runs_use_lambda_and_others_do_not(cfg):
    runs = phase_runs(cfg, "phase5_ch", generator="G")
    for r in runs:
        assert (r.lam is None) == (r.recipe not in ("ch", "ch_shuffled"))


def test_phase4_uses_secondary_generator_but_primary_calibration(cfg):
    runs = phase_runs(cfg, "phase4_gen2", generator="G")
    assert len(runs) == 6 and all(r.generator == "llama3.2-3b" for r in runs)
    assert all(r.calib_generator == "G" for r in runs)


def test_run_names_are_filesystem_safe_and_unique_across_phases(cfg):
    names = []
    for ph in ("phase3_main", "phase4_gen2", "phase5_ch", "dev_numeric_controls"):
        names += [r.name for r in phase_runs(cfg, ph, generator="G")]
    assert len(names) == len(set(names))
    assert all("/" not in n and " " not in n for n in names)


def test_bf16_run_has_no_quantization(cfg):
    r = [x for x in phase_runs(cfg, "phase3_main", "G") if x.condition == "bf16"][0]
    assert r.method == "none" and r.recipe is None


def test_phase1_signal_runs(cfg):
    runs = phase_runs(cfg, "phase1_signal", generator="G")
    assert {(r.verifier, r.condition) for r in runs} == {
        (v, c) for v in ("sky-4b", "sky-8b") for c in ("bf16", "bf16-null", "nf4", "gptq-w4", "gptq-w3")}
    assert all(r.recipe in (None, "generic") for r in runs)  # no dev-calibrated method in the early signal
