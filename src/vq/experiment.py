"""Turns configs/experiment.yaml into concrete scoring runs per phase."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import yaml

FROZEN_LAMBDA = "frozen"  # CH lambda picked on dev (phase 5 prelude), then fixed


@dataclass(frozen=True)
class ScoreRun:
    generator: str
    verifier: str
    condition: str
    method: str
    bits: int | None = None
    group_size: int | None = None
    recipe: str | None = None  # calibration recipe (None for uncalibrated)
    seed: int = 0
    lam: str | float | None = None
    calib_generator: str | None = None  # dev pools used for calibration (default: generator)
    batch_tokens: int | None = None  # scoring batch budget override (BF16 null control)

    @property
    def calib_gen(self) -> str:
        return self.calib_generator or self.generator

    @property
    def name(self) -> str:
        return f"{self.generator}__{self.verifier}__{self.condition}"


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _from_condition(cfg, generator, verifier, condition) -> ScoreRun:
    if condition == "ch-gptq-w4":
        return ScoreRun(generator, verifier, condition, "gptq", 4, 128, "ch", 0, FROZEN_LAMBDA)
    c = cfg["conditions"][condition]
    return ScoreRun(
        generator, verifier, condition, c["method"], c.get("bits"), c.get("group_size"),
        c.get("calib"), 0, None, batch_tokens=c.get("batch_tokens"),
    )


def phase_runs(cfg: dict, phase: str, generator: str | None) -> list[ScoreRun]:
    p = cfg["phases"][phase]
    if phase == "phase1_signal":
        return [_from_condition(cfg, generator, v, c) for v in p["grid"]["verifiers"] for c in p["grid"]["conditions"]]
    if phase == "phase3_main":
        runs = [_from_condition(cfg, generator, v, c) for v in p["grid"]["verifiers"] for c in p["grid"]["conditions"]]
        runs += [_from_condition(cfg, generator, e["verifier"], e["condition"]) for e in p["extra"]]
        return runs
    if phase == "phase4_gen2":
        # evaluated on the secondary generator, quantized/calibrated exactly as
        # for the primary one (transfer test, no re-calibration)
        gen = cfg["generation"]["secondary"]
        return [replace(_from_condition(cfg, gen, r["verifier"], r["condition"]), calib_generator=generator)
                for r in p["runs"]]
    if phase == "phase5_ch":
        runs = []
        for verifier, seeds in p["seeds"].items():
            for recipe in p["recipes"]:
                for s in seeds:
                    lam = FROZEN_LAMBDA if recipe in ("ch", "ch_shuffled") else None
                    runs.append(ScoreRun(generator, verifier, f"gptq-w{p['bits']}-{recipe}-s{s}",
                                         "gptq", p["bits"], 128, recipe, s, lam))
        return runs
    if phase == "phase7_h6":
        # H6 generator; generic calibration only (no dev pools of its own)
        gen = cfg["h6"]["generator"]
        return [_from_condition(cfg, gen, r["verifier"], r["condition"]) for r in p["runs"]]
    if phase in ("dev_numeric_controls", "phase3b_bits"):
        return [_from_condition(cfg, generator, r["verifier"], r["condition"]) for r in p["runs"]]
    raise ValueError(f"unknown phase {phase!r}")
