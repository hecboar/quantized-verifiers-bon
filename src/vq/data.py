"""Datasets and hashed manifests (plan §4.1).

Evaluation: GSM8K test + MATH500 (a subset of the MATH *test* split).
Development/calibration: GSM8K train + MATH train, so there is no overlap
with evaluation problems by construction.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .grading import extract_boxed, gsm8k_gold


@dataclass
class Item:
    id: str
    question: str
    gold: str
    kind: str  # "numeric" | "math"
    source: str


def _hash_ids(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def make_manifest(items: list[Item], n: int, seed: int, name: str, exclude: set[str] | None = None) -> dict:
    pool = sorted(it.id for it in items if not exclude or it.id not in exclude)
    if len(pool) < n:
        raise ValueError(f"manifest {name}: need {n} items, only {len(pool)} available")
    rng = np.random.default_rng(seed)
    ids = sorted(rng.choice(pool, size=n, replace=False).tolist())
    return {"name": name, "seed": seed, "n": n, "ids": ids, "sha256": _hash_ids(ids)}


def save_manifest(manifest: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(manifest, indent=1))


def load_manifest(path: str | Path) -> dict:
    m = json.loads(Path(path).read_text())
    if _hash_ids(m["ids"]) != m["sha256"]:
        raise ValueError(f"manifest {path} was modified after creation (hash mismatch)")
    return m


def save_items(items: list[Item], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")


def load_items(path: str | Path) -> list[Item]:
    with open(path, encoding="utf-8") as f:
        return [Item(**json.loads(line)) for line in f]


# ---------------------------------------------------------------- loaders
def _token():
    return os.environ.get("HF_TOKEN")


def load_gsm8k(split: str) -> list[Item]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split, token=_token())
    return [Item(f"gsm8k-{split}-{i}", r["question"], gsm8k_gold(r["answer"]), "numeric", f"gsm8k-{split}")
            for i, r in enumerate(ds)]


def load_math500() -> list[Item]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test", token=_token())
    return [Item(f"math500-{r['unique_id']}", r["problem"], r["answer"], "math", "math500") for r in ds]


MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                 "number_theory", "prealgebra", "precalculus"]


def load_math_train() -> list[Item]:
    from datasets import load_dataset

    out = []
    for subj in MATH_SUBJECTS:
        ds = load_dataset("EleutherAI/hendrycks_math", subj, split="train", token=_token())
        for i, r in enumerate(ds):
            gold = extract_boxed(r["solution"])
            if gold:
                out.append(Item(f"mathtrain-{subj}-{i}", r["problem"], gold, "math", "math-train"))
    return out


def load_math_test() -> list[Item]:
    """MATH test split (all subjects). MATH500 is a subset of it: callers must
    exclude MATH500 problems (by text) when sampling extra evaluation items."""
    from datasets import load_dataset

    out = []
    for subj in MATH_SUBJECTS:
        ds = load_dataset("EleutherAI/hendrycks_math", subj, split="test", token=_token())
        for i, r in enumerate(ds):
            gold = extract_boxed(r["solution"])
            if gold:
                out.append(Item(f"mathtest-{subj}-{i}", r["problem"], gold, "math", "math-test"))
    return out


def math_test_levels() -> dict[str, int]:
    """Difficulty level (1-5) of every MATH test problem, keyed like load_math_test."""
    from datasets import load_dataset

    out = {}
    for subj in MATH_SUBJECTS:
        ds = load_dataset("EleutherAI/hendrycks_math", subj, split="test", token=_token())
        for i, r in enumerate(ds):
            lv = r["level"].removeprefix("Level ")
            if lv.isdigit():
                out[f"mathtest-{subj}-{i}"] = int(lv)
    return out


def norm_text(s: str) -> str:
    return " ".join(s.split())
