"""Difficulty level (1-5) of every MATH item in the manifests, from the source
datasets (exploratory analyses; not used by any registered test)."""
import json
import re
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vq.data import MATH_SUBJECTS  # noqa: E402


def main(out=Path("data/math_levels.json")):
    levels = {}
    for r in load_dataset("HuggingFaceH4/MATH-500", split="test"):
        levels[f"math500-{r['unique_id']}"] = int(r["level"])
    for subj in MATH_SUBJECTS:
        for split, prefix in (("test", "mathtest"), ("train", "mathtrain")):
            for i, r in enumerate(load_dataset("EleutherAI/hendrycks_math", subj, split=split)):
                m = re.search(r"(\d)", str(r["level"]))
                if m:
                    levels[f"{prefix}-{subj}-{i}"] = int(m.group(1))
    out.write_text(json.dumps(levels, indent=0, sort_keys=True))
    print(f"{len(levels)} levels -> {out}")


if __name__ == "__main__":
    main()
