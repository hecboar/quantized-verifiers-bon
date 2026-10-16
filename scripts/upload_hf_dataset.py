"""Upload the candidate pools, verifier scores and result tables to the Hugging Face Hub.

usage: python scripts/upload_hf_dataset.py [--repo hecboar/quantized-verifiers-bon] [--public]

Reads HF_TOKEN from the environment or from .env. Uploads (no shard directories, no
backups): outputs/store/{candidates,raw_candidates,scores}/*.{parquet,json},
outputs/tables/**, outputs/memory*.json, data/items.jsonl, data/math_levels.json,
data/manifest_*.json, and the dataset card docs/hf_dataset_card.md as README.md.
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def token() -> str:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("HF_TOKEN not set")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="hecboar/quantized-verifiers-bon")
    ap.add_argument("--public", action="store_true", help="make the dataset public after upload")
    a = ap.parse_args()
    api = HfApi(token=token())
    api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=str(ROOT / "docs/hf_dataset_card.md"), path_in_repo="README.md",
                    repo_id=a.repo, repo_type="dataset", commit_message="Dataset card")
    for kind in ("candidates", "raw_candidates", "scores"):
        api.upload_folder(folder_path=str(ROOT / "outputs/store" / kind), path_in_repo=f"store/{kind}",
                          repo_id=a.repo, repo_type="dataset", allow_patterns=["*.parquet", "*.json"],
                          ignore_patterns=["*.shards/*", "*.bak"], commit_message=f"store/{kind}")
    api.upload_folder(folder_path=str(ROOT / "outputs/tables"), path_in_repo="tables", repo_id=a.repo,
                      repo_type="dataset", commit_message="result tables")
    for f in ("outputs/memory.json", "outputs/memory_revision.json", "data/items.jsonl", "data/math_levels.json"):
        api.upload_file(path_or_fileobj=str(ROOT / f), path_in_repo=f.split("/", 1)[1] if f.startswith("outputs/") else f,
                        repo_id=a.repo, repo_type="dataset", commit_message=f)
    api.upload_folder(folder_path=str(ROOT / "data"), path_in_repo="data", repo_id=a.repo, repo_type="dataset",
                      allow_patterns=["manifest_*.json"], commit_message="manifests")
    if a.public:
        api.update_repo_settings(a.repo, repo_type="dataset", private=False)
    info = api.dataset_info(a.repo)
    print(a.repo, "private" if info.private else "public", info.sha)


if __name__ == "__main__":
    main()
