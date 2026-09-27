"""Idempotent Parquet store: one file per campaign, atomic rename, JSON
sidecar metadata. Shards let long campaigns resume after a pod interruption."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


class CampaignStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, kind: str, name: str) -> Path:
        return self.root / kind / f"{name}.parquet"

    def done(self, kind: str, name: str) -> bool:
        return self._path(kind, name).exists()

    def write(self, kind: str, name: str, df: pd.DataFrame, meta: dict | None = None, overwrite: bool = False) -> Path:
        path = self._path(kind, name)
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True to replace")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        if meta is not None:
            path.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))
        os.replace(tmp, path)
        return path

    def read(self, kind: str, name: str) -> pd.DataFrame:
        return pd.read_parquet(self._path(kind, name))

    def meta(self, kind: str, name: str) -> dict:
        p = self._path(kind, name).with_suffix(".json")
        return json.loads(p.read_text()) if p.exists() else {}

    def list(self, kind: str) -> list[str]:
        d = self.root / kind
        return sorted(p.stem for p in d.glob("*.parquet")) if d.exists() else []

    # ----------------------------------------------------------- shards
    def _shard_dir(self, kind: str, name: str) -> Path:
        return self.root / kind / f"{name}.shards"

    def write_shard(self, kind: str, name: str, shard: int, df: pd.DataFrame) -> None:
        d = self._shard_dir(kind, name)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f"{shard:05d}.parquet.tmp"
        df.to_parquet(tmp, index=False)
        os.replace(tmp, d / f"{shard:05d}.parquet")

    def shard_ids(self, kind: str, name: str) -> list[int]:
        d = self._shard_dir(kind, name)
        return sorted(int(p.stem) for p in d.glob("*.parquet")) if d.exists() else []

    def read_shard(self, kind: str, name: str, shard: int) -> pd.DataFrame:
        return pd.read_parquet(self._shard_dir(kind, name) / f"{shard:05d}.parquet")

    def check_shard_fingerprint(self, kind: str, name: str, fingerprint: str) -> None:
        """Pin a campaign's shards to one run fingerprint (first writer wins)."""
        d = self._shard_dir(kind, name)
        f = d / "_fingerprint.txt"
        if f.exists():
            if f.read_text().strip() != fingerprint:
                raise RuntimeError(f"{kind}/{name}: existing shards come from a different run fingerprint")
        else:
            d.mkdir(parents=True, exist_ok=True)
            f.write_text(fingerprint)

    def merge_shards(self, kind: str, name: str, meta: dict | None = None) -> Path:
        d = self._shard_dir(kind, name)
        parts = [pd.read_parquet(p) for p in sorted(d.glob("*.parquet"))]
        return self.write(kind, name, pd.concat(parts, ignore_index=True), meta=meta, overwrite=True)
