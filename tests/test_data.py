import json

import pytest

from vq.data import Item, load_manifest, make_manifest, save_manifest


def _items(prefix, n):
    return [Item(id=f"{prefix}-{i}", question=f"q{i}", gold=str(i), kind="numeric", source=prefix) for i in range(n)]


def test_manifest_deterministic_and_sized():
    a = make_manifest(_items("g", 100), n=20, seed=0, name="x")
    b = make_manifest(_items("g", 100), n=20, seed=0, name="x")
    assert a["ids"] == b["ids"] and len(a["ids"]) == 20 and a["sha256"] == b["sha256"]


def test_manifest_excludes_ids():
    excl = {f"g-{i}" for i in range(90)}
    m = make_manifest(_items("g", 100), n=10, seed=0, name="x", exclude=excl)
    assert not set(m["ids"]) & excl


def test_manifest_too_small_raises():
    with pytest.raises(ValueError):
        make_manifest(_items("g", 5), n=10, seed=0, name="x")


def test_roundtrip_and_tamper_detection(tmp_path):
    m = make_manifest(_items("g", 50), n=10, seed=0, name="eval")
    p = tmp_path / "m.json"
    save_manifest(m, p)
    assert load_manifest(p)["ids"] == m["ids"]
    d = json.loads(p.read_text())
    d["ids"][0] = "g-49" if d["ids"][0] != "g-49" else "g-48"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        load_manifest(p)
