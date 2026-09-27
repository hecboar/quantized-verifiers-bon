"""Real Hub access for every dataset the pipeline loads (the first real pilot
failed on an un-namespaced dataset id that newer huggingface_hub rejects).
Run with: pytest -m network tests/test_network_data.py"""
import pytest

pytestmark = pytest.mark.network


def test_generic_calibration_texts_load_from_the_hub():
    from vq.cli import _generic_texts

    texts = _generic_texts()
    assert len(texts) > 1000 and all(len(t) > 200 for t in texts[:50])


def test_every_dataset_id_is_namespaced():
    import inspect

    import vq.cli
    import vq.data

    import re

    src = inspect.getsource(vq.data) + inspect.getsource(vq.cli)
    ids = re.findall(r'load_dataset\(\s*"([^"]+)"', src)
    assert ids, "no load_dataset calls found"
    assert all("/" in i for i in ids), f"un-namespaced dataset ids: {[i for i in ids if '/' not in i]}"
