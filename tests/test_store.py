import pandas as pd
import pytest

from vq.store import CampaignStore


def _df(n=4):
    return pd.DataFrame({"question_id": [f"q{i}" for i in range(n)], "cand": range(n), "score": [0.1 * i for i in range(n)]})


def test_write_then_exists_and_read(tmp_path):
    st = CampaignStore(tmp_path)
    assert not st.done("scores", "sky-4b__gptq-w4")
    st.write("scores", "sky-4b__gptq-w4", _df(), meta={"bits": 4})
    assert st.done("scores", "sky-4b__gptq-w4")
    pd.testing.assert_frame_equal(st.read("scores", "sky-4b__gptq-w4"), _df())
    assert st.meta("scores", "sky-4b__gptq-w4")["bits"] == 4


def test_refuses_overwrite_without_flag(tmp_path):
    st = CampaignStore(tmp_path)
    st.write("scores", "a", _df())
    with pytest.raises(FileExistsError):
        st.write("scores", "a", _df())
    st.write("scores", "a", _df(2), overwrite=True)
    assert len(st.read("scores", "a")) == 2


def test_partial_write_is_not_marked_done(tmp_path):
    st = CampaignStore(tmp_path)
    (tmp_path / "scores").mkdir()
    (tmp_path / "scores" / "b.parquet.tmp").write_bytes(b"garbage")
    assert not st.done("scores", "b")


def test_list_campaigns(tmp_path):
    st = CampaignStore(tmp_path)
    st.write("scores", "x", _df())
    st.write("scores", "y", _df())
    assert st.list("scores") == ["x", "y"]


def test_shards_accumulate_and_merge(tmp_path):
    st = CampaignStore(tmp_path)
    st.write_shard("scores", "c", 0, _df(2))
    st.write_shard("scores", "c", 1, _df(3))
    assert st.shard_ids("scores", "c") == [0, 1]
    st.merge_shards("scores", "c", meta={"k": 1})
    assert st.done("scores", "c") and len(st.read("scores", "c")) == 5
