import pytest

from vq.budget import Campaign, forecast, weight_bytes


def test_bf16_bytes():
    assert weight_bytes(target_params=1000, other_params=200, scheme="bf16") == 2400


def test_gptq_w4_g128_bytes_include_metadata():
    # 4 bits/param + per-group fp16 scale (2 B) + packed 4-bit zero (0.5 B)
    b = weight_bytes(target_params=128 * 10, other_params=0, scheme="gptq", bits=4, group_size=128)
    assert b == pytest.approx(128 * 10 * 0.5 + 10 * 2.5)


def test_nf4_bytes_blockwise_absmax():
    b = weight_bytes(target_params=64 * 4, other_params=0, scheme="nf4")
    assert b == pytest.approx(64 * 4 * 0.5 + 4 * 4)


def test_unknown_scheme():
    with pytest.raises(ValueError):
        weight_bytes(1, 1, scheme="fp3")


def test_forecast_hours_and_euros():
    camps = [
        Campaign("score-a", tokens=3_600_000, kind="score"),
        Campaign("gen-a", tokens=1_800_000, kind="generate"),
    ]
    f = forecast(camps, tok_per_s={"score": 1000, "generate": 500}, eur_per_hour=0.69, overhead_frac=0.0)
    assert f["hours"] == pytest.approx(1.0 + 1.0)
    assert f["eur"] == pytest.approx(1.38)
    assert f["by_campaign"]["score-a"]["hours"] == pytest.approx(1.0)


def test_forecast_flags_ceiling():
    camps = [Campaign("x", tokens=100_000_000, kind="score")]  # ~320 h -> ~220 EUR
    f = forecast(camps, tok_per_s={"score": 100}, eur_per_hour=0.69, ceiling_eur=71.0)
    assert f["over_ceiling"] is True
    ok = forecast([Campaign("y", tokens=1_000_000, kind="score")], {"score": 1000}, ceiling_eur=71.0)
    assert ok["over_ceiling"] is False
