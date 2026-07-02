"""Phase 3.4 tests: align_fx (extracted) and convert_price_result (additive).

convert_price itself is unchanged (its behaviour is pinned by the golden fixture
and the existing tests/test_price_conv.py). These cover the new public surface:
the extracted align_fx helper and the result-bearing convert_price_result.
"""

import logging

import pandas as pd
import pytest

from commodutil import convfactors
from commodutil.convfactors import ConversionResult


# ---- align_fx direct (mirrors the convert_price fx alignment tests) ----


def test_align_fx_strict_raises_on_pre_start():
    fx = pd.Series([1.10, 1.11, 1.12], index=pd.date_range("2026-01-05", periods=3))
    idx = pd.date_range("2026-01-01", periods=3)  # entirely before fx start
    with pytest.raises(ValueError, match="FX missing or stale"):
        convfactors.align_fx(fx, idx, policy="strict")


def test_align_fx_strict_within_staleness_ok():
    fx = pd.Series([1.07, 1.08, 1.09], index=pd.date_range("2026-01-01", periods=3))
    idx = pd.date_range("2026-01-01", periods=5)  # 2 days past fx end, within 7d
    out = convfactors.align_fx(fx, idx, policy="strict")
    assert out.isna().sum() == 0
    assert out.iloc[-1] == pytest.approx(1.09)  # forward-filled within staleness


def test_align_fx_ffill_warns_and_fills(caplog):
    fx = pd.Series([1.10, 1.11, 1.12], index=pd.date_range("2026-01-05", periods=3))
    idx = pd.date_range("2026-01-01", periods=3)  # pre-start
    with caplog.at_level(logging.WARNING, logger="commodutil.convfactors"):
        out = convfactors.align_fx(fx, idx, policy="ffill")
    assert out.isna().sum() == 0
    assert any("ffill_policy='ffill'" in rec.message for rec in caplog.records)


def test_align_fx_ffill_all_nan_raises():
    fx = pd.Series([float("nan")] * 3, index=pd.date_range("2026-01-01", periods=3))
    idx = pd.date_range("2026-01-01", periods=3)
    with pytest.raises(ValueError, match="entirely NaN"):
        convfactors.align_fx(fx, idx, policy="ffill")


def test_align_fx_unknown_policy_raises():
    fx = pd.Series([1.07], index=pd.date_range("2026-01-01", periods=1))
    with pytest.raises(ValueError, match="Unknown ffill_policy"):
        convfactors.align_fx(fx, fx.index, policy="bogus")


def test_align_fx_matches_convert_price_series_path():
    # convert_price delegates to align_fx; a manual compose reproduces it.
    idx = pd.date_range("2026-01-01", periods=3)
    p = pd.Series([35.0, 36.5, 34.2], index=idx)
    fx = pd.Series([1.07, 1.08, 1.06], index=idx)
    via_cp = convfactors.convert_price(p, "EUR/MWh", "USD/MMBtu", fx=fx)
    factor = convfactors.convfactor("MWh", "MMBtu")
    fx_aligned = convfactors.align_fx(fx, p.index)
    manual = (p / factor) * fx_aligned  # EUR major, divisor 1.0
    pd.testing.assert_series_equal(via_cp, manual)


# ---- convert_price_result ----

_SAMPLE = [
    (100.0, "mt", "bbl", "gasoline", None),
    (2.5, "gal", "bbl", None, None),
    (250.0, "USc/gal", "USD/gal", None, None),
    (100.0, "USD/mt", "USD/bbl", "diesel", None),
    (35.0, "EUR/MWh", "USD/MMBtu", None, 1.07),
    (80.0, "GBp/therm", "USD/MMBtu", None, 1.25),
]


def test_convert_price_result_value_matches_convert_price():
    for value, frm, to, commodity, fx in _SAMPLE:
        res = convfactors.convert_price_result(value, frm, to, commodity, fx=fx)
        expected = convfactors.convert_price(value, frm, to, commodity, fx=fx)
        assert isinstance(res, ConversionResult)
        assert res.value == pytest.approx(expected, rel=1e-12)
        assert res.applied is True
        assert res.note.startswith("convert_price:")


def test_convert_price_result_is_frozen():
    res = convfactors.convert_price_result(2.5, "gal", "bbl")
    with pytest.raises(Exception):
        res.value = 0.0  # frozen dataclass


def test_convert_price_result_identity_unchanged():
    res = convfactors.convert_price_result(100.0, "USD/bbl", "USD/bbl")
    assert res.value == 100.0
    assert res.applied is False
    assert res.note == "unchanged"


def test_convert_price_result_currency_leg_fallback_rin():
    # convert_price can't do convfactor('RIN','RIN'); the fallback applies the
    # pure currency scale and records the currency_leg route.
    res = convfactors.convert_price_result(250.0, "USc/RIN", "USD/RIN")
    assert res.value == pytest.approx(2.5, rel=1e-12)
    assert res.applied is True
    assert res.note.startswith("currency_leg:")


def test_convert_price_result_keep_raw_on_unconvertible():
    # Different, pint-unknown denominators: no route works -> keep-raw.
    res = convfactors.convert_price_result(100.0, "GBp/XYZ", "USD/ABC", on_error="keep")
    assert res.value == 100.0
    assert res.applied is False
    assert res.note.startswith("kept-raw:")


def test_convert_price_result_raise_is_default():
    with pytest.raises(Exception):
        convfactors.convert_price_result(100.0, "GBp/XYZ", "USD/ABC")


def test_convert_price_result_invalid_on_error():
    with pytest.raises(ValueError, match="on_error must be"):
        convfactors.convert_price_result(
            1.0, "USD/mt", "USD/bbl", "diesel", on_error="x"
        )
