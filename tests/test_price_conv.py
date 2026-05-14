import pandas as pd
import pytest

from commodutil import convfactors


def test_convert_price_scalar_mt_to_bbl_gasoline():
    # 100 $/mt gasoline � 12 $/bbl
    out = convfactors.convert_price(100.0, "mt", "bbl", "gasoline")
    assert out == pytest.approx(12.0, rel=1e-2)


def test_convert_price_scalar_gal_to_bbl():
    # 2.5 $/gal ? $/bbl (�42)
    out = convfactors.convert_price(2.5, "gal", "bbl")
    assert out == pytest.approx(105.0, rel=1e-6)


def test_convert_price_series_diesel():
    s = pd.Series([700.0, 750.0, 800.0])  # $/mt
    out = convfactors.convert_price(s, "mt", "bbl", "diesel")
    # Diesel mt?bbl factor ~7.45.. -> prices scale down accordingly
    ratios = out / s
    # Use convfactor as expected ratio
    expected = 1.0 / convfactors.convfactor("mt", "bbl", "diesel")
    assert ratios.median() == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# NGL species ($/gal <-> $/MMBtu) — added 2026-05 to support the NGL/natgas
# pricing pages. The ethane gal->MMBtu factor of ~15.13 supersedes the
# historical hardcode of 15.15 in oilpricingcharts/genchart.py.
# ---------------------------------------------------------------------------


def test_convert_price_ethane_gal_to_mmbtu():
    # 1 gal ethane HHV ~ 0.0661 MMBtu => price multiplier ~15.13
    out = convfactors.convert_price(0.20, "gal", "MMBtu", "ethane")
    assert out == pytest.approx(3.025, rel=1e-3)


def test_convert_price_butane_gal_to_mmbtu():
    # n-butane HHV ~ 103,000 BTU/gal -> multiplier ~9.8
    out = convfactors.convert_price(1.0, "gal", "MMBtu", "butane")
    assert out == pytest.approx(9.8, rel=2e-2)


def test_convert_price_propane_gal_to_mt():
    # 1 mt propane ~ 521 gal at density 0.507 kg/L
    out = convfactors.convert_price(1.0, "gal", "mt", "propane")
    assert out == pytest.approx(521.05, rel=1e-3)


def test_convert_price_isobutane_distinct_from_butane():
    # Iso-butane and n-butane share the same alkane family but differ in density
    # and HHV/gal — verify the entries are physically distinct.
    n = convfactors.convert_price(1.0, "gal", "MMBtu", "butane")
    iso = convfactors.convert_price(1.0, "gal", "MMBtu", "isobutane")
    assert n != iso
    # iso has slightly higher gal->MMBtu multiplier (denser BTU per gal here)
    assert abs(n - iso) > 0.05


# ---------------------------------------------------------------------------
# FX-bearing energy-price conversions (added 2026-05).
# ---------------------------------------------------------------------------


def test_convert_price_with_fx_eur_mwh_to_usd_mmbtu():
    # TTF: 35 EUR/MWh @ EURUSD=1.07 -> ~35*1.07/3.412 ~= 10.97 $/MMBtu
    out = convfactors.convert_price_with_fx(35.0, "EUR/MWh", "USD/MMBtu", fx=1.07)
    assert out == pytest.approx(10.97, rel=1e-3)


def test_convert_price_with_fx_gbp_pence_per_therm_to_usd_mmbtu():
    # NBP: 80 GBp/therm @ GBPUSD=1.25 -> 80*1.25/100*10 = 10.00 $/MMBtu
    # GBp prefix auto-detected and divided by 100.
    out = convfactors.convert_price_with_fx(80.0, "GBp/therm", "USD/MMBtu", fx=1.25)
    assert out == pytest.approx(10.00, rel=1e-6)


def test_convert_price_with_fx_series_aligned():
    p = pd.Series(
        [35.0, 36.5, 34.2],
        index=pd.date_range("2026-01-01", periods=3, freq="D"),
    )
    fx = pd.Series([1.07, 1.08, 1.06], index=p.index)
    out = convfactors.convert_price_with_fx(p, "EUR/MWh", "USD/MMBtu", fx=fx)
    # Spot-check first element
    assert out.iloc[0] == pytest.approx(35.0 * 1.07 / 3.412142, rel=1e-3)


def test_convert_price_with_fx_passthrough_for_usd():
    # If source currency is USD (or absent), no FX needed — falls back to convert_price
    out = convfactors.convert_price_with_fx(2.5, "USD/gal", "USD/bbl", fx=None)
    assert out == pytest.approx(105.0, rel=1e-6)


def test_convert_price_with_fx_missing_fx_raises():
    with pytest.raises(ValueError):
        convfactors.convert_price_with_fx(35.0, "EUR/MWh", "USD/MMBtu", fx=None)


def test_convert_price_with_fx_nonusd_target_rejected():
    with pytest.raises(ValueError):
        convfactors.convert_price_with_fx(10.0, "USD/MMBtu", "EUR/MMBtu", fx=1.0)
