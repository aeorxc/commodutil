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
    # n-butane rebased to EIA gross 4.353 MMBtu/bbl (~103,600 Btu/gal) ->
    # multiplier ~9.65 (was ~9.8 on the old 28.44 GJ/m^3 folklore value).
    out = convfactors.convert_price(1.0, "gal", "MMBtu", "butane")
    assert out == pytest.approx(9.65, rel=1e-2)


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


def test_convert_price_fx_eur_mwh_to_usd_mmbtu():
    # TTF: 35 EUR/MWh @ EURUSD=1.07 -> ~35*1.07/3.412 ~= 10.97 $/MMBtu
    out = convfactors.convert_price(35.0, "EUR/MWh", "USD/MMBtu", fx=1.07)
    assert out == pytest.approx(10.97, rel=1e-3)


def test_convert_price_fx_gbp_pence_per_therm_to_usd_mmbtu():
    # NBP: 80 GBp/therm @ GBPUSD=1.25 -> 80*1.25/100*10 = 10.00 $/MMBtu
    # GBp prefix auto-detected and divided by 100.
    out = convfactors.convert_price(80.0, "GBp/therm", "USD/MMBtu", fx=1.25)
    assert out == pytest.approx(10.00, rel=1e-6)


def test_convert_price_fx_series_aligned():
    p = pd.Series(
        [35.0, 36.5, 34.2],
        index=pd.date_range("2026-01-01", periods=3, freq="D"),
    )
    fx = pd.Series([1.07, 1.08, 1.06], index=p.index)
    out = convfactors.convert_price(p, "EUR/MWh", "USD/MMBtu", fx=fx)
    # Spot-check first element
    assert out.iloc[0] == pytest.approx(35.0 * 1.07 / 3.412142, rel=1e-3)


def test_convert_price_fx_passthrough_for_usd():
    # If source currency is USD (or absent), no FX needed — falls back to convert_price
    out = convfactors.convert_price(2.5, "USD/gal", "USD/bbl", fx=None)
    assert out == pytest.approx(105.0, rel=1e-6)


def test_convert_price_fx_missing_fx_raises():
    with pytest.raises(ValueError):
        convfactors.convert_price(35.0, "EUR/MWh", "USD/MMBtu", fx=None)


def test_convert_price_fx_nonusd_target_rejected():
    with pytest.raises(ValueError):
        convfactors.convert_price(10.0, "USD/MMBtu", "EUR/MMBtu", fx=1.0)


def test_convert_price_fx_series_future_index_extends_via_ffill():
    """Forward-curve case: value.index has dates beyond fx's last observation.

    With `ffill_policy='ffill'` (legacy permissive behaviour), a plain
    `fx.reindex(value.index).ffill()` would yield NaN for all post-fx-end
    dates because ffill operates only within the new index. The aligner
    unions, ffills across the union, then reindexes back so future-dated
    rows pick up the most-recent FX observation.

    Under the default 'strict' policy this would raise because the dates
    past fx's last observation exceed max_staleness; we opt in with 'ffill'.
    """
    fx_idx = pd.date_range("2026-01-01", periods=3, freq="D")
    fx = pd.Series([1.07, 1.08, 1.09], index=fx_idx)

    # Forward curve extends past fx's last date by a long way (well past 7d)
    value_idx = pd.date_range("2026-01-02", periods=30, freq="D")
    value = pd.Series([35.0] * 30, index=value_idx)

    out = convfactors.convert_price(
        value,
        "EUR/MWh",
        "USD/MMBtu",
        fx=fx,
        commodity="natural_gas",
        ffill_policy="ffill",
    )

    assert out.isna().sum() == 0, "no NaNs should remain after forward-fill alignment"
    # 2026-01-02 picks up FX=1.08, 2026-01-03 picks up 1.09;
    # later rows also use the latest FX = 1.09 (ffill past edge)
    assert out.iloc[0] == pytest.approx(35.0 * 1.08 / 3.412142, rel=1e-3)
    assert out.iloc[1] == pytest.approx(35.0 * 1.09 / 3.412142, rel=1e-3)
    assert out.iloc[-1] == pytest.approx(35.0 * 1.09 / 3.412142, rel=1e-3)


def test_convert_price_fx_series_pre_start_falls_back_to_latest():
    """Value index starts before fx index — under `ffill_policy='ffill'` the
    function falls back to the most-recent FX for those pre-start rows
    (this is the documented backtest-future-leakage risk; the strict policy
    refuses to do it)."""
    fx_idx = pd.date_range("2026-01-05", periods=3, freq="D")
    fx = pd.Series([1.10, 1.11, 1.12], index=fx_idx)

    value_idx = pd.date_range("2026-01-01", periods=3, freq="D")  # all before fx start
    value = pd.Series([35.0, 35.0, 35.0], index=value_idx)

    out = convfactors.convert_price(
        value,
        "EUR/MWh",
        "USD/MMBtu",
        fx=fx,
        commodity="natural_gas",
        ffill_policy="ffill",
    )

    assert out.isna().sum() == 0
    # All rows fall back to most-recent FX (1.12)
    expected = 35.0 * 1.12 / 3.412142
    for v in out:
        assert v == pytest.approx(expected, rel=1e-3)


# ---------------------------------------------------------------------------
# Bug-fix regression tests (added 2026-05). See codex_convfactors_review.md.
# ---------------------------------------------------------------------------


def test_bug1_pint_default_barrel_preserved():
    """Bug 1: redefining `barrel` to mean the oil barrel silently changed
    pint's default unit for non-oil callers. The module now defines
    `oil_barrel` (= 158.99 L, = `bbl`) and leaves pint's default `barrel`
    (= US dry ~ 119.24 L) untouched.
    """
    from commodutil.convfactors import ureg

    # Pint's default barrel is the US dry barrel ~ 119.24 L
    assert (1 * ureg("barrel")).to("liter").magnitude == pytest.approx(119.24, abs=0.1)
    # The oil barrel = 158.987 L = 42 US gallons
    assert (1 * ureg("oil_barrel")).to("liter").magnitude == pytest.approx(
        158.987294928, abs=1e-6
    )
    # bbl is aliased to the oil barrel, not pint's default barrel
    assert (1 * ureg("bbl")).to("liter").magnitude == pytest.approx(
        158.987294928, abs=1e-6
    )
    # And convert_price still works on bbl <-> oil_barrel
    assert convfactors.convert_price(1.0, "USD/bbl", "USD/oil_barrel") == pytest.approx(
        1.0, rel=1e-9
    )


def test_bug2_usc_to_usd_no_fx_required():
    """Bug 2: USc -> USD is a pure /100 scale within the same major currency,
    so no FX should be required.
    """
    out = convfactors.convert_price(0.50, "USc/gal", "USD/gal")
    assert out == pytest.approx(0.005, rel=1e-9)


def test_usc_per_pound_to_usd_per_pound_no_fx_required():
    out = convfactors.convert_price(71.26, "USc/LBS", "USD/lb")
    assert out == pytest.approx(0.7126, rel=1e-9)


def test_bug2_gbp_pence_to_gbp_no_fx_required():
    """Bug 2: GBp -> GBP is a pure /100 scale (no FX)."""
    out = convfactors.convert_price(50.0, "GBp/therm", "GBP/therm")
    assert out == pytest.approx(0.50, rel=1e-9)


def test_bug2_eur_cents_to_eur_no_fx_required():
    """Bug 2: EUc -> EUR is a pure /100 scale (no FX)."""
    out = convfactors.convert_price(50.0, "EUc/MWh", "EUR/MWh")
    assert out == pytest.approx(0.50, rel=1e-9)


def test_bug2_same_currency_unit_only():
    """Bug 2 sanity check: USc/gal -> USc/bbl is no currency change at all,
    just a 42-gal-per-bbl scaling. Should work without fx today, and Bug 2
    must not regress it.
    """
    out = convfactors.convert_price(2.5, "USc/gal", "USC/bbl".replace("USC", "USc"))
    # 2.5 USc/gal -> 105 USc/bbl
    assert out == pytest.approx(105.0, rel=1e-6)


def test_bug3_strict_raises_on_insufficient_fx():
    """Bug 3: default `ffill_policy='strict'` must refuse to silently back-fill
    or extend stale FX. Pre-FX-start dates with no observation within
    max_staleness must raise."""
    fx_idx = pd.date_range("2026-01-05", periods=3, freq="D")
    fx = pd.Series([1.10, 1.11, 1.12], index=fx_idx)

    # value dates are entirely before fx.index.min() — strict must raise.
    value_idx = pd.date_range("2026-01-01", periods=3, freq="D")
    value = pd.Series([35.0, 35.0, 35.0], index=value_idx)

    with pytest.raises(ValueError, match="FX missing or stale"):
        convfactors.convert_price(
            value,
            "EUR/MWh",
            "USD/MMBtu",
            fx=fx,
            commodity="natural_gas",
            # ffill_policy defaults to 'strict'
        )


def test_bug3_ffill_policy_works_and_warns(caplog):
    """Bug 3: opt-in `ffill_policy='ffill'` succeeds but emits a logging
    warning about future-leakage risk."""
    import logging

    fx_idx = pd.date_range("2026-01-05", periods=3, freq="D")
    fx = pd.Series([1.10, 1.11, 1.12], index=fx_idx)

    value_idx = pd.date_range("2026-01-01", periods=3, freq="D")
    value = pd.Series([35.0, 35.0, 35.0], index=value_idx)

    with caplog.at_level(logging.WARNING, logger="commodutil.convfactors"):
        out = convfactors.convert_price(
            value,
            "EUR/MWh",
            "USD/MMBtu",
            fx=fx,
            commodity="natural_gas",
            ffill_policy="ffill",
        )

    assert out.isna().sum() == 0
    assert any("ffill_policy='ffill'" in rec.message for rec in caplog.records), (
        "expected a future-leakage warning when ffill_policy='ffill'"
    )


def test_bug3_strict_within_staleness_works():
    """Bug 3: strict policy DOES allow forward-fill up to max_staleness.
    A 2-day gap is well within the 7-day default and should succeed."""
    fx_idx = pd.date_range("2026-01-01", periods=3, freq="D")
    fx = pd.Series([1.07, 1.08, 1.09], index=fx_idx)
    # Value extends 2 days past fx end — within 7-day staleness budget.
    value_idx = pd.date_range("2026-01-01", periods=5, freq="D")
    value = pd.Series([35.0] * 5, index=value_idx)

    out = convfactors.convert_price(
        value, "EUR/MWh", "USD/MMBtu", fx=fx, commodity="natural_gas"
    )
    assert out.isna().sum() == 0
    # Last row uses fx=1.09 (forward-filled within staleness)
    assert out.iloc[-1] == pytest.approx(35.0 * 1.09 / 3.412142, rel=1e-3)


def test_bug3_unknown_ffill_policy_raises():
    fx_idx = pd.date_range("2026-01-01", periods=3, freq="D")
    fx = pd.Series([1.07, 1.08, 1.09], index=fx_idx)
    value = pd.Series([35.0] * 3, index=fx_idx)
    with pytest.raises(ValueError, match="Unknown ffill_policy"):
        convfactors.convert_price(
            value,
            "EUR/MWh",
            "USD/MMBtu",
            fx=fx,
            commodity="natural_gas",
            ffill_policy="bogus",
        )


def test_bug4_split_currency_unit_does_not_treat_bbl_as_currency():
    """Bug 4: `bbl/day` is a bare rate unit, NOT a currency-qualified price.
    The parser must return ('', 'bbl/day'), not ('bbl', 'day')."""
    from commodutil.standards.currency import (
        split_currency_unit as _split_currency_unit,
    )

    assert _split_currency_unit("bbl/day") == ("", "bbl/day")
    assert _split_currency_unit("kt/month") == ("", "kt/month")
    # Valid currency tokens still split correctly
    assert _split_currency_unit("USD/bbl") == ("USD", "bbl")
    assert _split_currency_unit("EUR/MWh") == ("EUR", "MWh")
    assert _split_currency_unit("GBp/therm") == ("GBp", "therm")
    # No slash -> no currency
    assert _split_currency_unit("mt") == ("", "mt")


def test_bug5_natural_gas_mass_volume_raises():
    """Bug 5: natural_gas now has density=None (no sentinel 0.0).
    Mass<->volume conversion should raise a useful error referencing the
    missing density rather than yielding garbage."""
    with pytest.raises(ValueError, match="no density defined"):
        convfactors.convert(1.0, "kg", "m^3", commodity="natural_gas")


def test_bug5_commodity_accepts_density_none():
    """Bug 5: the Commodity dataclass accepts density_kg_m3=None and
    natural_gas in COMMODITIES uses None (not 0.0)."""
    from commodutil.convfactors import COMMODITIES, Commodity

    ng = COMMODITIES["natural_gas"]
    assert ng.density is None

    # Direct construction with density=None works.
    c = Commodity(name="test", density=None, energy_content=None)
    assert c.density is None


def test_crude_naphtha_energy_content_enabled():
    # crude/naphtha carry canonical energy_content; volume<->energy conversions
    # that previously raised ("No energy content defined") work.
    # crude stays on its BP world-crude gross basis (39.043 GJ/m^3); naphtha was
    # rebased to GROSS/HHV (EIA petrochemical naphtha 5.248 MMBtu/bbl =
    # 34.826266 GJ/m^3, was BP NCV 31.732). See convfactors.py basis-policy
    # block and conversion-architecture-plan.md decision 2b.
    crude_gj = convfactors.convert(1.0, "bbl", "GJ", "crude")
    assert crude_gj == pytest.approx(0.158987294928 * 39.043, rel=1e-6)
    naphtha_gj = convfactors.convert(1.0, "bbl", "GJ", "naphtha")
    assert naphtha_gj == pytest.approx(0.158987294928 * 34.826266, rel=1e-6)


# ---------------------------------------------------------------------------
# Coal: mass-basis energy_content (GJ/t) with density=None. Exercises the
# [energy]/[mass] branch of _commodity_context added for API2 thermal coal.
# ---------------------------------------------------------------------------


def test_coal_mass_basis_mt_to_energy():
    # API2 gross ~25.0 MMBtu/t (6,300 kcal/kg GAR -> 26.377 GJ/t).
    mmbtu = convfactors.convert(1.0, "mt", "MMBtu", "coal")
    assert mmbtu == pytest.approx(25.0, abs=0.05)
    assert convfactors.convert(1.0, "mt", "GJ", "coal") == pytest.approx(
        26.377, abs=0.01
    )
    # Round-trips back through the direct mass<->energy transformation.
    assert convfactors.convert(mmbtu, "MMBtu", "mt", "coal") == pytest.approx(
        1.0, rel=1e-9
    )
    # And the price form: 80 $/t / 25.0 MMBtu/t ~ 3.20 $/MMBtu.
    out = convfactors.convert_price(80.0, "USD/mt", "USD/MMBtu", "coal")
    assert out == pytest.approx(80.0 / 25.0, abs=0.02)


def test_coal_mt_to_bbl_raises_no_density():
    # density=None: mass<->volume must stay illegal even though energy works.
    with pytest.raises(ValueError, match="no density defined"):
        convfactors.convert(1.0, "mt", "bbl", "coal")
    with pytest.raises(ValueError, match="no density defined"):
        convfactors.convert_price(80.0, "USD/mt", "USD/bbl", "coal")


def test_volume_basis_commodity_unchanged_by_mass_basis_support():
    # Adding the [energy]/[mass] branch must not disturb volume-basis liquids:
    # diesel still converts volume<->energy and mass<->energy via density.
    assert convfactors.convert(1.0, "m^3", "GJ", "diesel") == pytest.approx(
        38.290312, rel=1e-6
    )
    assert convfactors.convert(1.0, "mt", "GJ", "diesel") == pytest.approx(
        45.353, abs=0.01
    )
