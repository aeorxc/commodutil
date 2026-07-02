"""Phase 3.3 tests: rate conversions lean on pint (separable design).

The hand-rolled scalar rate machinery (_parse_rate_unit / _rate_factor_scalar)
is deleted. A rate conversion is now the base-unit factor (the SAME
_convert_scalar path as a non-rate conversion) times a period ratio derived from
pint's calendar lengths (_period_rate_factor), so the calendar constants are no
longer duplicated. The Series path keeps calendar-aware days_in_month. Both are
fed by the single PriceUnit parser. Numerics are pinned by the golden fixture;
these tests assert the structural change and preserved edge behaviours.
"""

import inspect
import re

import pandas as pd
import pytest

from commodutil import convfactors


def test_hand_rolled_rate_machinery_deleted():
    converter = convfactors.converter
    assert not hasattr(converter, "_parse_rate_unit")
    assert not hasattr(converter, "_rate_factor_scalar")


def test_no_hardcoded_calendar_constants_in_convfactors():
    # The dedup guarantee: pint is the single source of month/year lengths, so
    # the old literals must not reappear anywhere in the module.
    src = inspect.getsource(convfactors)
    assert "30.4375" not in src
    assert not re.search(r"\b365\.25\b", src)


def test_rate_consistent_with_base_factor_times_pint_ratio():
    # Rate result == base-unit factor x pint period ratio, for the golden
    # commodities. Proves the separable design and that the ratio is pint's.
    month_len = (1 * convfactors.ureg("month")).to("day").magnitude
    for commodity in ("crude", "diesel", "gasoline", "fuel_oil"):
        base = convfactors.convfactor("kt", "bbl", commodity)
        rate = convfactors.convert(1.0, "kt/month", "bbl/day", commodity)
        assert rate == pytest.approx(base / month_len, rel=1e-12), commodity


def test_scalar_rate_uses_pint_calendar_constants():
    # pint's month=30.4375d, year=365.25d — the exact constants the deleted
    # code hand-rolled.
    assert convfactors.convert(1.0, "bbl/day", "bbl/month") == pytest.approx(
        30.4375, rel=1e-12
    )
    assert convfactors.convert(1.0, "bbl/day", "bbl/year") == pytest.approx(
        365.25, rel=1e-12
    )


def test_none_commodity_rate_raises_bare_unit_message():
    # Error messages are preserved character-for-character (bare units, not the
    # rate form) — these are the two pinned golden rate errors.
    with pytest.raises(ValueError, match=r"^Commodity required for kt to bbl$"):
        convfactors.convert(1.0, "kt/month", "bbl/day")
    with pytest.raises(ValueError, match=r"^Commodity required for bbl to mt$"):
        convfactors.convert(1.0, "bbl/day", "mt/year")


def test_series_month_aware_uses_calendar_days():
    # The Series path keeps calendar-aware days_in_month (2020 is a leap year,
    # so February has 29 days) — pint's fixed-length month cannot express this.
    dates = pd.date_range("2020-01-01", periods=3, freq="MS")  # Jan, Feb, Mar 2020
    s = pd.Series([100.0, 100.0, 100.0], index=dates)  # kt/month
    out = convfactors.convert(s, "kt/month", "bbl/day", "diesel")
    kt_to_bbl = convfactors.convfactor("kt", "bbl", "diesel")
    assert out.iloc[0] == pytest.approx(100 * kt_to_bbl / 31, abs=1)  # Jan 31d
    assert out.iloc[1] == pytest.approx(100 * kt_to_bbl / 29, abs=1)  # Feb 29d (leap)
    assert out.iloc[2] == pytest.approx(100 * kt_to_bbl / 31, abs=1)  # Mar 31d


def test_series_non_month_period_matches_pint_ratio():
    # A calendar-unaware index falls back to the pint period ratio (year->day).
    s = pd.Series([365250.0, 365250.0])  # RangeIndex, no days_in_month
    out = convfactors.convert(s, "bbl/year", "bbl/day")
    assert out.iloc[0] == pytest.approx(1000.0, abs=1e-6)
