"""Tests for the PriceUnit value type, the convert_currency_leg helper
(conversion-architecture-plan.md Phase 3.1), and the price-unit resolution
precedence that shares the module.

Covers:
  * PriceUnit.parse / __str__ round-trip property matrix;
  * parse edge cases (bare units, rate units NOT mistaken for currency,
    fractional currencies, the '$' token);
  * convert_currency_leg incl. the RIN case and error paths;
  * PriceUnit-object args to convert / convert_price producing byte-identical
    results to string args, parametrised over a sample of golden-fixture cases;
  * resolve_price_unit / resolve_price_unit_from_attrs precedence.
"""

import json
import os
from dataclasses import dataclass

import pytest

from commodutil import convfactors
from commodutil.standards import PriceUnit
from commodutil.standards import resolve_price_unit as facade_resolve_price_unit
from commodutil.standards.currency import VALID_CURRENCY_TOKENS, split_currency_unit
from commodutil.standards.price_unit import (
    resolve_price_unit,
    resolve_price_unit_from_attrs,
)


# ---------------------------------------------------------------------------
# Round-trip property: parse(str(pu)) == pu across a full matrix.
# ---------------------------------------------------------------------------

_UNITS = ["bbl", "mt", "kt", "gal", "m^3", "MMBtu", "MWh", "GJ", "therm", "L", "kg"]
_PERIODS = [None, "day", "month", "year"]


def test_round_trip_matrix():
    combos = 0
    for currency in list(VALID_CURRENCY_TOKENS) + [None]:
        for unit in _UNITS:
            for period in _PERIODS:
                pu = PriceUnit(currency=currency, quantity_unit=unit, period=period)
                assert PriceUnit.parse(str(pu)) == pu, f"round-trip failed for {pu!r}"
                combos += 1
    # Sanity: the matrix actually exercised a non-trivial number of combinations.
    assert combos == (len(VALID_CURRENCY_TOKENS) + 1) * len(_UNITS) * len(_PERIODS)


def test_construction_normalises_period_and_whitespace():
    # Plural / mixed-case periods and stray whitespace normalise so equality and
    # the round-trip guarantee hold regardless of how the instance was built.
    assert PriceUnit(currency="USD", quantity_unit=" bbl ", period="Days") == PriceUnit(
        currency="USD", quantity_unit="bbl", period="day"
    )
    assert PriceUnit(currency="", quantity_unit="bbl").currency is None


# ---------------------------------------------------------------------------
# parse edge cases.
# ---------------------------------------------------------------------------


def test_parse_bare_unit():
    pu = PriceUnit.parse("bbl")
    assert (pu.currency, pu.quantity_unit, pu.period) == (None, "bbl", None)
    assert not pu.is_currency_qualified
    assert pu.major_currency is None
    assert pu.fractional_divisor == 1.0
    assert str(pu) == "bbl"


def test_parse_rate_unit_not_treated_as_currency():
    # 'bbl/day' must be a bare rate unit, NOT currency='bbl'.
    for text, unit, period in [("bbl/day", "bbl", "day"), ("kt/month", "kt", "month")]:
        pu = PriceUnit.parse(text)
        assert pu.currency is None
        assert pu.quantity_unit == unit
        assert pu.period == period
        assert pu.quantity_leg() == text
        # matches split_currency_unit's ('', bare) behaviour
        assert (pu.currency or "", pu.quantity_leg()) == split_currency_unit(text)


def test_parse_currency_qualified():
    pu = PriceUnit.parse("USD/bbl")
    assert (pu.currency, pu.quantity_unit, pu.period) == ("USD", "bbl", None)
    assert pu.is_currency_qualified
    assert pu.major_currency == "USD"
    assert pu.quantity_leg() == "bbl"


def test_parse_fractional_currency():
    pu = PriceUnit.parse("GBp/therm")
    assert pu.currency == "GBp"
    assert pu.major_currency == "GBP"
    assert pu.fractional_divisor == 100.0
    usc = PriceUnit.parse("USc/gal")
    assert usc.major_currency == "USD"
    assert usc.fractional_divisor == 100.0


def test_parse_dollar_token():
    pu = PriceUnit.parse("$/bbl")
    assert pu.currency == "$"
    assert pu.major_currency == "USD"  # fractional_to_major resolves '$' -> USD
    assert str(pu) == "$/bbl"


def test_parse_currency_plus_rate():
    # Not currently emitted anywhere, but supported for completeness so the
    # (currency x unit x period) round-trip is total.
    pu = PriceUnit.parse("USD/bbl/day")
    assert (pu.currency, pu.quantity_unit, pu.period) == ("USD", "bbl", "day")
    assert pu.quantity_leg() == "bbl/day"
    assert str(pu) == "USD/bbl/day"


def test_parse_passthrough_and_rejects_empty():
    pu = PriceUnit.parse("USD/bbl")
    assert PriceUnit.parse(pu) is pu  # PriceUnit passes through unchanged
    for bad in ["", "   ", None]:
        with pytest.raises(ValueError):
            PriceUnit.parse(bad)


# ---------------------------------------------------------------------------
# convert_currency_leg — the RIN fix.
# ---------------------------------------------------------------------------


def test_currency_leg_rin_usc_to_usd():
    # The motivating case: a non-physical denominator ('RIN') that convert_price
    # cannot handle (convfactor('RIN','RIN') raises), but pure currency scaling
    # must still work.
    assert convfactors.convert_currency_leg(
        250.0, "USc/RIN", "USD/RIN"
    ) == pytest.approx(2.5, rel=1e-9)
    # And convert_price genuinely cannot do this today (documents the motivation).
    with pytest.raises(Exception):
        convfactors.convert_price(250.0, "USc/RIN", "USD/RIN")


def test_currency_leg_fractional_same_base():
    assert convfactors.convert_currency_leg(50.0, "GBp/therm", "GBP/therm") == (
        pytest.approx(0.5, rel=1e-9)
    )
    assert convfactors.convert_currency_leg(5.0, "USD/RIN", "USD/RIN") == 5.0


def test_currency_leg_no_source_currency_is_noop():
    assert convfactors.convert_currency_leg(7.0, "RIN", "USD/RIN") == 7.0


def test_currency_leg_cross_currency_requires_fx():
    assert convfactors.convert_currency_leg(
        10.0, "EUR/RIN", "USD/RIN", fx=1.07
    ) == pytest.approx(10.7, rel=1e-9)
    with pytest.raises(ValueError, match="FX rate required"):
        convfactors.convert_currency_leg(10.0, "EUR/RIN", "USD/RIN")


def test_currency_leg_mismatched_denominators_raise():
    with pytest.raises(ValueError, match="identical quantity denominators"):
        convfactors.convert_currency_leg(1.0, "USc/gal", "USD/bbl")


def test_currency_leg_nonusd_target_rejected():
    with pytest.raises(ValueError, match="only supports USD"):
        convfactors.convert_currency_leg(1.0, "USD/RIN", "EUR/RIN", fx=1.0)


def test_currency_leg_accepts_priceunit_args():
    out = convfactors.convert_currency_leg(
        100.0, PriceUnit.parse("USc/RIN"), PriceUnit.parse("USD/RIN")
    )
    assert out == pytest.approx(1.0, rel=1e-9)


# ---------------------------------------------------------------------------
# PriceUnit-object args == string args, over golden-fixture cases.
# ---------------------------------------------------------------------------

_FIXTURE = os.path.join(os.path.dirname(__file__), "golden", "golden_factors.json")


def _load_golden():
    with open(_FIXTURE, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _sample_quantity_value_keys(golden, limit=60):
    # Deterministic sample of quantity cases that produced a numeric value
    # (skip pinned-error pairs). Every Nth key across the sorted set.
    keys = sorted(k for k, v in golden["quantity_factors"].items() if "value" in v)
    step = max(1, len(keys) // limit)
    return keys[::step]


def test_priceunit_args_match_string_args_quantity():
    golden = _load_golden()
    sampled = _sample_quantity_value_keys(golden)
    assert sampled, "expected a non-empty sample of golden quantity cases"
    for key in sampled:
        commodity_key, from_unit, to_unit = key.split("::")
        commodity = None if commodity_key == "None" else commodity_key
        str_result = convfactors.convfactor(from_unit, to_unit, commodity)
        pu_result = convfactors.convfactor(
            PriceUnit.parse(from_unit), PriceUnit.parse(to_unit), commodity
        )
        assert pu_result == str_result, f"{key}: PU arg diverged from string arg"


def _parse_convert_price_key(key):
    # "value::from::to::commodity::fx=..."
    value_s, from_unit, to_unit, commodity_s, fx_s = key.split("::")
    commodity = None if commodity_s == "None" else commodity_s
    fx_raw = fx_s[len("fx=") :]
    fx = None if fx_raw == "None" else float(fx_raw)
    return float(value_s), from_unit, to_unit, commodity, fx


def test_priceunit_args_match_string_args_convert_price():
    golden = _load_golden()
    cases = [(k, v) for k, v in golden["convert_price"].items() if "value" in v]
    assert cases, "expected convert_price value cases in the fixture"
    for key, _ in cases:
        value, from_unit, to_unit, commodity, fx = _parse_convert_price_key(key)
        str_result = convfactors.convert_price(
            value, from_unit, to_unit, commodity, fx=fx
        )
        pu_result = convfactors.convert_price(
            value,
            PriceUnit.parse(from_unit),
            PriceUnit.parse(to_unit),
            commodity,
            fx=fx,
        )
        assert pu_result == str_result, f"{key}: PU arg diverged from string arg"


# ---------------------------------------------------------------------------
# resolve_price_unit / resolve_price_unit_from_attrs precedence.
# ---------------------------------------------------------------------------


def test_source_price_unit_wins_when_currency_qualified():
    assert (
        resolve_price_unit(
            source_price_unit="USD/gal",
            quote_unit="mt",
            currency="USD",
            contract_unit="bbl",
        )
        == "USD/gal"
    )


def test_invalid_source_price_unit_falls_back_to_quote_precedence():
    assert (
        resolve_price_unit(
            source_price_unit="gal",
            quote_unit="mt",
            currency="USD",
            contract_unit="bbl",
        )
        == "USD/mt"
    )


def test_quote_unit_with_valid_currency_prefix_wins():
    assert (
        resolve_price_unit(
            quote_unit="GBp/therm",
            currency="USD",
            contract_unit="bbl",
        )
        == "GBp/therm"
    )


def test_quote_unit_with_valid_currency_prefix_is_normalized():
    assert resolve_price_unit(quote_unit=" USD / bbl ") == "USD/bbl"


def test_quote_unit_with_invalid_currency_prefix_does_not_win():
    assert (
        resolve_price_unit(
            quote_unit="bbl/day",
            currency="USD",
            contract_unit="mt",
        )
        == "USD/mt"
    )


def test_invalid_currency_with_contract_unit_falls_back_to_contract_unit():
    assert resolve_price_unit(currency="XYZ", contract_unit="bbl") == "bbl"


def test_invalid_currency_with_bare_quote_unit_falls_back_to_quote_unit():
    assert resolve_price_unit(currency="XYZ", quote_unit="MMBTU") == "MMBTU"


def test_currency_and_bare_quote_unit_wins_over_contract_unit():
    assert (
        resolve_price_unit(
            quote_unit="mt",
            currency="EUR",
            contract_unit="bbl",
        )
        == "EUR/mt"
    )


def test_currency_and_bare_quote_unit_when_contract_unit_absent():
    assert resolve_price_unit(quote_unit="mt", currency="USD") == "USD/mt"


def test_quote_unit_wins_when_currency_absent():
    assert resolve_price_unit(quote_unit="mt", contract_unit="bbl") == "mt"


def test_currency_and_contract_unit_when_quote_unit_absent():
    assert resolve_price_unit(currency="USD", contract_unit="bbl") == "USD/bbl"


def test_quote_unit_fallback():
    assert resolve_price_unit(quote_unit="mt") == "mt"


def test_empty_metadata_returns_empty_string():
    assert resolve_price_unit() == ""
    assert resolve_price_unit(quote_unit=float("nan"), currency="nan") == ""


def test_resolve_price_unit_from_mapping_attrs():
    attrs = {
        "quote_unit": "mt",
        "currency": "USD",
        "contract_unit": "bbl",
    }
    assert resolve_price_unit_from_attrs(attrs) == "USD/mt"


def test_resolve_price_unit_from_object_attrs():
    @dataclass
    class Attrs:
        quote_unit: str
        currency: str
        contract_unit: str

    assert resolve_price_unit_from_attrs(Attrs("mt", "USD", "gal")) == "USD/mt"


def test_resolve_price_unit_exposed_via_standards_facade():
    assert facade_resolve_price_unit(quote_unit="USc/gal") == "USc/gal"
