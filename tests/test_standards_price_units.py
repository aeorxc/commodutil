"""Tests for commodutil.standards.price_units."""

from dataclasses import dataclass

from commodutil.standards import resolve_price_unit as facade_resolve_price_unit
from commodutil.standards.price_units import (
    resolve_price_unit,
    resolve_price_unit_from_attrs,
)


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
