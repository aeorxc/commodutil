"""Tests for commodutil.standards.currency."""

from commodutil.standards.currency import (
    CURRENCY_MAP,
    FRACTIONAL_CURRENCY_DIVISORS,
    FRACTIONAL_TO_MAJOR,
    VALID_CURRENCY_TOKENS,
    fractional_to_major,
    is_fractional_currency,
    required_fx_pair,
    split_currency_unit,
    to_symbol,
)


def test_is_fractional_currency():
    assert is_fractional_currency("USc") is True
    assert is_fractional_currency("GBp") is True
    assert is_fractional_currency("EUc") is True
    assert is_fractional_currency("USD") is False
    assert is_fractional_currency("EUR") is False
    assert is_fractional_currency("") is False


def test_fractional_to_major():
    assert fractional_to_major("GBp") == "GBP"
    assert fractional_to_major("USc") == "USD"
    assert fractional_to_major("EUc") == "EUR"
    assert fractional_to_major("JPy") == "JPY"
    # CAc/AUc were in VALID_CURRENCY_TOKENS but missing from fractional rules
    # before 2026-05-14 — bug fixed in PR A.
    assert fractional_to_major("CAc") == "CAD"
    assert fractional_to_major("AUc") == "AUD"
    # Already major — returned unchanged (upper-cased)
    assert fractional_to_major("USD") == "USD"
    assert fractional_to_major("EUR") == "EUR"
    # Dollar shorthand
    assert fractional_to_major("$") == "USD"
    # Empty
    assert fractional_to_major("") == ""


def test_fractional_currency_cac_auc_correctly_routed():
    # Regression guard for the pre-2026-05-14 bug where CAc/AUc were valid
    # currency tokens but had no fractional rule, so required_fx_pair
    # returned the wrong pair name and is_fractional_currency lied.
    assert is_fractional_currency("CAc") is True
    assert is_fractional_currency("AUc") is True
    assert required_fx_pair("CAc", "USD") == "CADUSD"
    assert required_fx_pair("AUc", "USD") == "AUDUSD"
    assert required_fx_pair("CAc", "CAD") is None  # same major, /100 scale only
    assert required_fx_pair("AUc", "AUD") is None


def test_split_currency_unit():
    assert split_currency_unit("EUR/MWh") == ("EUR", "MWh")
    assert split_currency_unit("USD/bbl") == ("USD", "bbl")
    assert split_currency_unit("GBp/therm") == ("GBp", "therm")
    # bbl is NOT a currency — must NOT split
    assert split_currency_unit("bbl/day") == ("", "bbl/day")
    assert split_currency_unit("kt/month") == ("", "kt/month")
    # No slash — unchanged
    assert split_currency_unit("mt") == ("", "mt")
    assert split_currency_unit("bbl") == ("", "bbl")


def test_required_fx_pair():
    assert required_fx_pair("EUR", "USD") == "EURUSD"
    assert required_fx_pair("GBP", "USD") == "GBPUSD"
    # Same currency — no FX needed
    assert required_fx_pair("USD", "USD") is None
    # Fractional resolves to same major — no FX
    assert required_fx_pair("USc", "USD") is None
    assert required_fx_pair("GBp", "GBP") is None
    # Empty source
    assert required_fx_pair("", "USD") is None
    # Fractional source against different major
    assert required_fx_pair("GBp", "USD") == "GBPUSD"
    assert required_fx_pair("EUc", "USD") == "EURUSD"


def test_to_symbol_canonical():
    assert to_symbol("USD") == "$"
    assert to_symbol("EUR") == "€"
    assert to_symbol("GBP") == "£"
    assert to_symbol("JPY") == "¥"
    assert to_symbol("USc") == "¢"
    assert to_symbol("GBp") == "p"


def test_to_symbol_legacy_aliases():
    # oilpricingcharts parity
    assert to_symbol("GBX") == "p"
    assert to_symbol("USC") == "¢"
    assert to_symbol("YEN") == "¥"
    assert to_symbol("MYR") == "RM"


def test_to_symbol_fallback_and_empty():
    # Unknown — returns input unchanged
    assert to_symbol("ZZZ") == "ZZZ"
    # Empty / None — returns empty string
    assert to_symbol("") == ""
    assert to_symbol(None) == ""


def test_vocab_constants_present():
    assert "USD" in VALID_CURRENCY_TOKENS
    assert "GBp" in VALID_CURRENCY_TOKENS
    assert "$" in VALID_CURRENCY_TOKENS
    assert FRACTIONAL_TO_MAJOR["GBp"] == "GBP"
    assert FRACTIONAL_CURRENCY_DIVISORS["USc"] == 100.0


# ---- CURRENCY_MAP (vendor-spec free-text -> ISO 4217 token) -------------


def test_currency_map_keys_lowercase():
    # Vendor parsers lowercase input before lookup; all keys must be lowercase.
    for key in CURRENCY_MAP:
        assert key == key.lower(), f"CURRENCY_MAP key {key!r} is not lowercase"


def test_currency_map_values_are_iso_codes():
    # All values are canonical 3-letter ISO 4217 codes.
    for code in CURRENCY_MAP.values():
        assert len(code) == 3 and code.isupper(), f"Bad currency token {code!r}"


def test_currency_map_known_entries():
    assert CURRENCY_MAP["us dollars and cents"] == "USD"
    assert CURRENCY_MAP["euros"] == "EUR"
    assert CURRENCY_MAP["pounds sterling"] == "GBP"
    assert CURRENCY_MAP["canadian dollars"] == "CAD"


# ---- Legacy import-path regression tests ----


def test_legacy_imports_from_convfactors():
    """Downstream callers must still be able to import the moved names from
    commodutil.convfactors (backwards-compat re-exports)."""
    from commodutil.convfactors import (
        FRACTIONAL_TO_MAJOR as cf_FRACTIONAL_TO_MAJOR,
        VALID_CURRENCY_TOKENS as cf_VALID_CURRENCY_TOKENS,
        fractional_to_major as cf_fractional_to_major,
        is_fractional_currency as cf_is_fractional_currency,
        split_currency_unit as cf_split_currency_unit,
    )

    assert cf_VALID_CURRENCY_TOKENS is VALID_CURRENCY_TOKENS
    assert cf_FRACTIONAL_TO_MAJOR is FRACTIONAL_TO_MAJOR
    assert cf_is_fractional_currency("USc") is True
    assert cf_fractional_to_major("GBp") == "GBP"
    assert cf_split_currency_unit("EUR/MWh") == ("EUR", "MWh")
