"""Tests for commodutil.standards.units."""

import pytest

from commodutil.standards.units import (
    canonical_price_unit_token,
    canonical_quantity_unit,
    canonical_unit_token,
    is_canonical_price_unit,
    normalize_price_unit_strict,
    quantity_unit_from_price_unit,
)


def test_default_unit_for_commodity_removed():
    # default_unit_for_commodity / _DEFAULT_UNIT fabricated per-commodity
    # quote-unit metadata that the gold source_price_unit guarantee now answers
    # per-instrument. Removed with zero desk call sites; lock the removal so the
    # old name can't quietly reappear.
    import commodutil.standards as standards
    from commodutil.standards import units

    with pytest.raises(AttributeError):
        units.default_unit_for_commodity
    with pytest.raises(AttributeError):
        standards.default_unit_for_commodity
    with pytest.raises(ImportError):
        from commodutil.standards.units import default_unit_for_commodity  # noqa: F401


# ---- UNIT_MAP tests ----


def test_unit_map_canonical_values():
    from commodutil.standards.units import UNIT_MAP

    assert UNIT_MAP["barrel"] == "bbl"
    assert UNIT_MAP["barrels"] == "bbl"
    assert UNIT_MAP["bbl"] == "bbl"
    assert UNIT_MAP["bbls"] == "bbl"


def test_unit_map_gallon_variants():
    from commodutil.standards.units import UNIT_MAP

    assert UNIT_MAP["gallon"] == "gal"
    assert UNIT_MAP["gallons"] == "gal"
    assert UNIT_MAP["gal"] == "gal"


def test_unit_map_metric_ton_variants():
    from commodutil.standards.units import UNIT_MAP

    assert UNIT_MAP["metric ton"] == "mt"
    assert UNIT_MAP["metric tons"] == "mt"
    assert UNIT_MAP["metric tonne"] == "mt"
    assert UNIT_MAP["metric tonnes"] == "mt"
    assert UNIT_MAP["tonne"] == "mt"
    assert UNIT_MAP["tonnes"] == "mt"


def test_unit_map_pound_variants():
    from commodutil.standards.units import UNIT_MAP

    assert UNIT_MAP["pound"] == "lb"
    assert UNIT_MAP["pounds"] == "lb"
    assert UNIT_MAP["lb"] == "lb"
    assert UNIT_MAP["lbs"] == "lb"


def test_unit_map_canonical_set():
    """UNIT_MAP canonical set.

    The ICE unit-parse gap fix widened UNIT_MAP beyond bbl/gal/mt/lb to also
    carry the physical energy/power quote units (MMBtu/GJ/MWh/MW/m^3) and the
    structural non-physical denominators (RIN/FEU/day) that ICE bronze rows
    quote, so curvemetadata parse_unit stops dropping them.

    Phase 2.1 residual consolidation added 'kg' as a canonical (kg + kilogram
    variants promoted from PUBLIC-only into UNIT_MAP; 'm^3' was already present
    via the spelled-out cubic-metre spellings).

    Exchange gap fix (A1) added 'therm' as a canonical: the ICE 'THM' spelling
    ('thm'/'thms') maps to it. It introduces no NEW physical unit — 'therm' was
    already a pint alias — only a UNIT_MAP entry so parse_unit resolves 'THM'.
    'MTONS'/'CBM'/'KL' folded into the pre-existing 'mt'/'m^3' canonicals, so
    they add no new canonical value.
    """
    from commodutil.standards.units import UNIT_MAP

    assert set(UNIT_MAP.values()) == {
        "bbl",
        "gal",
        "mt",
        "lb",
        "kg",
        "GJ",
        "MMBtu",
        "MWh",
        "MW",
        "m^3",
        "therm",
        "RIN",
        "FEU",
        "day",
    }


def test_canonical_quantity_unit_normalizes_common_labels():
    assert canonical_quantity_unit("BBL") == "bbl"
    assert canonical_quantity_unit("barrels") == "bbl"
    assert canonical_quantity_unit("GAL") == "gal"
    assert canonical_quantity_unit("Metric Tonnes") == "mt"
    # ICE gap fix: 'mmbtu' is now a recognised UNIT_MAP quote unit.
    assert canonical_quantity_unit("mmbtu") == "MMBtu"
    assert canonical_quantity_unit(None) is None


def test_quantity_unit_from_price_unit_uses_quote_denominator():
    assert quantity_unit_from_price_unit("USD/MT") == "mt"
    assert quantity_unit_from_price_unit("usd/mt") == "mt"
    assert quantity_unit_from_price_unit("USc/GAL") == "gal"
    assert quantity_unit_from_price_unit("USC/GAL") == "gal"
    assert quantity_unit_from_price_unit("$/BBL") == "bbl"
    assert quantity_unit_from_price_unit("BBL") == "bbl"
    assert quantity_unit_from_price_unit("bbl/day") is None


# ---- Public token tests ----


def test_canonical_unit_token_normalizes_public_market_units():
    assert canonical_unit_token("barrel") == "bbl"
    assert canonical_unit_token("BBL") == "bbl"
    assert canonical_unit_token("metric tonnes") == "mt"
    assert canonical_unit_token("MT") == "mt"
    assert canonical_unit_token("mmbtu") == "MMBtu"
    assert canonical_unit_token("MWH") == "MWh"
    assert canonical_unit_token("m3") == "m^3"
    assert canonical_unit_token("M3") == "m^3"


def test_canonical_unit_token_preserves_unknown_spelling():
    assert canonical_unit_token("  WeirdUnit  ") == "WeirdUnit"
    assert canonical_unit_token("") is None
    assert canonical_unit_token(None) is None


def test_canonical_price_unit_token_normalizes_currency_and_unit():
    assert canonical_price_unit_token("USD/BBL") == "USD/bbl"
    assert canonical_price_unit_token("USC/GAL") == "USc/gal"
    assert canonical_price_unit_token("EUR/MWH") == "EUR/MWh"
    assert canonical_price_unit_token("usd/metric tonnes") == "USD/mt"
    assert canonical_price_unit_token("US cents / gallon") == "USc/gal"


def test_canonical_price_unit_token_preserves_unknown_fragments():
    assert canonical_price_unit_token("FOO/BBL") == "FOO/bbl"
    assert canonical_price_unit_token("USD/StrangeUnit") == "USD/StrangeUnit"
    assert canonical_price_unit_token("StrangeUnit") == "StrangeUnit"
    assert canonical_price_unit_token(None) is None


# ---- strict price-unit validator (normalize_price_unit_strict / is_canonical_price_unit) ----


@pytest.mark.parametrize(
    "value,expected",
    [
        ("USD/mt", "USD/mt"),  # already canonical -> unchanged
        ("USD/MT", "USD/mt"),  # denominator case folded
        ("usd/mt", "USD/mt"),  # currency case folded (twin of lenient sibling)
        ("USC/GAL", "USc/gal"),
        ("US cents / gallon", "USc/gal"),
        ("USc/RIN", "USc/RIN"),  # RIN is a registered quantity unit
        ("$/bbl", "$/bbl"),
    ],
)
def test_normalize_price_unit_strict_resolves_both_legs(value, expected):
    assert normalize_price_unit_strict(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "usd_ton",  # no currency leg, and 'usd_ton' is not a unit
        "USD/ton",  # 'ton' is not a registry quantity unit (must be 'mt')
        "FOO/bbl",  # unrecognised currency
        "USD/StrangeUnit",  # unrecognised denominator (lenient sibling preserves)
        "mt",  # bare unit: no currency leg
        "USD",  # currency only: no denominator
        "",
        "   ",
        None,
    ],
)
def test_normalize_price_unit_strict_refuses_unresolved(value):
    # Unlike the lenient canonical_price_unit_token, unknown fragments are
    # refused (None), not preserved.
    assert normalize_price_unit_strict(value) is None


def test_normalize_price_unit_strict_is_stricter_than_lenient_sibling():
    # Same input where the two deliberately diverge: lenient preserves, strict refuses.
    assert canonical_price_unit_token("USD/StrangeUnit") == "USD/StrangeUnit"
    assert normalize_price_unit_strict("USD/StrangeUnit") is None


def test_normalize_price_unit_strict_output_is_canonical():
    for raw in ("USD/MT", "usd/mt", "USC/GAL", "US cents / gallon"):
        normalized = normalize_price_unit_strict(raw)
        assert normalized is not None
        assert is_canonical_price_unit(normalized)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("USD/mt", True),
        ("USc/gal", True),
        ("USc/RIN", True),
        ("$/bbl", True),
        ("USD/MT", False),  # resolvable but not exact spelling
        ("usd/mt", False),
        ("usd_ton", False),
        ("USD/ton", False),
        ("mt", False),
        ("", False),
        (None, False),
    ],
)
def test_is_canonical_price_unit(value, expected):
    assert is_canonical_price_unit(value) is expected


def test_public_facades_export_unit_token_helpers():
    import commodutil
    import commodutil.standards as standards

    assert standards.canonical_unit_token("BBL") == "bbl"
    assert standards.canonical_price_unit_token("USC/GAL") == "USc/gal"
    assert commodutil.canonical_unit_token("MT") == "mt"
    assert commodutil.canonical_price_unit_token("EUR/MWH") == "EUR/MWh"


# ---- to_pint_token tests ----


def test_to_pint_token_cubic_meter_variants():
    from commodutil.standards.units import to_pint_token

    assert to_pint_token("m³") == "m^3"
    assert to_pint_token("m**3") == "m^3"
    assert to_pint_token("cubic_meter") == "m^3"
    assert to_pint_token("CUBIC_METER") == "m^3"
    assert to_pint_token("m3") == "m^3"
    assert to_pint_token("M3") == "m^3"


def test_to_pint_token_cubic_meter_rate_forms():
    from commodutil.standards.units import to_pint_token

    assert to_pint_token("m3/day") == "m^3/day"
    assert to_pint_token("M3/day") == "m^3/day"


def test_to_pint_token_energy_casing():
    from commodutil.standards.units import to_pint_token

    assert to_pint_token("BTU") == "Btu"
    assert to_pint_token("MMBTU") == "MMBtu"


def test_to_pint_token_pound_casing():
    from commodutil.standards.units import to_pint_token

    assert to_pint_token("LBS") == "lb"
    assert to_pint_token("lbs") == "lb"


def test_to_pint_token_whitespace_stripped():
    from commodutil.standards.units import to_pint_token

    assert to_pint_token("  bbl  ") == "bbl"
    assert to_pint_token("\tMMBTU\n") == "MMBtu"


def test_to_pint_token_none_passthrough():
    from commodutil.standards.units import to_pint_token

    assert to_pint_token(None) is None


def test_to_pint_token_passthrough_for_unknown_tokens():
    from commodutil.standards.units import to_pint_token

    # Tokens outside the rule table pass through unchanged.
    assert to_pint_token("bbl") == "bbl"
    assert to_pint_token("kg") == "kg"
    assert to_pint_token("GJ") == "GJ"
    # Aliases handled by pint registration (not here)
    assert to_pint_token("barrel") == "barrel"
    assert to_pint_token("tonne") == "tonne"


def test_to_pint_token_exposed_via_standards_facade():
    """to_pint_token must be importable from commodutil.standards."""
    from commodutil.standards import to_pint_token

    assert to_pint_token("m³") == "m^3"


def test_unit_map_parity_with_curvemetadata():
    """Regression guard: UNIT_MAP must match curvemetadata's previous local
    copy byte-for-byte until curvemetadata's re-export PR lands. Skipped if
    curvemetadata is unavailable."""
    try:
        from curvemetadata.common_maps import UNIT_MAP as cm_UNIT_MAP
    except ImportError:
        import pytest

        pytest.skip("curvemetadata not available in this environment")

    from commodutil.standards.units import UNIT_MAP

    # Parity: every key/value in the curvemetadata copy is in commodutil's
    assert cm_UNIT_MAP == UNIT_MAP
