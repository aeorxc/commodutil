"""Tests for commodutil.standards.units."""

from commodutil.standards.units import default_unit_for_commodity


def test_natgas_defaults_to_mmbtu():
    assert default_unit_for_commodity("natgas") == "mmbtu"
    assert default_unit_for_commodity("natural_gas") == "mmbtu"


def test_refined_products_default_to_gal():
    assert default_unit_for_commodity("gasoline") == "gal"
    assert default_unit_for_commodity("diesel") == "gal"
    assert default_unit_for_commodity("jet") == "gal"


def test_crude_and_unknown_default_to_bbl():
    assert default_unit_for_commodity("crude") == "bbl"
    # Not in map — fallback to bbl
    assert default_unit_for_commodity("butane") == "bbl"
    assert default_unit_for_commodity("fuel_oil") == "bbl"


def test_empty_and_none_default_to_bbl():
    assert default_unit_for_commodity("") == "bbl"
    assert default_unit_for_commodity(None) == "bbl"


def test_case_insensitive():
    assert default_unit_for_commodity("NATGAS") == "mmbtu"
    assert default_unit_for_commodity("Gasoline") == "gal"


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


def test_unit_map_canonical_set_only_three():
    """UNIT_MAP normalises to exactly the canonical units bbl / gal / mt."""
    from commodutil.standards.units import UNIT_MAP

    assert set(UNIT_MAP.values()) == {"bbl", "gal", "mt"}


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


def test_converter_normalize_unit_delegates_to_to_pint_token():
    """CommodityConverter._normalize_unit is a thin shim around the
    standalone normaliser. Regression guard against accidental drift."""
    from commodutil.convfactors import converter
    from commodutil.standards.units import to_pint_token

    for sample in [
        "m³",
        "m**3",
        "cubic_meter",
        "CUBIC_METER",
        "BTU",
        "MMBTU",
        "  bbl  ",
        "m3/day",
        None,
    ]:
        assert converter._normalize_unit(sample) == to_pint_token(sample)


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
