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
