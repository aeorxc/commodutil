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
