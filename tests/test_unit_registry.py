"""Tests for the single unit registry (Phase 3.2).

Proves:
  * derived UNIT_MAP equals the historical literal exactly (Item 1);
  * PUBLIC_UNIT_MAP is the historical map plus ONLY the additive curvemetadata
    spellings (Items 1 + 2);
  * the pint registry resolves a token matrix to identical magnitudes (Item 1);
  * to_pint_token behaviour is unchanged (Item 1);
  * curvemetadata's _EXTRA_UNIT_ALIASES all resolve via canonical_unit_token so
    the local table can be deleted (Item 2);
  * the one-row promise: every alias / pint spec traces to exactly one registry
    row and units.py holds no hand-maintained parallel dicts (Item 3).
"""

import re

import pytest

from commodutil.convfactors import ureg
from commodutil.standards import unit_registry as registry
from commodutil.standards import units


# Historical literals captured BEFORE the registry refactor (the contract).
_FROZEN_UNIT_MAP = {
    "barrel": "bbl",
    "barrels": "bbl",
    "bbl": "bbl",
    "bbls": "bbl",
    "gallon": "gal",
    "gallons": "gal",
    "gal": "gal",
    "metric ton": "mt",
    "metric tons": "mt",
    "metric tonne": "mt",
    "metric tonnes": "mt",
    "mt": "mt",
    "tonne": "mt",
    "tonnes": "mt",
    "pound": "lb",
    "pounds": "lb",
    "lb": "lb",
    "lbs": "lb",
}

_FROZEN_PUBLIC_UNIT_MAP = {
    **_FROZEN_UNIT_MAP,
    "btu": "Btu",
    "mmbtu": "MMBtu",
    "mm btu": "MMBtu",
    "mwh": "MWh",
    "mw h": "MWh",
    "m3": "m^3",
    "m^3": "m^3",
    "m**3": "m^3",
    "cubic meter": "m^3",
    "cubic meters": "m^3",
    "cubic metre": "m^3",
    "cubic metres": "m^3",
    "cubic_meter": "m^3",
    "cubic_meters": "m^3",
    "cubic_metre": "m^3",
    "cubic_metres": "m^3",
}

# The exact pint-definition sequence (order load-bearing) captured pre-refactor.
_FROZEN_PINT_DEFINITIONS = [
    "oil_barrel = 158.987294928 liter = bbl",
    "gallon = 3.785411784 liter = gal",
    "metric_ton = 1000 kilogram = mt",
    "kiloton = 1000 metric_ton = kt",
    "cubic_kilometer = 1e9 meter**3 = km3",
    "gigajoule = 1e9 joule = gj = GJ",
    "petajoule = 1e15 joule = pj = PJ",
    "billion_cubic_meter = 1e9 meter**3 = bcm = BCM",
    "billion_cubic_foot = 1e9 foot**3 = bcf = BCF",
    "tonne_of_oil_equivalent = 41.868e9 joule = toe = TOE",
    "million_tonne_of_oil_equivalent = 1e6 tonne_of_oil_equivalent = Mtoe",
    "barrel_of_oil_equivalent = 6.119e9 joule = boe = BOE",
    "million_barrel_of_oil_equivalent = 1e6 barrel_of_oil_equivalent = Mboe",
    "megatonne = 1e6 metric_ton = Mt",
    "million_british_thermal_unit = 1e6 Btu = MMBtu",
    "@alias million_british_thermal_unit = mmbtu = MMBTU = million_btu",
    "@alias therm = Therm = THERM",
    "@alias british_thermal_unit = btu",
    "mw = 1e6 watt",
    "mwh = 1e6 watt * hour",
    "MWH = 1e6 watt * hour",
]

# Additive curvemetadata spellings absorbed in Item 2 (Phase 2.1).
_CURVEMETADATA_EXTRA = {
    "mwh": "MWh",
    "megawatt hour": "MWh",
    "megawatt hours": "MWh",
    "mmbtu": "MMBtu",
    "mm btu": "MMBtu",
    "million british thermal units": "MMBtu",
    "gj": "GJ",
    "gigajoule": "GJ",
    "gigajoules": "GJ",
    "m3": "m^3",
    "m^3": "m^3",
    "m**3": "m^3",
    "cubic meter": "m^3",
    "cubic meters": "m^3",
    "cubic metre": "m^3",
    "cubic metres": "m^3",
    "kg": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "kilogramme": "kg",
    "kilogrammes": "kg",
}


# ---- Item 1: exact derivation ----


def test_unit_map_derived_exactly():
    assert units.UNIT_MAP == _FROZEN_UNIT_MAP
    assert registry.UNIT_MAP == _FROZEN_UNIT_MAP


def test_public_unit_map_is_frozen_plus_only_additive():
    pub = units._PUBLIC_UNIT_MAP
    # Every historical key maps identically (no behaviour change).
    for key, value in _FROZEN_PUBLIC_UNIT_MAP.items():
        assert pub[key] == value, f"{key}: {pub.get(key)} != {value}"
    # Any new key is an intended additive (curvemetadata) spelling.
    new_keys = {k: v for k, v in pub.items() if k not in _FROZEN_PUBLIC_UNIT_MAP}
    for key, value in new_keys.items():
        assert _CURVEMETADATA_EXTRA.get(key) == value, f"unexpected new key {key!r}"


def test_pint_definitions_exact_and_ordered():
    # Byte-identical strings AND order (order is load-bearing for pint).
    assert list(registry.PINT_DEFINITIONS) == _FROZEN_PINT_DEFINITIONS


def test_pint_registry_resolves_tokens_to_expected_magnitudes():
    # token -> (target unit, expected magnitude) — pins the derived defines.
    cases = {
        "bbl": ("liter", 158.987294928),
        "gal": ("liter", 3.785411784),
        "mt": ("kg", 1000.0),
        "kt": ("kg", 1e6),
        "Mt": ("kg", 1e9),
        "km3": ("meter**3", 1e9),
        "bcm": ("meter**3", 1e9),
        "GJ": ("joule", 1e9),
        "PJ": ("joule", 1e15),
        "MMBtu": ("Btu", 1e6),
        "toe": ("joule", 41.868e9),
        "boe": ("joule", 6.119e9),
        "mw": ("watt", 1e6),
        "mwh": ("GJ", 3.6),
        "MWH": ("GJ", 3.6),
    }
    for token, (target, expected) in cases.items():
        assert (1 * ureg(token)).to(target).magnitude == pytest.approx(
            expected, rel=1e-9
        ), token
    # Case aliases resolve to the same magnitude as their canonical spelling.
    for alias, canonical in [
        ("mmbtu", "MMBtu"),
        ("MMBTU", "MMBtu"),
        ("btu", "Btu"),
        ("THERM", "therm"),
    ]:
        assert (1 * ureg(alias)).to_base_units().magnitude == pytest.approx(
            (1 * ureg(canonical)).to_base_units().magnitude, rel=1e-12
        ), alias


def test_to_pint_token_behaviour_unchanged():
    expected = {
        "BTU": "Btu",
        "MMBTU": "MMBtu",
        "LBS": "lb",
        "lbs": "lb",
        "m3": "m^3",
        "m^3": "m^3",
        "m**3": "m^3",
        "cubic_meter": "m^3",
        "CUBIC_METER": "m^3",
        "m3/day": "m^3/day",
        "M3/day": "m^3/day",
        "m³": "m^3",
        "bbl": "bbl",
        "MMBtu": "MMBtu",
        "MWh": "MWh",
        "GJ": "GJ",
        "kg": "kg",
        "therm": "therm",
        "  bbl  ": "bbl",
    }
    for token, want in expected.items():
        assert units.to_pint_token(token) == want, token


# ---- Item 2: curvemetadata shadow vocab absorbed ----


def test_curvemetadata_extra_aliases_resolve_via_canonical_unit_token():
    # Once these all resolve here, curvemetadata_helpers.units._EXTRA_UNIT_ALIASES
    # can be deleted and normalize_unit_token can defer to canonical_unit_token.
    for spelling, canonical in _CURVEMETADATA_EXTRA.items():
        assert units.canonical_unit_token(spelling) == canonical, spelling


def test_curvemetadata_additions_do_not_disturb_unit_map():
    # Item 2 is additive to the PUBLIC map only; the bbl/gal/mt/lb UNIT_MAP that
    # curvemetadata.parse_unit consumes is untouched.
    assert units.UNIT_MAP == _FROZEN_UNIT_MAP


# ---- Item 3: one-row promise + no parallel dicts ----


def test_every_spelling_traces_to_exactly_one_row():
    seen = {}
    for row in registry.UNIT_ROWS:
        for spelling in list(row.unit_map_spellings) + list(row.public_only_spellings):
            assert spelling not in seen, (
                f"spelling {spelling!r} appears in rows {seen.get(spelling)!r} "
                f"and {row.canonical!r}"
            )
            seen[spelling] = row.canonical
    # The derived maps contain exactly the rows' spellings — nothing extra.
    assert set(units.UNIT_MAP) | set(units._PUBLIC_UNIT_MAP) == set(seen)


def test_every_pint_spec_traces_to_exactly_one_row():
    flat = [spec for row in registry.UNIT_ROWS for spec in row.pint_specs]
    assert flat == list(registry.PINT_DEFINITIONS)
    assert len(flat) == len(set(flat)), "a pint spec is duplicated across rows"


def test_units_module_holds_no_parallel_alias_dicts():
    # Structural guard: units.py must derive the maps from the registry, not
    # redefine them. Same object identity + no dict literals in the source.
    assert units.UNIT_MAP is registry.UNIT_MAP
    assert units._PUBLIC_UNIT_MAP is registry.PUBLIC_UNIT_MAP
    import inspect

    src = inspect.getsource(units)
    assert not re.search(r"^\s*UNIT_MAP\s*=\s*\{", src, re.M)
    assert not re.search(r"^\s*_PUBLIC_UNIT_MAP\s*=\s*\{", src, re.M)
    assert "replacements = {" not in src  # to_pint_token no longer hand-rolls it


# ---- Consumer contract: pyoilprice-used helpers still behave ----


def test_public_helper_functions_behaviour():
    assert units.canonical_quantity_unit("BBL") == "bbl"
    assert units.canonical_quantity_unit("metric tonnes") == "mt"
    assert units.canonical_unit_token("MWH") == "MWh"
    assert units.canonical_unit_token("unknown_x") == "unknown_x"
    assert units.canonical_price_unit_token("usc/gal") == "USc/gal"
    assert units.quantity_unit_from_price_unit("USD/MT") == "mt"
    assert units.quantity_unit_from_price_unit("bbl/day") is None
    assert units.default_unit_for_commodity("natgas") == "mmbtu"
    assert units.default_unit_for_commodity("crude") == "bbl"
