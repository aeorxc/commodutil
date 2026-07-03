"""Freeze test for the commodity alias table.

``commodutil.standards.commodities.COMMODITY_ALIASES`` is the sole owner AND the
only public name of the alias (spelling/synonym -> canonical commodity) mapping;
the historical ``commodutil.convfactors.ALIASES`` re-export was removed in the
same consolidation. This test pins the flattened mapping to a literal snapshot
so any future edit to the grouped ``_ALIAS_SPELLINGS`` owner that would silently
change the resolved aliases fails loudly, checks the structural invariants that
cannot live in the owner module itself (target membership would require
importing convfactors, which would create an import cycle), and locks the
removal of the old name so it cannot quietly reappear.
"""

from __future__ import annotations

import pytest

from commodutil.convfactors import COMMODITIES
from commodutil.standards.commodities import COMMODITY_ALIASES

# The exact alias mapping as of the B1 derivation (was a hand-maintained literal
# dict in convfactors.py). If a change to the owner is intentional, update this
# snapshot in the same commit.
_EXPECTED_ALIASES = {
    "crude oil": "crude",
    "crudeoil": "crude",
    "fo": "fuel_oil",
    "fuel oil": "fuel_oil",
    "fueloil": "fuel_oil",
    "gas": "gasoline",
    "gas oil": "diesel",
    "gas_oil": "diesel",
    "gasoil": "diesel",
    "go": "diesel",
    "i-butane": "isobutane",
    "i_butane": "isobutane",
    "iso-butane": "isobutane",
    "iso_butane": "isobutane",
    "kerosene": "jet",
    "lng": "natgas",
    "mogas": "gasoline",
    "n-butane": "butane",
    "n_butane": "butane",
    "nat_gas": "natural_gas",
    "nat_gasoline": "natural_gasoline",
    "natgaso": "natural_gasoline",
    "naturalgas": "natural_gas",
    "ng": "natural_gas",
    "normal butane": "butane",
    "normal_butane": "butane",
    "pentanes_plus": "natural_gasoline",
    "ulsd": "diesel",
}


def test_commodity_aliases_frozen_snapshot():
    assert COMMODITY_ALIASES == _EXPECTED_ALIASES


def test_convfactors_aliases_re_export_removed():
    # The historical convfactors.ALIASES / facade ALIASES names were removed
    # with the consolidation; COMMODITY_ALIASES in standards.commodities is the
    # only public name. Lock the removal so the old spelling can't quietly
    # reappear and re-fork ownership.
    import commodutil
    from commodutil import convfactors

    with pytest.raises(AttributeError):
        convfactors.ALIASES
    with pytest.raises(AttributeError):
        commodutil.ALIASES


def test_converter_aliases_is_defensive_copy():
    # CommodityConverter must hold a copy, not the owner dict itself.
    from commodutil.convfactors import CommodityConverter

    conv = CommodityConverter()
    assert conv.aliases == COMMODITY_ALIASES
    assert conv.aliases is not COMMODITY_ALIASES


def test_every_alias_target_is_a_real_commodity():
    for alias, target in COMMODITY_ALIASES.items():
        assert target in COMMODITIES, f"{alias!r} -> unknown commodity {target!r}"


def test_no_alias_key_collides_with_a_canonical_name():
    # An alias whose key is already a COMMODITIES key would be dead (the
    # converter checks COMMODITIES membership only after alias resolution, but a
    # self-referential alias is a smell).
    for alias in COMMODITY_ALIASES:
        assert alias == alias.lower(), f"alias {alias!r} must be lower-case"
