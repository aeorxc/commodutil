"""commodutil.standards.units: canonical unit vocabulary.

Owns:
- UNIT_MAP: alias -> canonical unit map for normalising free-form unit
  strings from vendor contract specs ("barrel", "Barrels", "BBL" -> "bbl").
- default_unit_for_commodity(): returns the canonical quoted unit for a
  commodity (volume basis).

Pure vocab -- no pint, no pandas. The pint registry in
commodutil.convfactors handles unit algebra; this module handles the
string-normalisation layer that runs BEFORE algebra.
"""

from __future__ import annotations

from typing import Optional


# ---- Alias -> canonical normalisation ----

# Maps lowercase aliases (singular / plural / abbreviated forms) to the
# canonical unit token used by downstream code. Used by vendor-spec
# parsers (e.g. curvemetadata.ice_util.parse_unit). Keys are matched
# case-insensitively at call time -- callers should lowercase input.
UNIT_MAP = {
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
    "tonne": "mt",
    "tonnes": "mt",
}


# ---- Default unit per commodity ----

_DEFAULT_UNIT = {
    "natgas": "mmbtu",
    "natural_gas": "mmbtu",
    "gasoline": "gal",
    "diesel": "gal",
    "jet": "gal",
}


def default_unit_for_commodity(commodity: Optional[str]) -> str:
    """Return the canonical quoted unit for a commodity (volume basis).

    Falls back to 'bbl' for any commodity not in the explicit map (covers
    crude / fuel oil / naphtha / VGO / NGL species etc.).
    """
    if not commodity:
        return "bbl"
    return _DEFAULT_UNIT.get(str(commodity).lower(), "bbl")


__all__ = ["UNIT_MAP", "default_unit_for_commodity"]
