"""commodutil.standards.units: default-unit helpers per commodity.

Lifted from pyoilprice/conversion.py's inline fallback. Pure vocab — no
pint, no pandas.
"""

from __future__ import annotations


_DEFAULT_UNIT = {
    "natgas": "mmbtu",
    "natural_gas": "mmbtu",
    "gasoline": "gal",
    "diesel": "gal",
    "jet": "gal",
}


def default_unit_for_commodity(commodity: str) -> str:
    """Return the canonical quoted unit for a commodity (volume basis).

    Falls back to 'bbl' for any commodity not in the explicit map (covers
    crude / fuel oil / naphtha / VGO / NGL species etc.).
    """
    if not commodity:
        return "bbl"
    return _DEFAULT_UNIT.get(str(commodity).lower(), "bbl")


__all__ = ["default_unit_for_commodity"]
