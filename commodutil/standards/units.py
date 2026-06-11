"""commodutil.standards.units: canonical unit vocabulary.

Owns:
- UNIT_MAP: alias -> canonical unit map for normalising free-form unit
  strings from vendor contract specs ("barrel", "Barrels", "BBL" -> "bbl").
- default_unit_for_commodity(): returns the canonical quoted unit for a
  commodity (volume basis).
- to_pint_token(): cleans a unit string into a form pint can parse
  (encoding fixes, cubic-meter notation, BTU casing, whitespace).

Pure string-shaped — no pint imports, no pandas. The pint registry in
commodutil.convfactors handles unit algebra. Two sibling normalisation
layers live here:

1. Vocab normalisation (UNIT_MAP): free-form vendor text -> canonical
   token. Used pre-pint by vendor-spec parsers (curvemetadata).
2. Pint-token normalisation (to_pint_token): canonical token -> pint-
   parseable string. Used by commodutil.convfactors before feeding the
   pint registry.

They solve different problems and do NOT share a vocabulary; see the
merge plan in commodutil 3.11.0 notes for the trade-off discussion.
"""

from __future__ import annotations

from typing import Optional

from commodutil.standards.currency import split_currency_unit


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
    "mt": "mt",
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


def canonical_quantity_unit(unit: object) -> str | None:
    """Return the canonical quantity unit token for free-form unit text.

    This intentionally covers quoted physical units only: ``bbl``, ``gal``,
    and ``mt``. Rate periods and currencies are outside this vocabulary.
    """
    if unit is None:
        return None
    text = str(unit).strip().lower()
    if not text:
        return None
    return UNIT_MAP.get(text)


def quantity_unit_from_price_unit(price_unit: object) -> str | None:
    """Return the denominator quantity unit from a price-unit string.

    Examples:
        ``USD/MT`` -> ``mt``
        ``USc/GAL`` -> ``gal``
        ``$/BBL`` -> ``bbl``

    Bare quantity units are also accepted. Bare rate units such as
    ``bbl/day`` return ``None`` because the denominator is a time period, not
    a physical quantity.
    """
    if price_unit is None:
        return None
    text = str(price_unit).strip()
    if not text:
        return None
    _, unit_text = split_currency_unit(text)
    if "/" in unit_text:
        _, _, unit_text = text.partition("/")
    return canonical_quantity_unit(unit_text)


# ---- Pint-token normalisation ----


def to_pint_token(unit: Optional[str]) -> Optional[str]:
    """Normalize a unit string into a pint-parseable token.

    Fixes ASCII / encoding / casing pitfalls so the resulting string can
    be fed to ``pint.UnitRegistry`` without raising. Does NOT canonicalise
    to the bbl/gal/mt vocabulary (that's UNIT_MAP's job).

    Rules:
    - ``None`` -> ``None`` (passthrough).
    - Strip whitespace.
    - Cubic-meter notations: ``m��`` / ``m³`` / ``m**3`` / ``cubic_meter``
      / ``CUBIC_METER`` / ``m3`` (case-insensitive exact match) -> ``m^3``.
    - Rate forms: ``m3/...`` / ``M3/...`` -> ``m^3/...``.
    - Energy casing: ``BTU`` -> ``Btu``; ``MMBTU`` -> ``MMBtu``.

    Other tokens pass through unchanged. Aliases like ``barrel``, ``tonne``,
    ``gallon`` are not handled here -- they are registered as pint aliases
    in commodutil.convfactors at module load and resolved by the pint
    registry itself.
    """
    if unit is None:
        return unit
    u = unit.strip()
    # Fix cubic meter notations and encoding issues
    replacements = {
        "m��": "m^3",  # UTF-8 mojibake of m^3 written as "m??"
        "m³": "m^3",
        "m**3": "m^3",
        "cubic_meter": "m^3",
        "CUBIC_METER": "m^3",
    }
    for bad, good in replacements.items():
        u = u.replace(bad, good)

    # Additional robust normalizations using ASCII-only fallbacks
    if u.lower() == "m3":
        u = "m^3"
    # Handle rate-style variants like 'm3/day' or 'M3/day'
    u = u.replace("m3/", "m^3/").replace("M3/", "m^3/")
    # Energy unit common uppercase forms
    if u == "BTU":
        u = "Btu"
    if u == "MMBTU":
        u = "MMBtu"
    return u


__all__ = [
    "UNIT_MAP",
    "canonical_quantity_unit",
    "default_unit_for_commodity",
    "quantity_unit_from_price_unit",
    "to_pint_token",
]
