"""commodutil.standards.units: canonical unit vocabulary.

Owns:
- UNIT_MAP: alias -> canonical unit map for normalising free-form unit
  strings from vendor contract specs ("barrel", "Barrels", "BBL" -> "bbl").
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

from commodutil.standards.currency import (
    VALID_CURRENCY_TOKENS,
    normalize_currency_token,
    split_currency_unit,
)
from commodutil.standards.unit_registry import (
    ENCODING_REPAIRS as _ENCODING_REPAIRS,
)
from commodutil.standards.unit_registry import (
    PUBLIC_UNIT_MAP as _PUBLIC_UNIT_MAP,
)
from commodutil.standards.unit_registry import (
    UNIT_MAP as UNIT_MAP,
)


# ---- Alias -> canonical normalisation ----
#
# UNIT_MAP (the bbl/gal/mt/lb vendor-spec vocabulary, consumed by
# curvemetadata.ice_util.parse_unit) and _PUBLIC_UNIT_MAP (the wider public /
# metadata spelling map) are DERIVED from the single unit registry table in
# commodutil.standards.unit_registry. To add a unit spelling, add it to a
# UnitRow there — not here. Keys are matched case-insensitively at call time
# (callers lower-case their input).


# ---- Quantity-unit parsing ----


def canonical_quantity_unit(unit: object) -> Optional[str]:
    """Return the canonical physical quantity token for free-form unit text.

    This intentionally covers quoted physical units only: ``bbl``, ``gal``,
    and ``mt``. Currencies and rate periods are outside this vocabulary.
    """
    if unit is None:
        return None
    text = str(unit).strip().lower()
    if not text:
        return None
    return UNIT_MAP.get(text)


def canonical_unit_token(unit: object) -> Optional[str]:
    """Return the canonical public token for a bare market unit string.

    This is pure string vocabulary for metadata/public quote labels. Known
    aliases normalise to commodutil public tokens (for example ``BBL`` ->
    ``bbl`` and ``MWH`` -> ``MWh``). Unknown tokens are only stripped and then
    returned with their original spelling.
    """
    if unit is None:
        return None
    cleaned = str(unit).strip()
    if not cleaned:
        return None
    return _PUBLIC_UNIT_MAP.get(cleaned.lower(), cleaned)


def quantity_unit_from_price_unit(price_unit: object) -> Optional[str]:
    """Return the denominator quantity unit from a price-unit string.

    Examples:
      * ``USD/MT`` -> ``mt``
      * ``USc/GAL`` -> ``gal``
      * ``$/BBL`` -> ``bbl``

    Bare quantity units are accepted. Bare rate units such as ``bbl/day``
    return ``None`` because the denominator is a time period, not a physical
    quantity.
    """
    if price_unit is None:
        return None
    text = str(price_unit).strip()
    if not text:
        return None

    _currency, unit_text = split_currency_unit(text)
    if not _currency and "/" in text:
        head, _, tail = text.partition("/")
        if head.strip().lower() in {token.lower() for token in VALID_CURRENCY_TOKENS}:
            unit_text = tail.strip()
    if "/" in unit_text:
        return None
    return canonical_quantity_unit(unit_text)


def canonical_price_unit_token(price_unit: object) -> Optional[str]:
    """Return the canonical public token for a price-unit string.

    Known currency/scale tokens are normalised with
    ``commodutil.standards.currency.normalize_currency_token``; known bare
    units use ``canonical_unit_token``. Unknown currency or unit fragments are
    stripped and preserved rather than inferred.
    """
    if price_unit is None:
        return None
    cleaned = str(price_unit).strip()
    if not cleaned:
        return None
    if "/" not in cleaned:
        return canonical_unit_token(cleaned)

    currency_text, _, unit_text = cleaned.partition("/")
    currency_text = currency_text.strip()
    unit_text = unit_text.strip()
    currency_token = normalize_currency_token(currency_text) or currency_text
    unit_token = canonical_unit_token(unit_text) or unit_text
    return f"{currency_token}/{unit_token}"


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
    - Pound casing/plurals: ``LBS`` / ``lbs`` -> ``lb``.

    Other tokens pass through unchanged. Aliases like ``barrel``, ``tonne``,
    ``gallon`` are not handled here -- they are registered as pint aliases
    in commodutil.convfactors at module load and resolved by the pint
    registry itself.
    """
    if unit is None:
        return unit
    u = unit.strip()
    # Fix cubic meter notations and encoding issues. The repair table lives in
    # the unit registry (single source of unit-vocabulary data); applied here as
    # ordered substring replacements.
    for bad, good in _ENCODING_REPAIRS.items():
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
    if u.upper() == "LBS":
        u = "lb"
    return u


__all__ = [
    "UNIT_MAP",
    "canonical_price_unit_token",
    "canonical_quantity_unit",
    "canonical_unit_token",
    "quantity_unit_from_price_unit",
    "to_pint_token",
]
