"""commodutil.standards.commodities: canonical commodity vocabulary.

Owns:
- COMMODITY_KEYWORDS: ordered list of (display_name, group, [keywords])
  used by free-text inference. Ordering matters — "Natural Gasoline"
  must precede "Natural Gas" so the substring "natural gas" inside
  "natural gasoline" doesn't win.
- COMMODITY_CONVERSION_MAP: display_name -> commodutil.convfactors.COMMODITIES
  key, for downstream conversion routing.
- infer_commodity_and_group(text): free-text inference helper that walks
  COMMODITY_KEYWORDS in order and returns the first hit.
- normalize_commodity_for_conversion(commodity): map a free-form commodity
  string to a commodutil.convfactors conversion key.
- infer_commodity_from_exchange_symbol(symbol): last-resort short-substring
  fallback for raw exchange symbols (e.g. "CL_Mar25" -> "crude").

Previously lived in curvemetadata.common_maps / curvemetadata.taxonomy;
relocated to eliminate divergence risk between curvemetadata and
commodutil's commodity lists.
"""

from __future__ import annotations

import re
from typing import Optional


COMMODITY_KEYWORDS = [
    ("Brent", "Crude Oil", ["brent"]),
    ("WTI", "Crude Oil", ["wti"]),
    ("Crude Oil", "Crude Oil", ["crude oil", "crude"]),
    # NB: 'Natural Gasoline' MUST come before 'Natural Gas' — the substring
    # "natural gas" is contained in "natural gasoline" and would otherwise win.
    ("Natural Gasoline", "NGL", ["natural gasoline"]),
    (
        "Natural Gas",
        "Natural Gas",
        [
            "natural gas",
            "nat gas",
            "natgas",
            "jkm",
            "ttf",
            "nbp",
            "henry hub",
            "henry",
        ],
    ),
    ("Jet", "Refined Products", ["jet fuel", "jet"]),
    ("Diesel", "Refined Products", ["diesel", "ulsd", "gasoil", "heating oil"]),
    ("Gasoline", "Refined Products", ["gasoline", "rbob", "cbob", "mogas", "eurobob"]),
    ("Fuel Oil", "Refined Products", ["fuel oil", "hsfo", "lsfo", "marine fuel"]),
    ("Naphtha", "Refined Products", ["naphtha"]),
    ("Product Basket", "Refined Products", ["refined products", "product basket"]),
    ("VGO", "Refined Products", ["vgo"]),
    ("FAME", "Biofuel", ["fame"]),
    ("HVO", "Biofuel", ["hvo"]),
    ("Isobutane", "NGL", ["isobutane"]),
    ("Butane", "NGL", ["butane"]),
    ("Ethane", "NGL", ["ethane"]),
    ("Propane", "NGL", ["propane"]),
    ("NGL", "NGL", ["ngl"]),
    ("FFA", "Freight", ["freight", "ffa"]),
]

COMMODITY_CONVERSION_MAP = {
    "Crude Oil": "crude",
    "Brent": "crude",
    "WTI": "crude",
    "Natural Gas": "natgas",
    "Jet": "jet",
    "Diesel": "diesel",
    "Gasoline": "gasoline",
    "Fuel Oil": "fuel_oil",
    "Naphtha": "naphtha",
    "Product Basket": "product_basket",
    "VGO": "vgo",
    "FAME": "fame",
    "HVO": "hvo",
    # NGL species — switched from the generic 'lpg' blend to first-class species
    # in commodutil 2026-05 (each has its own density / HHV for $/gal<->$/MMBtu).
    # Keep the generic 'NGL' bucket on 'lpg' as a safe blend default.
    "Natural Gasoline": "natural_gasoline",
    "Isobutane": "isobutane",
    "Butane": "butane",
    "Propane": "propane",
    "NGL": "lpg",
    "Ethane": "ethane",
}


def _normalize_text(value: str) -> str:
    """Normalise text for keyword matching: lowercase, replace separators, collapse whitespace.

    Args:
        value: Input text string.

    Returns:
        Normalised lowercase text with single spaces and no `/` or `-` separators.
    """
    text = value.strip().lower()
    text = text.replace("/", " ").replace("-", " ")
    text = " ".join(text.split())
    return text


def infer_commodity_and_group(
    text: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """Infer commodity and group from free-form text using COMMODITY_KEYWORDS.

    Args:
        text: Product name or description text.

    Returns:
        Tuple of (commodity_name, group_name), or (None, None) if not found.

    Examples:
        >>> infer_commodity_and_group("ICE Brent Crude Futures")
        ('Brent', 'Crude Oil')
        >>> infer_commodity_and_group("Henry Hub Natural Gas")
        ('Natural Gas', 'Natural Gas')
        >>> infer_commodity_and_group("Natural Gasoline OPIS")
        ('Natural Gasoline', 'NGL')
        >>> infer_commodity_and_group("Unknown Widget") == (None, None)
        True
    """
    if not text:
        return None, None
    haystack = _normalize_text(str(text))
    for commodity_name, group_name, keywords in COMMODITY_KEYWORDS:
        for keyword in keywords:
            if keyword in haystack:
                return commodity_name, group_name
    return None, None


def normalize_commodity_for_conversion(commodity: Optional[str]) -> Optional[str]:
    """Normalise a free-form commodity string to a commodutil conversion key.

    Args:
        commodity: Commodity name (free-form text or canonical display name).

    Returns:
        Normalised key for use with commodutil.convfactors conversion
        functions (e.g. ``"crude"``, ``"natgas"``), or ``None`` for empty
        input. Falls back to a slugged form of the input if no
        COMMODITY_KEYWORDS hit.

    Examples:
        >>> normalize_commodity_for_conversion("Brent")
        'crude'
        >>> normalize_commodity_for_conversion("ICE Brent Crude")
        'crude'
        >>> normalize_commodity_for_conversion(None) is None
        True
    """
    if not commodity:
        return None

    text = _normalize_text(str(commodity))

    commodity_name, _ = infer_commodity_and_group(text)
    if commodity_name:
        mapped = COMMODITY_CONVERSION_MAP.get(commodity_name)
        if mapped:
            return mapped
        return _normalize_text(commodity_name).replace(" ", "_")

    return text.replace(" ", "_")


_EXCHANGE_SYMBOL_TOKEN_SPLIT = re.compile(r"[_:.\-\s/]+")

# Token sets for ``infer_commodity_from_exchange_symbol`` — matched via
# whole-token equality after splitting on ``_ : . - whitespace /``. Token sets
# are checked in order; first hit wins. All tokens are lower-case.
#
# We deliberately AVOID short ambiguous 2-char tokens (``cl`` / ``rb`` /
# ``ho`` / ``ng``) that used to live here as substrings — they caused
# false-positives across real feed-prefixed identifiers (e.g.
# ``Ice_ClearedGas:JKM`` → 'crude' via ``cl``; ``Singapore_Spot:Naphtha`` →
# 'natgas' via ``ng``; ``Hong_Kong:HKD`` → 'gasoil' via ``ho``).
#
# Natgas tokens include LNG/European/US hub acronyms (jkm/ttf/nbp/hh/henry)
# so feed-prefixed gas symbols classify correctly instead of falling through
# to ``cl``-style false matches.
_EXCHANGE_SYMBOL_TOKENS: list[tuple[str, frozenset[str]]] = [
    # Order matters only when token sets could overlap; they don't here.
    ("crude", frozenset({"wti", "brent", "brn"})),
    ("gasoline", frozenset({"rbob", "gasoline", "mogas"})),
    ("gasoil", frozenset({"gasoil", "diesel", "heating"})),
    (
        "natgas",
        frozenset(
            {
                "natural",
                "natgas",
                "jkm",
                "ttf",
                "nbp",
                "hh",
                "henry",
            }
        ),
    ),
]


def infer_commodity_from_exchange_symbol(symbol: Optional[str]) -> Optional[str]:
    """Infer commodity from a raw exchange symbol name (token-based match).

    Last-resort fallback when description-based ``infer_commodity_and_group``
    fails (no Description, or Description didn't match COMMODITY_KEYWORDS).
    The symbol is lower-cased and split on ``_ : . - whitespace /`` into
    tokens; commodity is inferred via WHOLE-TOKEN equality against
    ``_EXCHANGE_SYMBOL_TOKENS``.

    This replaces an older substring-based implementation that used short
    2-char tokens (``cl`` / ``rb`` / ``ho`` / ``ng``); those caused
    false-positives on feed-prefixed identifiers — e.g.
    ``"Ice_ClearedGas:JKM"`` matched ``cl`` and returned 'crude' instead of
    'natgas'; ``"Singapore_Spot:Naphtha"`` matched ``ng`` and returned
    'natgas' instead of 'naphtha' (no match). The token-equality rewrite
    eliminates the substring leak.

    Returns:
        Canonical commodity name ('crude' / 'gasoline' / 'gasoil' / 'natgas')
        or None if no match. Symbols that don't match any known token return
        ``None`` — callers should treat ``None`` as "skip / unknown" rather
        than guess.

    Examples:
        >>> infer_commodity_from_exchange_symbol("ICE_EuroFutures:BRN")
        'crude'
        >>> infer_commodity_from_exchange_symbol("RBOB_Apr25")
        'gasoline'
        >>> infer_commodity_from_exchange_symbol("Ice_ClearedGas:JKM")
        'natgas'
        >>> infer_commodity_from_exchange_symbol("Ice_ClearedGas:TTF")
        'natgas'
        >>> infer_commodity_from_exchange_symbol("Singapore_Spot:Naphtha") is None
        True
        >>> infer_commodity_from_exchange_symbol("Hong_Kong:HKD") is None
        True
        >>> infer_commodity_from_exchange_symbol("LME_Copper:Long") is None
        True
        >>> infer_commodity_from_exchange_symbol("XYZ_Spot") is None
        True
    """
    if not symbol:
        return None
    tokens = {t for t in _EXCHANGE_SYMBOL_TOKEN_SPLIT.split(str(symbol).lower()) if t}
    if not tokens:
        return None
    for commodity, token_set in _EXCHANGE_SYMBOL_TOKENS:
        if tokens & token_set:
            return commodity
    return None


__all__ = [
    "COMMODITY_KEYWORDS",
    "COMMODITY_CONVERSION_MAP",
    "infer_commodity_and_group",
    "normalize_commodity_for_conversion",
    "infer_commodity_from_exchange_symbol",
]
