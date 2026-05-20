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


COMMODITY_KEYWORDS = [
    ("Brent", "Crude Oil", ["brent"]),
    ("WTI", "Crude Oil", ["wti"]),
    ("Crude Oil", "Crude Oil", ["crude oil", "crude"]),
    # NB: 'Natural Gasoline' MUST come before 'Natural Gas' — the substring
    # "natural gas" is contained in "natural gasoline" and would otherwise win.
    ("Natural Gasoline", "NGL", ["natural gasoline"]),
    ("Natural Gas", "Natural Gas", ["natural gas", "nat gas", "natgas"]),
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


def infer_commodity_from_exchange_symbol(symbol: Optional[str]) -> Optional[str]:
    """Infer commodity from a raw exchange symbol name (loose substring match).

    Last-resort fallback when description-based ``infer_commodity_and_group``
    fails (no Description, or Description didn't match COMMODITY_KEYWORDS).
    Mirrors legacy substring-fallback logic that lived inline in
    ``pyoilprice.conversion`` and then in ``curvemetadata.taxonomy``. Patterns
    are SHORT substrings (cl, rb, ho, ng) matched anywhere in the input —
    ``"close_value"`` will match ``cl`` and return ``"crude"``. This is
    acceptable on raw exchange-symbol identifiers (which are short and
    predictable) but **UNSAFE on free-text inputs** — use
    ``infer_commodity_and_group()`` for descriptions or product names.

    Returns:
        Canonical commodity name ('crude' / 'gasoline' / 'gasoil' / 'natgas')
        or None if no match.

    Examples (raw exchange symbols only):
        >>> infer_commodity_from_exchange_symbol("CL_Mar25")
        'crude'
        >>> infer_commodity_from_exchange_symbol("ICE_EuroFutures:BRN")
        'crude'
        >>> infer_commodity_from_exchange_symbol("RBOB_Apr25")
        'gasoline'
        >>> infer_commodity_from_exchange_symbol("HO_May25")
        'gasoil'
        >>> infer_commodity_from_exchange_symbol("NG_Jun25")
        'natgas'
        >>> infer_commodity_from_exchange_symbol("XYZ_Spot") is None
        True
    """
    if not symbol:
        return None
    s = str(symbol).lower()
    if any(x in s for x in ["cl", "wti", "brent", "brn"]):
        return "crude"
    if any(x in s for x in ["rb", "gasoline", "mogas"]):
        return "gasoline"
    if any(x in s for x in ["ho", "diesel", "gasoil"]):
        return "gasoil"
    if any(x in s for x in ["ng", "natural"]):
        return "natgas"
    return None


__all__ = [
    "COMMODITY_KEYWORDS",
    "COMMODITY_CONVERSION_MAP",
    "infer_commodity_and_group",
    "normalize_commodity_for_conversion",
    "infer_commodity_from_exchange_symbol",
]
