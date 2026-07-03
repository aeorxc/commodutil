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
    # Petrochemicals: keep polyethylene-specific entries before "Ethylene",
    # because the ordered substring matcher would otherwise match the
    # "ethylene" inside "polyethylene".
    ("HDPE", "Petrochemical", ["hdpe", "high density polyethylene"]),
    ("LLDPE", "Petrochemical", ["lldpe", "linear low density polyethylene"]),
    # Keep "Polypropylene" before "Propylene" for the same substring reason.
    # Bare "pp" is deliberately omitted: it is too short for naive substring
    # matching (e.g. copper, shipping). Rely on long forms instead.
    ("Polypropylene", "Petrochemical", ["polypropylene"]),
    ("Propylene", "Petrochemical", ["polymer grade propylene", "pgp", "propylene"]),
    ("Ethylene", "Petrochemical", ["ethylene"]),
    ("Isobutane", "NGL", ["isobutane"]),
    ("Butane", "NGL", ["butane"]),
    ("Ethane", "NGL", ["ethane"]),
    ("Propane", "NGL", ["propane"]),
    ("NGL", "NGL", ["ngl"]),
    ("FFA", "Freight", ["freight", "ffa"]),
    # Coal (API2 gross basis, added to COMMODITIES 2026-07). Group "Coal" is not
    # in commodity_groups.COMMODITY_GROUPS (closed set mirroring the legacy
    # MetadataDB2 CHECK constraint) — that tuple is deliberately NOT extended.
    # Gold's commodity_group column is free-form and already carries values
    # outside the closed set (e.g. 'Emissions'); no live consumer validates
    # groups (2026-07 census).
    ("Coal", "Coal", ["api2", "api4", "coal"]),
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
    # Coal (API2 gross basis) added to COMMODITIES 2026-07; keywords entry above
    # enables display-name inference ("API2 Rotterdam Coal Futures" -> Coal).
    "Coal": "coal",
}


# -----------------------------------------------------------------------------
# Commodity aliases: exact-key spelling/synonym table for
# commodutil.convfactors.CommodityConverter (which resolves
# aliases.get(name.lower(), name) before COMMODITIES lookup). This is the SOLE
# owner of the alias data; convfactors.ALIASES is a derived view of
# COMMODITY_ALIASES below (see convfactors.py).
#
# Deliberately grouped by canonical target (the natural axis for maintenance:
# "what are the accepted spellings of butane?") and flattened programmatically
# into COMMODITY_ALIASES. Every value MUST be a key in
# commodutil.convfactors.COMMODITIES; the freeze test asserts this (it cannot
# be checked here without importing convfactors, which would cycle).
#
# INTENTIONALLY INDEPENDENT of COMMODITY_KEYWORDS above. That table is a
# free-text *substring* inference vocabulary (brent/wti/jkm/ttf/rbob/api2...)
# used to classify descriptions; this is an *exact whole-string* alias lookup
# for convert(). They diverge on purpose:
#   * COMMODITY_KEYWORDS routes "Natural Gas" -> "natgas" (the LNG entry) for
#     inference, whereas the gaseous-pipeline aliases here (ng/naturalgas/
#     nat_gas) resolve to the distinct "natural_gas" COMMODITIES entry.
#   * COMMODITY_KEYWORDS carries ~37 substrings that are canonical names or
#     exchange tickers, not convert() aliases.
# Only 5 spellings overlap both tables; deriving this table from
# COMMODITY_KEYWORDS would over-generate and couple alias resolution to
# inference-vocabulary edits, so the two are kept separate by design.
_ALIAS_SPELLINGS: dict[str, list[str]] = {
    # Middle distillates
    "diesel": ["ulsd", "gasoil", "gas_oil", "gas oil", "go"],
    "jet": ["kerosene"],
    # Motor gasoline
    "gasoline": ["gas", "mogas"],
    # Fuel oil
    "fuel_oil": ["fueloil", "fuel oil", "fo"],
    # Crude
    "crude": ["crude oil", "crudeoil"],
    # NGL species. 'propane'/'butane'/'isobutane'/'natural_gasoline' became
    # first-class COMMODITIES entries (2026-05, $/gal<->$/MMBtu for MB OPIS NGL
    # futures); these cover common separator/name spellings of each species.
    "butane": ["n_butane", "n-butane", "normal_butane", "normal butane"],
    "isobutane": ["iso_butane", "iso-butane", "i_butane", "i-butane"],
    "natural_gasoline": ["natgaso", "nat_gasoline", "pentanes_plus"],
    # Natural gas: 'lng' -> liquefied ('natgas'); gaseous-pipeline spellings ->
    # the distinct 'natural_gas' entry (see divergence note above).
    "natgas": ["lng"],
    "natural_gas": ["ng", "naturalgas", "nat_gas"],
}


def _build_commodity_aliases() -> dict[str, str]:
    """Flatten ``_ALIAS_SPELLINGS`` (target -> [spellings]) into a
    spelling -> target lookup, guarding against a spelling being assigned to
    two different targets."""
    aliases: dict[str, str] = {}
    for canonical, spellings in _ALIAS_SPELLINGS.items():
        for spelling in spellings:
            if spelling in aliases and aliases[spelling] != canonical:
                raise ValueError(
                    f"alias {spelling!r} maps to both {aliases[spelling]!r} "
                    f"and {canonical!r}"
                )
            aliases[spelling] = canonical
    return aliases


COMMODITY_ALIASES = _build_commodity_aliases()


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


def _ngl_species_from_keywords() -> frozenset[str]:
    """Return the discrete NGL species currently declared in COMMODITY_KEYWORDS."""
    return frozenset(
        commodity_name
        for commodity_name, group_name, _ in COMMODITY_KEYWORDS
        if group_name == "NGL" and commodity_name != "NGL"
    )


def infer_ngl_species(text: object) -> Optional[str]:
    """Return a discrete NGL species from free text, or ``None``.

    This is consumed by the curvemetadata silver classifier when upgrading
    generic NGL exchange rows. The policy is upgrade-only-when-unambiguous:
    only an inferred NGL commodity that is one of the single-species keyword
    entries is returned; spreads, baskets, and non-NGL rows stay generic at the
    caller.
    """
    commodity_name, group_name = infer_commodity_and_group(str(text or ""))
    if group_name == "NGL" and commodity_name in _ngl_species_from_keywords():
        return commodity_name
    return None


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
    "COMMODITY_ALIASES",
    "infer_commodity_and_group",
    "infer_ngl_species",
    "normalize_commodity_for_conversion",
    "infer_commodity_from_exchange_symbol",
]
