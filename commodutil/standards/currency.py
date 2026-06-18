"""commodutil.standards.currency: canonical currency tokens, fractional rules, and FX-pair routing.

Owns the immutable vocabulary used by commodutil.convfactors.convert_price and any
downstream consumer needing to validate / parse / route between currencies.
Pure stdlib — no pint, no pandas, no convfactors imports.
"""

from __future__ import annotations

from typing import Optional


_FRACTIONAL_CURRENCY_DIVISORS = {
    # Quoted-in-fractional-units of a major currency. Multiplying by the FX rate
    # for the major currency (e.g. GBP/USD) AND dividing by 100 lifts the
    # fractional-currency quote to its USD equivalent.
    "GBp": 100.0,  # pence in sterling, e.g. NBP at "70 GBp/therm"
    "USc": 100.0,  # US cents
    "EUc": 100.0,  # euro cents
    "JPy": 100.0,  # rare, but for symmetry
    "CAc": 100.0,  # Canadian cents
    "AUc": 100.0,  # Australian cents
}

# Mapping fractional currency -> its parent major currency. Used to short-circuit
# the FX requirement when the source and target only differ by a fractional
# prefix (e.g. USc -> USD is a pure /100 scale, no FX needed).
_FRACTIONAL_TO_MAJOR = {
    "USc": "USD",
    "GBp": "GBP",
    "EUc": "EUR",
    "JPy": "JPY",
    "CAc": "CAD",
    "AUc": "AUD",
}

# Known currency tokens for the price-unit parser. Anything not in this set
# is NOT treated as a currency — so e.g. `bbl/day` is correctly identified
# as a bare rate unit rather than parsed as currency=`bbl`, unit=`day`.
#
# PUBLIC API: exported as `VALID_CURRENCY_TOKENS` (alias below) so downstream
# callers (e.g. pyoilprice.conversion) can reuse the same set instead of
# duplicating it.
# TODO: if this list grows beyond ~30 tokens, switch to the `iso4217` package
# rather than hand-curating.
_VALID_CURRENCY_TOKENS = {
    # ISO 4217 majors relevant to commodity trading
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "CAD",
    "AUD",
    "CHF",
    "CNY",
    "CNH",
    "INR",
    "MXN",
    "BRL",
    "ZAR",
    "KRW",
    "SGD",
    "HKD",
    "NZD",
    "NOK",
    "SEK",
    "DKK",
    "PLN",
    "TRY",
    "RUB",
    "ILS",
    # Fractional / minor units
    "USc",
    "GBp",
    "EUc",
    "JPy",
    "CAc",
    "AUc",
    # Currency symbol shorthand also accepted as a currency token
    "$",
}

_CURRENCY_TOKEN_LOOKUP = {token.upper(): token for token in _VALID_CURRENCY_TOKENS}
_CURRENCY_TOKEN_ALIASES = {
    "USC": "USc",
    "US CENTS": "USc",
    "U S CENTS": "USc",
    "GBX": "GBp",
    "GBPENCE": "GBp",
    "GB PENCE": "GBp",
    "PENCE": "GBp",
    "RMB": "CNY",
}

# Public aliases — these are the import names downstream code should use.
VALID_CURRENCY_TOKENS = _VALID_CURRENCY_TOKENS
FRACTIONAL_TO_MAJOR = _FRACTIONAL_TO_MAJOR
FRACTIONAL_CURRENCY_DIVISORS = _FRACTIONAL_CURRENCY_DIVISORS


def _normalized_currency_lookup_key(token: str) -> str:
    return " ".join(str(token).replace(".", " ").strip().upper().split())


def normalize_currency_token(token: Optional[str]) -> Optional[str]:
    """Return the canonical commodutil currency token for common aliases.

    The canonical tokens remain ``VALID_CURRENCY_TOKENS``. This helper accepts
    mixed case and legacy/fractional aliases used in vendor specs, such as
    ``USC``/``US cents`` -> ``USc`` and ``GBX``/``pence`` -> ``GBp``. Unknown
    values return ``None`` so callers can distinguish currency-like tokens
    from physical units.
    """
    if token is None:
        return None
    cleaned = str(token).strip()
    if not cleaned:
        return None
    if cleaned in _VALID_CURRENCY_TOKENS:
        return cleaned

    lower = cleaned.lower()
    mapped = CURRENCY_MAP.get(lower)
    if mapped:
        return mapped

    lookup_key = _normalized_currency_lookup_key(cleaned)
    return _CURRENCY_TOKEN_LOOKUP.get(lookup_key) or _CURRENCY_TOKEN_ALIASES.get(
        lookup_key
    )


def is_fractional_currency(token: str) -> bool:
    """Return True if `token` is a recognised fractional currency
    (e.g. 'GBp', 'USc'). Useful for detecting pure-scale conversions that
    don't require an FX leg.
    """
    return token in _FRACTIONAL_TO_MAJOR


def fractional_to_major(token: str) -> str:
    """Resolve a fractional currency token to its parent major currency.

    Examples:
        fractional_to_major('GBp') -> 'GBP'
        fractional_to_major('USc') -> 'USD'
        fractional_to_major('EUR') -> 'EUR'   (already major; returned unchanged)
        fractional_to_major('$')   -> 'USD'   ('$' shorthand resolved to USD)
    """
    if not token:
        return ""
    if token == "$":
        return "USD"
    return _FRACTIONAL_TO_MAJOR.get(token, token.upper())


def split_currency_unit(token: str) -> tuple[str, str]:
    """Split a 'CCY/unit' token into ('CCY', 'unit').

    Only splits when the prefix before the first `/` is a recognised
    currency token (see `_VALID_CURRENCY_TOKENS`). Otherwise returns
    ('', token) — so bare rate units like 'bbl/day' or 'kt/month' are
    left intact for the downstream rate-unit parser.
    """
    if "/" not in token:
        return "", token
    head, _, tail = token.partition("/")
    head = head.strip()
    if head in _VALID_CURRENCY_TOKENS:
        return head, tail.strip()
    return "", token


def required_fx_pair(from_ccy: str, to_ccy: str) -> Optional[str]:
    """Return the FX pair name (e.g. 'EURUSD') needed to convert from->to.

    Returns None when no FX leg is needed (same major currency on both sides,
    or one side has no currency at all).

    Lifted from pyoilprice/util.py. Pure vocab; no FX-fetching dependency.
    """
    if not from_ccy:
        return None
    from_major = fractional_to_major(from_ccy)
    to_major = fractional_to_major(to_ccy) if to_ccy else "USD"
    if from_major == to_major:
        return None
    # Only USD-target is supported by commodutil.convfactors.convert_price
    # today (matches its own restriction). Build the pair quoted as
    # foreign->USD when targeting USD.
    if to_major == "USD":
        return f"{from_major}USD"
    return f"{from_major}{to_major}"


# Currency-symbol display map. Bundled here to deduplicate the parallel dict
# that lived in oilpricingcharts/util.py. Symbols are display policy; keep
# this map narrow (one mapping per ISO/fractional token; fallback returns
# the code itself).
_SYMBOLS = {
    # Canonical commodutil tokens
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "CAD": "C$",
    "AUD": "A$",
    "CHF": "Fr",
    "CNY": "¥",
    "CNH": "¥",
    "INR": "₹",
    "BRL": "R$",
    "ZAR": "R",
    "KRW": "₩",
    "SGD": "S$",
    "HKD": "HK$",
    "NZD": "NZ$",
    "NOK": "kr",
    "SEK": "kr",
    "DKK": "kr",
    "MXN": "Mex$",
    "PLN": "zł",
    "TRY": "₺",
    "RUB": "₽",
    "ILS": "₪",
    # Fractional units
    "USc": "¢",
    "GBp": "p",
    "EUc": "c",
    "JPy": "¥",
    "CAc": "¢",
    "AUc": "¢",
    # Literal dollar symbol
    "$": "$",
    # Legacy aliases used by oilpricingcharts (codex-required parity)
    "GBX": "p",  # alias of GBp
    "USC": "¢",  # alias of USc (uppercase)
    "YEN": "¥",  # alias of JPY
    "MYR": "RM",  # Malaysian ringgit — not a commodutil canonical token,
    # but oilpricingcharts had it; preserved for parity.
}


def to_symbol(code: Optional[str]) -> str:
    """Return display symbol for a currency code (or code itself if unknown).

    Honors both canonical commodutil tokens and legacy aliases used by
    oilpricingcharts (GBX, USC, YEN, MYR). Returns the input code unchanged
    for tokens not in the map (graceful fallback for new currencies).
    Empty/None input returns empty string.
    """
    if not code:
        return ""
    return _SYMBOLS.get(str(code), str(code))


# ---- Vendor-spec free-text -> canonical-token map ------------------------
#
# Maps lowercase free-form currency phrases (as they appear in CME/ICE
# contract spec descriptions) to canonical ISO 4217 codes. Used by
# vendor-spec parsers (e.g. curvemetadata.ice_util.map_currency) to lift
# strings like "US Dollars and Cents" -> "USD". Keys are matched
# case-insensitively at call time — callers should lowercase input.
#
# Lifted from curvemetadata.common_maps so commodutil owns the single
# source of truth for currency-token vocabulary.
CURRENCY_MAP = {
    "us dollars and cents": "USD",
    "u.s. dollars and cents": "USD",
    "us dollars": "USD",
    "u.s. dollars": "USD",
    "usd": "USD",
    "euros": "EUR",
    "euro": "EUR",
    "pounds sterling": "GBP",
    "british pounds": "GBP",
    "canadian dollars": "CAD",
    "cad": "CAD",
}


__all__ = [
    "VALID_CURRENCY_TOKENS",
    "FRACTIONAL_TO_MAJOR",
    "FRACTIONAL_CURRENCY_DIVISORS",
    "CURRENCY_MAP",
    "normalize_currency_token",
    "is_fractional_currency",
    "fractional_to_major",
    "split_currency_unit",
    "required_fx_pair",
    "to_symbol",
]
