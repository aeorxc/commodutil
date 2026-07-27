"""Canonical trading-hub and pricing-location vocabulary.

Owns canonical trading-hub codes (NYH, USGC, NWE, ARA, Med, Sing, MEG, and
others) and their pattern-matching logic for commodity trading.

Country-membership groups/baskets live in ``geography.py`` — a different vocabulary.
"""

from __future__ import annotations

import re
from typing import Optional

# Canonical trading-hub codes and pattern matchers
TRADING_HUB_PATTERNS = [
    ("NYH", ["nyh", "new york harbor", "new york harbour", "new york"]),
    ("USGC", ["usgc", "gulf coast", "us gulf", "u.s. gulf"]),
    ("USAC", ["us atlantic coast", "u.s. atlantic coast", "usac"]),
    ("LA", ["los angeles"]),
    ("NWE", ["nwe", "northwest europe", "north west europe"]),
    ("Rott", ["rotterdam", "rdam", "rott"]),
    ("ARA", ["ara"]),
    ("Med", ["mediterranean", "med"]),
    ("Sing", ["singapore", "sing"]),
    ("MEG", ["meg", "middle east gulf", "arabian gulf", "persian gulf"]),
    ("Japan", ["japan"]),
]

# Canonical trading-hub codes as frozenset for fast membership checks
VALID_TRADING_HUBS = frozenset(code for code, _ in TRADING_HUB_PATTERNS)


def normalize_trading_hub(text: Optional[str]) -> Optional[str]:
    """Infer a trading hub or pricing location from a product description.

    Includes an RBOB-to-NYH heuristic.

    Args:
        text: Product name or description (e.g., "Brent NYH", "WTI USGC").

    Returns:
        Canonical trading-hub code (e.g., "NYH", "USGC", "NWE") or None.

    Examples:
        >>> normalize_trading_hub("Brent NYH")
        'NYH'
        >>> normalize_trading_hub("RBOB Feb25")
        'NYH'
        >>> normalize_trading_hub("Unknown Location")
        None
    """
    if not text:
        return None

    lower = text.lower()

    # CARBOB short-circuit: California Reformulated Blendstock for Oxygenate
    # Blending — Los Angeles / US West Coast, NOT NY Harbor. Must run BEFORE
    # the RBOB check so "carbob" doesn't fall through to the NYH heuristic.
    if re.search(r"\bcarbob\b", lower):
        return "LA"

    # RBOB convention: always NY Harbor. Use word boundary so "carbob" (which
    # has 'a' before 'rbob' — no word boundary) does not match here.
    if re.search(r"\brbob\b", lower):
        return "NYH"

    # Pattern-match against TRADING_HUB_PATTERNS
    for code, patterns in TRADING_HUB_PATTERNS:
        for pattern in patterns:
            if len(pattern) <= 3:
                # Short patterns: word-boundary match only
                if re.search(rf"\b{re.escape(pattern)}\b", lower):
                    return code
            elif pattern in lower:
                # Long patterns: substring match
                return code

    return None


def is_valid_trading_hub(code: str) -> bool:
    """Return True if code is a canonical trading-hub code.

    Args:
        code: Trading-hub code string (e.g., "NYH", "USGC").

    Returns:
        True if code is in VALID_TRADING_HUBS, False otherwise.
    """
    return code in VALID_TRADING_HUBS


# ---- Crude grade regions ----
#
# Producer-region groupings for crude grades, used by crude-differentials
# charts. Lifted from oilpricingcharts.symbols_config_crudediffs (keys kept
# byte-identical to the source so chart configs can switch over without
# re-mapping). Values are ordered tuples of display grade names — they are
# NOT pricing symbols and do NOT carry vendor (Platts/Argus) IDs. Symbol
# resolution stays in the chart-config layer.
CRUDE_GRADE_REGIONS = {
    "north_sea": (
        "Forties",
        "Oseberg",
        "Ekofisk",
        "Troll",
        "Johan Sverdrup",
        "FOB N Sea WTI Midland",
    ),
    "waf": (
        "Bonny Light",
        "Forcados",
        "Qua Iboe",
        "Cabinda",
        "Doba",
    ),
    "nafrica": (
        "Nile Blend",
        "Dar Blend",
        "Es Sider",
    ),
    "russian": (
        "Urals Rott",
        "Urals Med",
        "ESPO",
        "Siberian Light",
        "Sokol",
    ),
    "us_midcon": (
        "Bakken Clearbook",
        "Light Sweet Guernsey",
        "Denver Julesburg Light",
    ),
    "us_texas": (
        "WTI Houston",
        "WTI Midland",
        "WTS",
        "Southern Green Canyon",
        "WCS Houston",
    ),
    "us_louisiana": (
        "LLS",
        "HLS",
        "Thunder Horse",
        "Poseidon",
        "Mars",
    ),
    "canadian": (
        "WCS",
        "CDB",
        "AWB",
        "CLK",
        "MSW",
        "Syn",
    ),
    "latam_wti": (
        "Vasconia",
        "Castilla",
        "Maya",
        "Liza",
        "Buzios",
        "Mero",
        "Tupi",
        "Unity Gold",
    ),
    "asia_pacific": (
        "Tapis",
        "Duri",
        "Vincent",
    ),
    "middle_east": (
        "Dubai",
        "Oman",
        "Murban",
        "Al Shaheen",
        "Upper Zakum",
        "Qatar Land",
        "Qatar Marine",
    ),
}

VALID_CRUDE_GRADE_REGIONS = frozenset(CRUDE_GRADE_REGIONS.keys())


def is_crude_grade_region(key: str) -> bool:
    """Return True if key is a canonical crude grade-region key."""
    return key in VALID_CRUDE_GRADE_REGIONS


__all__ = [
    "TRADING_HUB_PATTERNS",
    "VALID_TRADING_HUBS",
    "normalize_trading_hub",
    "is_valid_trading_hub",
    "CRUDE_GRADE_REGIONS",
    "VALID_CRUDE_GRADE_REGIONS",
    "is_crude_grade_region",
]
