"""commodutil.standards.regions: Region code canonicalization.

Owns canonical region codes and pattern-matching logic for commodity trading.
Single source of truth; curvemetadata is expected to re-export from here in a
follow-up PR.

Known divergence from curvemetadata.taxonomy.infer_region (as of 2026-05-14):
the legacy implementation in curvemetadata uses `rf"\\b...\\b"` in a raw
f-string, which is a *literal* backslash-b, not a regex word boundary. As a
result, the short region patterns ("nyh", "usgc", "usac", "nwe", "rott",
"ara", "med", "sing") never matched in curvemetadata production. This module
uses `rf"\b...\b"` (a true word boundary), so short codes will start
matching for the first time once consumers migrate.
"""

from __future__ import annotations

import re
from typing import Optional

# Canonical region codes and pattern matchers
REGION_PATTERNS = [
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

# Canonical region codes as frozenset for fast membership checks
VALID_REGIONS = frozenset(code for code, _ in REGION_PATTERNS)


def normalize_region(text: Optional[str]) -> Optional[str]:
    """Infer geographic region from product name or description.

    Consolidates pattern-matching logic from curvemetadata.taxonomy.infer_region
    with RBOB-to-NYH heuristic.

    Args:
        text: Product name or description (e.g., "Brent NYH", "WTI USGC").

    Returns:
        Canonical region code (e.g., "NYH", "USGC", "NWE") or None.

    Examples:
        >>> normalize_region("Brent NYH")
        'NYH'
        >>> normalize_region("RBOB Feb25")
        'NYH'
        >>> normalize_region("Unknown Location")
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

    # Pattern-match against REGION_PATTERNS
    for code, patterns in REGION_PATTERNS:
        for pattern in patterns:
            if len(pattern) <= 3:
                # Short patterns: word-boundary match only
                if re.search(rf"\b{re.escape(pattern)}\b", lower):
                    return code
            elif pattern in lower:
                # Long patterns: substring match
                return code

    return None


def is_valid_region(code: str) -> bool:
    """Return True if code is a canonical region code.

    Args:
        code: Region code string (e.g., "NYH", "USGC").

    Returns:
        True if code is in VALID_REGIONS, False otherwise.
    """
    return code in VALID_REGIONS


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
    "REGION_PATTERNS",
    "VALID_REGIONS",
    "normalize_region",
    "is_valid_region",
    "CRUDE_GRADE_REGIONS",
    "VALID_CRUDE_GRADE_REGIONS",
    "is_crude_grade_region",
]
