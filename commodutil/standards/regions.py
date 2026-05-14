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

    # RBOB convention: always NY Harbor
    if "rbob" in lower:
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


__all__ = [
    "REGION_PATTERNS",
    "VALID_REGIONS",
    "normalize_region",
    "is_valid_region",
]
