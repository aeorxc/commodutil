"""commodutil.standards.analysis_types: canonical analysis-type vocabulary.

Owns the inference function for classifying a curve as one of:
- 'crack': cracking spread
- 'arb': geographic arbitrage spread
- 'diff': differential / generic spread
- 'outright': none of the above (flat price)

Previously duplicated in curvemetadata.ice._infer_analysis_type (substring
matching) and curvemetadata.cme._infer_analysis_type (regex word-boundary).
Consolidated here using the stricter regex approach -- substring 'crack' in
'crackers' would have been a false positive in the ICE version.

Future subtypes (hi5, regrade, freight) are NOT included in this round --
they require validation against actual Curve.Definitions.Analysis values
in MetadataDB2, which is a separate piece of work.
"""

from __future__ import annotations

import re
from typing import Optional

# Closed set of types recognised by infer_analysis_type. Callers downstream
# may treat the return as a canonical tag.
ANALYSIS_TYPES = ("crack", "arb", "diff", "outright")


def infer_analysis_type(
    text: Optional[str],
    extra_text: Optional[str] = None,
    category: Optional[str] = None,
) -> str:
    """Infer analysis type from product metadata.

    Returns one of: 'crack', 'arb', 'diff', 'outright'. Defaults to
    'outright' when no pattern matches.

    Args:
        text: Primary product name or description.
        extra_text: Optional secondary text (e.g. CME sub-category) --
            appended to `text` for matching.
        category: Optional explicit category field. If equals 'arb'
            (case-insensitive), short-circuits to 'arb' regardless of
            other text content (CME convention).
    """
    if isinstance(category, str) and category.strip().lower() == "arb":
        return "arb"
    combined = f"{text or ''} {extra_text or ''}".lower()
    if re.search(r"\bcrack\b", combined):
        return "crack"
    if re.search(r"\barb\b", combined):
        return "arb"
    if re.search(r"\bdiff\b", combined) or re.search(r"\bspread\b", combined):
        return "diff"
    return "outright"


__all__ = ["ANALYSIS_TYPES", "infer_analysis_type"]
