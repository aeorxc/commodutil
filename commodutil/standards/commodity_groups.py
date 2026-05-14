"""commodutil.standards.commodity_groups: canonical commodity group taxonomy.

Mirrors the CommodityGroup CHECK constraint on Curve.Definitions in
MetadataDB2 (see curvemetadata/sql/001_create_curvemetadata_tables.sql).
Single source of truth for the Python side -- curvemetadata.ice and
curvemetadata.cme currently hard-code these strings; they can adopt
this constant in a follow-up.
"""

from __future__ import annotations

# Ordered to match the SQL constraint order (preserves what consumers see
# when iterating). Treat as a closed set -- additions require a coordinated
# SQL migration.
COMMODITY_GROUPS = (
    "Agriculture",
    "Biofuel",
    "Crude Oil",
    "Freight",
    "LNG",
    "Natural Gas",
    "NGL",
    "Petrochemical",
    "Refined Products",
)

# Set for O(1) membership checks
VALID_COMMODITY_GROUPS = frozenset(COMMODITY_GROUPS)


def is_valid_commodity_group(value: str) -> bool:
    """Return True if `value` is a recognised commodity group string."""
    return value in VALID_COMMODITY_GROUPS


__all__ = [
    "COMMODITY_GROUPS",
    "VALID_COMMODITY_GROUPS",
    "is_valid_commodity_group",
]
