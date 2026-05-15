"""commodutil.standards: canonical vocabularies for commodity trading.

Re-exports the public surface of each submodule so callers can write
`from commodutil.standards import normalize_region` instead of reaching
into the submodule directly.
"""

from commodutil.standards.analysis_types import (
    ANALYSIS_TYPES,
    infer_analysis_type,
)
from commodutil.standards.commodities import (
    COMMODITY_CONVERSION_MAP,
    COMMODITY_KEYWORDS,
)
from commodutil.standards.commodity_groups import (
    COMMODITY_GROUPS,
    VALID_COMMODITY_GROUPS,
    is_valid_commodity_group,
)
from commodutil.standards.currency import (
    FRACTIONAL_CURRENCY_DIVISORS,
    FRACTIONAL_TO_MAJOR,
    VALID_CURRENCY_TOKENS,
    fractional_to_major,
    is_fractional_currency,
    required_fx_pair,
    split_currency_unit,
    to_symbol,
)
from commodutil.standards.regions import (
    REGION_PATTERNS,
    VALID_REGIONS,
    is_valid_region,
    normalize_region,
)
from commodutil.standards.units import (
    UNIT_MAP,
    default_unit_for_commodity,
)

__all__ = [
    # analysis_types
    "ANALYSIS_TYPES",
    "infer_analysis_type",
    # commodities
    "COMMODITY_CONVERSION_MAP",
    "COMMODITY_KEYWORDS",
    # commodity_groups
    "COMMODITY_GROUPS",
    "VALID_COMMODITY_GROUPS",
    "is_valid_commodity_group",
    # currency
    "FRACTIONAL_CURRENCY_DIVISORS",
    "FRACTIONAL_TO_MAJOR",
    "VALID_CURRENCY_TOKENS",
    "fractional_to_major",
    "is_fractional_currency",
    "required_fx_pair",
    "split_currency_unit",
    "to_symbol",
    # regions
    "REGION_PATTERNS",
    "VALID_REGIONS",
    "is_valid_region",
    "normalize_region",
    # units
    "UNIT_MAP",
    "default_unit_for_commodity",
]
