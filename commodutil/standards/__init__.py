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
    infer_commodity_and_group,
    infer_commodity_from_exchange_symbol,
    normalize_commodity_for_conversion,
)
from commodutil.standards.commodity_groups import (
    COMMODITY_GROUPS,
    VALID_COMMODITY_GROUPS,
    is_valid_commodity_group,
)
from commodutil.standards.currency import (
    CURRENCY_MAP,
    FRACTIONAL_CURRENCY_DIVISORS,
    FRACTIONAL_TO_MAJOR,
    VALID_CURRENCY_TOKENS,
    fractional_to_major,
    is_fractional_currency,
    normalize_currency_token,
    required_fx_pair,
    split_currency_unit,
    to_symbol,
)
from commodutil.standards.price_unit import (
    PriceUnit,
)
from commodutil.standards.price_units import (
    resolve_price_unit,
    resolve_price_unit_from_attrs,
)
from commodutil.standards.regions import (
    CRUDE_GRADE_REGIONS,
    REGION_PATTERNS,
    VALID_CRUDE_GRADE_REGIONS,
    VALID_REGIONS,
    is_crude_grade_region,
    is_valid_region,
    normalize_region,
)
from commodutil.standards.units import (
    UNIT_MAP,
    canonical_price_unit_token,
    canonical_quantity_unit,
    canonical_unit_token,
    default_unit_for_commodity,
    quantity_unit_from_price_unit,
    to_pint_token,
)

__all__ = [
    # analysis_types
    "ANALYSIS_TYPES",
    "infer_analysis_type",
    # commodities
    "COMMODITY_CONVERSION_MAP",
    "COMMODITY_KEYWORDS",
    "infer_commodity_and_group",
    "infer_commodity_from_exchange_symbol",
    "normalize_commodity_for_conversion",
    # commodity_groups
    "COMMODITY_GROUPS",
    "VALID_COMMODITY_GROUPS",
    "is_valid_commodity_group",
    # currency
    "CURRENCY_MAP",
    "FRACTIONAL_CURRENCY_DIVISORS",
    "FRACTIONAL_TO_MAJOR",
    "VALID_CURRENCY_TOKENS",
    "fractional_to_major",
    "is_fractional_currency",
    "normalize_currency_token",
    "required_fx_pair",
    "split_currency_unit",
    "to_symbol",
    # price_unit
    "PriceUnit",
    # price_units
    "resolve_price_unit",
    "resolve_price_unit_from_attrs",
    # regions
    "CRUDE_GRADE_REGIONS",
    "REGION_PATTERNS",
    "VALID_CRUDE_GRADE_REGIONS",
    "VALID_REGIONS",
    "is_crude_grade_region",
    "is_valid_region",
    "normalize_region",
    # units
    "UNIT_MAP",
    "canonical_price_unit_token",
    "canonical_quantity_unit",
    "canonical_unit_token",
    "default_unit_for_commodity",
    "quantity_unit_from_price_unit",
    "to_pint_token",
]
