"""commodutil.standards: canonical vocabularies for commodity trading.

Re-exports the public surface of each submodule so callers can write
`from commodutil.standards import normalize_trading_hub` instead of reaching
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
    infer_ngl_species,
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
from commodutil.standards.geography import (
    COUNTRY_ALIASES,
    COUNTRY_GROUP_MEMBERS,
    COUNTRY_GROUP_ROWS,
    COUNTRY_NAME_MAP,
    VALID_COUNTRY_CODES,
    VALID_COUNTRY_GROUP_KEYS,
    CountryGroupRow,
    canonical_country,
    country_group_members,
    country_in_group,
    is_canonical_country,
    is_canonical_country_group,
)
from commodutil.standards.price_unit import (
    PriceUnit,
    resolve_price_unit,
    resolve_price_unit_from_attrs,
)
from commodutil.standards.trading_hubs import (
    CRUDE_GRADE_REGIONS,
    TRADING_HUB_PATTERNS,
    VALID_CRUDE_GRADE_REGIONS,
    VALID_TRADING_HUBS,
    is_crude_grade_region,
    is_valid_trading_hub,
    normalize_trading_hub,
)
from commodutil.standards.units import (
    UNIT_MAP,
    canonical_price_unit_token,
    canonical_quantity_unit,
    canonical_unit_token,
    is_canonical_price_unit,
    normalize_price_unit_strict,
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
    "infer_ngl_species",
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
    # geography
    "COUNTRY_ALIASES",
    "VALID_COUNTRY_CODES",
    "COUNTRY_NAME_MAP",
    "canonical_country",
    "is_canonical_country",
    "CountryGroupRow",
    "COUNTRY_GROUP_ROWS",
    "VALID_COUNTRY_GROUP_KEYS",
    "COUNTRY_GROUP_MEMBERS",
    "is_canonical_country_group",
    "country_group_members",
    "country_in_group",
    # price_unit
    "PriceUnit",
    "resolve_price_unit",
    "resolve_price_unit_from_attrs",
    # trading_hubs
    "CRUDE_GRADE_REGIONS",
    "TRADING_HUB_PATTERNS",
    "VALID_CRUDE_GRADE_REGIONS",
    "VALID_TRADING_HUBS",
    "is_crude_grade_region",
    "is_valid_trading_hub",
    "normalize_trading_hub",
    # units
    "UNIT_MAP",
    "canonical_price_unit_token",
    "canonical_quantity_unit",
    "canonical_unit_token",
    "is_canonical_price_unit",
    "normalize_price_unit_strict",
    "quantity_unit_from_price_unit",
    "to_pint_token",
]
