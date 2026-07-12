"""commodutil: Commodity market standards backbone.

Public API for unit conversions, forward curve transforms, date utilities,
and pandas helpers. Symbols are loaded lazily on first access via PEP 562
`__getattr__` so `import commodutil` stays cheap (~10ms vs. ~3s if the
facade eagerly imported convfactors and its pint registry).

Cheap paths:
    import commodutil                                  # near-instant
    from commodutil.standards.regions import ...       # stdlib only
    from commodutil.standards.currency import ...      # stdlib only

Heavier paths (lazy-load their sub-module on first attribute access):
    from commodutil import convert_price               # triggers convfactors
    from commodutil import curyear                     # triggers dates
"""

from __future__ import annotations

# Map exported name -> dotted submodule path. Single source of truth for the
# public facade. Keys in this dict are what `from commodutil import X` will
# resolve via __getattr__ below.
_LAZY_EXPORTS = {
    # convfactors (heaviest -- pint registry + Commodity dataclass init)
    "COMMODITIES": "commodutil.convfactors",
    "Commodity": "commodutil.convfactors",
    "convert": "commodutil.convfactors",
    "convert_price": "commodutil.convfactors",
    "convfactor": "commodutil.convfactors",
    "list_commodities": "commodutil.convfactors",
    "list_units": "commodutil.convfactors",
    # standards.currency (stdlib-only -- cheap, no pint)
    "FRACTIONAL_TO_MAJOR": "commodutil.standards.currency",
    "VALID_CURRENCY_TOKENS": "commodutil.standards.currency",
    "fractional_to_major": "commodutil.standards.currency",
    "is_fractional_currency": "commodutil.standards.currency",
    "normalize_currency_token": "commodutil.standards.currency",
    "split_currency_unit": "commodutil.standards.currency",
    # standards.units (stdlib-only -- cheap, no pint)
    "canonical_price_unit_token": "commodutil.standards.units",
    "canonical_quantity_unit": "commodutil.standards.units",
    "canonical_unit_token": "commodutil.standards.units",
    "is_canonical_price_unit": "commodutil.standards.units",
    "normalize_price_unit_strict": "commodutil.standards.units",
    "quantity_unit_from_price_unit": "commodutil.standards.units",
    # dates
    "curmon": "commodutil.dates",
    "curmonyear": "commodutil.dates",
    "curmonyear_str": "commodutil.dates",
    "curyear": "commodutil.dates",
    "find_year": "commodutil.dates",
    "last_day_of_prev_month": "commodutil.dates",
    "nextyear": "commodutil.dates",
    "prevmon": "commodutil.dates",
    "prevmon_str": "commodutil.dates",
    "prevyear": "commodutil.dates",
    "start_day_of_prev_month": "commodutil.dates",
    # forwards
    "all_spread_combinations": "commodutil.forwards",
    "time_spreads": "commodutil.forwards",
    # pandasutil
    "fillna_downbet": "commodutil.pandasutil",
    "mergets": "commodutil.pandasutil",
    # transforms
    "seasonalize": "commodutil.transforms",
}

__all__ = sorted(_LAZY_EXPORTS.keys())


def __getattr__(name: str):
    """PEP 562 lazy attribute access — loads the source submodule on first
    use of a facade-exported symbol, caches the resolved value back into
    the module's globals so subsequent accesses skip the dispatch.
    """
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'commodutil' has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
