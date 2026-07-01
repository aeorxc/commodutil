"""Resolve source price-unit tokens from metadata attributes.

This module owns only string precedence for price-unit metadata. It has no
database access and does not perform unit conversion.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from commodutil.standards.currency import VALID_CURRENCY_TOKENS, split_currency_unit


def _clean_token(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:  # NaN without importing pandas/numpy.
            return ""
    except Exception:
        pass
    token = str(value).strip()
    if token.lower() in {"", "none", "nan", "nat", "<na>"}:
        return ""
    return token


def _currency_qualified_unit(unit: str) -> Optional[str]:
    currency, bare_unit = split_currency_unit(unit)
    if currency and bare_unit:
        return f"{currency}/{bare_unit}"
    return None


def resolve_price_unit(
    *,
    source_price_unit: Any = None,
    quote_unit: Any = None,
    currency: Any = None,
    contract_unit: Any = None,
) -> str:
    """Resolve a source price-unit token from metadata fields.

    Precedence:
    1. ``source_price_unit`` when already ``CCY/unit`` with a valid currency.
    2. ``quote_unit`` when it is already ``CCY/unit`` with a valid currency.
    3. ``currency/quote_unit`` when ``quote_unit`` is a bare unit.
    4. ``currency/contract_unit``.
    5. ``quote_unit``.
    6. ``contract_unit``.
    """
    source_price = _clean_token(source_price_unit)
    quote = _clean_token(quote_unit)
    ccy = _clean_token(currency)
    contract = _clean_token(contract_unit)

    if source_price:
        qualified_source = _currency_qualified_unit(source_price)
        if qualified_source:
            return qualified_source

    if quote:
        qualified_quote = _currency_qualified_unit(quote)
        if qualified_quote:
            return qualified_quote

    if ccy in VALID_CURRENCY_TOKENS and quote and "/" not in quote:
        return f"{ccy}/{quote}"
    if ccy in VALID_CURRENCY_TOKENS and contract:
        return f"{ccy}/{contract}"
    if quote:
        return quote
    return contract


def _metadata_value(attrs: Any, name: str) -> Any:
    if attrs is None:
        return None
    if isinstance(attrs, Mapping):
        return attrs.get(name)
    return getattr(attrs, name, None)


def resolve_price_unit_from_attrs(attrs: Any) -> str:
    """Resolve a source price-unit token from an attrs mapping or object.

    Reads ``quote_unit``, ``currency``, and ``contract_unit`` attributes/keys
    and applies :func:`resolve_price_unit`.
    """
    return resolve_price_unit(
        source_price_unit=_metadata_value(attrs, "source_price_unit"),
        quote_unit=_metadata_value(attrs, "quote_unit"),
        currency=_metadata_value(attrs, "currency"),
        contract_unit=_metadata_value(attrs, "contract_unit"),
    )


__all__ = ["resolve_price_unit", "resolve_price_unit_from_attrs"]
