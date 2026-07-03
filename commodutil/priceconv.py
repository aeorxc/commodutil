"""Price/currency/FX conversion helpers for commodutil.

Module map: price/currency/FX conversion lives here; quantity/pint conversion lives in ``commodutil.convfactors``; vocabulary lives in ``commodutil.standards.unit_registry``; ``PriceUnit`` grammar and resolve precedence live in ``commodutil.standards.price_unit``.
"""

from dataclasses import dataclass
from typing import Optional, Union

import pandas as pd

from commodutil.convfactors import convfactor, logger
from commodutil.standards import currency as _currency
from commodutil.standards.price_unit import PriceUnit


def align_fx(
    fx: pd.Series,
    index: pd.Index,
    policy: str = "strict",
    max_staleness: pd.Timedelta = pd.Timedelta(days=7),
) -> pd.Series:
    """Align an FX Series onto ``index`` for a price conversion.

    Extracted verbatim from ``convert_price`` (Phase 3.4) so new code can compose
    FX alignment explicitly; ``convert_price`` delegates here, so behaviour and
    error/warning text are unchanged.

    - ``policy='strict'`` (default): forward-fill FX onto ``index`` bounded by
      ``max_staleness``; any target date left uncovered (pre-FX-start, or stale
      past the limit) raises ``ValueError`` rather than silently back-filling or
      extrapolating.
    - ``policy='ffill'``: permissive — union the indices, ffill across the union,
      reindex back, and fill any residual NaN with the most-recent non-null FX.
      Emits a logging warning (future-leakage risk in backtests). Raises only if
      the FX series is entirely NaN.
    """
    target_idx = index

    if policy == "strict":
        # Bounded ffill: only forward-fill within `max_staleness`. Anything
        # uncovered (e.g. value-dates before fx.index.min() or stale past
        # the limit) stays NaN and triggers a loud raise — no silent
        # back-fill, no silent stale extrapolation.
        union_idx = fx.index.union(target_idx)
        fx_union = fx.reindex(union_idx).sort_index().ffill()
        # Track how stale each ffilled value is and zero-out anything older
        # than max_staleness.
        valid_mask = ~fx.reindex(union_idx).isna()
        last_valid_pos = pd.Series(union_idx, index=union_idx).where(valid_mask).ffill()
        staleness = pd.Series(union_idx, index=union_idx) - last_valid_pos
        fx_union = fx_union.where(staleness <= max_staleness)
        fx_aligned = fx_union.reindex(target_idx)
        if fx_aligned.isna().any():
            missing = target_idx[fx_aligned.isna()]
            first_missing = missing[0]
            first_missing_str = (
                first_missing.date()
                if hasattr(first_missing, "date")
                else first_missing
            )
            raise ValueError(
                f"FX missing or stale (>{max_staleness}) for "
                f"{len(missing)} target date(s) (first: {first_missing_str}). "
                f"Pass ffill_policy='ffill' to fill with the last non-null "
                f"FX (BACKTEST FUTURE LEAKAGE RISK)."
            )
    elif policy == "ffill":
        logger.warning(
            "convert_price: ffill_policy='ffill' — pre-FX-start dates will "
            "be back-filled with the latest FX. Future-leakage risk in "
            "backtests; prefer 'strict' for historical research."
        )
        if not target_idx.isin(fx.index).all():
            union_idx = fx.index.union(target_idx)
            fx_aligned = fx.reindex(union_idx).ffill().reindex(target_idx)
        else:
            fx_aligned = fx.reindex(target_idx).ffill()
        if fx_aligned.isna().any():
            fx_nonnull = fx.dropna()
            if fx_nonnull.size == 0:
                raise ValueError(
                    "FX series is entirely NaN; refusing the silent "
                    "multiply-by-1.0 fallback."
                )
            fx_aligned = fx_aligned.fillna(fx_nonnull.iloc[-1])
    else:
        raise ValueError(
            f"Unknown ffill_policy: {policy!r} (expected 'strict' or 'ffill')"
        )

    return fx_aligned


def convert_price(
    value: Union[float, pd.Series],
    from_unit: Union[str, PriceUnit],
    to_unit: Union[str, PriceUnit],
    commodity: Optional[str] = None,
    fx: Union[float, pd.Series, None] = None,
    ffill_policy: str = "strict",
    max_staleness: pd.Timedelta = pd.Timedelta(days=7),
) -> Union[float, pd.Series]:
    """
    Convert price values ($/unit) between units, optionally bearing an FX rate.

    Price conversion is the inverse of quantity conversion:
        price_to = price_from / convfactor(from_unit, to_unit, commodity)

    If `from_unit` and `to_unit` differ in currency (e.g. EUR/MWh -> USD/MMBtu),
    `fx` must be supplied (scalar or pandas.Series indexed by date), quoted as
    target/source — i.e. USD-per-foreign-currency. Fractional-currency
    prefixes ('GBp', 'USc', ...) are auto-detected and divided by 100.

    If `fx` is a Series and `value` is a Series, alignment policy is controlled
    by `ffill_policy` and `max_staleness`:

    - `ffill_policy='strict'` (default): FX is forward-filled onto value.index
      with a bounded ffill of `max_staleness`. If any target dates remain
      uncovered, a `ValueError` is raised — refusing to silently back-fill
      pre-FX-start dates (which would be future leakage in backtests) or
      to extend stale FX values indefinitely.
    - `ffill_policy='ffill'`: legacy permissive behaviour — union the two
      indices, ffill across the union, then reindex back; any remaining
      NaNs are filled with the most-recent non-null FX. Emits a logging
      warning because this is unsafe for backtests.

    There is no all-NaN `1.0` fallback. If FX is unusable, the function raises.

    `from_unit` / `to_unit` are EITHER bare units ('mt', 'bbl', 'gal', 'MMBtu',
    'MWh', 'therm') OR currency-qualified ('USD/bbl', 'EUR/MWh', 'GBp/therm').
    Currency-qualified targets are currently restricted to USD (anything else
    raises ValueError — extend in future if non-USD targets are needed).

    Examples:
        # Gasoline: $/mt -> $/bbl (divide by ~8.33)
        convert_price(100, 'mt', 'bbl', commodity='gasoline')  # ~12.0

        # US gallon to barrel: $/gal -> $/bbl (multiply by 42)
        convert_price(2.5, 'gal', 'bbl')  # ~105.0

        # TTF EUR/MWh -> $/MMBtu (EURUSD = 1.07)
        convert_price(35.0, 'EUR/MWh', 'USD/MMBtu', fx=1.07)  # ~10.98

        # NBP GBp/therm -> $/MMBtu (GBPUSD = 1.25); GBp auto-detected & /100
        convert_price(80.0, 'GBp/therm', 'USD/MMBtu', fx=1.25)  # ~10.00

        # Time-varying FX with a pandas Series
        p = pd.Series([35.0, 36.5, 34.2], index=pd.date_range('2026', periods=3))
        fx_series = pd.Series([1.07, 1.08, 1.06], index=p.index)
        convert_price(p, 'EUR/MWh', 'USD/MMBtu', fx=fx_series)
    """
    # Parse each side once at the boundary into a PriceUnit and read its legs,
    # instead of re-splitting raw strings. PriceUnit.parse is built on
    # split_currency_unit, so (currency, quantity_leg) is byte-identical to the
    # previous split_currency_unit(...) result — behaviour and error strings are
    # unchanged. from_unit/to_unit are re-bound to the canonical string form for
    # use in the error messages / examples below.
    from_pu = (
        from_unit if isinstance(from_unit, PriceUnit) else PriceUnit.parse(from_unit)
    )
    to_pu = to_unit if isinstance(to_unit, PriceUnit) else PriceUnit.parse(to_unit)
    from_unit = str(from_pu)
    to_unit = str(to_pu)
    from_ccy, from_bare_unit = (from_pu.currency or ""), from_pu.quantity_leg()
    to_ccy, to_bare_unit = (to_pu.currency or ""), to_pu.quantity_leg()

    # Resolve the underlying "major" currency on each side for same-base detection
    # (e.g. USc and USD share major USD — pure scale, no FX needed).
    from_major = _currency.FRACTIONAL_TO_MAJOR.get(
        from_ccy, from_ccy.upper() if from_ccy else ""
    )
    to_major = _currency.FRACTIONAL_TO_MAJOR.get(
        to_ccy, to_ccy.upper() if to_ccy else ""
    )
    # Treat '$' as 'USD' for the purpose of major-currency comparison.
    if from_major == "$":
        from_major = "USD"
    if to_major == "$":
        to_major = "USD"

    same_base_fractional = bool(from_ccy and to_ccy and from_major == to_major)

    # Validate target currency — explicit USD only for now (only enforced when
    # the target is currency-qualified at all AND we're not in a same-base
    # fractional case like GBp/therm -> GBP/therm, which is a pure scale).
    if to_ccy and to_major != "USD" and not same_base_fractional:
        raise ValueError(
            f"convert_price currently only supports USD/* as target; got '{to_unit}'"
        )

    # Unit-leg conversion (no FX yet — uses commodity factors).
    # convfactor returns a nonzero float or raises, so no None/zero guard is
    # needed here.
    factor = convfactor(from_bare_unit, to_bare_unit, commodity)
    unit_converted = value / factor

    # Same-base fractional case: USc -> USD, GBp -> GBP, EUc -> EUR, JPy -> JPY.
    # This is a pure /100 scale (or *100 in the reverse direction) — no FX
    # needed even though the literal currency tokens differ. Handle BEFORE the
    # `fx is None` raise below.
    if same_base_fractional:
        from_div = _currency.FRACTIONAL_CURRENCY_DIVISORS.get(from_ccy, 1.0)
        to_div = _currency.FRACTIONAL_CURRENCY_DIVISORS.get(to_ccy, 1.0)
        # value is in source-currency units; divide by from_div to get majors,
        # multiply by to_div to get target-currency units.
        return unit_converted * (to_div / from_div)

    # If no source currency or it's already USD, no FX leg needed
    if not from_ccy or from_major == "USD":
        return unit_converted

    # Apply FX leg
    if fx is None:
        raise ValueError(
            f"FX rate required to convert {from_unit} -> {to_unit} "
            f"(source currency '{from_ccy}' is non-USD)"
        )

    fractional_divisor = _currency.FRACTIONAL_CURRENCY_DIVISORS.get(from_ccy, 1.0)

    if isinstance(unit_converted, pd.Series) and isinstance(fx, pd.Series):
        fx_aligned = align_fx(
            fx, unit_converted.index, policy=ffill_policy, max_staleness=max_staleness
        )
        return unit_converted * fx_aligned / fractional_divisor

    return unit_converted * fx / fractional_divisor


def convert_currency_leg(
    value: Union[float, pd.Series],
    from_unit: Union[str, PriceUnit],
    to_unit: Union[str, PriceUnit],
    fx: Union[float, pd.Series, None] = None,
) -> Union[float, pd.Series]:
    """Convert ONLY the currency leg of a price whose quantity denominator is
    unchanged.

    This is the separable-currency-leg helper from the conversion architecture
    plan (Phase 3.1). Unlike :func:`convert_price`, it performs NO physical-unit
    conversion and never calls :func:`convfactor`, so it works when the quantity
    denominator is non-physical or unknown to pint — e.g. ``'USc/RIN' ->
    'USD/RIN'`` (a pure ``/100`` scale). ``convert_price`` cannot do that today
    because it tries to compute ``convfactor('RIN', 'RIN')`` and raises on the
    unknown unit.

    It reimplements, as a first-class API, the broad-except ``/100.0`` currency
    fallback shim in ``oilrisk``'s ``artis.py`` (lines ~75-98): scale a
    fractional-currency quote (USc, GBp, ...) to its major unit without needing a
    commodity or a physical factor.

    Rules:
      * The two quantity denominators must be **string-equal** after parsing
        (``from`` and ``to`` refer to the same thing priced per identical unit);
        otherwise ``ValueError``. No physical-unit validation is performed.
      * Same-base fractional (``USc``->``USD``, ``GBp``->``GBP``): pure divisor
        scale, no ``fx`` needed.
      * No source currency, or source already the USD major: no-op (returns
        ``value`` unchanged).
      * Cross major-currency (e.g. ``EUR``->``USD``): ``fx`` is required (quoted
        target-per-source) or ``ValueError`` is raised. ``fx`` is applied
        element-wise; index-aligned FX Series are the caller's responsibility
        (use :func:`convert_price` for the strict/ffill staleness machinery).
      * Target currency is restricted to USD (matching ``convert_price``),
        except pure same-base fractional scaling like ``GBp``->``GBP``.

    Examples:
        convert_currency_leg(250.0, 'USc/RIN', 'USD/RIN')   # -> 2.5  (/100)
        convert_currency_leg(50.0, 'GBp/therm', 'GBP/therm')  # -> 0.5
        convert_currency_leg(10.0, 'EUR/RIN', 'USD/RIN', fx=1.07)  # -> 10.7
    """
    from_pu = (
        from_unit if isinstance(from_unit, PriceUnit) else PriceUnit.parse(from_unit)
    )
    to_pu = to_unit if isinstance(to_unit, PriceUnit) else PriceUnit.parse(to_unit)

    # Quantity denominators must match exactly — string equality only, so a
    # non-physical denominator like 'RIN' is fine (no pint lookup).
    if from_pu.quantity_leg() != to_pu.quantity_leg():
        raise ValueError(
            f"convert_currency_leg requires identical quantity denominators; "
            f"got '{from_pu.quantity_leg()}' vs '{to_pu.quantity_leg()}'"
        )

    from_ccy = from_pu.currency or ""
    to_ccy = to_pu.currency or ""
    from_major = from_pu.major_currency or ""
    to_major = to_pu.major_currency or ""

    same_base_fractional = bool(from_ccy and to_ccy and from_major == to_major)

    # USD-only currency target, except pure same-base fractional scaling.
    if to_ccy and to_major != "USD" and not same_base_fractional:
        raise ValueError(
            f"convert_currency_leg currently only supports USD/* as target; "
            f"got '{to_pu}'"
        )

    # Same-base fractional (USc->USD, GBp->GBP, ...): pure divisor scale.
    if same_base_fractional:
        return value * (to_pu.fractional_divisor / from_pu.fractional_divisor)

    # No source currency, or already USD major: currency leg is a no-op.
    if not from_ccy or from_major == "USD":
        return value

    # Cross major-currency: an FX leg is required.
    if fx is None:
        raise ValueError(
            f"FX rate required to convert {from_pu} -> {to_pu} "
            f"(source currency '{from_ccy}' is non-USD)"
        )
    return value * fx / from_pu.fractional_divisor


@dataclass(frozen=True)
class ConversionResult:
    """Result of :func:`convert_price_result`.

    - ``value``: the converted price (float or pandas Series).
    - ``applied``: True when a non-identity conversion route ran (from/to differ);
      False for an identity ('unchanged') or a kept-raw error.
    - ``note``: which route produced ``value`` — e.g.
      ``'convert_price:USD/mt->USD/bbl[diesel]'``, ``'currency_leg:USc/RIN->USD/RIN'``,
      ``'unchanged'``, or ``'kept-raw:<error>'``.
    """

    value: Union[float, pd.Series]
    applied: bool
    note: str


def convert_price_result(
    value: Union[float, pd.Series],
    from_unit: Union[str, PriceUnit],
    to_unit: Union[str, PriceUnit],
    commodity: Optional[str] = None,
    fx: Union[float, pd.Series, None] = None,
    ffill_policy: str = "strict",
    max_staleness: pd.Timedelta = pd.Timedelta(days=7),
    on_error: str = "raise",
) -> ConversionResult:
    """Result-bearing wrapper over :func:`convert_price`: returns a
    :class:`ConversionResult` (value, applied, note) instead of a bare value.

    Both oilrisk price loaders (``artis.py`` and
    ``load_marketplace_price_snapshots.py``) independently invented exactly this
    ``(value, changed?, source_note)`` shape around convert_price; this is the
    shared primitive they unify onto in Phase 4.2.

    Same parameters as :func:`convert_price`, plus:
      * ``on_error='raise'`` (default): a conversion failure propagates, so plain
        use stays strict.
      * ``on_error='keep'``: on failure (and when no currency-leg fallback
        applies) return the ORIGINAL ``value`` with ``applied=False`` and a
        ``'kept-raw:...'`` note — the keep-raw semantic oilrisk wants for rows it
        can't convert but must not drop.

    Fallback: if the full conversion fails but both units are currency-qualified
    with equal quantity denominators, the currency leg alone is applied via
    :func:`convert_currency_leg` (e.g. ``'USc/RIN'->'USD/RIN'``), with a
    ``'currency_leg:...'`` note — mirroring the fallback oilrisk's artis.py does
    at its call site so Phase 4.2 can delete that logic there.
    """
    if on_error not in ("raise", "keep"):
        raise ValueError(f"on_error must be 'raise' or 'keep', got {on_error!r}")

    from_pu = (
        from_unit if isinstance(from_unit, PriceUnit) else PriceUnit.parse(from_unit)
    )
    to_pu = to_unit if isinstance(to_unit, PriceUnit) else PriceUnit.parse(to_unit)
    label = f"{from_pu}->{to_pu}"
    if commodity:
        label += f"[{commodity}]"

    # Identity: same currency + unit + period -> nothing to do.
    if from_pu == to_pu:
        return ConversionResult(value=value, applied=False, note="unchanged")

    try:
        out = convert_price(
            value,
            from_pu,
            to_pu,
            commodity,
            fx=fx,
            ffill_policy=ffill_policy,
            max_staleness=max_staleness,
        )
        return ConversionResult(value=out, applied=True, note=f"convert_price:{label}")
    except Exception as exc:
        # Currency-leg fallback: the full conversion failed, but if both sides
        # are currency-qualified with equal denominators the currency scale is
        # still well-defined (e.g. convert_price can't do convfactor('RIN',
        # 'RIN'), but 'USc/RIN'->'USD/RIN' is a clean /100).
        if (
            from_pu.is_currency_qualified
            and to_pu.is_currency_qualified
            and from_pu.quantity_leg() == to_pu.quantity_leg()
        ):
            try:
                out = convert_currency_leg(value, from_pu, to_pu, fx=fx)
                return ConversionResult(
                    value=out, applied=True, note=f"currency_leg:{label}"
                )
            except Exception:
                pass  # fall through to on_error handling
        if on_error == "keep":
            return ConversionResult(value=value, applied=False, note=f"kept-raw:{exc}")
        raise
