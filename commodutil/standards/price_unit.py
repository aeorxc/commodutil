"""commodutil.standards.price_unit: the ``PriceUnit`` value type.

A ``PriceUnit`` is the structured form of a price-unit string such as
``'USD/bbl'``, ``'GBp/therm'``, ``'kt/month'`` or a bare ``'bbl'``. It splits a
quote into three independently-usable legs:

* ``currency``      — canonical currency token per ``standards.currency``
                      (``'USD'``, ``'GBp'``, ``'$'``, ...), or ``None`` for a
                      bare / rate-only unit.
* ``quantity_unit`` — the bare physical unit token (``'bbl'``, ``'MWh'`` ...).
* ``period``        — the rate period for forms like ``'bbl/day'`` (``'day'`` /
                      ``'month'`` / ``'year'``), or ``None``.

Design goals (conversion-architecture-plan.md, Phase 3.1):

* ``PriceUnit.parse(str(pu)) == pu`` is guaranteed for any ``PriceUnit`` — the
  string form is canonical and losslessly round-trips.
* ``parse`` is built directly on ``standards.currency.split_currency_unit`` plus
  the same rate-suffix logic ``convfactors`` uses, so the legs it produces are
  byte-for-byte what those callers computed before — letting ``convert`` /
  ``convert_price`` adopt it at the boundary with zero behaviour change.
* The **currency leg is usable on its own** (``major_currency``,
  ``fractional_divisor``), which is what lets a currency-only conversion (USc ->
  USD) succeed even when the quantity denominator is non-physical/unknown (the
  RIN class of bug — see ``convfactors.convert_currency_leg``).

Pure stdlib: depends only on ``standards.currency`` (itself pure stdlib). No
pint, no pandas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from commodutil.standards.currency import (
    FRACTIONAL_CURRENCY_DIVISORS,
    fractional_to_major,
    split_currency_unit,
)


@dataclass(frozen=True)
class PriceUnit:
    """Immutable, structured representation of a price-unit string.

    See module docstring. Construct directly with canonical legs or, more
    commonly, via :meth:`parse`. Directly-constructed instances are normalised
    (whitespace stripped; period lower-cased and de-pluralised; empty strings
    coerced to ``None``) so equality and the round-trip guarantee hold
    regardless of how the instance was built.
    """

    currency: Optional[str] = None
    quantity_unit: str = ""
    period: Optional[str] = None

    def __post_init__(self) -> None:
        # frozen dataclass -> mutate via object.__setattr__ during init only.
        object.__setattr__(self, "quantity_unit", str(self.quantity_unit).strip())
        if self.currency is not None:
            currency = str(self.currency).strip()
            object.__setattr__(self, "currency", currency or None)
        if self.period is not None:
            # Mirror convfactors._parse_rate_unit: day(s)/month(s)/year(s).
            period = str(self.period).strip().lower().rstrip("s")
            object.__setattr__(self, "period", period or None)

    # ---- construction -----------------------------------------------------

    @classmethod
    def parse(cls, text: Union["PriceUnit", str]) -> "PriceUnit":
        """Parse a price-unit string into a ``PriceUnit``.

        Handles every form the codebase round-trips today:
          * bare unit:            ``'bbl'``, ``'MMBtu'``, ``'m^3'``
          * rate unit:            ``'bbl/day'``, ``'kt/month'`` (currency=None)
          * currency-qualified:   ``'USD/bbl'``, ``'GBp/therm'``, ``'$/bbl'``
          * currency + rate:      ``'USD/bbl/day'`` (accepted for completeness;
            not currently emitted anywhere in the codebase)

        A currency prefix is recognised only when it is a valid token per
        ``standards.currency`` (via ``split_currency_unit``), so a bare rate
        unit like ``'bbl/day'`` is NOT mis-parsed as currency ``'bbl'`` — it
        stays quantity ``'bbl'`` / period ``'day'``.

        Passing an existing ``PriceUnit`` returns it unchanged.
        """
        if isinstance(text, cls):
            return text
        if text is None:
            raise ValueError("PriceUnit.parse requires a non-None unit string")
        raw = str(text).strip()
        if not raw:
            raise ValueError("PriceUnit.parse requires a non-empty unit string")

        # split_currency_unit returns ('', raw) when the prefix is not a valid
        # currency token, leaving bare rate units intact.
        currency, bare = split_currency_unit(raw)
        quantity_unit, period = cls._split_period(bare)
        return cls(
            currency=currency or None, quantity_unit=quantity_unit, period=period
        )

    @staticmethod
    def _split_period(bare: str) -> tuple[str, Optional[str]]:
        """Split a bare unit into (quantity_unit, period), mirroring
        convfactors._parse_rate_unit's period handling."""
        if "/" in bare:
            base, _, period = bare.partition("/")
            period = period.strip().lower().rstrip("s")
            return base.strip(), (period or None)
        return bare.strip(), None

    # ---- string form ------------------------------------------------------

    def __str__(self) -> str:
        denom = self.quantity_unit
        if self.period:
            denom = f"{denom}/{self.period}"
        if self.currency:
            return f"{self.currency}/{denom}"
        return denom

    # ---- leg accessors ----------------------------------------------------

    def currency_leg(self) -> Optional[str]:
        """The currency token, or ``None`` for a bare/rate-only unit."""
        return self.currency

    def quantity_leg(self) -> str:
        """The full quantity denominator string (unit plus any rate period).

        Byte-identical to the bare-unit element returned by
        ``standards.currency.split_currency_unit`` — e.g. ``'bbl'`` for
        ``'USD/bbl'``, ``'bbl/day'`` for ``'bbl/day'``.
        """
        denom = self.quantity_unit
        if self.period:
            denom = f"{denom}/{self.period}"
        return denom

    @property
    def is_currency_qualified(self) -> bool:
        """True when a currency leg is present."""
        return self.currency is not None

    @property
    def major_currency(self) -> Optional[str]:
        """The parent major currency (USc -> USD, GBp -> GBP, $ -> USD), or
        ``None`` when there is no currency leg."""
        if not self.currency:
            return None
        return fractional_to_major(self.currency)

    @property
    def fractional_divisor(self) -> float:
        """Divisor lifting a fractional-currency quote to its major unit
        (GBp/USc/... -> 100.0); 1.0 for major or absent currencies."""
        return FRACTIONAL_CURRENCY_DIVISORS.get(self.currency, 1.0)


__all__ = ["PriceUnit"]
