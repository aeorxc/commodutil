"""
Modern implementation of commodity unit conversions using Pint.
Clean-slate design with no backward-compatibility constraints.
"""

import pint
from pint.errors import DimensionalityError
from typing import Union, Optional
from dataclasses import dataclass
import pandas as pd
from functools import lru_cache

# Initialize pint with custom definitions
ureg = pint.UnitRegistry()

# Define oil & gas specific units
ureg.define("barrel = 158.987294928 liter = bbl")
ureg.define("gallon = 3.785411784 liter = gal")
ureg.define("metric_ton = 1000 kilogram = mt")
ureg.define("kiloton = 1000 metric_ton = kt")
ureg.define("cubic_kilometer = 1e9 meter**3 = km3")  # 1 km^3 = 1 billion m^3
ureg.define("gigajoule = 1e9 joule = gj = GJ")
ureg.define("petajoule = 1e15 joule = pj = PJ")
ureg.define("billion_cubic_meter = 1e9 meter**3 = bcm = BCM")
ureg.define("billion_cubic_foot = 1e9 foot**3 = bcf = BCF")
ureg.define("tonne_of_oil_equivalent = 41.868e9 joule = toe = TOE")
ureg.define("million_tonne_of_oil_equivalent = 1e6 tonne_of_oil_equivalent = Mtoe")
ureg.define("barrel_of_oil_equivalent = 6.119e9 joule = boe = BOE")
ureg.define("million_barrel_of_oil_equivalent = 1e6 barrel_of_oil_equivalent = Mboe")
ureg.define("megatonne = 1e6 metric_ton = Mt")
# Pint defines `Btu` (= 1055.056 J) and `therm` (= 1e5 Btu) but NOT `MMBtu`.
# The natgas / NGL pages quote in $/MMBtu, so register it explicitly (and the
# normalize_unit helper already maps 'MMBTU' -> 'MMBtu').
ureg.define("million_british_thermal_unit = 1e6 Btu = MMBtu")


@dataclass
class Commodity:
    """Represents a commodity with its physical properties"""

    name: str
    density: pint.Quantity  # kg/L or API gravity
    energy_content: Optional[pint.Quantity] = None  # GJ/m^3 or similar

    def __post_init__(self):
        # Ensure quantities have correct dimensions
        if not isinstance(self.density, pint.Quantity):
            self.density = self.density * ureg.kg / ureg.liter
        if self.energy_content and not isinstance(self.energy_content, pint.Quantity):
            self.energy_content = self.energy_content * ureg.GJ / ureg.m**3


# Define commodities with their properties and correct industry factors
COMMODITIES = {
    # Crude oil (BP approximate conversion factors)
    # 1 mt ≈ 7.33 bbl and ≈ 1.165 kL => density ≈ 0.85809 kg/L
    "crude": Commodity("crude", 0.85809151 * ureg.kg / ureg.L, None),
    # Light ends - tuned to match kbbl/kt figures exactly
    "gasoline": Commodity(
        "gasoline", 0.755079324 * ureg.kg / ureg.L, 33.7898 * ureg.GJ / ureg.m**3
    ),  # BP: 44.75 GJ/t
    "naphtha": Commodity(
        "naphtha", 0.706720311 * ureg.kg / ureg.L, None
    ),  # 8.90 kbbl/kt
    "ethanol": Commodity(
        "ethanol", 0.755079324 * ureg.kg / ureg.L, 21 * ureg.GJ / ureg.m**3
    ),  # 8.33 kbbl/kt
    # Middle distillates
    "diesel": Commodity(
        "diesel", 0.844269902 * ureg.kg / ureg.L, 36.624428 * ureg.GJ / ureg.m**3
    ),  # BP: 43.38 GJ/t
    "jet": Commodity(
        "jet", 0.798199336 * ureg.kg / ureg.L, 35.056915 * ureg.GJ / ureg.m**3
    ),  # BP: 43.92 GJ/t
    "fame": Commodity(
        "fame", 0.892001564 * ureg.kg / ureg.L, 33 * ureg.GJ / ureg.m**3
    ),  # 7.051345 kbbl/kt
    "hvo": Commodity(
        "hvo", 0.781731391 * ureg.kg / ureg.L, 34 * ureg.GJ / ureg.m**3
    ),  # 8.046 kbbl/kt
    # Heavy products
    "vgo": Commodity("vgo", 0.911566778 * ureg.kg / ureg.L, None),  # 6.90 kbbl/kt
    "fuel_oil": Commodity(
        "fuel_oil", 0.990521381 * ureg.kg / ureg.L, 41.175974 * ureg.GJ / ureg.m**3
    ),  # BP: 41.57 GJ/t
    # LPG and Natural gas (liquefied)
    "lpg": Commodity(
        "lpg", 0.541 * ureg.kg / ureg.L, 24.96715 * ureg.GJ / ureg.m**3
    ),  # BP: LPG 46.15 GJ/t
    "natgas": Commodity(
        "natgas", 0.542225066 * ureg.kg / ureg.L, 26.137 * ureg.GJ / ureg.m**3
    ),
    # Natural gas (gaseous, pipeline): BP approx 36 PJ per bcm => 0.036 GJ/m**3
    "natural_gas": Commodity(
        "natural_gas", 0.0 * ureg.kg / ureg.L, 0.036 * ureg.GJ / ureg.m**3
    ),  # LNG figures
    # Light gases (NGLs as discrete species — supersede the generic 'lpg' alias
    # for unit conversions where pure-component density/HHV matter, e.g.
    # $/gal <-> $/MMBtu for the MB OPIS futures (AC0, B0, AD0, A8I)).
    "ethane": Commodity(
        "ethane", 0.373 * ureg.kg / ureg.L, 18.4262 * ureg.GJ / ureg.m**3
    ),  # BP: 49.4 GJ/t; 1 gal ~ 0.0661 MMBtu (HHV)
    "propane": Commodity(
        "propane", 0.507 * ureg.kg / ureg.L, 25.5375 * ureg.GJ / ureg.m**3
    ),  # ~91,690 BTU/gal HHV; ~50.35 GJ/t
    "butane": Commodity(
        "butane", 0.574 * ureg.kg / ureg.L, 28.4400 * ureg.GJ / ureg.m**3
    ),  # n-butane; ~103,000 BTU/gal HHV liquid; ~49.5 GJ/t
    "isobutane": Commodity(
        "isobutane", 0.557 * ureg.kg / ureg.L, 27.7700 * ureg.GJ / ureg.m**3
    ),  # iso-C4; ~99,000 BTU/gal HHV liquid
    "natural_gasoline": Commodity(
        "natural_gasoline", 0.6675 * ureg.kg / ureg.L, 31.4 * ureg.GJ / ureg.m**3
    ),  # pentanes-plus; ~118,000 BTU/gal HHV
    # BP product basket (optional reference)
    "product_basket": Commodity(
        "product_basket", 0.781 * ureg.kg / ureg.L, 33.642356 * ureg.GJ / ureg.m**3
    ),
}

# Aliases for compatibility
ALIASES = {
    # Common synonyms and marketing terms
    "ulsd": "diesel",
    "gasoil": "diesel",
    "gas_oil": "diesel",
    "gas oil": "diesel",
    "go": "diesel",
    "kerosene": "jet",
    # Motor gasoline
    "gas": "gasoline",
    "mogas": "gasoline",
    # Fuel oil
    "fueloil": "fuel_oil",
    "fuel oil": "fuel_oil",
    "fo": "fuel_oil",
    # Crude
    "crude oil": "crude",
    "crudeoil": "crude",
    # LPG and nat gas
    # Note: 'propane'/'butane'/'isobutane'/'natural_gasoline' are now first-class
    # entries in COMMODITIES (added 2026-05 for $/gal<->$/MMBtu work on the
    # MB OPIS NGL futures). The legacy 'propane -> lpg' alias is gone; existing
    # callers using commodity='lpg' for a generic LPG blend still work via the
    # 'lpg' entry. NGL-species aliases below cover common spellings.
    "n_butane": "butane",
    "n-butane": "butane",
    "normal_butane": "butane",
    "normal butane": "butane",
    "iso_butane": "isobutane",
    "iso-butane": "isobutane",
    "i_butane": "isobutane",
    "i-butane": "isobutane",
    "natgaso": "natural_gasoline",
    "nat_gasoline": "natural_gasoline",
    "pentanes_plus": "natural_gasoline",
    "lng": "natgas",  # LNG (liquefied natural gas)
    "ng": "natural_gas",  # pipeline natural gas (gaseous)
    "naturalgas": "natural_gas",
    "nat_gas": "natural_gas",
}


class CommodityConverter:
    """Clean, modern interface for commodity unit conversions"""

    def __init__(self):
        self.ureg = ureg
        self.commodities = COMMODITIES
        self.aliases = ALIASES

    @lru_cache(maxsize=128)
    def get_commodity(self, name: str) -> Commodity:
        """Get commodity object, resolving aliases"""
        name = name.lower()
        name = self.aliases.get(name, name)
        if name not in self.commodities:
            raise ValueError(f"Unknown commodity: {name}")
        return self.commodities[name]

    def convert(
        self,
        value: Union[float, pd.Series],
        from_unit: str,
        to_unit: str,
        commodity: Optional[str] = None,
    ) -> Union[float, pd.Series]:
        """
        Convert between units, using commodity properties when needed

        Examples:
            # Simple unit conversion (no commodity needed)
            convert(100, 'bbl', 'L')

            # Mass to volume (needs commodity density)
            convert(100, 'kt', 'bbl', commodity='diesel')

            # Energy conversions
            convert(1000, 'm^3', 'GJ', commodity='diesel')

            # With pandas Series and daily rates
            convert(series, 'kt/month', 'bbl/day', commodity='gasoline')
        """
        # Normalize and parse units to handle daily/monthly rates
        from_unit = self._normalize_unit(from_unit)
        to_unit = self._normalize_unit(to_unit)
        from_rate = self._parse_rate_unit(from_unit)
        to_rate = self._parse_rate_unit(to_unit)

        # Get base units
        from_base = from_rate["base"]
        to_base = to_rate["base"]

        # Create quantity
        if isinstance(value, pd.Series):
            result = self._convert_series(
                value,
                from_base,
                to_base,
                commodity,
                from_rate["period"],
                to_rate["period"],
            )
        else:
            result = self._convert_scalar(value, from_base, to_base, commodity)
            if from_rate["period"] or to_rate["period"]:
                factor = self._rate_factor_scalar(
                    from_rate["period"], to_rate["period"]
                )
                result = result * factor

        return result

    def _convert_scalar(
        self, value: float, from_unit: str, to_unit: str, commodity: Optional[str]
    ) -> float:
        """Convert a scalar value across mass/volume/energy using commodity context when needed."""
        from_unit = self._normalize_unit(from_unit)
        to_unit = self._normalize_unit(to_unit)
        qty = value * self.ureg(from_unit)

        # Try direct conversion first
        try:
            return qty.to(to_unit).magnitude
        except DimensionalityError:
            pass

        # Determine unit types
        is_from_energy = self._is_energy(from_unit)
        is_to_energy = self._is_energy(to_unit)
        is_from_mass = self._is_mass(from_unit)
        is_to_mass = self._is_mass(to_unit)
        is_from_volume = self._is_volume(from_unit)
        is_to_volume = self._is_volume(to_unit)

        # Energy conversions
        if is_from_energy or is_to_energy:
            if not commodity:
                raise ValueError("Commodity required for energy conversion")
            comm = self.get_commodity(commodity)
            if not comm.energy_content:
                raise ValueError(f"No energy content defined for {commodity}")

            ec = comm.energy_content.to("J/m^3")

            if is_from_energy:
                energy_J = qty.to("J")
                # Energy -> Volume or Mass
                volume_m3 = (energy_J / ec).to("m^3")
                if is_to_volume:
                    return volume_m3.to(to_unit).magnitude
                elif is_to_mass:
                    density_kg_m3 = comm.density.to("kg/m^3")
                    if density_kg_m3.magnitude == 0:
                        raise ValueError(
                            f"Density not defined for {commodity}; cannot convert energy to mass"
                        )
                    mass_kg = (volume_m3 * density_kg_m3).to("kg")
                    return mass_kg.to(to_unit).magnitude
                else:
                    raise ValueError(f"Cannot convert energy to {to_unit}")
            else:
                # Volume/Mass -> Energy
                if is_from_mass:
                    density_kg_m3 = comm.density.to("kg/m^3")
                    if density_kg_m3.magnitude == 0:
                        raise ValueError(
                            f"Density not defined for {commodity}; cannot convert mass to energy"
                        )
                    mass_kg = qty.to("kg")
                    volume_m3 = (mass_kg / density_kg_m3).to("m^3")
                elif is_from_volume:
                    volume_m3 = qty.to("m^3")
                else:
                    raise ValueError(f"Cannot convert {from_unit} to energy")
                energy_J = (volume_m3 * ec).to("J")
                return energy_J.to(to_unit).magnitude

        # Mass <-> Volume conversions require density
        if (is_from_mass and is_to_volume) or (is_from_volume and is_to_mass):
            if not commodity:
                raise ValueError(f"Commodity required for {from_unit} to {to_unit}")
            comm = self.get_commodity(commodity)
            density_kg_L = comm.density.to("kg/L")
            density_kg_m3 = comm.density.to("kg/m^3")
            if density_kg_L.magnitude == 0 or density_kg_m3.magnitude == 0:
                raise ValueError(
                    f"Density not defined for {commodity}; cannot convert between mass and volume"
                )
            if is_from_mass and is_to_volume:
                mass_kg = qty.to("kg")
                volume_L = (mass_kg / density_kg_L).to("L")
                return volume_L.to(to_unit).magnitude
            else:
                volume_L = qty.to("L")
                mass_kg = (volume_L * density_kg_L).to("kg")
                return mass_kg.to(to_unit).magnitude

        raise ValueError(
            f"Cannot convert from {from_unit} to {to_unit} - incompatible dimensions"
        )

    def _convert_series(
        self,
        series: pd.Series,
        from_unit: str,
        to_unit: str,
        commodity: Optional[str],
        from_period: Optional[str],
        to_period: Optional[str],
    ) -> pd.Series:
        """Convert a pandas Series with optional rate handling.

        - Month conversions use the index's actual days_in_month when available.
        - Other time conversions use standard averages (365.25 days/year, 30.4375 days/month).
        """
        result = series.copy()

        # Handle period conversions for rates
        if from_period or to_period:
            if from_period != to_period:
                if hasattr(series.index, "days_in_month") and (
                    (from_period == "day" and to_period == "month")
                    or (from_period == "month" and to_period == "day")
                ):
                    # Month-aware conversions using calendar days per month
                    if from_period == "day" and to_period == "month":
                        result = result * series.index.days_in_month
                    else:
                        result = result / series.index.days_in_month
                else:
                    # Fallback to scalar factor for other period conversions
                    factor = self._rate_factor_scalar(from_period, to_period)
                    result = result * factor

        # Apply unit conversion
        factor_units = self._convert_scalar(1.0, from_unit, to_unit, commodity)
        result = result * factor_units

        return result

    def _parse_rate_unit(self, unit: str) -> dict:
        """Parse units like 'bbl/day' or 'kt/month'."""
        if "/" in unit:
            base, period = unit.split("/", 1)
            base = self._normalize_unit(base)
            period = period.strip().lower().rstrip("s")  # day(s), month(s), year(s)
            return {"base": base, "period": period}
        return {"base": self._normalize_unit(unit), "period": None}

    def _rate_factor_scalar(
        self, from_period: Optional[str], to_period: Optional[str]
    ) -> float:
        """Scalar factor to convert between rate periods for scalars.

        Uses average calendar lengths when months/years are involved.
        - Average days per year: 365.25
        - Average days per month: 365.25 / 12 = 30.4375
        """
        if from_period == to_period:
            return 1.0
        if from_period is None and to_period is None:
            return 1.0
        if from_period is None or to_period is None:
            # Ambiguous to add/remove a time dimension for scalar; no-op to preserve behavior
            return 1.0

        avg_days_per_year = 365.25
        avg_days_per_month = avg_days_per_year / 12.0

        # Helper to express rates as per-day factors
        def per_day_factor(period: str) -> float:
            if period == "day":
                return 1.0
            if period == "year":
                return 1.0 / avg_days_per_year
            if period == "month":
                return 1.0 / avg_days_per_month
            # Fallback to Pint if it's a known time unit
            try:
                return (1 * (self.ureg(period) ** -1)).to(self.ureg.day**-1).magnitude
            except Exception:
                raise ValueError(f"Unsupported rate period: {period}")

        from_per_day = per_day_factor(from_period)
        to_per_day = per_day_factor(to_period)
        # Convert from per-from_period to per-to_period
        return from_per_day / to_per_day

    def _is_energy(self, unit: str) -> bool:
        try:
            (1 * self.ureg(unit)).to("J")
            return True
        except DimensionalityError:
            return False

    def _is_mass(self, unit: str) -> bool:
        try:
            (1 * self.ureg(unit)).to("kg")
            return True
        except DimensionalityError:
            return False

    def _is_volume(self, unit: str) -> bool:
        try:
            (1 * self.ureg(unit)).to("m^3")
            return True
        except DimensionalityError:
            return False

    def _normalize_unit(self, unit: str) -> str:
        """Normalize common aliases and fix encoding issues.

        - Map 'm��'/'m³'/'m**3'/'cubic_meter' -> 'm^3'
        - Map energy aliases 'BTU' -> 'Btu', 'MMBTU' -> 'MMBtu'
        - Trim whitespace
        """
        if unit is None:
            return unit
        u = unit.strip()
        # Fix cubic meter notations and encoding issues
        replacements = {
            "m��": "m^3",
            "m³": "m^3",
            "m**3": "m^3",
            "cubic_meter": "m^3",
            "CUBIC_METER": "m^3",
        }
        for bad, good in replacements.items():
            u = u.replace(bad, good)

        # Additional robust normalizations using ASCII-only fallbacks
        if u.lower() == "m3":
            u = "m^3"
        # Handle rate-style variants like 'm3/day' or 'M3/day'
        u = u.replace("m3/", "m^3/").replace("M3/", "m^3/")
        # Energy unit common uppercase forms
        if u == "BTU":
            u = "Btu"
        if u == "MMBTU":
            u = "MMBtu"
        return u

    @property
    def available_commodities(self) -> list:
        """List all available commodities"""
        return list(self.commodities.keys())

    @property
    def available_units(self) -> list:
        """List common units for oil & gas"""
        return [
            # Volume
            "bbl",
            "barrel",
            "L",
            "liter",
            "m³",
            "cubic_meter",
            "gal",
            "gallon",
            "bcm",
            "bcf",
            # Mass
            "kg",
            "mt",
            "metric_ton",
            "kt",
            "kiloton",
            "t",
            "tonne",
            "Mt",
            # Energy
            "J",
            "GJ",
            "gigajoule",
            "MJ",
            "megajoule",
            "PJ",
            "toe",
            "Mtoe",
            "boe",
            "Mboe",
            "BTU",
            "MMBTU",
            # Rates
            "bbl/day",
            "kt/month",
            "m³/day",
            "mt/year",
        ]


# Global converter instance for convenience
converter = CommodityConverter()


# Convenience functions for direct use
def convert(value, from_unit: str, to_unit: str, commodity: Optional[str] = None):
    """Convert values between units"""
    return converter.convert(value, from_unit, to_unit, commodity)


def convfactor(from_unit: str, to_unit: str, commodity: Optional[str] = None) -> float:
    """Get conversion factor between units"""
    return converter.convert(1.0, from_unit, to_unit, commodity)


def convert_price(
    value: Union[float, pd.Series],
    from_unit: str,
    to_unit: str,
    commodity: Optional[str] = None,
) -> Union[float, pd.Series]:
    """
    Convert price values ($/unit) between units using commodity-aware quantity factors.

    Price conversion is the inverse of quantity conversion:
        price_to = price_from / convfactor(from_unit, to_unit, commodity)

    Examples:
        # Gasoline: $/mt -> $/bbl (divide by ~8.33)
        convert_price(100, 'mt', 'bbl', 'gasoline')  # ~12.0

        # US gallon to barrel: $/gal -> $/bbl (multiply by 42)
        convert_price(2.5, 'gal', 'bbl')  # ~105.0
    """
    factor = convfactor(from_unit, to_unit, commodity)
    if factor is None or factor == 0:
        return value
    if isinstance(value, pd.Series):
        return value / factor
    return value / factor


# ---- FX scaling helpers for $/{currency}/{energy-or-volume-or-mass} prices ----
#
# Design note (2026-05): callers building cross-currency / cross-unit energy
# prices (e.g. TTF EUR/MWh -> $/MMBtu, NBP GBp/therm -> $/MMBtu) previously had
# to roll their own helper because `convert_price` is FX-agnostic.
# `convert_price_with_fx` keeps the FX leg explicit (an FX *series* or scalar,
# fetched and aligned by the caller — convfactors does not pull from any FX
# feed itself) and delegates the unit-leg to `convert_price`. This keeps the
# single-conversion-call ergonomics promised by the saved feedback memory
# without coupling commodutil to any pricing feed.

_FRACTIONAL_CURRENCY_DIVISORS = {
    # Quoted-in-fractional-units of a major currency. Multiplying by the FX rate
    # for the major currency (e.g. GBP/USD) AND dividing by 100 lifts the
    # fractional-currency quote to its USD equivalent.
    "GBp": 100.0,  # pence in sterling, e.g. NBP at "70 GBp/therm"
    "USc": 100.0,  # US cents
    "EUc": 100.0,  # euro cents
    "JPy": 100.0,  # rare, but for symmetry
}


def _split_currency_unit(token: str) -> tuple[str, str]:
    """Split a 'CCY/unit' token into ('CCY', 'unit'). Returns ('', token) if no slash."""
    if "/" not in token:
        return "", token
    ccy, unit = token.split("/", 1)
    return ccy.strip(), unit.strip()


def convert_price_with_fx(
    value: Union[float, pd.Series],
    from_unit: str,
    to_unit: str,
    fx: Union[float, pd.Series, None] = None,
    commodity: Optional[str] = None,
) -> Union[float, pd.Series]:
    """
    Convert a price across both a currency leg AND a unit leg in one call.

    Use this when you have prices quoted in a foreign currency per a non-USD-
    standard unit (e.g. EUR/MWh, GBp/therm) and you want $/MMBtu.

    The currency token is the prefix before the '/' (e.g. 'EUR' in 'EUR/MWh',
    'GBp' in 'GBp/therm'). Pass `fx` as the rate (or pandas Series of rates)
    quoted as **target/source** — i.e. USD-per-foreign-currency. So for
    EUR/MWh -> $/MMBtu, pass `fx = EURUSD` (the EURUSD spot, ~1.07).
    For GBp/therm -> $/MMBtu, pass `fx = GBPUSD` — the fractional 'GBp' suffix
    is detected automatically and divided by 100.

    `from_unit` / `to_unit` are EITHER bare units ('mt', 'bbl', 'gal', 'MMBtu',
    'MWh', 'therm') OR currency-qualified ('USD/bbl', 'EUR/MWh', 'GBp/therm').
    If a currency is supplied on `to_unit`, it is currently assumed to be USD
    (passing anything else raises ValueError — extend in future if non-USD
    targets are needed).

    Examples
    --------
    >>> # TTF EUR/MWh -> $/MMBtu (EURUSD = 1.07)
    >>> convert_price_with_fx(35.0, 'EUR/MWh', 'USD/MMBtu', fx=1.07)  # ~10.98
    >>>
    >>> # NBP GBp/therm -> $/MMBtu (GBPUSD = 1.25)
    >>> # GBp prefix auto-detected & /100; therm->MMBtu *10; net /10
    >>> convert_price_with_fx(80.0, 'GBp/therm', 'USD/MMBtu', fx=1.25)  # ~10.00
    >>>
    >>> # Time-varying FX with a pandas Series
    >>> p = pd.Series([35.0, 36.5, 34.2], index=pd.date_range('2026', periods=3))
    >>> fx_series = pd.Series([1.07, 1.08, 1.06], index=p.index)
    >>> convert_price_with_fx(p, 'EUR/MWh', 'USD/MMBtu', fx=fx_series)

    Notes
    -----
    - If FX is None and the currency token is USD-equivalent, this falls back
      to `convert_price` (pure unit-leg).
    - The FX series is aligned to `value`'s index when both are pandas Series.
    - Returning a numeric type (not None) is always preferred — if `fx` is
      missing for some dates in a Series, those rows become NaN (pandas default).
    """
    from_ccy, from_bare_unit = _split_currency_unit(from_unit)
    to_ccy, to_bare_unit = _split_currency_unit(to_unit)

    # Validate target currency — explicit USD only for now
    if to_ccy and to_ccy.upper() not in {"USD", "$"}:
        raise ValueError(
            f"convert_price_with_fx currently only supports USD/* as target; got '{to_unit}'"
        )

    # Unit-leg conversion (no FX yet — uses commodity factors)
    unit_converted = convert_price(value, from_bare_unit, to_bare_unit, commodity)

    # If no source currency or it's already USD, no FX leg needed
    if not from_ccy or from_ccy.upper() in {"USD", "$"}:
        return unit_converted

    # Apply FX leg
    if fx is None:
        raise ValueError(
            f"FX rate required to convert {from_unit} -> {to_unit} "
            f"(source currency '{from_ccy}' is non-USD)"
        )

    fractional_divisor = _FRACTIONAL_CURRENCY_DIVISORS.get(from_ccy, 1.0)

    if isinstance(unit_converted, pd.Series) and isinstance(fx, pd.Series):
        # Align FX to value index
        fx_aligned = fx.reindex(unit_converted.index).ffill()
        return unit_converted * fx_aligned / fractional_divisor

    return unit_converted * fx / fractional_divisor


def list_commodities():
    """List all available commodities"""
    return converter.available_commodities


def list_units():
    """List common units"""
    # Return normalized forms to avoid encoding issues
    return [converter._normalize_unit(u) for u in converter.available_units]


# Example usage
if __name__ == "__main__":
    print("Modern Commodity Converter Examples\n" + "=" * 50)

    # Simple conversions
    print("\n1. Simple unit conversions (no commodity needed):")
    print(f"100 bbl = {convert(100, 'bbl', 'L'):.0f} L")
    print(f"1000 L = {convert(1000, 'L', 'bbl'):.2f} bbl")

    # Commodity-specific conversions
    print("\n2. Mass-Volume conversions (needs commodity):")
    print(f"100 kt diesel = {convert(100, 'kt', 'bbl', 'diesel'):.0f} bbl")
    print(f"1000 bbl gasoline = {convert(1000, 'bbl', 'mt', 'gasoline'):.2f} mt")

    # Energy conversions (now implemented across mass/volume/energy)
    print("\n3. Energy conversions:")
    print("(Energy conversions implemented for mass/volume/energy)")

    # Series with daily rates
    print("\n4. Pandas Series with rate conversions:")
    dates = pd.date_range("2024-01", periods=3, freq="MS")
    series = pd.Series([100, 110, 105], index=dates)
    result = convert(series, "kt/month", "bbl/day", "diesel")
    print(f"January: {result.iloc[0]:.0f} bbl/day")

    # Available commodities
    print(f"\n5. Available commodities: {', '.join(list_commodities())}")

    # Error handling
    print("\n6. Error handling:")
    try:
        convert(100, "kt", "bbl")  # Missing commodity
    except ValueError as e:
        print(f"Error: {e}")

    print("\n" + "=" * 50)
    print("Key improvements over original:")
    print("• Type hints and dataclasses for clarity")
    print("• Automatic dimensional analysis")
    print("• Clean separation of concerns")
    print("• Caching for performance")
    print("• Better error messages")
    print("• Extensible commodity definitions")
    print("• Modern Python patterns")
