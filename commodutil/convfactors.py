"""
Modern implementation of commodity unit conversions using Pint.
Clean-slate design with no backward-compatibility constraints.
"""

import logging
import pint
from pint.errors import DimensionalityError
from typing import Union, Optional
from dataclasses import dataclass
import pandas as pd
from functools import lru_cache

from commodutil.standards import currency as _currency
from commodutil.standards.price_unit import PriceUnit
from commodutil.standards.unit_registry import PINT_DEFINITIONS as _PINT_DEFINITIONS
from commodutil.standards.units import to_pint_token as _to_pint_token

logger = logging.getLogger(__name__)

# Initialize pint with custom definitions.
#
# NOTE: kept case-sensitive on purpose. `pint.UnitRegistry(case_sensitive=False)`
# looks tempting (it would drop the hand-registered casing aliases below), but it
# is UNSAFE for this registry: pint's case-folded lookup makes our domain units
# collide with SI units that differ only by case -- `kt`/`mt`/`gal` vs `kT`/`mT`
# (kilo/milli-tesla, T = tesla) and `Gal` (galileo, cm/s^2), while lowercase
# `mw`/`mwh` resolve to milli-watt instead of mega-. Collision winners are chosen
# from a set and are NON-DETERMINISTIC (PYTHONHASHSEED-dependent), so `kt`->`km3`
# etc. intermittently raise "incompatible dimensions". We therefore register the
# exact spellings we need explicitly and leave the registry case-sensitive.
ureg = pint.UnitRegistry()

# Define oil & gas specific units.
#
# The exact pint definitions (and their required ordering) live as the
# `pint_specs` of each row in commodutil.standards.unit_registry; they are
# applied here in registry order. This includes the load-bearing re-routes and
# case spellings that keep this registry correct:
#   * `oil_barrel` (158.987294928 L = 42 US gal) with `bbl` routed to it —
#     pint's default `barrel` is the US dry barrel (~119.24 L), left untouched.
#   * `metric_ton = mt` — pint's `mt` would otherwise be milli-tonne.
#   * MMBtu, plus case aliases (mmbtu/MMBTU, therm/THERM, btu) and lowercase /
#     all-caps power spellings (mw, mwh, MWH) — the registry stays CASE-SENSITIVE
#     (kt/kilotesla etc. collide non-deterministically under case-folding).
# To add or change a unit definition, edit its UnitRow in the registry.
for _definition in _PINT_DEFINITIONS:
    ureg.define(_definition)


@dataclass
class Commodity:
    """Represents a commodity with its physical properties.

    `density` is `Optional[pint.Quantity]`: `None` means "no liquid density
    defined" (e.g. pipeline natural gas — it's a gas, not a liquid, so
    mass<->volume conversion is undefined and must raise).
    Previously this used `0.0 kg/L` as a sentinel; magnitude==0 checks
    were scattered through the codebase. `None` makes the intent explicit.

    `energy_content` may be a pint Quantity of EITHER dimensionality:
      - `[energy]/[volume]` (e.g. GJ/m^3) — the default for liquids; registers a
        `[volume] <-> [energy]` transformation (and chains to mass via density).
      - `[energy]/[mass]` (e.g. GJ/t) — for density-less solids like coal;
        registers a DIRECT `[mass] <-> [energy]` transformation so `mt <-> MMBtu`
        works with `density=None`, while `mt <-> bbl` still correctly raises.
    A bare float is coerced to the volume-basis default (GJ/m^3) for back-compat.
    """

    name: str
    density: Optional[pint.Quantity] = None  # kg/L or API gravity; None = not a liquid
    # [energy]/[volume] (GJ/m^3, liquids) OR [energy]/[mass] (GJ/t, e.g. coal)
    energy_content: Optional[pint.Quantity] = None

    def __post_init__(self):
        # Ensure quantities have correct dimensions
        if self.density is not None and not isinstance(self.density, pint.Quantity):
            self.density = self.density * ureg.kg / ureg.liter
        # Bare float -> volume-basis default (GJ/m^3). Callers wanting a mass
        # basis pass an explicit Quantity (e.g. `26.377 * ureg.GJ / ureg.mt`),
        # which is left untouched here and dispatched on in _commodity_context.
        if self.energy_content is not None and not isinstance(
            self.energy_content, pint.Quantity
        ):
            self.energy_content = self.energy_content * ureg.GJ / ureg.m**3


# -----------------------------------------------------------------------------
# Calorific basis policy: GROSS (higher heating value, HHV).
#
# `energy_content` for every liquid commodity below is stated on a GROSS/HHV
# basis, anchored on the U.S. EIA Monthly Energy Review Appendix A, Table A1
# ("Approximate Heat Content of Petroleum and Biofuels", explicitly gross —
# "Gross heat content rates are applied in all Btu calculations for the Monthly
# Energy Review") and, for the NGL species, GPA/EIA figures (EIA A1 which cites
# NIST enthalpy-of-combustion + API liquid densities). NGL/natural-gas hubs
# (Henry Hub, TTF, NBP, JKM) all quote HHV, so a cross-fuel $/MMBtu comparison
# is only internally consistent if every commodity is on the same gross basis.
#
# The stored value is GJ/m^3 (energy per unit VOLUME), set directly from the
# EIA gross figure in MMBtu/bbl via the fixed, density-independent conversion
#     GJ/m^3 = MMBtu/bbl * 1.055056 GJ/MMBtu / 0.158987294928 m^3/bbl
# so that convfactor('bbl','MMBtu', <commodity>) reproduces the EIA number
# exactly. Densities are NOT part of this conversion; they remain the
# BP/Energy-Institute values unchanged (the resulting GJ/t is therefore a
# derived reporting figure = energy_content / density). Each entry's comment
# records the source EIA MMBtu/bbl figure and the implied gross GJ/t.
#
# Historical note: several products (diesel/naphtha/jet/gasoline/fuel_oil) and
# the biofuels previously carried BP-style NET (NCV) energy contents, which
# understated $/MMBtu by ~1-12% vs gross. Standardising on gross was the
# owner-approved decision (conversion-architecture-plan.md, decision 2b).
#
# NOT covered by this policy (see per-entry comments / flagged for desk review):
# crude (kept on its documented BP gross basis), the LPG/LNG blends (`lpg`,
# `natgas`), gaseous `natural_gas`, and `product_basket` — all blend- or
# construction-dependent rather than single-species EIA rows.
# -----------------------------------------------------------------------------
# Define commodities with their properties and correct industry factors
COMMODITIES = {
    # Crude oil (BP approximate conversion factors)
    # 1 mt ≈ 7.33 bbl and ≈ 1.165 kL => density ≈ 0.85809 kg/L
    "crude": Commodity(
        "crude", 0.85809151 * ureg.kg / ureg.L, 39.043 * ureg.GJ / ureg.m**3
    ),  # HHV; BP/EI ~45.5 GJ/t gross (implies 5.883 MMBtu/bbl). EIA Table A2 crude ~5.69-5.80 MMBtu/bbl (US, lighter) -> commodutil is ~1-3% higher; kept on documented BP world-crude gross basis, not changed
    # Light ends - tuned to match kbbl/kt figures exactly
    "gasoline": Commodity(
        "gasoline", 0.755079324 * ureg.kg / ureg.L, 34.653728 * ureg.GJ / ureg.m**3
    ),  # HHV; EIA motor gasoline blending components (2007+, GREET) 5.222 MMBtu/bbl -> 45.89 GJ/t gross (was BP NCV 44.75; RBOB/blendstock basis, not E10-finished ~5.05)
    "naphtha": Commodity(
        "naphtha", 0.706720311 * ureg.kg / ureg.L, 34.826266 * ureg.GJ / ureg.m**3
    ),  # HHV; EIA petrochemical naphtha <401F 5.248 MMBtu/bbl -> 49.28 GJ/t gross (was NCV 44.90)
    "ethanol": Commodity(
        "ethanol", 0.7937 * ureg.kg / ureg.L, 23.485167 * ureg.GJ / ureg.m**3
    ),  # HHV; EIA fuel ethanol (undenatured) 3.539 MMBtu/bbl. Density: ASTM D4806 undenatured fuel ethanol SG 60/60F lower bound 0.7937 (Energy Transfer spec note; IEA AMF rounds ethanol density to 0.79 kg/L). Energy_content is per-volume, so bbl->MMBtu stays fixed while mt factors move.
    # Middle distillates
    "diesel": Commodity(
        "diesel", 0.844269902 * ureg.kg / ureg.L, 38.290312 * ureg.GJ / ureg.m**3
    ),  # HHV; EIA distillate (15 ppm & under) 5.770 MMBtu/bbl -> 45.35 GJ/t gross (was BP NCV 43.38)
    "jet": Commodity(
        "jet", 0.798199336 * ureg.kg / ureg.L, 37.626702 * ureg.GJ / ureg.m**3
    ),  # HHV; EIA jet fuel kerosene-type 5.670 MMBtu/bbl -> 47.14 GJ/t gross (was BP NCV 43.92)
    "fame": Commodity(
        "fame", 0.892001564 * ureg.kg / ureg.L, 35.562874 * ureg.GJ / ureg.m**3
    ),  # HHV; EIA biodiesel 5.359 MMBtu/bbl -> 39.87 GJ/t gross (was net ~37.00)
    "hvo": Commodity(
        "hvo", 0.781731391 * ureg.kg / ureg.L, 36.458748 * ureg.GJ / ureg.m**3
    ),  # HHV; EIA renewable diesel fuel 5.494 MMBtu/bbl -> 46.64 GJ/t gross (was net ~43.49)
    # Heavy products
    "vgo": Commodity("vgo", 0.911566778 * ureg.kg / ureg.L, None),  # 6.90 kbbl/kt
    "fuel_oil": Commodity(
        "fuel_oil", 0.990521381 * ureg.kg / ureg.L, 41.721177 * ureg.GJ / ureg.m**3
    ),  # HHV; EIA residual fuel oil 6.287 MMBtu/bbl -> 42.12 GJ/t gross (was BP NCV 41.57)
    # LPG and Natural gas (liquefied)
    "lpg": Commodity(
        "lpg", 0.541 * ureg.kg / ureg.L, 24.96715 * ureg.GJ / ureg.m**3
    ),  # BP: LPG 46.15 GJ/t (implies 3.762 MMBtu/bbl). FLAGGED: generic propane/butane blend; likely NET-basis, but the gross value is composition-dependent (undocumented mix) -> not changed. Desk: specify a C3/C4 split, or price via the discrete NGL species instead.
    "natgas": Commodity(
        "natgas", 0.542225066 * ureg.kg / ureg.L, 26.137 * ureg.GJ / ureg.m**3
    ),  # liquefied NG / LNG (alias 'lng'); implies 48.20 GJ/t, 3.939 MMBtu/bbl. FLAGGED: density 0.542 is high for LNG (~0.43-0.47) and basis is unclear -> not changed pending desk clarification of what this represents vs 'natural_gas'.
    # Natural gas (gaseous, pipeline): BP approx 36 PJ per bcm => 0.036 GJ/m**3
    # density=None: not a liquid, so mass<->volume conversion is undefined.
    # FLAGGED (HHV policy): 0.036 GJ/m^3 (=36 MJ/m^3) is a rounded BP approx sitting
    # between net (~34.6) and US-pipeline gross (~38.3 MJ/m^3, ~1028 Btu/scf). Not
    # clearly net, and it has broad blast radius (bcm<->energy in many callers +
    # explicit tests), so left unchanged; desk to confirm gross ~0.0383 if wanted.
    "natural_gas": Commodity(
        "natural_gas", density=None, energy_content=0.036 * ureg.GJ / ureg.m**3
    ),
    # Light gases (NGLs as discrete species — supersede the generic 'lpg' alias
    # for unit conversions where pure-component density/HHV matter, e.g.
    # $/gal <-> $/MMBtu for the MB OPIS futures (AC0, B0, AD0, A8I)).
    "ethane": Commodity(
        "ethane", 0.373 * ureg.kg / ureg.L, 18.4262 * ureg.GJ / ureg.m**3
    ),  # HHV; implies 2.777 MMBtu/bbl vs EIA ethane 2.783 (+0.2%, within tol) -> left as-is
    "propane": Commodity(
        "propane", 0.507 * ureg.kg / ureg.L, 25.5375 * ureg.GJ / ureg.m**3
    ),  # HHV; implies 3.848 MMBtu/bbl vs EIA propane 3.841 (-0.2%, within tol) -> left as-is
    "butane": Commodity(
        "butane", 0.574 * ureg.kg / ureg.L, 28.887 * ureg.GJ / ureg.m**3
    ),  # HHV; EIA MER App A n-butane 4.353 MMBtu/bbl gross (was 28.44 -> 4.286 MMBtu/bbl, -1.57% off EIA; aligned 2026-07 per EIA-gross basis decision)
    "isobutane": Commodity(
        "isobutane", 0.557 * ureg.kg / ureg.L, 27.7700 * ureg.GJ / ureg.m**3
    ),  # HHV; implies 4.185 MMBtu/bbl vs EIA isobutane 4.183 (-0.0%, within tol) -> left as-is
    "natural_gasoline": Commodity(
        "natural_gasoline", 0.6675 * ureg.kg / ureg.L, 30.778 * ureg.GJ / ureg.m**3
    ),  # HHV; EIA MER App A pentanes-plus 4.638 MMBtu/bbl gross (was 31.4 -> 4.732 MMBtu/bbl, +1.98% off EIA; aligned 2026-07 per EIA-gross basis decision)
    # BP product basket (optional reference)
    "product_basket": Commodity(
        "product_basket", 0.781 * ureg.kg / ureg.L, 33.642356 * ureg.GJ / ureg.m**3
    ),  # implies 43.08 GJ/t, 5.070 MMBtu/bbl. FLAGGED: weighted product basket, construction (component weights) undocumented -> not changed; should be recomputed from the now-gross component products with documented weights rather than guessed.
    # Solid fuels: MASS-basis energy_content (GJ/t) with density=None. mt<->energy
    # (the only meaningful jump, $/t <-> $/MMBtu) works; mt<->bbl stays illegal.
    "coal": Commodity(
        "coal",
        density=None,
        energy_content=26.377 * ureg.GJ / ureg.mt,
    ),  # API2 CIF ARA thermal coal. GROSS/HHV basis: traded spec is 6,000 kcal/kg
    # NAR (net); the gross equivalent is ~6,300 kcal/kg GAR (Argus/CoalIndo NAR/GAR
    # markers show a ~300 kcal/kg NAR->GAR gap for high-rank export coal, e.g.
    # 6,200 NAR = 6,500 GAR). 6,300 kcal/kg x 4.1868 kJ/kcal (IT calorie) = 26.377
    # GJ/t = 25.00 MMBtu/t. Replaces the legacy /27.76 tce divisor in
    # oilpricingcharts energy_mmbtu (generic tonne-of-coal-equivalent, 7,000 kcal/kg
    # GCV = 29.29 GJ/t) which understated coal $/MMBtu by ~11%.
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
        from_unit: Union[str, PriceUnit],
        to_unit: Union[str, PriceUnit],
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
        # Accept PriceUnit | str at the boundary: a PriceUnit collapses to its
        # canonical string form, which the existing rate/pint logic then handles
        # unchanged (convert/convfactor operate on bare & rate units, never on
        # currency-qualified ones). str(PriceUnit) is byte-identical to the
        # equivalent canonical string, so numeric behaviour is unchanged.
        if isinstance(from_unit, PriceUnit):
            from_unit = str(from_unit)
        if isinstance(to_unit, PriceUnit):
            to_unit = str(to_unit)

        # Normalize, then split rate units with the single PriceUnit parser
        # (quantity leg + rate period). No private rate parser anymore.
        from_unit = _to_pint_token(from_unit)
        to_unit = _to_pint_token(to_unit)
        from_pu = PriceUnit.parse(from_unit)
        to_pu = PriceUnit.parse(to_unit)
        from_base = _to_pint_token(from_pu.quantity_unit)
        to_base = _to_pint_token(to_pu.quantity_unit)
        from_period = from_pu.period
        to_period = to_pu.period

        if isinstance(value, pd.Series):
            # Series keeps the calendar-aware days_in_month path (pint's
            # fixed-length month can't express it).
            return self._convert_series(
                value, from_base, to_base, commodity, from_period, to_period
            )

        # Scalar: the base-unit factor goes through the SAME _convert_scalar path
        # as a non-rate conversion (so a rate result is consistent-by-construction
        # with the base one), multiplied by the period ratio. That ratio comes
        # from pint's own calendar month/year lengths, replacing the deleted
        # hand-rolled duplicates of those constants.
        result = self._convert_scalar(value, from_base, to_base, commodity)
        if from_period or to_period:
            result = result * self._period_rate_factor(from_period, to_period)
        return result

    @lru_cache(maxsize=128)
    def _commodity_context(self, name: str) -> pint.Context:
        """Build (and cache) a pint Context expressing a commodity's dimensional
        transformations.

        A commodity's physical properties define which cross-dimension jumps are
        legal:
          - ``density`` (kg/m^3) enables ``[mass] <-> [volume]``.
          - ``energy_content`` on a VOLUME basis (J/m^3) enables
            ``[volume] <-> [energy]`` (and chains to ``[mass] <-> [energy]`` via
            density when present).
          - ``energy_content`` on a MASS basis (J/kg) enables ``[mass] <-> [energy]``
            DIRECTLY — for density-less solids (coal) whose only meaningful jump
            is $/t <-> $/MMBtu. No volume transform is registered, so a
            mass<->volume jump still finds no path and raises.
        pint composes the registered transformations into a graph and finds the
        conversion path itself — no manual branching on the path is needed.

        Properties that are ``None`` register no transformation, so pint simply
        finds no path and raises ``DimensionalityError`` (translated to a domain
        ``ValueError`` by the caller). This is how density=None commodities
        (natural_gas, naphtha, vgo) reject any mass<->volume(<->energy) jump.
        """
        comm = self.get_commodity(name)
        ctx = pint.Context(comm.name)
        if comm.density is not None:
            d = comm.density.to("kg/m^3")
            ctx.add_transformation("[mass]", "[volume]", lambda ureg, x, d=d: x / d)
            ctx.add_transformation("[volume]", "[mass]", lambda ureg, x, d=d: x * d)
        if comm.energy_content is not None:
            ec = comm.energy_content
            if ec.check("[energy]/[volume]"):
                e = ec.to("J/m^3")
                ctx.add_transformation(
                    "[volume]", "[energy]", lambda ureg, x, e=e: x * e
                )
                ctx.add_transformation(
                    "[energy]", "[volume]", lambda ureg, x, e=e: x / e
                )
            elif ec.check("[energy]/[mass]"):
                em = ec.to("J/kg")
                ctx.add_transformation(
                    "[mass]", "[energy]", lambda ureg, x, em=em: x * em
                )
                ctx.add_transformation(
                    "[energy]", "[mass]", lambda ureg, x, em=em: x / em
                )
            else:
                raise ValueError(
                    f"{comm.name}: energy_content must be [energy]/[volume] or "
                    f"[energy]/[mass], got dimensionality {dict(ec.dimensionality)}"
                )
        return ctx

    def _convert_scalar(
        self, value: float, from_unit: str, to_unit: str, commodity: Optional[str]
    ) -> float:
        """Convert a scalar value across mass/volume/energy using a commodity's
        pint Context when a cross-dimension jump is needed."""
        from_unit = _to_pint_token(from_unit)
        to_unit = _to_pint_token(to_unit)
        qty = value * self.ureg(from_unit)

        # Same-dimension conversion (e.g. kt->mt, bbl->L): no commodity needed.
        try:
            return qty.to(to_unit).magnitude
        except DimensionalityError:
            pass

        # Cross-dimensional: needs commodity physical properties. Decide the
        # right error up front when no commodity was supplied (mirrors the
        # historical messages the callers/tests grep for).
        is_from_energy = self._is_energy(from_unit)
        is_to_energy = self._is_energy(to_unit)
        involves_energy = is_from_energy or is_to_energy

        if not commodity:
            if involves_energy:
                raise ValueError("Commodity required for energy conversion")
            is_from_mass = self._is_mass(from_unit)
            is_to_mass = self._is_mass(to_unit)
            is_from_volume = self._is_volume(from_unit)
            is_to_volume = self._is_volume(to_unit)
            if (is_from_mass and is_to_volume) or (is_from_volume and is_to_mass):
                raise ValueError(f"Commodity required for {from_unit} to {to_unit}")
            raise ValueError(
                f"Cannot convert from {from_unit} to {to_unit} - incompatible dimensions"
            )

        # Let the commodity's context find the conversion path (including
        # mass<->energy chaining). A missing property leaves no path -> pint
        # raises DimensionalityError, which we translate to the domain error.
        comm = self.get_commodity(commodity)
        ctx = self._commodity_context(commodity)
        try:
            return qty.to(to_unit, ctx).magnitude
        except DimensionalityError as exc:
            if involves_energy and comm.energy_content is None:
                raise ValueError(f"No energy content defined for {commodity}") from exc
            if comm.density is None and (
                self._is_mass(from_unit) or self._is_mass(to_unit)
            ):
                raise ValueError(
                    f"Mass<->volume conversion not supported for {commodity!r} "
                    f"(no density defined)"
                ) from exc
            raise ValueError(
                f"Cannot convert from {from_unit} to {to_unit} - incompatible dimensions"
            ) from exc

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
        - Other time conversions use pint's calendar period lengths.
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
                    # Other period conversions (e.g. year<->day): pint's ratio
                    # of period lengths.
                    result = result * self._period_rate_factor(from_period, to_period)

        # Apply unit conversion
        factor_units = self._convert_scalar(1.0, from_unit, to_unit, commodity)
        result = result * factor_units

        return result

    def _period_rate_factor(
        self, from_period: Optional[str], to_period: Optional[str]
    ) -> float:
        """Ratio converting a per-``from_period`` rate to per-``to_period``,
        derived from pint's calendar period lengths (day/month/year) — the
        single source that replaces the deleted hand-rolled constants. A
        ``None`` period on either side is a no-op (1.0), matching historical
        behaviour.

        Feeds both the scalar path (period factor) and the Series non-month
        path; the calendar-aware month<->day Series case is handled separately
        with days_in_month.
        """
        if from_period == to_period or from_period is None or to_period is None:
            return 1.0
        from_days = (1 * self.ureg(from_period)).to("day").magnitude
        to_days = (1 * self.ureg(to_period)).to("day").magnitude
        return to_days / from_days

    def _is_energy(self, unit: str) -> bool:
        return self.ureg(unit).check("[energy]")

    def _is_mass(self, unit: str) -> bool:
        return self.ureg(unit).check("[mass]")

    def _is_volume(self, unit: str) -> bool:
        return self.ureg(unit).check("[length]**3")

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
            "oil_barrel",
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
def convert(
    value,
    from_unit: Union[str, PriceUnit],
    to_unit: Union[str, PriceUnit],
    commodity: Optional[str] = None,
):
    """Convert values between units. Units may be strings or ``PriceUnit``."""
    return converter.convert(value, from_unit, to_unit, commodity)


def convfactor(
    from_unit: Union[str, PriceUnit],
    to_unit: Union[str, PriceUnit],
    commodity: Optional[str] = None,
) -> float:
    """Get conversion factor between units. Units may be strings or ``PriceUnit``."""
    return converter.convert(1.0, from_unit, to_unit, commodity)


# ---- Currency-aware price conversion helpers ----
#
# Currency vocabulary lives in commodutil.standards.currency (importable
# without dragging in pint / pandas). convfactors owns only the integrated
# unit + currency `convert_price` math (which depends on the pint registry
# above) and reads currency vocabulary directly from _currency. Callers
# wanting currency vocabulary should import from commodutil.standards.currency.


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


def list_commodities():
    """List all available commodities"""
    return converter.available_commodities


def list_units():
    """List common units"""
    # Return normalized forms to avoid encoding issues
    return [_to_pint_token(u) for u in converter.available_units]


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
