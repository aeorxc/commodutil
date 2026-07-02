"""commodutil.standards.unit_registry: the single source-of-truth unit table.

Phase 3.2 of the conversion architecture plan. Every unit the package knows
appears here exactly once, as a :class:`UnitRow`. From this one table we DERIVE,
at import time:

* ``UNIT_MAP``            — the bbl/gal/mt/lb vocabulary (vendor-spec
                            normalisation; consumed by curvemetadata).
* ``PUBLIC_UNIT_MAP``     — the wider public/metadata spelling map behind
                            ``standards.units.canonical_unit_token`` etc.
* ``PINT_DEFINITIONS``    — the ordered list of ``ureg.define(...)`` strings the
                            pint registry in ``convfactors`` applies.

To ADD A UNIT SPELLING, add it to the relevant :class:`UnitRow` in ``UNIT_ROWS``
below — nowhere else. ``standards.units`` and ``convfactors`` derive their maps
and pint registrations from this table; there are no hand-maintained parallel
dicts.

Registry invariants:
* CASE-SENSITIVE by design — the pint registry must stay case-sensitive
  (``kt``/kilotesla, ``mt``/millitesla, ``gal``/galileo collide otherwise; the
  decision is settled). Spellings are matched case-insensitively only by the
  string-vocab maps (which lower-case their keys), never by pint.
* ``unit_map_spellings`` feed both ``UNIT_MAP`` and ``PUBLIC_UNIT_MAP``;
  ``public_only_spellings`` feed only ``PUBLIC_UNIT_MAP`` — mirroring the
  historical ``_PUBLIC_UNIT_MAP = {**UNIT_MAP, <extras>}`` construction.
* ``UNIT_ROWS`` order IS the pint-definition order: a definition may reference
  an earlier one (``kiloton`` needs ``metric_ton``; ``Mtoe`` needs ``toe``;
  the MMBtu ``@alias`` needs the MMBtu ``define``). Do not reorder casually.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class UnitRow:
    """One unit's full vocabulary + pint registration.

    canonical              canonical token (e.g. ``'bbl'``, ``'m^3'``, ``'MMBtu'``).
    dimension              coarse dimension label ('volume' | 'mass' | 'energy' |
                           'power'); documentation/grouping only.
    unit_map_spellings     lower-case spellings that go into UNIT_MAP (and, by
                           inheritance, PUBLIC_UNIT_MAP).
    public_only_spellings  lower-case spellings that go into PUBLIC_UNIT_MAP only.
    pint_specs             exact strings passed to ``ureg.define(...)``, in order.
    substring_risky        UNIT_MAP spellings on this row that are short/ambiguous
                           as SUBSTRINGS (e.g. ``'mw'`` inside ``'mwh'``). Consumers
                           that match UNIT_MAP entries as substrings (curvemetadata
                           ``parse_unit``) should enforce word boundaries for these;
                           exposed as ``RISKY_SUBSTRING_SPELLINGS``.
    """

    canonical: str
    dimension: str
    unit_map_spellings: Tuple[str, ...] = ()
    public_only_spellings: Tuple[str, ...] = ()
    pint_specs: Tuple[str, ...] = ()
    substring_risky: Tuple[str, ...] = ()


# The one table. Order is load-bearing for PINT_DEFINITIONS (see module docstring).
#
# Item-2 note (Phase 2.1): the spellings tagged "curvemetadata" below were
# absorbed from curvemetadata_helpers.units._EXTRA_UNIT_ALIASES so that local
# table can be deleted. They are purely ADDITIVE — no pre-existing mapping
# changes.
UNIT_ROWS: Tuple[UnitRow, ...] = (
    # ---- bbl/gal/mt/lb: the UNIT_MAP vocabulary ----
    UnitRow(
        "bbl",
        "volume",
        unit_map_spellings=("barrel", "barrels", "bbl", "bbls"),
        pint_specs=("oil_barrel = 158.987294928 liter = bbl",),
    ),
    UnitRow(
        "gal",
        "volume",
        unit_map_spellings=("gallon", "gallons", "gal"),
        pint_specs=("gallon = 3.785411784 liter = gal",),
    ),
    UnitRow(
        "mt",
        "mass",
        unit_map_spellings=(
            "metric ton",
            "metric tons",
            "metric tonne",
            "metric tonnes",
            "mt",
            "tonne",
            "tonnes",
        ),
        pint_specs=("metric_ton = 1000 kilogram = mt",),
    ),
    # ---- pint-only definitions (no string-vocab spellings) ----
    UnitRow("kt", "mass", pint_specs=("kiloton = 1000 metric_ton = kt",)),
    UnitRow("km3", "volume", pint_specs=("cubic_kilometer = 1e9 meter**3 = km3",)),
    UnitRow(
        "GJ",
        "energy",
        # ICE gap fix: 'gj' + plural 'gjs' in UNIT_MAP. curvemetadata parse_unit
        # uses word-boundary matching, so 'gj' does NOT match "100 GJs" (trailing
        # s) — the plural must be listed explicitly. curvemetadata: gigajoule(s).
        unit_map_spellings=("gj", "gjs"),
        public_only_spellings=("gigajoule", "gigajoules"),
        pint_specs=("gigajoule = 1e9 joule = gj = GJ",),
        substring_risky=("gj",),
    ),
    UnitRow("PJ", "energy", pint_specs=("petajoule = 1e15 joule = pj = PJ",)),
    UnitRow(
        "bcm",
        "volume",
        pint_specs=("billion_cubic_meter = 1e9 meter**3 = bcm = BCM",),
    ),
    UnitRow(
        "bcf",
        "volume",
        pint_specs=("billion_cubic_foot = 1e9 foot**3 = bcf = BCF",),
    ),
    UnitRow(
        "toe",
        "energy",
        pint_specs=("tonne_of_oil_equivalent = 41.868e9 joule = toe = TOE",),
    ),
    UnitRow(
        "Mtoe",
        "energy",
        pint_specs=(
            "million_tonne_of_oil_equivalent = 1e6 tonne_of_oil_equivalent = Mtoe",
        ),
    ),
    UnitRow(
        "boe",
        "energy",
        pint_specs=("barrel_of_oil_equivalent = 6.119e9 joule = boe = BOE",),
    ),
    UnitRow(
        "Mboe",
        "energy",
        pint_specs=(
            "million_barrel_of_oil_equivalent = 1e6 barrel_of_oil_equivalent = Mboe",
        ),
    ),
    UnitRow("Mt", "mass", pint_specs=("megatonne = 1e6 metric_ton = Mt",)),
    # ---- energy/power with public spellings + pint casing aliases ----
    UnitRow(
        "MMBtu",
        "energy",
        # ICE gap fix: 'mmbtu' + plural 'mmbtus' in UNIT_MAP so word-boundary
        # parse_unit catches "2500 MMBtus" / "100 MMBtus per lot" / "25,000
        # MMBtus" (the 752-row plural category) as well as the singular.
        # curvemetadata: adds "million british thermal units".
        unit_map_spellings=("mmbtu", "mmbtus"),
        public_only_spellings=("mm btu", "million british thermal units"),
        pint_specs=(
            "million_british_thermal_unit = 1e6 Btu = MMBtu",
            "@alias million_british_thermal_unit = mmbtu = MMBTU = million_btu",
        ),
    ),
    UnitRow("therm", "energy", pint_specs=("@alias therm = Therm = THERM",)),
    UnitRow(
        "Btu",
        "energy",
        public_only_spellings=("btu",),
        pint_specs=("@alias british_thermal_unit = btu",),
    ),
    # NOTE: MWh is listed BEFORE MW so that in UNIT_MAP iteration order the
    # longer 'mwh' spelling precedes the shorter (substring-risky) 'mw' — a
    # substring matcher then resolves "100 MWh" to MWh, not MW. mw/mwh/MWH pint
    # defs are mutually independent, so this ordering is pint-safe.
    UnitRow(
        "MWh",
        "energy",
        # ICE gap fix: 'mwh' + plural 'mwhs' into UNIT_MAP.
        # curvemetadata: adds "megawatt hour" / "megawatt hours".
        unit_map_spellings=("mwh", "mwhs"),
        public_only_spellings=("mw h", "megawatt hour", "megawatt hours"),
        pint_specs=("mwh = 1e6 watt * hour", "MWH = 1e6 watt * hour"),
    ),
    UnitRow(
        "MW",
        "power",
        # ICE gap fix: 'mw' + plural 'mws' (power capacity, "1 MW"). Under
        # word-boundary matching 'mw' cannot match inside 'mwh'/'mwhs' anyway;
        # MWh is still listed first as belt-and-braces.
        unit_map_spellings=("mw", "mws"),
        pint_specs=("mw = 1e6 watt",),
        substring_risky=("mw",),
    ),
    # ---- vocabulary-only rows (unit resolved by pint's own defaults) ----
    UnitRow(
        "lb",
        "mass",
        unit_map_spellings=("pound", "pounds", "lb", "lbs"),
    ),
    UnitRow(
        "m^3",
        "volume",
        # ICE gap fix: the spelled-out cubic-metre forms (singular + plural) go
        # into UNIT_MAP. A prefix like 'cubic met' does NOT work under
        # word-boundary matching (trailing letters kill the lookahead), so each
        # form is listed explicitly.
        unit_map_spellings=(
            "cubic meter",
            "cubic meters",
            "cubic metre",
            "cubic metres",
        ),
        public_only_spellings=(
            "m3",
            "m^3",
            "m**3",
            "cubic_meter",
            "cubic_meters",
            "cubic_metre",
            "cubic_metres",
        ),
    ),
    UnitRow(
        "kg",
        "mass",
        # curvemetadata: kg / kilogram(s) / kilogramme(s)
        public_only_spellings=(
            "kg",
            "kilogram",
            "kilograms",
            "kilogramme",
            "kilogrammes",
        ),
    ),
    # ---- structural / non-physical vocabulary-only tokens (ICE) ----
    # No pint specs: these carry no conversion factor. They exist only so a
    # source_price_unit can compose currency-qualified strings like 'USc/RIN'
    # (see convfactors.convert_currency_leg, oilrisk ARTIS USc/RIN flows) and so
    # freight-priced ICE rows get a denominator instead of being dropped.
    UnitRow(
        "RIN",
        "credit",  # biofuel Renewable Identification Number; dimensionless credit
        # plural 'rins' explicit for word-boundary parse ("50,000 RINs").
        unit_map_spellings=("rin", "rins"),
        substring_risky=("rin",),
    ),
    UnitRow(
        "FEU",
        "container",  # forty-foot equivalent unit (container freight)
        # plural 'feus' explicit; 'forty foot' stays boundary-delimited in text.
        unit_map_spellings=("feu", "feus", "forty foot"),
        substring_risky=("feu",),
    ),
    UnitRow(
        "day",
        "time",  # freight FFA charter day (rate denominator, not a quantity)
        unit_map_spellings=("charter day", "charter days"),
    ),
)


def _build_unit_map() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in UNIT_ROWS:
        for spelling in row.unit_map_spellings:
            out[spelling] = row.canonical
    return out


def _build_public_unit_map() -> Dict[str, str]:
    # Mirrors the historical `{**UNIT_MAP, <public extras>}` construction.
    out: Dict[str, str] = dict(_build_unit_map())
    for row in UNIT_ROWS:
        for spelling in row.public_only_spellings:
            out[spelling] = row.canonical
    return out


def _build_pint_definitions() -> Tuple[str, ...]:
    return tuple(spec for row in UNIT_ROWS for spec in row.pint_specs)


def _build_risky_substring_spellings() -> frozenset:
    return frozenset(s for row in UNIT_ROWS for s in row.substring_risky)


# Derived, import-time products of the one table.
UNIT_MAP: Dict[str, str] = _build_unit_map()
PUBLIC_UNIT_MAP: Dict[str, str] = _build_public_unit_map()
PINT_DEFINITIONS: Tuple[str, ...] = _build_pint_definitions()

# UNIT_MAP spellings that are ambiguous as bare substrings (e.g. 'mw' inside
# 'mwh'). Substring-matching consumers (curvemetadata parse_unit) should require
# a word boundary for these. UNIT_MAP iteration order already places safer,
# longer spellings first (e.g. 'mwh' before 'mw').
RISKY_SUBSTRING_SPELLINGS: frozenset = _build_risky_substring_spellings()


# Encoding / notation repairs used by ``standards.units.to_pint_token`` to turn
# mojibake and alternate cubic-metre notations into a pint-parseable token. Kept
# here (not as a literal inside units.py) so the registry module remains the sole
# home for unit-vocabulary data. Applied as ordered substring replacements.
ENCODING_REPAIRS: Dict[str, str] = {
    "m��": "m^3",  # UTF-8 mojibake of m^3 (two U+FFFD replacement chars)
    "m³": "m^3",  # m³ (superscript three, U+00B3)
    "m**3": "m^3",
    "cubic_meter": "m^3",
    "CUBIC_METER": "m^3",
}


__all__ = [
    "UnitRow",
    "UNIT_ROWS",
    "UNIT_MAP",
    "PUBLIC_UNIT_MAP",
    "PINT_DEFINITIONS",
    "RISKY_SUBSTRING_SPELLINGS",
    "ENCODING_REPAIRS",
]
