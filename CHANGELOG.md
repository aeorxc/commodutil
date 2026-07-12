# Changelog

## 5.4.0 - 2026-07-12

- Added `normalize_price_unit_strict` and `is_canonical_price_unit` to `commodutil.standards.units` (and the `commodutil` / `commodutil.standards` facades): the STRICT counterpart to the lenient `canonical_price_unit_token`. Where the lenient token preserves unrecognised currency/unit fragments so it can label any quote string, the strict variant returns a canonical `CCY/unit` token only when both legs resolve (recognised currency over a `canonical_quantity_unit` denominator) and `None` otherwise — for validating a curated price-unit qualifier that must round-trip through conversion (e.g. `usd_ton`/`USD/ton` refused, `USD/MT` -> `USD/mt`).

## 5.0.1 - 2026-07-03

- Breaking: removed `default_unit_for_commodity` and `_DEFAULT_UNIT` from `commodutil.standards.units`; these fabricated per-commodity quote units, and the gold instrument registry answers this per-instrument.
- Breaking: removed the `commodutil.standards.price_units` module. Price-unit resolution (`resolve_price_unit`, `resolve_price_unit_from_attrs`) moved into `commodutil.standards.price_unit` alongside the `PriceUnit` value type. Migrate imports to `commodutil.standards.price_unit`.
- First release of the 5.x line; the 5.0.0 version number is unusable on the feed.

## 4.10.0 - 2026-07-03

- Added exchange unit spellings for `MTONS`, `CBM`, `THM`, and `KL` to the unit registry, with `KL`/kilolitre spellings resolving to the canonical `m^3` unit.
- Moved commodity alias ownership to `commodutil.standards.commodities.COMMODITY_ALIASES`; `CommodityConverter` now consumes a defensive copy from that source.
- Breaking: removed the `commodutil.convfactors.ALIASES` re-export. Migrate callers to `from commodutil.standards.commodities import COMMODITY_ALIASES`.

## 4.8.0 - 2026-07-03

- Split price, currency, and FX conversion helpers into `commodutil.priceconv`, while keeping lazy compatibility exports from `commodutil.convfactors` for the price-conversion API.
- Consolidated petrochemical taxonomy keywords into `commodutil.standards.commodities`.
- Added `infer_ngl_species` for discrete NGL species inference from free text.

## 4.7.0 - 2026-07-03

- Fixed ethanol density to `0.7937 kg/L`, preserving the EIA gross per-volume energy content while correcting mass-based conversion factors.
- Promoted `kg` and bare cubic-metre spellings (`m3`, `m^3`, `m**3`) into the `UNIT_MAP` tier so vendor-spec parsers can resolve them directly.

## 4.6.0 - 2026-07-02

- Replaced hand-rolled scalar rate-period conversion with `PriceUnit` parsing and Pint-derived period ratios, while preserving calendar-aware month/day handling for series.
- Added `align_fx`, `ConversionResult`, and `convert_price_result` for explicit price-conversion results, currency-leg fallback, and keep-raw error handling.

## 4.5.0 - 2026-07-02

- Added the `PriceUnit` value type for structured price-unit parsing across currency, quantity-unit, and rate-period legs.
- Added `commodutil.standards.unit_registry` as the single source of truth for unit vocabulary, public unit maps, and Pint definitions.
- Added ICE unit spellings for `MMBtu`, `GJ`, `MWh`, `m^3`, `MW`, `RIN`, `FEU`, and `day`, covering the registry gap identified in ICE contract metadata.

## 4.4.0 - 2026-07-02

- Standardized covered liquid and biofuel energy-content factors on the EIA gross/HHV basis, documenting explicit exceptions and basis assumptions.
- Added API2/coal support with mass-basis energy content for density-less solid-fuel conversions and Coal taxonomy inference.
- Added the golden-factor regression fixture and validation tests for quantity, rate, and price conversions.
