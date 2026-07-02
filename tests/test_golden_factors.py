"""Golden-factor regression test.

This is the numeric safety net for the conversion-architecture refactor
(see docs/conversion-architecture-plan.md, Phase 0). It freezes EVERY
conversion number commodutil produces today so that any future refactor
which changes a factor -- even by a rounding ulp -- fails CI loudly.

Why this matters: oilrisk materialises ``convfactor()`` outputs into SQL
literals inside the live PnL identity. Numeric drift here is P&L drift.
Error *messages* are also pinned exactly, because convfactors deliberately
preserves historical error strings that tests and consumers grep for
(plan ground rule 5: error text is API until a census proves otherwise).

The fixture at ``tests/golden/golden_factors.json`` is checked in. This
module is the single source of truth for HOW each number is produced; the
JSON is just the frozen answer key. Both regeneration and validation call
the same ``build_records()`` so they can never drift apart.

Regeneration (never happens in a normal run -- guarded by an env var):

    GOLDEN_REGEN=1 python -m pytest tests/test_golden_factors.py
    # or:
    GOLDEN_REGEN=1 python tests/test_golden_factors.py

Regenerate ONLY when a factor change is intentional and desk-approved,
then eyeball the JSON diff before committing.
"""

import json
import math
import os

import pytest

from commodutil import convfactors


FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "golden", "golden_factors.json")

# Relative tolerance for float comparison. 1e-9 is tight enough to catch a
# genuine factor change while tolerating platform float noise.
REL_TOL = 1e-9

# The unit-pair matrix pinned for quantity conversions.
QUANTITY_UNITS = [
    "bbl",
    "mt",
    "kt",
    "gal",
    "m^3",
    "MMBtu",
    "MWh",
    "GJ",
    "therm",
    "L",
    "kg",
]

# Rate-conversion pairs. These exercise _parse_rate_unit / _rate_factor_scalar,
# which Phase 3.3 plans to delete in favour of native pint parsing -- so their
# outputs MUST be pinned before that refactor lands.
RATE_PAIRS = [
    ("kt/month", "bbl/day"),
    ("bbl/day", "mt/year"),
    ("m^3/day", "bbl/day"),
    ("mt/year", "kt/month"),
]
RATE_COMMODITIES = ["crude", "diesel", "gasoline", "fuel_oil", None]

# convert_price cases: (value, from_unit, to_unit, commodity, fx).
# Chosen to cover the mass<->volume price inversion per commodity, the
# NGL $/gal<->$/MMBtu energy legs, the fractional-currency pure-scale path
# (USc->USD), and the FX-bearing energy legs (EUR/MWh, GBp/therm).
CONVERT_PRICE_CASES = [
    (100.0, "mt", "bbl", "gasoline", None),
    (2.5, "gal", "bbl", None, None),
    (1.0, "USD/gal", "USD/MMBtu", "ethane", None),
    (1.0, "USD/gal", "USD/MMBtu", "propane", None),
    (1.0, "USD/gal", "USD/MMBtu", "butane", None),
    (35.0, "EUR/MWh", "USD/MMBtu", None, 1.07),
    (80.0, "GBp/therm", "USD/MMBtu", None, 1.25),
    (250.0, "USc/gal", "USD/gal", None, None),
    (100.0, "USD/mt", "USD/bbl", "jet", None),
    (100.0, "USD/mt", "USD/bbl", "diesel", None),
    (100.0, "USD/mt", "USD/bbl", "crude", None),
]


def _commodity_key(commodity):
    """Stable string form of a commodity for a JSON key (None -> 'None')."""
    return "None" if commodity is None else commodity


def _record_call(fn):
    """Run ``fn`` and return a JSON-serialisable record of what it produced.

    Success -> ``{"value": <float>}``.
    Exception -> ``{"error": {"type": <ClassName>, "message": <str(exc)>}}``.

    Error type + message are treated as API (plan ground rule 5) and are
    compared exactly on validation.
    """
    try:
        value = float(fn())
    except Exception as exc:  # noqa: BLE001 -- deliberately catch any error to pin it
        return {"error": {"type": type(exc).__name__, "message": str(exc)}}
    if not math.isfinite(value):
        # Non-finite results are not valid JSON numbers and would signal a
        # real bug; surface them rather than silently emitting `NaN`.
        return {"non_finite": repr(value)}
    return {"value": value}


def build_records():
    """Build the full golden record set from live commodutil calls.

    Returns a dict of three sorted-key categories. This function is the ONE
    place that defines every pinned number; regeneration writes exactly what
    this produces and validation compares against exactly what this produces.
    """
    quantity = {}
    for commodity in [None] + sorted(convfactors.list_commodities()):
        ck = _commodity_key(commodity)
        for from_unit in QUANTITY_UNITS:
            for to_unit in QUANTITY_UNITS:
                key = f"{ck}::{from_unit}::{to_unit}"
                quantity[key] = _record_call(
                    lambda f=from_unit, t=to_unit, c=commodity: convfactors.convfactor(
                        f, t, c
                    )
                )

    rates = {}
    for commodity in RATE_COMMODITIES:
        ck = _commodity_key(commodity)
        for from_unit, to_unit in RATE_PAIRS:
            key = f"{ck}::{from_unit}::{to_unit}"
            rates[key] = _record_call(
                lambda f=from_unit, t=to_unit, c=commodity: convfactors.convert(
                    1.0, f, t, c
                )
            )

    prices = {}
    for value, from_unit, to_unit, commodity, fx in CONVERT_PRICE_CASES:
        ck = _commodity_key(commodity)
        key = f"{value}::{from_unit}::{to_unit}::{ck}::fx={fx}"
        prices[key] = _record_call(
            lambda v=value,
            f=from_unit,
            t=to_unit,
            c=commodity,
            x=fx: convfactors.convert_price(v, f, t, c, fx=x)
        )

    return {
        "quantity_factors": quantity,
        "rate_conversions": rates,
        "convert_price": prices,
    }


def _fixture_payload():
    """The full on-disk payload: a self-describing header plus the records."""
    return {
        "_README": (
            "Golden-factor regression fixture for commodutil conversions. "
            "Frozen answer key for tests/test_golden_factors.py; DO NOT hand-edit. "
            "Regenerate intentionally with: "
            "GOLDEN_REGEN=1 python -m pytest tests/test_golden_factors.py "
            "(only after a desk-approved factor change; review the diff). "
            "value = convfactor/convert/convert_price output; "
            "error = pinned exception type + message (treated as API)."
        ),
        **build_records(),
    }


def _write_fixture():
    os.makedirs(os.path.dirname(FIXTURE_PATH), exist_ok=True)
    with open(FIXTURE_PATH, "w", encoding="utf-8") as fh:
        json.dump(_fixture_payload(), fh, indent=2, sort_keys=True)
        fh.write("\n")


def _load_fixture():
    with open(FIXTURE_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _assert_record_matches(key, expected, actual):
    """Assert one record matches: floats to REL_TOL, errors exactly."""
    assert set(actual.keys()) == set(expected.keys()), (
        f"{key}: record shape changed (expected {sorted(expected)}, "
        f"got {sorted(actual)})"
    )
    if "value" in expected:
        assert actual["value"] == pytest.approx(expected["value"], rel=REL_TOL), (
            f"{key}: value drift {actual['value']!r} != {expected['value']!r}"
        )
    elif "error" in expected:
        assert actual["error"] == expected["error"], (
            f"{key}: error changed {actual['error']!r} != {expected['error']!r}"
        )
    else:
        assert actual == expected, f"{key}: {actual!r} != {expected!r}"


# --- Regeneration guard: rewrite the fixture only when explicitly asked. ----
if os.environ.get("GOLDEN_REGEN") == "1":
    _write_fixture()


def _check_category(category):
    fixture = _load_fixture()
    assert category in fixture, f"category {category!r} missing from fixture"
    expected = fixture[category]
    actual = build_records()[category]
    # Coverage must not silently shrink or grow.
    assert set(actual.keys()) == set(expected.keys()), (
        f"{category}: key set changed. "
        f"missing={sorted(set(expected) - set(actual))} "
        f"new={sorted(set(actual) - set(expected))}"
    )
    for key in sorted(expected):
        _assert_record_matches(key, expected[key], actual[key])


def test_golden_quantity_factors():
    _check_category("quantity_factors")


def test_golden_rate_conversions():
    _check_category("rate_conversions")


def test_golden_convert_price():
    _check_category("convert_price")


def test_golden_fixture_has_readme():
    # The fixture must stay self-describing so a future reader knows what it
    # is and how to regenerate it.
    fixture = _load_fixture()
    assert "_README" in fixture and "GOLDEN_REGEN" in fixture["_README"]


if __name__ == "__main__":
    # Allow regeneration without pytest: `GOLDEN_REGEN=1 python tests/test_golden_factors.py`
    if os.environ.get("GOLDEN_REGEN") == "1":
        _write_fixture()
        print(f"Wrote golden fixture: {FIXTURE_PATH}")
    else:
        print(
            "No-op. Set GOLDEN_REGEN=1 to regenerate the fixture, or run via pytest "
            "to validate."
        )
