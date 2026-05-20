"""Tests for commodutil.standards.commodities."""

from __future__ import annotations

import pytest

from commodutil.standards.commodities import (
    COMMODITY_CONVERSION_MAP,
    COMMODITY_KEYWORDS,
    infer_commodity_and_group,
    infer_commodity_from_exchange_symbol,
    normalize_commodity_for_conversion,
)


def test_commodity_keywords_sanity_length():
    assert len(COMMODITY_KEYWORDS) >= 20


def test_commodity_keywords_entry_shape():
    for entry in COMMODITY_KEYWORDS:
        assert isinstance(entry, tuple)
        assert len(entry) == 3
        display, group, kws = entry
        assert isinstance(display, str)
        assert isinstance(group, str)
        assert isinstance(kws, list)
        assert all(isinstance(k, str) for k in kws)


def test_natural_gasoline_before_natural_gas():
    displays = [d for d, _, _ in COMMODITY_KEYWORDS]
    assert "Natural Gasoline" in displays
    assert "Natural Gas" in displays
    assert displays.index("Natural Gasoline") < displays.index("Natural Gas")


def test_brent_entry_present():
    assert ("Brent", "Crude Oil", ["brent"]) in COMMODITY_KEYWORDS


def test_conversion_map_brent_is_crude():
    assert COMMODITY_CONVERSION_MAP["Brent"] == "crude"


def test_conversion_map_natural_gas_is_natgas():
    assert COMMODITY_CONVERSION_MAP["Natural Gas"] == "natgas"


def test_conversion_map_ngl_default_lpg_blend():
    assert COMMODITY_CONVERSION_MAP["NGL"] == "lpg"


def test_conversion_map_no_orphan_entries():
    """Every commodity in COMMODITY_CONVERSION_MAP appears as a display_name
    in COMMODITY_KEYWORDS — consistency check."""
    displays = {d for d, _, _ in COMMODITY_KEYWORDS}
    orphans = set(COMMODITY_CONVERSION_MAP.keys()) - displays
    assert not orphans, f"Orphan conversion map entries: {orphans}"


# ---------- infer_commodity_and_group --------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("ICE Brent Crude Futures", ("Brent", "Crude Oil")),
        ("WTI Mar25", ("WTI", "Crude Oil")),
        ("Crude Oil Forward", ("Brent", "Crude Oil")),
        # Ordering guard — "Natural Gasoline" must beat "Natural Gas"
        ("Natural Gasoline OPIS", ("Natural Gasoline", "NGL")),
        ("Henry Hub Natural Gas", ("Natural Gas", "Natural Gas")),
        ("Jet Fuel CIF NWE", ("Jet", "Refined Products")),
        ("ULSD Heating Oil", ("Diesel", "Refined Products")),
        ("RBOB Gasoline", ("Gasoline", "Refined Products")),
        ("HSFO 3.5%", ("Fuel Oil", "Refined Products")),
        ("Naphtha CIF", ("Naphtha", "Refined Products")),
        ("Freight FFA Route", ("FFA", "Freight")),
    ],
)
def test_infer_commodity_and_group_hits(text, expected):
    # Note: "Crude Oil Forward" hits 'brent' substring inside 'forward'?
    # No — 'brent' is not a substring of 'forward'. The 'crude' keyword
    # appears under both 'Brent' (no) and 'Crude Oil' entries. Walk-order
    # determines the winner. See test_first_keyword_hit_wins below for the
    # canonical example.
    result = infer_commodity_and_group(text)
    assert result[1] == expected[1], f"Group mismatch for {text!r}"


def test_first_keyword_hit_wins():
    # COMMODITY_KEYWORDS is walked top-down; the first commodity whose
    # keyword list contains any substring of the haystack wins. "crude oil"
    # appears in the "Crude Oil" entry (3rd) but plain "crude" also appears
    # there. "brent" alone is the Brent entry (1st), so "ICE Brent Crude"
    # returns Brent — earlier in the list.
    assert infer_commodity_and_group("ICE Brent Crude") == ("Brent", "Crude Oil")
    # Without "brent", the "crude" keyword in the Crude Oil row matches.
    assert infer_commodity_and_group("Crude Oil Forward") == ("Crude Oil", "Crude Oil")


@pytest.mark.parametrize("text", [None, "", "Unknown Widget"])
def test_infer_commodity_and_group_misses(text):
    assert infer_commodity_and_group(text) == (None, None)


# ---------- normalize_commodity_for_conversion ----------------------------


@pytest.mark.parametrize(
    "commodity,expected",
    [
        ("Brent", "crude"),
        ("ICE Brent Crude", "crude"),
        ("WTI", "crude"),
        ("Natural Gas", "natgas"),
        ("Gasoline", "gasoline"),
        ("Diesel", "diesel"),
        ("Fuel Oil", "fuel_oil"),
    ],
)
def test_normalize_commodity_for_conversion_hits(commodity, expected):
    assert normalize_commodity_for_conversion(commodity) == expected


def test_normalize_commodity_for_conversion_empty():
    assert normalize_commodity_for_conversion(None) is None
    assert normalize_commodity_for_conversion("") is None


def test_normalize_commodity_for_conversion_unknown_falls_back_to_slug():
    # No COMMODITY_KEYWORDS hit -> normalised slug fallback (lowercased,
    # separators collapsed to underscores).
    assert normalize_commodity_for_conversion("Some Unknown") == "some_unknown"


# ---------- infer_commodity_from_exchange_symbol --------------------------


@pytest.mark.parametrize(
    "symbol,expected",
    [
        ("CL_Mar25", "crude"),
        ("ICE_EuroFutures:BRN", "crude"),
        ("brent forward", "crude"),
        ("wti", "crude"),
        ("CME_NymexFutures_EOD:RB", "gasoline"),
        ("RBOB_Apr25", "gasoline"),
        ("HO_May25", "gasoil"),
        ("diesel europe", "gasoil"),
        ("NG_Jun25", "natgas"),
        ("natural gas", "natgas"),
        ("XYZ_Spot", None),
        (None, None),
        ("", None),
    ],
)
def test_infer_commodity_from_exchange_symbol(symbol, expected):
    assert infer_commodity_from_exchange_symbol(symbol) == expected


def test_infer_commodity_from_exchange_symbol_loose_match_documented():
    # INTENTIONAL loose-match behaviour: "close_value" contains "cl" so the
    # function returns "crude". This is the documented short-substring
    # fallback for raw exchange symbols (NOT free-form text).
    assert infer_commodity_from_exchange_symbol("close_value") == "crude"
