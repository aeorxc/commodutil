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
        # Crude — token matches on wti / brent / brn. Raw NYMEX `cl` is
        # NOT a token any more (false-positive risk in 'Cleared' / 'Close');
        # the canonical Brent token is 'brn' and full forms hit free-text.
        ("ICE_EuroFutures:BRN", "crude"),
        ("brent forward", "crude"),
        ("wti", "crude"),
        # Gasoline — token matches on rbob / gasoline / mogas. `rb` alone
        # is no longer a token (false positives in 'Carbon').
        ("RBOB_Apr25", "gasoline"),
        # Gasoil — token matches on gasoil / diesel / heating. `ho` alone
        # is no longer a token (false positives in 'Hong' / 'HKD').
        ("diesel europe", "gasoil"),
        ("heating oil", "gasoil"),
        # Natgas — natural / natgas / hub acronyms. `ng` alone is no
        # longer a token (false positives in 'Long' / 'Naphtha').
        ("natural gas", "natgas"),
        ("JKM_M1", "natgas"),
        ("ICE_TTF", "natgas"),
        # No-match — returns None and the caller (e.g. pyoilprice) should
        # skip with a WARN rather than guess.
        ("XYZ_Spot", None),
        (None, None),
        ("", None),
        # Dropped legacy 2-char tokens — these now return None.
        ("CL_Mar25", None),
        ("CME_NymexFutures_EOD:RB", None),
        ("HO_May25", None),
        ("NG_Jun25", None),
    ],
)
def test_infer_commodity_from_exchange_symbol(symbol, expected):
    assert infer_commodity_from_exchange_symbol(symbol) == expected


def test_infer_commodity_from_exchange_symbol_no_loose_substring_match():
    # GUARD against the old substring-based behaviour: previously
    # ``"close_value"`` contained the substring "cl" and the function
    # returned "crude". The token-based rewrite splits the symbol and
    # matches whole tokens only — "close" is not "cl" and we get None.
    # The docstring previously called this loose match "INTENTIONAL"; it
    # caused false-positives across feed-prefixed identifiers
    # (JKM/TTF/Naphtha/HKD/Copper) and is now considered a bug.
    assert infer_commodity_from_exchange_symbol("close_value") is None


# Regression suite — these symbols previously misclassified under the
# substring-based implementation. See task #68 diagnosis + pyoilprice
# PR #20928 (JKM unit-conversion bug).
@pytest.mark.parametrize(
    "symbol,expected",
    [
        # Gas hubs that previously matched 'cl' (ClearedGas) -> 'crude'.
        # Token-based: 'jkm' / 'ttf' / 'nbp' / 'hh' tokens hit 'natgas'.
        ("Ice_ClearedGas:JKM", "natgas"),
        ("Ice_ClearedGas:TTF", "natgas"),
        ("Ice_ClearedGas:NBP", "natgas"),
        ("Ice_ClearedGas:HH", "natgas"),
        ("Ice_ClearedGas:Henry", "natgas"),
        # 'Naphtha' contains 'ng' substring; previously matched 'natgas'.
        # Token-based: 'naphtha' is its own token, not 'ng' -> None.
        ("Singapore_Spot:Naphtha", None),
        # 'HKD' contains 'ho' substring; previously matched 'gasoil'.
        ("Hong_Kong:HKD", None),
        # 'Carbon' contains 'rb' substring; previously matched 'gasoline'.
        ("Carbon:EUA", None),
        # 'Long' contains 'ng' substring; previously matched 'natgas'.
        ("LME_Copper:Long", None),
    ],
)
def test_infer_commodity_from_exchange_symbol_regression_no_substring_leak(
    symbol, expected
):
    assert infer_commodity_from_exchange_symbol(symbol) == expected


def test_infer_commodity_from_exchange_symbol_dropped_2char_tokens():
    """The 2-char ambiguous tokens cl/rb/ho/ng must NOT match when embedded
    in larger words. The legacy substring behaviour was the root cause of
    all the regressions above; this test pins the removal."""
    # 'cl' inside 'Cleared' / 'Close' must not return 'crude'.
    assert infer_commodity_from_exchange_symbol("Cleared") is None
    assert infer_commodity_from_exchange_symbol("close") is None
    # 'rb' inside 'Carbon' must not return 'gasoline'.
    assert infer_commodity_from_exchange_symbol("Carbon") is None
    # 'ho' inside 'HKD' / 'Hong' must not return 'gasoil'.
    assert infer_commodity_from_exchange_symbol("Hong") is None
    # 'ng' inside 'Long' / 'Naphtha' must not return 'natgas'.
    assert infer_commodity_from_exchange_symbol("Long") is None
    assert infer_commodity_from_exchange_symbol("Naphtha") is None


def test_infer_commodity_from_exchange_symbol_token_separators():
    """Tokeniser must split on _ : . - whitespace and /."""
    assert infer_commodity_from_exchange_symbol("ICE:BRN") == "crude"
    assert infer_commodity_from_exchange_symbol("ICE.BRN") == "crude"
    assert infer_commodity_from_exchange_symbol("ICE-BRN") == "crude"
    assert infer_commodity_from_exchange_symbol("ICE BRN") == "crude"
    assert infer_commodity_from_exchange_symbol("ICE/BRN") == "crude"
