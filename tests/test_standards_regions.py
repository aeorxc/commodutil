"""Tests for commodutil.standards.regions."""

from __future__ import annotations

from commodutil.standards.regions import (
    REGION_PATTERNS,
    VALID_REGIONS,
    is_valid_region,
    normalize_region,
)


def test_normalize_region_brent_nyh():
    assert normalize_region("Brent NYH") == "NYH"


def test_normalize_region_rbob_heuristic():
    # RBOB convention: always NY Harbor regardless of other tokens
    assert normalize_region("RBOB Feb25") == "NYH"


def test_normalize_region_rdam_substring():
    assert normalize_region("RDam fuel oil") == "Rott"


def test_normalize_region_singapore_substring():
    assert normalize_region("Singapore gasoil") == "Sing"


def test_normalize_region_sahara_does_not_match_ara():
    # "ara" is a short pattern; must use word boundaries, so "Saharan" must NOT match.
    assert normalize_region("Saharan crude") is None


def test_normalize_region_none_input():
    assert normalize_region(None) is None


def test_normalize_region_empty_string():
    assert normalize_region("") is None


def test_is_valid_region_true():
    assert is_valid_region("NYH") is True


def test_is_valid_region_false():
    assert is_valid_region("XXX") is False


def test_region_patterns_codes_match_valid_regions():
    codes = {code for code, _ in REGION_PATTERNS}
    assert codes == set(VALID_REGIONS)


def test_rbob_heuristic_takes_precedence_over_other_regions():
    # RBOB convention overrides any geographic pattern.
    assert normalize_region("RBOB Singapore") == "NYH"


def test_first_match_wins_for_overlapping_patterns():
    # REGION_PATTERNS is ordered; first hit wins. "New York USGC" matches
    # NYH before USGC because NYH is listed first.
    assert normalize_region("New York USGC") == "NYH"


def test_parity_with_curvemetadata_long_patterns_only():
    """Parity guard for LONG (substring) patterns only.

    curvemetadata.taxonomy.infer_region has a latent regex bug: it uses
    `rf"\\b...\\b"` (literal backslash-b) instead of `rf"\b...\b"` (word
    boundary), so its short patterns never match in production. This test
    confirms parity for the long patterns that ACTUALLY worked in
    curvemetadata. Short-pattern divergence is documented in
    regions.py module docstring.
    """
    try:
        from curvemetadata.taxonomy import infer_region
    except ImportError:
        import pytest

        pytest.skip("curvemetadata not available in this environment")

    long_pattern_cases = [
        "RBOB Feb25",  # RBOB heuristic — same in both
        "RDam fuel oil",
        "Singapore gasoil",
        "Saharan crude",
        "Los Angeles diesel",
        "Mediterranean fuel oil",
        "Unknown Location",
        "",
        None,
        "U.S. Atlantic Coast",
        "Rotterdam fuel",
    ]
    for text in long_pattern_cases:
        assert normalize_region(text) == infer_region(text), (
            f"Parity divergence for {text!r}: "
            f"new={normalize_region(text)!r}, "
            f"curvemetadata={infer_region(text)!r}"
        )


def test_short_pattern_parity_with_curvemetadata():
    """Short-pattern (len <= 3) regions used to diverge between
    normalize_region and curvemetadata.taxonomy.infer_region because
    the latter had a `\\b` regex bug (literal backslash-b instead of
    a word boundary). The bug was fixed in curvemetadata 2026-05;
    this test is now a regression guard against the bug returning.
    """
    try:
        from curvemetadata.taxonomy import infer_region
    except ImportError:
        import pytest

        pytest.skip("curvemetadata not available in this environment")

    # Only patterns of len <= 3 hit the previously-broken regex branch.
    # Among the configured patterns those are: "nyh", "ara", "med", "nwe".
    short_pattern_cases = [
        ("Brent NYH", "NYH"),
        ("NWE jet", "NWE"),
        ("ARA gasoil", "ARA"),
        ("Med fuel oil", "Med"),
        # Word-boundary protection: "ara" must NOT match inside "Saharan"
        ("Saharan crude", None),
    ]
    for text, expected in short_pattern_cases:
        assert normalize_region(text) == expected, (
            f"normalize_region({text!r}) should match {expected!r}"
        )
        assert infer_region(text) == expected, (
            f"curvemetadata.infer_region({text!r}) should match {expected!r} "
            f"(got {infer_region(text)!r}). If you see None for one of the "
            f"short patterns, the curvemetadata `\\b` regex bug has regressed."
        )


def test_facade_reexports_visible_at_top_level():
    # Smoke check: the eager facade exposes key symbols at top level.
    import commodutil

    names = set(dir(commodutil))
    assert "convert_price" in names
    assert "VALID_CURRENCY_TOKENS" in names
    assert "curyear" in names


def test_facade_unknown_attribute_raises():
    import commodutil

    import pytest

    with pytest.raises(AttributeError):
        _ = commodutil.this_symbol_does_not_exist
