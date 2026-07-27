"""Tests for commodutil.standards.trading_hubs."""

from __future__ import annotations

from commodutil.standards.trading_hubs import (
    TRADING_HUB_PATTERNS,
    VALID_TRADING_HUBS,
    is_valid_trading_hub,
    normalize_trading_hub,
)


def test_normalize_trading_hub_brent_nyh():
    assert normalize_trading_hub("Brent NYH") == "NYH"


def test_normalize_trading_hub_rbob_heuristic():
    # RBOB convention: always NY Harbor regardless of other tokens
    assert normalize_trading_hub("RBOB Feb25") == "NYH"


def test_normalize_trading_hub_carbob_is_not_nyh():
    """CARBOB is California gasoline (Los Angeles / USWC), NOT NY Harbor.

    Regression guard against the prior substring bug where `"rbob" in lower`
    matched "carbob" and mis-classified California gasoline as NYH. The fix
    uses a word-boundary regex plus an explicit CARBOB short-circuit that
    maps to "LA".
    """
    assert normalize_trading_hub("CARBOB") != "NYH"
    assert normalize_trading_hub("CARBOB") == "LA"
    assert normalize_trading_hub("Los Angeles CARBOB") == "LA"
    # Sanity: plain RBOB still resolves to NYH.
    assert normalize_trading_hub("RBOB") == "NYH"


def test_normalize_trading_hub_rdam_substring():
    assert normalize_trading_hub("RDam fuel oil") == "Rott"


def test_normalize_trading_hub_singapore_substring():
    assert normalize_trading_hub("Singapore gasoil") == "Sing"


def test_normalize_trading_hub_sahara_does_not_match_ara():
    # "ara" is a short pattern; must use word boundaries, so "Saharan" must NOT match.
    assert normalize_trading_hub("Saharan crude") is None


def test_normalize_trading_hub_none_input():
    assert normalize_trading_hub(None) is None


def test_normalize_trading_hub_empty_string():
    assert normalize_trading_hub("") is None


def test_is_valid_trading_hub_true():
    assert is_valid_trading_hub("NYH") is True


def test_is_valid_trading_hub_false():
    assert is_valid_trading_hub("XXX") is False


def test_trading_hub_patterns_codes_match_valid_trading_hubs():
    codes = {code for code, _ in TRADING_HUB_PATTERNS}
    assert codes == set(VALID_TRADING_HUBS)


def test_rbob_heuristic_takes_precedence_over_other_trading_hubs():
    # RBOB convention overrides any geographic pattern.
    assert normalize_trading_hub("RBOB Singapore") == "NYH"


def test_first_match_wins_for_overlapping_patterns():
    # TRADING_HUB_PATTERNS is ordered; first hit wins. "New York USGC" matches
    # NYH before USGC because NYH is listed first.
    assert normalize_trading_hub("New York USGC") == "NYH"


def test_parity_with_curvemetadata_long_patterns_only():
    """Parity guard for LONG (substring) patterns only.

    curvemetadata.taxonomy.infer_region has a latent regex bug: it uses
    `rf"\\b...\\b"` (literal backslash-b) instead of `rf"\b...\b"` (word
    boundary), so its short patterns never match in production. This test
    confirms parity for the long patterns that ACTUALLY worked in
    curvemetadata. Short-pattern divergence is documented in
    trading_hubs.py module docstring.
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
        assert normalize_trading_hub(text) == infer_region(text), (
            f"Parity divergence for {text!r}: "
            f"new={normalize_trading_hub(text)!r}, "
            f"curvemetadata={infer_region(text)!r}"
        )


def test_short_pattern_parity_with_curvemetadata():
    """Short-pattern (len <= 3) trading hubs used to diverge between
    normalize_trading_hub and curvemetadata.taxonomy.infer_region because
    the latter had a `\\b` regex bug (literal backslash-b instead of
    a word boundary). The bug was fixed in curvemetadata 1.4.0 (commit
    e8ba20e on the 1.2.0 release branch); this test is a regression
    guard once that fix is published.

    Skips gracefully when the installed curvemetadata still has the
    bug (e.g. CI envs pulling curvemetadata < 1.4.0 from the package
    index during the release-window between commodutil 3.10.0 and
    curvemetadata 1.4.0 publishing).
    """
    import pytest

    try:
        from curvemetadata.taxonomy import infer_region
    except ImportError:
        pytest.skip("curvemetadata not available in this environment")

    # Probe: does the installed curvemetadata have the `\b` fix? Any of
    # the short patterns would return None if the bug is still present.
    if infer_region("Brent NYH") is None:
        pytest.skip(
            "Installed curvemetadata has the pre-1.4.0 `\\b` regex bug "
            "(short-pattern trading hubs never match). Upgrade curvemetadata "
            "to >=1.4.0 to enable this parity test."
        )

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
        assert normalize_trading_hub(text) == expected, (
            f"normalize_trading_hub({text!r}) should match {expected!r}"
        )
        assert infer_region(text) == expected, (
            f"curvemetadata.infer_region({text!r}) should match {expected!r} "
            f"(got {infer_region(text)!r}). The `\\b` fix is present (probe "
            f"above passed) but parity broke on this input -- investigate."
        )


def test_facade_reexports_visible_at_top_level():
    # Smoke check: the lazy facade exposes key symbols via dir().
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


def test_facade_is_lazy_bare_import_does_not_load_convfactors():
    """Regression guard for the PEP 562 lazy facade. `import commodutil`
    must NOT eagerly load convfactors (pint registry + Commodity
    dataclass init). Reverting the facade to eager would trip this and
    the cost evidence is documented in the commit message that
    introduced the lazy pattern (~3.3s -> ~2ms speedup)."""
    import importlib
    import sys

    # Force a clean import to measure the bare cost.
    for mod in [
        "commodutil",
        "commodutil.convfactors",
        "commodutil.dates",
        "commodutil.forwards",
        "commodutil.pandasutil",
        "commodutil.transforms",
    ]:
        sys.modules.pop(mod, None)

    importlib.import_module("commodutil")
    assert "commodutil.convfactors" not in sys.modules, (
        "Lazy facade is broken — `import commodutil` triggered convfactors "
        "load. Restore the PEP 562 __getattr__ pattern in commodutil/__init__.py."
    )


def test_facade_lazy_load_resolves_and_caches():
    """First access of a facade symbol loads the source submodule and
    caches the resolved value back into globals."""
    import importlib
    import sys

    for mod in [
        "commodutil",
        "commodutil.convfactors",
    ]:
        sys.modules.pop(mod, None)

    commodutil = importlib.import_module("commodutil")
    assert "commodutil.convfactors" not in sys.modules

    fn = commodutil.convert_price  # triggers lazy load
    assert "commodutil.convfactors" in sys.modules
    assert callable(fn)

    # Second access hits the cache (still works, same object)
    assert commodutil.convert_price is fn


# ---- CRUDE_GRADE_REGIONS tests ----


def test_crude_grade_regions_keys_match_source_file():
    """CRUDE_GRADE_REGIONS keys are byte-identical to the source layout in
    oilpricingcharts.symbols_config_crudediffs. Re-keying would break the
    follow-up migration."""
    from commodutil.standards.trading_hubs import CRUDE_GRADE_REGIONS

    expected = {
        "north_sea",
        "waf",
        "nafrica",
        "russian",
        "us_midcon",
        "us_texas",
        "us_louisiana",
        "canadian",
        "latam_wti",
        "asia_pacific",
        "middle_east",
    }
    assert set(CRUDE_GRADE_REGIONS.keys()) == expected


def test_crude_grade_regions_values_are_non_empty_tuples():
    from commodutil.standards.trading_hubs import CRUDE_GRADE_REGIONS

    for key, grades in CRUDE_GRADE_REGIONS.items():
        assert isinstance(grades, tuple), f"{key} must be a tuple"
        assert grades, f"{key} must have at least one grade"
        for g in grades:
            assert isinstance(g, str) and g, (
                f"{key}: grade {g!r} must be a non-empty str"
            )


def test_crude_grade_regions_known_grades_present():
    """Spot-check a few well-known grades land in the right region."""
    from commodutil.standards.trading_hubs import CRUDE_GRADE_REGIONS

    assert "Forties" in CRUDE_GRADE_REGIONS["north_sea"]
    assert "Bonny Light" in CRUDE_GRADE_REGIONS["waf"]
    assert "ESPO" in CRUDE_GRADE_REGIONS["russian"]
    assert "Dubai" in CRUDE_GRADE_REGIONS["middle_east"]
    assert "Tapis" in CRUDE_GRADE_REGIONS["asia_pacific"]


def test_valid_crude_grade_regions_matches_dict():
    from commodutil.standards.trading_hubs import (
        CRUDE_GRADE_REGIONS,
        VALID_CRUDE_GRADE_REGIONS,
    )

    assert VALID_CRUDE_GRADE_REGIONS == frozenset(CRUDE_GRADE_REGIONS.keys())


def test_is_crude_grade_region():
    from commodutil.standards.trading_hubs import is_crude_grade_region

    assert is_crude_grade_region("north_sea") is True
    assert is_crude_grade_region("waf") is True
    assert is_crude_grade_region("not_a_region") is False
    assert (
        is_crude_grade_region("NYH") is False
    )  # canonical region code, not a grade region


# ---- MEG / Japan additions ----


def test_meg_is_valid_trading_hub():
    assert is_valid_trading_hub("MEG") is True


def test_japan_is_valid_trading_hub():
    assert is_valid_trading_hub("Japan") is True


def test_normalize_trading_hub_meg_aliases():
    assert normalize_trading_hub("MEG fuel oil") == "MEG"
    assert normalize_trading_hub("Middle East Gulf naphtha") == "MEG"
    assert normalize_trading_hub("Arabian Gulf gasoil") == "MEG"
    assert normalize_trading_hub("Persian Gulf cargo") == "MEG"


def test_normalize_trading_hub_japan():
    assert normalize_trading_hub("Japan gasoil") == "Japan"


# ---- Sibling-vocab invariant ----


def test_crude_grade_regions_and_valid_trading_hubs_do_not_overlap():
    """CRUDE_GRADE_REGIONS (producer-side grade groupings, snake_case e.g.
    'north_sea') and VALID_TRADING_HUBS (canonical delivery / pricing-location
    codes, e.g. 'NWE') describe different things and MUST stay disjoint.
    Any overlap would be a vocabulary bug."""
    from commodutil.standards.trading_hubs import VALID_CRUDE_GRADE_REGIONS

    assert VALID_TRADING_HUBS.isdisjoint(VALID_CRUDE_GRADE_REGIONS)


def test_new_vocabs_exposed_via_standards_facade():
    from commodutil.standards import (
        CRUDE_GRADE_REGIONS,
        VALID_TRADING_HUBS,
        is_crude_grade_region,
        is_valid_trading_hub,
    )

    assert "north_sea" in CRUDE_GRADE_REGIONS
    assert "NWE" in VALID_TRADING_HUBS
    assert "MEG" in VALID_TRADING_HUBS
    assert "Japan" in VALID_TRADING_HUBS
    assert is_crude_grade_region("waf")
    assert is_valid_trading_hub("USGC")
