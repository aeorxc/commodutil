"""Tests for commodutil.standards.analysis_types."""

from commodutil.standards.analysis_types import ANALYSIS_TYPES, infer_analysis_type


def test_recognises_crack():
    assert infer_analysis_type("Brent-Diesel Crack") == "crack"
    assert infer_analysis_type("WTI Crack Spread") == "crack"


def test_recognises_arb():
    assert infer_analysis_type("Singapore-LA Gasoline Arb") == "arb"


def test_recognises_diff():
    assert infer_analysis_type("Brent-WTI Diff") == "diff"
    assert infer_analysis_type("Gasoil-Heating Oil Spread") == "diff"


def test_default_outright():
    assert infer_analysis_type("Brent Front Month") == "outright"
    assert infer_analysis_type("WTI Cushing") == "outright"


def test_empty_and_none():
    assert infer_analysis_type("") == "outright"
    assert infer_analysis_type(None) == "outright"


def test_word_boundary_prevents_false_match():
    """Substring matching (old ICE behavior) would have wrongly returned
    'crack' for 'crackers' or 'arb' for 'arbitrary'. Word boundary fixes."""
    assert infer_analysis_type("crackers party") == "outright"
    assert infer_analysis_type("arbitrary value") == "outright"
    assert infer_analysis_type("trackers") == "outright"


def test_category_short_circuit():
    # CME convention: category="arb" wins regardless of text
    assert infer_analysis_type("WTI Front Month", category="arb") == "arb"
    assert infer_analysis_type("WTI Front Month", category="ARB") == "arb"
    # Other category values don't override
    assert infer_analysis_type("WTI Brent Diff", category="energy") == "diff"


def test_extra_text_appended():
    # ICE-style single-arg call still works (extra_text defaults to None)
    assert infer_analysis_type("WTI Crack Spread") == "crack"
    # CME-style multi-arg: sub_category contributes
    assert infer_analysis_type("WTI Brent", "crack") == "crack"
    assert infer_analysis_type("WTI Brent", None, None) == "outright"


def test_crack_priority_over_arb_diff():
    """Order matters: crack short-circuits before arb/diff checks."""
    assert infer_analysis_type("Gasoline Crack Arb") == "crack"


def test_analysis_types_constant():
    assert ANALYSIS_TYPES == ("crack", "arb", "diff", "outright")
