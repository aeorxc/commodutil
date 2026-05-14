"""Tests for commodutil.standards.commodities."""

from __future__ import annotations

import pytest

from commodutil.standards.commodities import (
    COMMODITY_CONVERSION_MAP,
    COMMODITY_KEYWORDS,
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


def test_parity_with_curvemetadata():
    """Verify COMMODITY_KEYWORDS / COMMODITY_CONVERSION_MAP are identical to
    the curvemetadata copies — guards against divergence during the migration."""
    try:
        from curvemetadata.common_maps import (
            COMMODITY_CONVERSION_MAP as cm_MAP,
            COMMODITY_KEYWORDS as cm_KW,
        )
    except ImportError:
        pytest.skip("curvemetadata not available")

    assert COMMODITY_KEYWORDS == cm_KW
    assert COMMODITY_CONVERSION_MAP == cm_MAP
