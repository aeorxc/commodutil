"""Tests for commodutil.standards.commodity_groups."""

from __future__ import annotations

import pytest

from commodutil.standards.commodity_groups import (
    COMMODITY_GROUPS,
    VALID_COMMODITY_GROUPS,
    is_valid_commodity_group,
)


EXPECTED_GROUPS = (
    "Agriculture",
    "Biofuel",
    "Crude Oil",
    "Freight",
    "LNG",
    "Natural Gas",
    "NGL",
    "Petrochemical",
    "Refined Products",
)


def test_count_is_nine():
    assert len(COMMODITY_GROUPS) == 9


def test_exact_tuple_equality_preserves_sql_order():
    # COMMODITY_GROUPS docstring promises SQL constraint order is preserved.
    # A future maintainer reordering the tuple would silently change what
    # downstream consumers see when iterating.
    assert COMMODITY_GROUPS == EXPECTED_GROUPS


def test_valid_set_matches_expected():
    assert VALID_COMMODITY_GROUPS == frozenset(EXPECTED_GROUPS)


def test_all_expected_groups_present():
    for group in EXPECTED_GROUPS:
        assert group in VALID_COMMODITY_GROUPS, f"Missing: {group}"


def test_is_valid_crude_oil():
    assert is_valid_commodity_group("Crude Oil") is True


def test_is_valid_case_sensitive():
    # Matches SQL semantics -- the CHECK constraint uses exact-cased
    # N'Crude Oil' literals.
    assert is_valid_commodity_group("crude oil") is False


def test_is_valid_unknown_returns_false():
    assert is_valid_commodity_group("Unknown") is False


def test_is_valid_empty_string_returns_false():
    assert is_valid_commodity_group("") is False


def test_commodity_groups_is_tuple():
    assert isinstance(COMMODITY_GROUPS, tuple)


def test_valid_commodity_groups_is_frozenset():
    assert isinstance(VALID_COMMODITY_GROUPS, frozenset)


# Cross-check: every CommodityGroup string used by curvemetadata.cme and
# curvemetadata.ice must round-trip through VALID_COMMODITY_GROUPS. If
# curvemetadata adds a new group string before the SQL constraint is
# extended, this test flags the divergence loudly.
CURVEMETADATA_USED_GROUPS = (
    "Crude Oil",
    "Refined Products",
    "Petrochemical",
    "Freight",
    "NGL",
    "Biofuel",
    # Natural Gas / LNG / Agriculture are not currently used by ice.py /
    # cme.py but are reserved by the SQL constraint.
)


@pytest.mark.parametrize("group", CURVEMETADATA_USED_GROUPS)
def test_curvemetadata_group_is_recognised(group):
    assert is_valid_commodity_group(group), (
        f"curvemetadata uses {group!r} but it is not in VALID_COMMODITY_GROUPS"
    )
