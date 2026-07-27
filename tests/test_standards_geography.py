"""Tests for commodutil.standards.geography."""

import pycountry
import pytest

from commodutil.standards.geography import (
    COUNTRY_ALIASES,
    COUNTRY_NAME_MAP,
    REGION_MEMBERS,
    REGION_ROWS,
    VALID_COUNTRY_CODES,
    VALID_REGION_KEYS,
    _build_country_name_map,
    canonical_country,
    country_in_region,
    is_canonical_country,
    is_canonical_region,
    region_members,
)


def test_country_universe_matches_pycountry():
    pycountry_codes = frozenset(country.alpha_2 for country in pycountry.countries)
    assert len(VALID_COUNTRY_CODES) >= 245
    assert VALID_COUNTRY_CODES == pycountry_codes


def test_country_alias_spot_checks():
    assert COUNTRY_ALIASES["UK"] == "GB"
    assert COUNTRY_ALIASES["UAE"] == "AE"
    assert COUNTRY_ALIASES["Holland"] == "NL"
    assert canonical_country("UK") == "GB"
    assert canonical_country("UAE") == "AE"
    assert canonical_country("Holland") == "NL"
    assert canonical_country("unknown") is None


@pytest.mark.parametrize(
    "value,expected",
    [
        ("Korea (South)", "KR"),
        ("Korea (North)", "KP"),
        ("Congo (Democratic Republic)", "CD"),
        ("Congo (Republic)", "CG"),
        ("Falkland Islands", "FK"),
        ("Bosnia Herzegovina", "BA"),
        ("Sint Eustatius", "BQ"),
    ],
)
def test_empirically_validated_country_aliases(value, expected):
    assert canonical_country(value) == expected


def test_kosovo_is_not_an_iso_country():
    # Kosovo has no ISO 3166-1 code and belongs in vendor crosswalks.
    assert canonical_country("Kosovo") is None


def test_country_name_builder_rejects_cross_country_alias_collision():
    with pytest.raises(ValueError, match="shadows"):
        _build_country_name_map({"United Kingdom": "US"})


def test_country_name_map_contains_pycountry_names_and_aliases():
    assert COUNTRY_NAME_MAP["united kingdom"] == "GB"
    assert COUNTRY_NAME_MAP["uk"] == "GB"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("GB", "GB"),
        ("gb", "GB"),
        ("UK", "GB"),
        ("uk", "GB"),
        ("U.K.", "GB"),
        ("United Kingdom", "GB"),
        ("USA", "US"),
        ("United States", "US"),
        ("South Korea", "KR"),
        ("Republic of Korea", "KR"),
        ("Turkey", "TR"),
        ("Turkiye", "TR"),
        ("UAE", "AE"),
        ("Holland", "NL"),
        ("Netherlands", "NL"),
        ("Russia", "RU"),
        ("  ivory coast  ", "CI"),
        ("unknown", None),
        ("", None),
        (None, None),
    ],
)
def test_canonical_country_is_lenient(value, expected):
    assert canonical_country(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("GB", True),
        ("US", True),
        ("gb", False),
        (" GB ", False),
        ("United Kingdom", False),
        ("", False),
        (None, False),
    ],
)
def test_is_canonical_country_is_strict(value, expected):
    assert is_canonical_country(value) is expected


def test_region_keys_are_exactly_the_four_starter_definitions():
    expected = {"EUROPE_WIDE", "EUROPE_OECD", "NWE_15", "MED_19"}
    assert {row.key for row in REGION_ROWS} == expected
    assert VALID_REGION_KEYS == frozenset(expected)
    assert set(REGION_MEMBERS) == expected


def test_region_member_counts_match_definitions():
    assert len(region_members("EUROPE_WIDE")) == 43
    assert len(region_members("EUROPE_OECD")) == 27
    assert len(region_members("NWE_15")) == 15
    assert len(region_members("MED_19")) == 19


def test_every_region_member_is_a_canonical_country():
    for key, members in REGION_MEMBERS.items():
        assert members <= VALID_COUNTRY_CODES, key


def test_documented_region_relationships():
    assert region_members("NWE_15") < region_members("EUROPE_WIDE")
    assert region_members("NWE_15").isdisjoint(region_members("MED_19"))
    assert region_members("MED_19") - region_members("EUROPE_WIDE") == {"PS"}


def test_region_predicate_is_strict():
    assert is_canonical_region("NWE_15") is True
    assert is_canonical_region("nwe_15") is False
    assert is_canonical_region(" NWE_15 ") is False
    assert is_canonical_region("Europe") is False


def test_region_members_rejects_unknown_key():
    with pytest.raises(KeyError):
        region_members("Europe")


def test_country_in_region_uses_lenient_country_resolution():
    assert country_in_region("GB", "NWE_15") is True
    assert country_in_region("uk", "NWE_15") is True
    assert country_in_region(" United Kingdom ", "NWE_15") is True
    assert country_in_region("Turkey", "MED_19") is True
    assert country_in_region("unknown", "NWE_15") is False


def test_geography_names_exported_from_standards_facade():
    from commodutil import standards

    assert standards.canonical_country("UAE") == "AE"
    assert standards.is_canonical_country("AE")
    assert standards.is_canonical_region("MED_19")
    assert standards.country_in_region("Turkey", "MED_19")
