"""Canonical countries and membership-defined country groups.

Countries are delegated to pycountry, the ISO 3166-1 authority. Desk aliases
and membership-defined country groups remain the local vocabulary. Each group key
identifies a specific country basket, so callers cannot request an ambiguous
bare "Europe".

Trading-hub/pricing-location codes live in ``trading_hubs.py`` — a different
vocabulary.

To add a desk country spelling or country group, edit ``COUNTRY_ALIASES`` or
``COUNTRY_GROUP_ROWS`` below — nowhere else. All lookup maps, valid-key sets, and
memberships are derived from those definitions at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pycountry


VALID_COUNTRY_CODES: frozenset[str] = frozenset(
    country.alpha_2 for country in pycountry.countries
)

# Trade-data spellings not exposed as pycountry names.
COUNTRY_ALIASES: dict[str, str] = {
    "UAE": "AE",
    "Aland Islands": "AX",
    "Bosnia": "BA",
    "Saint Barthelemy": "BL",
    "Brunei": "BN",
    "The Bahamas": "BS",
    "Cocos Islands": "CC",
    "DR Congo": "CD",
    "Congo-Kinshasa": "CD",
    "Congo-Brazzaville": "CG",
    "Cote d'Ivoire": "CI",
    "Ivory Coast": "CI",
    "PRC": "CN",
    "Cape Verde": "CV",
    "Curacao": "CW",
    "Faeroe Islands": "FO",
    "UK": "GB",
    "U.K.": "GB",
    "Great Britain": "GB",
    "Britain": "GB",
    "The Gambia": "GM",
    "North Korea": "KP",
    "South Korea": "KR",
    "Republic of Korea": "KR",
    "Saint Martin": "MF",
    "Macedonia": "MK",
    "Burma": "MM",
    "Macau": "MO",
    "Holland": "NL",
    "The Netherlands": "NL",
    "The Philippines": "PH",
    "Pitcairn Islands": "PN",
    "State of Palestine": "PS",
    "Reunion": "RE",
    "Russia": "RU",
    "São Tomé and Príncipe": "ST",
    "Sint Maarten": "SX",
    "Swaziland": "SZ",
    "East Timor": "TL",
    "Turkey": "TR",
    "Turkiye": "TR",
    "Republic of China": "TW",
    "USA": "US",
    "U.S.": "US",
    "U.S.A.": "US",
    "Holy See": "VA",
    "US Virgin Islands": "VI",
    # From empirical vendor-vocabulary validation (see commodflows research).
    # ISO merges Bonaire and Sint Eustatius into BQ; sub-island detail is
    # lost by design.
    "Bosnia Herzegovina": "BA",
    "Bonaire": "BQ",
    "Caribbean Netherlands": "BQ",
    "Sint Eustatius": "BQ",
    "Congo (Democratic Republic)": "CD",
    "Democratic Republic of the Congo": "CD",
    "Congo (Republic)": "CG",
    "Cape Verde Islands": "CV",
    "Falkland Islands": "FK",
    "Micronesia": "FM",
    "Korea (North)": "KP",
    "Korea (South)": "KR",
    "St Lucia": "LC",
    "Saint Helena": "SH",
    "St Helena Island": "SH",
    "Trinidad & Tobago": "TT",
    "St Vincent": "VC",
    "Virgin Islands (British)": "VG",
}


def _build_country_name_map(aliases: Dict[str, str]) -> Dict[str, str]:
    """Build the case-folded name map and reject conflicting aliases."""
    result: Dict[str, str] = {}
    for country in pycountry.countries:
        for attribute in ("name", "official_name", "common_name"):
            name = getattr(country, attribute, None)
            if name is None:
                continue
            key = name.casefold()
            existing = result.get(key)
            if existing is not None and existing != country.alpha_2:
                raise ValueError(
                    f"pycountry name {name!r} maps to both "
                    f"{existing!r} and {country.alpha_2!r}"
                )
            result[key] = country.alpha_2

    for alias, code in aliases.items():
        if code not in VALID_COUNTRY_CODES:
            raise ValueError(
                f"country alias {alias!r} has unknown country code {code!r}"
            )
        key = alias.casefold()
        existing = result.get(key)
        if existing is not None and existing != code:
            raise ValueError(
                f"country alias {alias!r} maps to {code!r} but shadows "
                f"the pycountry resolution {existing!r}"
            )
        result[key] = code
    return result


COUNTRY_NAME_MAP: Dict[str, str] = _build_country_name_map(COUNTRY_ALIASES)


def canonical_country(text: object) -> Optional[str]:
    """Return an alpha-2 code for a code, country name, or alias."""
    if text is None:
        return None
    cleaned = str(text).strip()
    if not cleaned:
        return None
    code = cleaned.upper()
    if code in VALID_COUNTRY_CODES:
        return code
    return COUNTRY_NAME_MAP.get(cleaned.casefold())


def is_canonical_country(code: object) -> bool:
    """Return True only for an exact canonical alpha-2 country code."""
    return isinstance(code, str) and code in VALID_COUNTRY_CODES


@dataclass(frozen=True)
class CountryGroupRow:
    """One definition-named country group and its membership."""

    key: str
    description: str
    members: Tuple[str, ...]


# The one country-group table. Additions belong here and nowhere else.
COUNTRY_GROUP_ROWS: Tuple[CountryGroupRow, ...] = (
    CountryGroupRow(
        "EUROPE_WIDE",
        "Broad 43-country Europe basket",
        (
            "AD",
            "AL",
            "AT",
            "BA",
            "BE",
            "BG",
            "CH",
            "CY",
            "CZ",
            "DE",
            "DK",
            "EE",
            "ES",
            "FI",
            "FO",
            "FR",
            "GB",
            "GG",
            "GI",
            "GR",
            "HR",
            "HU",
            "IE",
            "IS",
            "IT",
            "JE",
            "LI",
            "LT",
            "LU",
            "LV",
            "ME",
            "MK",
            "MT",
            "NL",
            "NO",
            "PL",
            "PT",
            "RO",
            "RS",
            "SE",
            "SI",
            "SK",
            "TR",
        ),
    ),
    CountryGroupRow(
        "EUROPE_OECD",
        "27-country OECD Europe basket",
        (
            "AT",
            "BE",
            "CH",
            "CZ",
            "DE",
            "DK",
            "EE",
            "ES",
            "FI",
            "FR",
            "GB",
            "GR",
            "HU",
            "IE",
            "IS",
            "IT",
            "LT",
            "LU",
            "LV",
            "NL",
            "NO",
            "PL",
            "PT",
            "SE",
            "SI",
            "SK",
            "TR",
        ),
    ),
    CountryGroupRow(
        "NWE_15",
        "15-country North West Europe basket",
        (
            "BE",
            "DE",
            "DK",
            "FI",
            "FO",
            "FR",
            "GB",
            "GG",
            "IE",
            "IS",
            "JE",
            "LU",
            "NL",
            "NO",
            "SE",
        ),
    ),
    CountryGroupRow(
        "MED_19",
        "19-country Mediterranean basket",
        (
            "AD",
            "AL",
            "AT",
            "BA",
            "CH",
            "CY",
            "ES",
            "GI",
            "GR",
            "HR",
            "IT",
            "ME",
            "MK",
            "MT",
            "PS",
            "PT",
            "RS",
            "SI",
            "TR",
        ),
    ),
)


def _build_country_group_members(
    rows: Tuple[CountryGroupRow, ...],
) -> Dict[str, frozenset[str]]:
    result: Dict[str, frozenset[str]] = {}
    for row in rows:
        if row.key in result:
            raise ValueError(f"duplicate country-group key {row.key!r}")
        unknown = set(row.members) - VALID_COUNTRY_CODES
        if unknown:
            raise ValueError(
                f"country group {row.key!r} has unknown country codes: "
                f"{sorted(unknown)!r}"
            )
        if len(set(row.members)) != len(row.members):
            raise ValueError(f"country group {row.key!r} has duplicate members")
        result[row.key] = frozenset(row.members)
    return result


COUNTRY_GROUP_MEMBERS: Dict[str, frozenset[str]] = _build_country_group_members(
    COUNTRY_GROUP_ROWS
)
VALID_COUNTRY_GROUP_KEYS: frozenset[str] = frozenset(COUNTRY_GROUP_MEMBERS)


def is_canonical_country_group(key: object) -> bool:
    """Return True only for an exact definition-named country-group key."""
    return isinstance(key, str) and key in VALID_COUNTRY_GROUP_KEYS


def country_group_members(key: str) -> frozenset[str]:
    """Return the immutable country membership for a canonical group key."""
    return COUNTRY_GROUP_MEMBERS[key]


def country_in_group(country: object, group: str) -> bool:
    """Return whether a lenient country input belongs to a canonical group."""
    members = COUNTRY_GROUP_MEMBERS[group]
    code = canonical_country(country)
    return code in members if code is not None else False


__all__ = [
    "COUNTRY_ALIASES",
    "VALID_COUNTRY_CODES",
    "COUNTRY_NAME_MAP",
    "canonical_country",
    "is_canonical_country",
    "CountryGroupRow",
    "COUNTRY_GROUP_ROWS",
    "VALID_COUNTRY_GROUP_KEYS",
    "COUNTRY_GROUP_MEMBERS",
    "is_canonical_country_group",
    "country_group_members",
    "country_in_group",
]
