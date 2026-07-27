"""Canonical countries and membership-defined physical regions.

Countries are delegated to pycountry, the ISO 3166-1 authority. Desk aliases
and membership-defined regions remain the local vocabulary. Each region key
identifies a specific country basket, so callers cannot request an ambiguous
bare "Europe".

To add a desk country spelling or region, edit ``COUNTRY_ALIASES`` or
``REGION_ROWS`` below — nowhere else. All lookup maps, valid-key sets, and
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
}


def _build_country_name_map(aliases: dict[str, str]) -> Dict[str, str]:
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
class RegionRow:
    """One definition-named physical region and its country membership."""

    key: str
    description: str
    members: Tuple[str, ...]


# The one region table. Additions belong here and nowhere else.
REGION_ROWS: Tuple[RegionRow, ...] = (
    RegionRow(
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
    RegionRow(
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
    RegionRow(
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
    RegionRow(
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


def _build_region_members(
    rows: Tuple[RegionRow, ...],
) -> Dict[str, frozenset[str]]:
    result: Dict[str, frozenset[str]] = {}
    for row in rows:
        if row.key in result:
            raise ValueError(f"duplicate region key {row.key!r}")
        unknown = set(row.members) - VALID_COUNTRY_CODES
        if unknown:
            raise ValueError(
                f"region {row.key!r} has unknown country codes: {sorted(unknown)!r}"
            )
        if len(set(row.members)) != len(row.members):
            raise ValueError(f"region {row.key!r} has duplicate members")
        result[row.key] = frozenset(row.members)
    return result


REGION_MEMBERS: Dict[str, frozenset[str]] = _build_region_members(REGION_ROWS)
VALID_REGION_KEYS: frozenset[str] = frozenset(REGION_MEMBERS)


def is_canonical_region(key: object) -> bool:
    """Return True only for an exact definition-named region key."""
    return isinstance(key, str) and key in VALID_REGION_KEYS


def region_members(key: str) -> frozenset[str]:
    """Return the immutable country membership for a canonical region key."""
    return REGION_MEMBERS[key]


def country_in_region(country: object, region: str) -> bool:
    """Return whether a lenient country input belongs to a canonical region."""
    code = canonical_country(country)
    return code in REGION_MEMBERS[region] if code is not None else False


__all__ = [
    "COUNTRY_ALIASES",
    "VALID_COUNTRY_CODES",
    "COUNTRY_NAME_MAP",
    "canonical_country",
    "is_canonical_country",
    "RegionRow",
    "REGION_ROWS",
    "VALID_REGION_KEYS",
    "REGION_MEMBERS",
    "is_canonical_region",
    "region_members",
    "country_in_region",
]
