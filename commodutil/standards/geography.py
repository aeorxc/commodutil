"""Canonical countries and membership-defined physical regions.

Regions are named by definition: each key identifies a specific country
basket, so callers cannot request an ambiguous bare "Europe".

To add a country or region, edit ``COUNTRY_ROWS`` or ``REGION_ROWS`` below —
nowhere else. All lookup maps, valid-key sets, and memberships are derived from
those tables at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class CountryRow:
    """One ISO 3166-1 alpha-2 country and its accepted names."""

    code: str
    name: str
    aliases: Tuple[str, ...] = ()


# The one country table. Codes and names follow ISO 3166-1; aliases capture
# common English alternatives used in commodity data.
COUNTRY_ROWS: Tuple[CountryRow, ...] = (
    CountryRow("AD", "Andorra"),
    CountryRow("AE", "United Arab Emirates", ("UAE",)),
    CountryRow("AF", "Afghanistan"),
    CountryRow("AG", "Antigua and Barbuda"),
    CountryRow("AI", "Anguilla"),
    CountryRow("AL", "Albania"),
    CountryRow("AM", "Armenia"),
    CountryRow("AO", "Angola"),
    CountryRow("AQ", "Antarctica"),
    CountryRow("AR", "Argentina"),
    CountryRow("AS", "American Samoa"),
    CountryRow("AT", "Austria"),
    CountryRow("AU", "Australia"),
    CountryRow("AW", "Aruba"),
    CountryRow("AX", "Åland Islands", ("Aland Islands",)),
    CountryRow("AZ", "Azerbaijan"),
    CountryRow("BA", "Bosnia and Herzegovina", ("Bosnia",)),
    CountryRow("BB", "Barbados"),
    CountryRow("BD", "Bangladesh"),
    CountryRow("BE", "Belgium"),
    CountryRow("BF", "Burkina Faso"),
    CountryRow("BG", "Bulgaria"),
    CountryRow("BH", "Bahrain"),
    CountryRow("BI", "Burundi"),
    CountryRow("BJ", "Benin"),
    CountryRow("BL", "Saint Barthélemy", ("Saint Barthelemy",)),
    CountryRow("BM", "Bermuda"),
    CountryRow("BN", "Brunei", ("Brunei Darussalam",)),
    CountryRow("BO", "Bolivia", ("Plurinational State of Bolivia",)),
    CountryRow("BQ", "Bonaire, Sint Eustatius and Saba"),
    CountryRow("BR", "Brazil"),
    CountryRow("BS", "Bahamas", ("The Bahamas",)),
    CountryRow("BT", "Bhutan"),
    CountryRow("BV", "Bouvet Island"),
    CountryRow("BW", "Botswana"),
    CountryRow("BY", "Belarus"),
    CountryRow("BZ", "Belize"),
    CountryRow("CA", "Canada"),
    CountryRow("CC", "Cocos (Keeling) Islands", ("Cocos Islands",)),
    CountryRow(
        "CD",
        "Democratic Republic of the Congo",
        ("DR Congo", "Congo-Kinshasa"),
    ),
    CountryRow("CF", "Central African Republic"),
    CountryRow("CG", "Republic of the Congo", ("Congo", "Congo-Brazzaville")),
    CountryRow("CH", "Switzerland"),
    CountryRow("CI", "Côte d'Ivoire", ("Cote d'Ivoire", "Ivory Coast")),
    CountryRow("CK", "Cook Islands"),
    CountryRow("CL", "Chile"),
    CountryRow("CM", "Cameroon"),
    CountryRow("CN", "China", ("People's Republic of China", "PRC")),
    CountryRow("CO", "Colombia"),
    CountryRow("CR", "Costa Rica"),
    CountryRow("CU", "Cuba"),
    CountryRow("CV", "Cabo Verde", ("Cape Verde",)),
    CountryRow("CW", "Curaçao", ("Curacao",)),
    CountryRow("CX", "Christmas Island"),
    CountryRow("CY", "Cyprus"),
    CountryRow("CZ", "Czechia", ("Czech Republic",)),
    CountryRow("DE", "Germany"),
    CountryRow("DJ", "Djibouti"),
    CountryRow("DK", "Denmark"),
    CountryRow("DM", "Dominica"),
    CountryRow("DO", "Dominican Republic"),
    CountryRow("DZ", "Algeria"),
    CountryRow("EC", "Ecuador"),
    CountryRow("EE", "Estonia"),
    CountryRow("EG", "Egypt"),
    CountryRow("EH", "Western Sahara"),
    CountryRow("ER", "Eritrea"),
    CountryRow("ES", "Spain"),
    CountryRow("ET", "Ethiopia"),
    CountryRow("FI", "Finland"),
    CountryRow("FJ", "Fiji"),
    CountryRow("FK", "Falkland Islands", ("Falkland Islands (Malvinas)",)),
    CountryRow("FM", "Micronesia", ("Federated States of Micronesia",)),
    CountryRow("FO", "Faroe Islands", ("Faeroe Islands",)),
    CountryRow("FR", "France"),
    CountryRow("GA", "Gabon"),
    CountryRow(
        "GB",
        "United Kingdom",
        ("UK", "U.K.", "Great Britain", "Britain"),
    ),
    CountryRow("GD", "Grenada"),
    CountryRow("GE", "Georgia"),
    CountryRow("GF", "French Guiana"),
    CountryRow("GG", "Guernsey"),
    CountryRow("GH", "Ghana"),
    CountryRow("GI", "Gibraltar"),
    CountryRow("GL", "Greenland"),
    CountryRow("GM", "Gambia", ("The Gambia",)),
    CountryRow("GN", "Guinea"),
    CountryRow("GP", "Guadeloupe"),
    CountryRow("GQ", "Equatorial Guinea"),
    CountryRow("GR", "Greece"),
    CountryRow("GS", "South Georgia and the South Sandwich Islands"),
    CountryRow("GT", "Guatemala"),
    CountryRow("GU", "Guam"),
    CountryRow("GW", "Guinea-Bissau"),
    CountryRow("GY", "Guyana"),
    CountryRow("HK", "Hong Kong"),
    CountryRow("HM", "Heard Island and McDonald Islands"),
    CountryRow("HN", "Honduras"),
    CountryRow("HR", "Croatia"),
    CountryRow("HT", "Haiti"),
    CountryRow("HU", "Hungary"),
    CountryRow("ID", "Indonesia"),
    CountryRow("IE", "Ireland"),
    CountryRow("IL", "Israel"),
    CountryRow("IM", "Isle of Man"),
    CountryRow("IN", "India"),
    CountryRow("IO", "British Indian Ocean Territory"),
    CountryRow("IQ", "Iraq"),
    CountryRow("IR", "Iran", ("Islamic Republic of Iran",)),
    CountryRow("IS", "Iceland"),
    CountryRow("IT", "Italy"),
    CountryRow("JE", "Jersey"),
    CountryRow("JM", "Jamaica"),
    CountryRow("JO", "Jordan"),
    CountryRow("JP", "Japan"),
    CountryRow("KE", "Kenya"),
    CountryRow("KG", "Kyrgyzstan"),
    CountryRow("KH", "Cambodia"),
    CountryRow("KI", "Kiribati"),
    CountryRow("KM", "Comoros"),
    CountryRow("KN", "Saint Kitts and Nevis"),
    CountryRow(
        "KP",
        "North Korea",
        ("Democratic People's Republic of Korea",),
    ),
    CountryRow("KR", "South Korea", ("Republic of Korea",)),
    CountryRow("KW", "Kuwait"),
    CountryRow("KY", "Cayman Islands"),
    CountryRow("KZ", "Kazakhstan"),
    CountryRow("LA", "Laos", ("Lao People's Democratic Republic",)),
    CountryRow("LB", "Lebanon"),
    CountryRow("LC", "Saint Lucia"),
    CountryRow("LI", "Liechtenstein"),
    CountryRow("LK", "Sri Lanka"),
    CountryRow("LR", "Liberia"),
    CountryRow("LS", "Lesotho"),
    CountryRow("LT", "Lithuania"),
    CountryRow("LU", "Luxembourg"),
    CountryRow("LV", "Latvia"),
    CountryRow("LY", "Libya"),
    CountryRow("MA", "Morocco"),
    CountryRow("MC", "Monaco"),
    CountryRow("MD", "Moldova", ("Republic of Moldova",)),
    CountryRow("ME", "Montenegro"),
    CountryRow("MF", "Saint Martin (French part)", ("Saint Martin",)),
    CountryRow("MG", "Madagascar"),
    CountryRow("MH", "Marshall Islands"),
    CountryRow("MK", "North Macedonia", ("Macedonia",)),
    CountryRow("ML", "Mali"),
    CountryRow("MM", "Myanmar", ("Burma",)),
    CountryRow("MN", "Mongolia"),
    CountryRow("MO", "Macao", ("Macau",)),
    CountryRow("MP", "Northern Mariana Islands"),
    CountryRow("MQ", "Martinique"),
    CountryRow("MR", "Mauritania"),
    CountryRow("MS", "Montserrat"),
    CountryRow("MT", "Malta"),
    CountryRow("MU", "Mauritius"),
    CountryRow("MV", "Maldives"),
    CountryRow("MW", "Malawi"),
    CountryRow("MX", "Mexico"),
    CountryRow("MY", "Malaysia"),
    CountryRow("MZ", "Mozambique"),
    CountryRow("NA", "Namibia"),
    CountryRow("NC", "New Caledonia"),
    CountryRow("NE", "Niger"),
    CountryRow("NF", "Norfolk Island"),
    CountryRow("NG", "Nigeria"),
    CountryRow("NI", "Nicaragua"),
    CountryRow("NL", "Netherlands", ("Holland", "The Netherlands")),
    CountryRow("NO", "Norway"),
    CountryRow("NP", "Nepal"),
    CountryRow("NR", "Nauru"),
    CountryRow("NU", "Niue"),
    CountryRow("NZ", "New Zealand"),
    CountryRow("OM", "Oman"),
    CountryRow("PA", "Panama"),
    CountryRow("PE", "Peru"),
    CountryRow("PF", "French Polynesia"),
    CountryRow("PG", "Papua New Guinea"),
    CountryRow("PH", "Philippines", ("The Philippines",)),
    CountryRow("PK", "Pakistan"),
    CountryRow("PL", "Poland"),
    CountryRow("PM", "Saint Pierre and Miquelon"),
    CountryRow("PN", "Pitcairn", ("Pitcairn Islands",)),
    CountryRow("PR", "Puerto Rico"),
    CountryRow("PS", "Palestine", ("State of Palestine",)),
    CountryRow("PT", "Portugal"),
    CountryRow("PW", "Palau"),
    CountryRow("PY", "Paraguay"),
    CountryRow("QA", "Qatar"),
    CountryRow("RE", "Réunion", ("Reunion",)),
    CountryRow("RO", "Romania"),
    CountryRow("RS", "Serbia"),
    CountryRow("RU", "Russian Federation", ("Russia",)),
    CountryRow("RW", "Rwanda"),
    CountryRow("SA", "Saudi Arabia"),
    CountryRow("SB", "Solomon Islands"),
    CountryRow("SC", "Seychelles"),
    CountryRow("SD", "Sudan"),
    CountryRow("SE", "Sweden"),
    CountryRow("SG", "Singapore"),
    CountryRow("SH", "Saint Helena, Ascension and Tristan da Cunha"),
    CountryRow("SI", "Slovenia"),
    CountryRow("SJ", "Svalbard and Jan Mayen"),
    CountryRow("SK", "Slovakia"),
    CountryRow("SL", "Sierra Leone"),
    CountryRow("SM", "San Marino"),
    CountryRow("SN", "Senegal"),
    CountryRow("SO", "Somalia"),
    CountryRow("SR", "Suriname"),
    CountryRow("SS", "South Sudan"),
    CountryRow("ST", "Sao Tome and Principe", ("São Tomé and Príncipe",)),
    CountryRow("SV", "El Salvador"),
    CountryRow("SX", "Sint Maarten (Dutch part)", ("Sint Maarten",)),
    CountryRow("SY", "Syria", ("Syrian Arab Republic",)),
    CountryRow("SZ", "Eswatini", ("Swaziland",)),
    CountryRow("TC", "Turks and Caicos Islands"),
    CountryRow("TD", "Chad"),
    CountryRow("TF", "French Southern Territories"),
    CountryRow("TG", "Togo"),
    CountryRow("TH", "Thailand"),
    CountryRow("TJ", "Tajikistan"),
    CountryRow("TK", "Tokelau"),
    CountryRow("TL", "Timor-Leste", ("East Timor",)),
    CountryRow("TM", "Turkmenistan"),
    CountryRow("TN", "Tunisia"),
    CountryRow("TO", "Tonga"),
    CountryRow("TR", "Türkiye", ("Turkey", "Turkiye")),
    CountryRow("TT", "Trinidad and Tobago"),
    CountryRow("TV", "Tuvalu"),
    CountryRow("TW", "Taiwan", ("Republic of China",)),
    CountryRow("TZ", "Tanzania", ("United Republic of Tanzania",)),
    CountryRow("UA", "Ukraine"),
    CountryRow("UG", "Uganda"),
    CountryRow("UM", "United States Minor Outlying Islands"),
    CountryRow(
        "US",
        "United States",
        ("USA", "U.S.", "U.S.A.", "United States of America"),
    ),
    CountryRow("UY", "Uruguay"),
    CountryRow("UZ", "Uzbekistan"),
    CountryRow("VA", "Vatican City", ("Holy See",)),
    CountryRow("VC", "Saint Vincent and the Grenadines"),
    CountryRow("VE", "Venezuela", ("Bolivarian Republic of Venezuela",)),
    CountryRow("VG", "British Virgin Islands", ("Virgin Islands, British",)),
    CountryRow(
        "VI",
        "U.S. Virgin Islands",
        ("US Virgin Islands", "Virgin Islands, U.S."),
    ),
    CountryRow("VN", "Vietnam", ("Viet Nam",)),
    CountryRow("VU", "Vanuatu"),
    CountryRow("WF", "Wallis and Futuna"),
    CountryRow("WS", "Samoa"),
    CountryRow("YE", "Yemen"),
    CountryRow("YT", "Mayotte"),
    CountryRow("ZA", "South Africa"),
    CountryRow("ZM", "Zambia"),
    CountryRow("ZW", "Zimbabwe"),
)


def _build_country_name_map(
    rows: Tuple[CountryRow, ...],
) -> Dict[str, str]:
    """Build the case-folded name map, rejecting cross-country collisions."""
    result: Dict[str, str] = {}
    for row in rows:
        for name in (row.name, *row.aliases):
            key = name.casefold()
            existing = result.get(key)
            if existing is not None and existing != row.code:
                raise ValueError(
                    f"country name or alias {name!r} maps to both "
                    f"{existing!r} and {row.code!r}"
                )
            result[key] = row.code
    return result


VALID_COUNTRY_CODES: frozenset[str] = frozenset(row.code for row in COUNTRY_ROWS)
if len(VALID_COUNTRY_CODES) != len(COUNTRY_ROWS):
    raise ValueError("COUNTRY_ROWS contains duplicate country codes")

COUNTRY_NAME_MAP: Dict[str, str] = _build_country_name_map(COUNTRY_ROWS)


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
    "CountryRow",
    "COUNTRY_ROWS",
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
