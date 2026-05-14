"""commodutil.standards.commodities: canonical commodity vocabulary.

Owns:
- COMMODITY_KEYWORDS: ordered list of (display_name, group, [keywords])
  used by free-text inference. Ordering matters — "Natural Gasoline"
  must precede "Natural Gas" so the substring "natural gas" inside
  "natural gasoline" doesn't win.
- COMMODITY_CONVERSION_MAP: display_name -> commodutil.convfactors.COMMODITIES
  key, for downstream conversion routing.

Previously lived in curvemetadata.common_maps; relocated to eliminate
divergence risk between curvemetadata and commodutil's commodity lists.
curvemetadata.common_maps re-exports for backwards compatibility.
"""

from __future__ import annotations

COMMODITY_KEYWORDS = [
    ("Brent", "Crude Oil", ["brent"]),
    ("WTI", "Crude Oil", ["wti"]),
    ("Crude Oil", "Crude Oil", ["crude oil", "crude"]),
    # NB: 'Natural Gasoline' MUST come before 'Natural Gas' — the substring
    # "natural gas" is contained in "natural gasoline" and would otherwise win.
    ("Natural Gasoline", "NGL", ["natural gasoline"]),
    ("Natural Gas", "Natural Gas", ["natural gas", "nat gas", "natgas"]),
    ("Jet", "Refined Products", ["jet fuel", "jet"]),
    ("Diesel", "Refined Products", ["diesel", "ulsd", "gasoil", "heating oil"]),
    ("Gasoline", "Refined Products", ["gasoline", "rbob", "cbob", "mogas", "eurobob"]),
    ("Fuel Oil", "Refined Products", ["fuel oil", "hsfo", "lsfo", "marine fuel"]),
    ("Naphtha", "Refined Products", ["naphtha"]),
    ("Product Basket", "Refined Products", ["refined products", "product basket"]),
    ("VGO", "Refined Products", ["vgo"]),
    ("FAME", "Biofuel", ["fame"]),
    ("HVO", "Biofuel", ["hvo"]),
    ("Isobutane", "NGL", ["isobutane"]),
    ("Butane", "NGL", ["butane"]),
    ("Ethane", "NGL", ["ethane"]),
    ("Propane", "NGL", ["propane"]),
    ("NGL", "NGL", ["ngl"]),
    ("FFA", "Freight", ["freight", "ffa"]),
]

COMMODITY_CONVERSION_MAP = {
    "Crude Oil": "crude",
    "Brent": "crude",
    "WTI": "crude",
    "Natural Gas": "natgas",
    "Jet": "jet",
    "Diesel": "diesel",
    "Gasoline": "gasoline",
    "Fuel Oil": "fuel_oil",
    "Naphtha": "naphtha",
    "Product Basket": "product_basket",
    "VGO": "vgo",
    "FAME": "fame",
    "HVO": "hvo",
    # NGL species — switched from the generic 'lpg' blend to first-class species
    # in commodutil 2026-05 (each has its own density / HHV for $/gal<->$/MMBtu).
    # Keep the generic 'NGL' bucket on 'lpg' as a safe blend default.
    "Natural Gasoline": "natural_gasoline",
    "Isobutane": "isobutane",
    "Butane": "butane",
    "Propane": "propane",
    "NGL": "lpg",
    "Ethane": "ethane",
}

__all__ = [
    "COMMODITY_KEYWORDS",
    "COMMODITY_CONVERSION_MAP",
]
