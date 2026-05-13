"""POI category mapping for OpenStreetMap tags.

Maps raw OSM tags onto eight categories used as node features in the
v2 London commuting model.

Categories
----------
1. Commercial   — generic shops and offices
2. F&B          — food and beverage amenities
3. Retail       — large retail / supermarkets
4. Office       — specific commercial offices
5. Education    — schools, universities, libraries
6. Healthcare   — hospitals, clinics, pharmacies
7. Transport    — public transport stops/stations
8. Public       — civic, government, emergency services

Public API
----------
POI_CATEGORY_MAP : dict
    Human-readable name for each integer category id.
categorize_osm_tags(tags) -> int | None
    Given a dict of OSM tags, returns the integer category (1-8) or
    None if the feature does not match any of the eight categories.

Resolution order matters where tags overlap (e.g. an office that is
also a shop). The function evaluates the more specific categories
first (Retail / Office / Transport / Public / Education / Healthcare /
F&B) before falling back to the generic Commercial bucket.
"""

from __future__ import annotations

from typing import Mapping, Optional

# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------

POI_CATEGORY_MAP: dict[int, str] = {
    1: "Commercial",
    2: "F&B",
    3: "Retail",
    4: "Office",
    5: "Education",
    6: "Healthcare",
    7: "Transport",
    8: "Public",
}

# Category 2 — F&B
FNB_AMENITIES = {
    "restaurant",
    "cafe",
    "bar",
    "food_court",
    "fast_food",
    "pub",
    "biergarten",
    "ice_cream",
}

# Category 3 — Retail (large-format / grocery)
RETAIL_SHOPS = {
    "supermarket",
    "mall",
    "convenience",
    "department_store",
    "wholesale",
}

# Category 5 — Education
EDUCATION_AMENITIES = {
    "school",
    "university",
    "college",
    "kindergarten",
    "library",
}

# Category 6 — Healthcare
HEALTHCARE_AMENITIES = {
    "hospital",
    "clinic",
    "doctors",
    "pharmacy",
    "dentist",
    "veterinary",
}

# Category 7 — Transport (also matched via public_transport=*)
TRANSPORT_AMENITIES = {"bus_station"}
TRANSPORT_RAILWAY = {"station", "halt", "tram_stop"}

# Category 8 — Public / civic
PUBLIC_AMENITIES = {
    "townhall",
    "courthouse",
    "police",
    "post_office",
    "fire_station",
    "community_centre",
}


# ---------------------------------------------------------------------------
# Classification function
# ---------------------------------------------------------------------------

def _get(tags: Mapping[str, object], key: str) -> str:
    """Safe string accessor for OSM tag dicts."""
    val = tags.get(key)
    if val is None:
        return ""
    return str(val).strip().lower()


def categorize_osm_tags(tags: Mapping[str, object]) -> Optional[int]:
    """Return the integer category (1-8) for a set of OSM tags or None.

    Parameters
    ----------
    tags : Mapping[str, object]
        Dictionary of OSM key -> value pairs for the feature.

    Returns
    -------
    int or None
        Integer in 1..8 matching ``POI_CATEGORY_MAP`` or ``None`` if
        the feature is not relevant to any of the eight categories.

    Notes
    -----
    Resolution order (most specific first):
        Transport -> Public -> Healthcare -> Education -> Retail
        -> F&B -> Office -> Commercial (fallback)
    """
    if not tags:
        return None

    amenity = _get(tags, "amenity")
    shop = _get(tags, "shop")
    office = _get(tags, "office")
    railway = _get(tags, "railway")
    public_transport = _get(tags, "public_transport")
    highway = _get(tags, "highway")

    # 7 — Transport (highest priority so a "bus_station" amenity
    # doesn't get swallowed by Public).
    if public_transport:
        return 7
    if amenity in TRANSPORT_AMENITIES:
        return 7
    if railway in TRANSPORT_RAILWAY:
        return 7
    # Treat plain bus stops as transport too, although they are
    # extracted separately for the bus-stop layer.
    if highway == "bus_stop":
        return 7

    # 8 — Public / civic
    if amenity in PUBLIC_AMENITIES:
        return 8
    if office == "government":
        return 8

    # 6 — Healthcare
    if amenity in HEALTHCARE_AMENITIES:
        return 6

    # 5 — Education
    if amenity in EDUCATION_AMENITIES:
        return 5

    # 3 — Retail (specific large-format shops)
    if shop in RETAIL_SHOPS:
        return 3

    # 2 — F&B
    if amenity in FNB_AMENITIES:
        return 2

    # 4 — Office (specific commercial offices, anything other than
    # generic government already classified above).
    if office:
        return 4

    # 1 — Commercial fallback (any other shop)
    if shop:
        return 1

    return None


__all__ = ["POI_CATEGORY_MAP", "categorize_osm_tags"]
