"""Small geodesy helpers: local tangent-plane conversions and distances.

Everything here uses a spherical-earth, flat-local-plane approximation, which
is accurate to well under a metre over the few-kilometre areas a single drone
flight covers.
"""

from __future__ import annotations

import math

EARTH_RADIUS_M = 6_371_008.8


def offset_latlon(lat: float, lon: float, north_m: float, east_m: float) -> tuple[float, float]:
    """Move (lat, lon) by north/east metres."""
    dlat = math.degrees(north_m / EARTH_RADIUS_M)
    dlon = math.degrees(east_m / (EARTH_RADIUS_M * math.cos(math.radians(lat))))
    return lat + dlat, lon + dlon


def to_local(lat0: float, lon0: float, lat: float, lon: float) -> tuple[float, float]:
    """(lat, lon) -> (east_m, north_m) relative to origin (lat0, lon0)."""
    north = math.radians(lat - lat0) * EARTH_RADIUS_M
    east = math.radians(lon - lon0) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
    return east, north


def from_local(lat0: float, lon0: float, east_m: float, north_m: float) -> tuple[float, float]:
    """(east_m, north_m) relative to origin -> (lat, lon)."""
    return offset_latlon(lat0, lon0, north_m, east_m)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def polygon_area_m2(points_en: list[tuple[float, float]]) -> float:
    """Shoelace area of a polygon given in local east/north metres."""
    n = len(points_en)
    s = 0.0
    for i in range(n):
        x1, y1 = points_en[i]
        x2, y2 = points_en[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def format_latlon(lat: float, lon: float) -> str:
    return f"{lat:.6f}, {lon:.6f}"
