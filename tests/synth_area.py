"""Synthetic area pack with known terrain for movement-model tests and demos."""

from __future__ import annotations

import math

import numpy as np

from isr.geo import from_local, to_local
from isr.track.terrain import AreaPack, lonlat_to_tile_px

LAT0, LON0 = 39.70, -105.20


def make_pack(half_km: float = 6.0, zoom: int = 13, ridge: bool = True, ways: bool = True) -> AreaPack:
    """Flat valley at 2000 m with an optional north-south ridge 1.5 km east of centre (+400 m),
    a river running north-south 1 km west of centre with one road bridge at the centre line,
    a lake south-west and a trail heading north-east."""
    s, w = from_local(LAT0, LON0, -half_km * 1000, -half_km * 1000)
    n, e = from_local(LAT0, LON0, half_km * 1000, half_km * 1000)
    x0, y0 = lonlat_to_tile_px(w, n, zoom)
    x1, y1 = lonlat_to_tile_px(e, s, zoom)
    x0, y0 = math.floor(x0), math.floor(y0)
    W, H = int(math.ceil(x1 - x0)) + 2, int(math.ceil(y1 - y0)) + 2
    nglob = 256 * 2**zoom
    xs = (np.arange(W) + x0) / nglob * 360 - 180
    ys = np.arange(H) + y0
    lats = np.degrees(np.arctan(np.sinh(math.pi * (1 - 2 * ys / nglob))))
    lon_g, lat_g = np.meshgrid(xs, lats)
    east = np.radians(lon_g - LON0) * 6_371_008.8 * math.cos(math.radians(LAT0))
    dem = np.full_like(east, 2000.0)
    if ridge:
        dem += 400 * np.exp(-((east - 1500) / 300) ** 2)
    pack = AreaPack(bbox=(s, w, n, e), dem=dem.astype(np.float32), dem_zoom=zoom, dem_origin=(float(x0), float(y0)))
    if ways:
        ll = lambda e_, n_: list(from_local(LAT0, LON0, e_, n_))  # noqa: E731
        pack.ways = [
            {"kind": "river", "tags": {"waterway": "river", "name": "Test River"},
             "coords": [ll(-1000, -5500), ll(-1000, 5500)], "closed": False},
            {"kind": "road", "tags": {"highway": "secondary", "name": "Bridge Rd"},
             "coords": [ll(-5500, 0), ll(5500, 0)], "closed": False},
            {"kind": "road", "tags": {"highway": "residential"},
             "coords": [ll(3000, -5500), ll(3000, 5500)], "closed": False},
            {"kind": "path", "tags": {"highway": "path", "name": "NE Trail"},
             "coords": [ll(0, 0), ll(800, 2500), ll(1200, 5000)], "closed": False},
            {"kind": "water", "tags": {"natural": "water"},
             "coords": [ll(-4000, -4000), ll(-2500, -4000), ll(-2500, -2500), ll(-4000, -2500), ll(-4000, -4000)],
             "closed": True},
        ]
    return pack


def at(e_m: float, n_m: float) -> tuple[float, float]:
    return from_local(LAT0, LON0, e_m, n_m)
