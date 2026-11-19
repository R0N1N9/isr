"""Movement model: where could the subject have gone, and where are they likely now?

The area is a grid of square cells. Moving between neighbouring cells takes
time that depends on the travel mode:

foot
    Tobler's hiking function on the local slope (fastest ~6 km/h walking
    slightly downhill, much slower uphill or down steep ground), times a
    terrain factor: 1.0 on trails and roads, 0.6 off-trail (Tobler's own
    off-path factor), 0.35 wading a stream. Rivers, lakes, cliffs and slopes
    over 45 degrees are impassable except where a trail or road crosses
    (bridges). A profile multiplier models slower or faster subjects.
vehicle
    Speeds by road class from OpenStreetMap (motorway 100 km/h down to tracks
    15 km/h) and a slow off-road speed so a car seen in a field can still
    reach the road. Water is impassable except on roads.

Shortest travel times come from Dijkstra's algorithm on the directed cell
graph (uphill and downhill differ). On top of that:

* corridor (space-time prism) between two sightings A (time tA) and B (tB):
  every cell x with time(A->x) + time(x->B) <= tB - tA. The subject cannot
  have been anywhere else in between, if the positions and times are right.
* probability now, from the last sighting: the subject kept moving for an
  unknown share of the elapsed time (uniform), so travel time is spread
  evenly over [0, elapsed]; within each travel-time band, cells ahead in the
  direction of travel and cells near trails/roads get more weight.

These are planning aids built on stated assumptions, not predictions of a
single route. All parameters are exposed so the search manager can tune them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from ..geo import from_local, to_local
from .terrain import AreaPack

# km/h by OSM highway class (vehicle mode)
ROAD_KMH = {
    "motorway": 100, "motorway_link": 60, "trunk": 90, "trunk_link": 50, "primary": 80, "primary_link": 50,
    "secondary": 70, "secondary_link": 40, "tertiary": 60, "tertiary_link": 40, "unclassified": 50,
    "residential": 40, "living_street": 15, "service": 20, "road": 40, "track": 15,
}
# relative walking speed by subject profile (assumptions; tune with --speed-factor)
FOOT_PROFILES = {"hiker": 1.0, "runner": 1.5, "child": 0.6, "elderly": 0.55, "dementia": 0.5, "injured": 0.35}
FOOT_MAX_KMH = 8.0  # sustained straight-line speed beyond which a leg is flagged
VEHICLE_MAX_KMH = 140.0

NEIGHBOURS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


@dataclass
class Grid:
    lat0: float
    lon0: float
    res: float
    nx: int
    ny: int

    @classmethod
    def around(cls, points: list[tuple[float, float]], margin_m: float, res: float, max_cells: int = 640) -> "Grid":
        lat0 = sum(p[0] for p in points) / len(points)
        lon0 = sum(p[1] for p in points) / len(points)
        en = [to_local(lat0, lon0, *p) for p in points]
        w = max(abs(e) for e, _ in en) * 2 + 2 * margin_m
        h = max(abs(n) for _, n in en) * 2 + 2 * margin_m
        res = max(res, max(w, h) / max_cells)
        return cls(lat0, lon0, res, max(8, int(math.ceil(w / res))), max(8, int(math.ceil(h / res))))

    def en(self, i, j):
        return (np.asarray(j) - (self.nx - 1) / 2) * self.res, ((self.ny - 1) / 2 - np.asarray(i)) * self.res

    def cell(self, lat: float, lon: float) -> tuple[int, int]:
        e, n = to_local(self.lat0, self.lon0, lat, lon)
        j = int(round(e / self.res + (self.nx - 1) / 2))
        i = int(round((self.ny - 1) / 2 - n / self.res))
        return min(max(i, 0), self.ny - 1), min(max(j, 0), self.nx - 1)

    def inside(self, lat: float, lon: float) -> bool:
        e, n = to_local(self.lat0, self.lon0, lat, lon)
        return abs(e) <= self.nx * self.res / 2 and abs(n) <= self.ny * self.res / 2

    def latlon(self, i, j) -> tuple[float, float]:
        e, n = self.en(i, j)
        return from_local(self.lat0, self.lon0, float(e), float(n))

    def latlon_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        ii, jj = np.mgrid[0:self.ny, 0:self.nx]
        e, n = self.en(ii, jj)
        lat = self.lat0 + np.degrees(n / 6_371_008.8)
        lon = self.lon0 + np.degrees(e / (6_371_008.8 * math.cos(math.radians(self.lat0))))
        return lat, lon

    def px(self, lat: float, lon: float) -> tuple[float, float]:
        """Continuous pixel coordinates for PIL drawing (cell centres at +0.5)."""
        e, n = to_local(self.lat0, self.lon0, lat, lon)
        return e / self.res + self.nx / 2, -n / self.res + self.ny / 2

    def bounds(self) -> tuple[float, float, float, float]:
        s, w = from_local(self.lat0, self.lon0, -self.nx * self.res / 2, -self.ny * self.res / 2)
        n, e = from_local(self.lat0, self.lon0, self.nx * self.res / 2, self.ny * self.res / 2)
        return s, w, n, e


@dataclass
class Surface:
    """Per-cell inputs to the travel model."""

    grid: Grid
    elev: np.ndarray | None
    speed: np.ndarray  # foot: terrain factor (0..1); vehicle: km/h
    features: dict[str, np.ndarray] = field(default_factory=dict)  # boolean rasters: path, road, water, ...
    mode: str = "foot"


def _raster(grid: Grid, ways: list[dict], kinds: set[str], width_m: float, fill_polygons: bool) -> np.ndarray:
    img = Image.new("L", (grid.nx, grid.ny), 0)
    d = ImageDraw.Draw(img)
    w = max(1, int(round(width_m / grid.res)))
    for way in ways:
        if way["kind"] not in kinds:
            continue
        pts = [grid.px(la, lo) for la, lo in way["coords"]]
        if fill_polygons and way.get("closed") and len(pts) >= 3:
            d.polygon(pts, fill=1, outline=1)
        else:
            d.line(pts, fill=1, width=w)
            if w > 1:  # round joints so barriers have no diagonal gaps
                for x, y in pts:
                    d.ellipse((x - w / 2, y - w / 2, x + w / 2, y + w / 2), fill=1)
    return np.asarray(img, dtype=bool)


def build_surface(grid: Grid, pack: AreaPack | None, mode: str = "foot", offroad_kmh: float = 3.0) -> Surface:
    ways = pack.ways if pack else []
    elev = None
    if pack is not None and pack.dem is not None:
        lat, lon = grid.latlon_arrays()
        elev = pack.elevation(lat, lon).astype(np.float32)
    barrier_w = max(2.0 * grid.res, 10.0)  # >= 2 cells so 8-connected moves can't slip through diagonally
    feats = {
        "water": _raster(grid, ways, {"water"}, barrier_w, True),
        "river": _raster(grid, ways, {"river"}, barrier_w, False),
        "cliff": _raster(grid, ways, {"cliff"}, barrier_w, False),
        "stream": _raster(grid, ways, {"stream"}, grid.res, False),
        "path": _raster(grid, ways, {"path"}, grid.res, False),
        "road": _raster(grid, ways, {"road"}, grid.res, False),
    }
    blocked = feats["water"] | feats["river"] | feats["cliff"]
    if mode == "foot":
        sp = np.full((grid.ny, grid.nx), 0.6, np.float32)
        sp[feats["stream"]] = 0.35
        sp[blocked] = 0.0
        sp[feats["path"] | feats["road"]] = 1.0  # trails/roads win: bridges, fords
    else:
        sp = np.full((grid.ny, grid.nx), float(offroad_kmh), np.float32)
        sp[blocked] = 0.0
        img = Image.new("F", (grid.nx, grid.ny), 0.0)
        d = ImageDraw.Draw(img)
        for way in sorted((w for w in ways if w["kind"] in ("road", "path")),
                          key=lambda w: ROAD_KMH.get(w["tags"].get("highway", ""), 0)):
            hw = way["tags"].get("highway", "")
            kmh = ROAD_KMH.get(hw, 0)
            if way["kind"] == "path" and hw != "track":
                continue
            ms = str(way["tags"].get("maxspeed", "")).split()[0] if way["tags"].get("maxspeed") else ""
            if ms.isdigit():
                kmh = min(kmh or 999, int(ms) * (1.609 if "mph" in str(way["tags"]["maxspeed"]) else 1))
            if kmh:
                d.line([grid.px(la, lo) for la, lo in way["coords"]], fill=float(kmh), width=1)
        roads = np.asarray(img, np.float32)
        sp = np.where(roads > 0, np.maximum(roads, sp), sp)
    return Surface(grid, elev, sp, feats, mode)


def tobler_kmh(slope: np.ndarray) -> np.ndarray:
    return 6.0 * np.exp(-3.5 * np.abs(slope + 0.05))


def build_graph(surf: Surface, speed_factor: float = 1.0) -> csr_matrix:
    """Directed cell graph with edge weights in seconds."""
    g = surf.grid
    ny, nx = g.ny, g.nx
    idx = np.arange(ny * nx).reshape(ny, nx)
    rows, cols, wts = [], [], []
    for dy, dx in NEIGHBOURS:
        ys = slice(max(0, -dy), ny - max(0, dy))
        xs = slice(max(0, -dx), nx - max(0, dx))
        yd = slice(max(0, dy), ny + min(0, dy))
        xd = slice(max(0, dx), nx + min(0, dx))
        length = g.res * math.hypot(dy, dx)
        si, sj = surf.speed[ys, xs], surf.speed[yd, xd]
        if surf.mode == "foot":
            if surf.elev is not None:
                slope = (surf.elev[yd, xd] - surf.elev[ys, xs]) / length
            else:
                slope = np.zeros_like(si)
            base = tobler_kmh(slope) / 3.6 * speed_factor
            vi, vj = base * si, base * sj
            ok = (si > 0) & (sj > 0) & (np.abs(slope) <= 1.0)
        else:
            vi, vj = si / 3.6 * speed_factor, sj / 3.6 * speed_factor
            ok = (si > 0) & (sj > 0)
        with np.errstate(divide="ignore"):
            t = length / 2 * (1 / np.where(ok, vi, 1) + 1 / np.where(ok, vj, 1))
        rows.append(idx[ys, xs][ok])
        cols.append(idx[yd, xd][ok])
        wts.append(t[ok].astype(np.float64))
    return csr_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(ny * nx, ny * nx))


@dataclass
class Model:
    surf: Surface
    graph: csr_matrix
    _rev: csr_matrix | None = None

    @property
    def grid(self) -> Grid:
        return self.surf.grid

    @property
    def reverse(self) -> csr_matrix:
        if self._rev is None:
            self._rev = self.graph.T.tocsr()
        return self._rev

    def node(self, lat: float, lon: float) -> int:
        """Grid node for a position, snapped to the nearest passable cell (GPS error, bank of a river...)."""
        i, j = self.grid.cell(lat, lon)
        passable = self.surf.speed > 0
        if not passable[i, j]:
            dist, (ii, jj) = ndimage.distance_transform_edt(~passable, return_indices=True)
            i, j = int(ii[i, j]), int(jj[i, j])
        return i * self.grid.nx + j

    def times_from(self, lat: float, lon: float, limit: float = np.inf, reverse: bool = False):
        d, pred = dijkstra(self.reverse if reverse else self.graph, directed=True, indices=self.node(lat, lon),
                           limit=limit, return_predecessors=True)
        return d.reshape(self.grid.ny, self.grid.nx), pred

    def path(self, pred: np.ndarray, target: int) -> list[tuple[float, float]]:
        out, k, guard = [], target, 0
        while k >= 0 and guard < self.grid.nx * self.grid.ny:
            out.append(self.grid.latlon(k // self.grid.nx, k % self.grid.nx))
            k = int(pred[k])
            guard += 1
        return out[::-1]


def corridor(model: Model, a: tuple[float, float], b: tuple[float, float], dt_s: float):
    """Space-time prism between two sightings.

    Returns (slack, min_time, best_path): ``slack`` is dt minus the time of the
    fastest route through each cell (NaN outside the prism), ``min_time`` the
    fastest A->B time, ``best_path`` the lat/lon cells of that route.
    """
    da, pred = model.times_from(*a, limit=max(dt_s, 1.0) * 1.5)
    db, _ = model.times_from(*b, limit=max(dt_s, 1.0) * 1.5, reverse=True)
    tb = model.node(*b)
    min_t = float(da.ravel()[tb])
    total = da + db
    slack = np.where(total <= dt_s, dt_s - total, np.nan)
    best = model.path(pred, tb) if np.isfinite(min_t) else []
    return slack, min_t, best


def bearing(a: tuple[float, float], b: tuple[float, float]) -> float:
    e, n = to_local(a[0], a[1], b[0], b[1])
    return math.atan2(e, n)


def probability(model: Model, last: tuple[float, float], elapsed_s: float, heading: float | None = None,
                heading_weight: float = 1.0, trail_bias: float = 1.5, bins: int = 40):
    """Probability per cell of the subject's current position, plus travel times from ``last``."""
    t, _ = model.times_from(*last, limit=elapsed_s)
    reach = np.isfinite(t) & (t <= elapsed_s)
    p = np.zeros_like(t, dtype=np.float64)
    if not reach.any():
        return p, t
    g = model.grid
    w = np.ones_like(p)
    if heading is not None and heading_weight > 0:
        ii, jj = np.mgrid[0:g.ny, 0:g.nx]
        li, lj = g.cell(*last)
        phi = np.arctan2((jj - lj).astype(float), (li - ii).astype(float))
        w *= np.exp(heading_weight * (np.cos(phi - heading) - 1))
    if trail_bias != 1 and model.surf.mode == "foot":
        near = ndimage.binary_dilation(model.surf.features["path"] | model.surf.features["road"],
                                       iterations=max(1, int(round(50 / g.res))))
        w[near] *= trail_bias
    band = np.minimum((t[reach] / max(elapsed_s, 1e-9) * bins).astype(int), bins - 1)
    wr = w[reach]
    band_sum = np.bincount(band, weights=wr, minlength=bins)
    # equal probability per travel-time band, shared within the band by weight
    pr = np.where(band_sum[band] > 0, wr / band_sum[band], 0) / bins
    p[reach] = pr
    s = p.sum()
    return (p / s if s > 0 else p), t


def coverage_mask(grid: Grid, footprints: list[list[tuple[float, float]]]) -> np.ndarray:
    img = Image.new("L", (grid.nx, grid.ny), 0)
    d = ImageDraw.Draw(img)
    for fp in footprints:
        if fp and len(fp) >= 3:
            d.polygon([grid.px(la, lo) for la, lo in fp], fill=1)
    return np.asarray(img, dtype=bool)


def segments(grid: Grid, p_remaining: np.ndarray, p_initial: np.ndarray, size_m: float, top: int = 8):
    """Square search segments ranked by remaining probability."""
    k = max(1, int(round(size_m / grid.res)))
    out = []
    for i0 in range(0, grid.ny, k):
        for j0 in range(0, grid.nx, k):
            rem = float(p_remaining[i0:i0 + k, j0:j0 + k].sum())
            if rem <= 0:
                continue
            i1, j1 = min(grid.ny, i0 + k), min(grid.nx, j0 + k)
            corners = [grid.latlon(i0 - .5, j0 - .5), grid.latlon(i0 - .5, j1 - .5),
                       grid.latlon(i1 - .5, j1 - .5), grid.latlon(i1 - .5, j0 - .5)]
            out.append({"p_remaining": rem, "p_initial": float(p_initial[i0:i0 + k, j0:j0 + k].sum()),
                        "corners": corners, "size_m": [round((j1 - j0) * grid.res), round((i1 - i0) * grid.res)]})
    out.sort(key=lambda s: -s["p_remaining"])
    for r, s in enumerate(out[:top], 1):
        s["rank"] = r
    return out[:top]


def hillshade(elev: np.ndarray, res: float, az: float = 315, alt: float = 45) -> np.ndarray:
    gy, gx = np.gradient(elev.astype(np.float64), res)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    a, z = math.radians(360 - az + 90), math.radians(90 - alt)
    hs = np.cos(z) * np.cos(slope) + np.sin(z) * np.sin(slope) * np.cos(a - aspect)
    return np.clip(hs, 0, 1)
