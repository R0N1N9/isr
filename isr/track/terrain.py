"""Offline "area packs": elevation + trails/roads/water for one search area.

Fetch once while you have internet (``isr track fetch``), then everything runs
offline in the field.

* Elevation comes from the public AWS Terrain Tiles (Terrarium PNG encoding,
  built from SRTM / USGS 3DEP and others; no account or key needed).
* Trails, roads, rivers, lakes and cliffs come from OpenStreetMap through the
  Overpass API, or from a file you exported yourself (Overpass JSON, GeoJSON
  or .osm XML), e.g. from overpass-turbo.eu or a Geofabrik extract.

A pack is a folder: ``dem.npz`` (elevation mosaic) and ``osm.json`` (ways).
"""

from __future__ import annotations

import io
import json
import math
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "isr-tools/0.2 (offline SAR planning)"


# --------------------------------------------------------------------------
# Web Mercator tile maths
# --------------------------------------------------------------------------


def lonlat_to_tile_px(lon: float, lat: float, z: int) -> tuple[float, float]:
    """Global pixel coordinates (256 px tiles) at zoom z."""
    n = 256 * 2**z
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(max(-85.0511, min(85.0511, lat)))
    y = (1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n
    return x, y


def decode_terrarium(rgb: np.ndarray) -> np.ndarray:
    """Terrarium PNG -> elevation in metres."""
    rgb = rgb.astype(np.float32)
    return rgb[..., 0] * 256.0 + rgb[..., 1] + rgb[..., 2] / 256.0 - 32768.0


# --------------------------------------------------------------------------
# Pack
# --------------------------------------------------------------------------


@dataclass
class AreaPack:
    bbox: tuple[float, float, float, float]  # south, west, north, east
    dem: np.ndarray | None = None  # elevation mosaic (Web Mercator pixels)
    dem_zoom: int = 0
    dem_origin: tuple[float, float] = (0.0, 0.0)  # global pixel x, y of dem[0, 0]
    ways: list[dict] = field(default_factory=list)  # {"kind", "tags", "coords": [[lat, lon], ...], "closed"}
    source: dict = field(default_factory=dict)

    def elevation(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray | None:
        """Bilinear elevation at arrays of lat/lon, or None without a DEM."""
        if self.dem is None:
            return None
        n = 256 * 2**self.dem_zoom
        x = (np.asarray(lon) + 180.0) / 360.0 * n - self.dem_origin[0]
        lat_r = np.radians(np.clip(lat, -85.0511, 85.0511))
        y = (1 - np.log(np.tan(lat_r) + 1 / np.cos(lat_r)) / math.pi) / 2 * n - self.dem_origin[1]
        return ndimage.map_coordinates(self.dem, [y.ravel(), x.ravel()], order=1, mode="nearest").reshape(
            np.shape(lat))

    def save(self, folder: Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        if self.dem is not None:
            np.savez_compressed(folder / "dem.npz", dem=self.dem.astype(np.float32), zoom=self.dem_zoom,
                                origin=np.array(self.dem_origin), bbox=np.array(self.bbox))
        (folder / "osm.json").write_text(json.dumps({"bbox": self.bbox, "ways": self.ways, "source": self.source}))

    @classmethod
    def load(cls, folder: Path) -> "AreaPack":
        folder = Path(folder)
        meta = json.loads((folder / "osm.json").read_text()) if (folder / "osm.json").exists() else {}
        pack = cls(bbox=tuple(meta.get("bbox", (0, 0, 0, 0))), ways=meta.get("ways", []),
                   source=meta.get("source", {}))
        if (folder / "dem.npz").exists():
            z = np.load(folder / "dem.npz")
            pack.dem = z["dem"]
            pack.dem_zoom = int(z["zoom"])
            pack.dem_origin = tuple(float(v) for v in z["origin"])
            if not meta:
                pack.bbox = tuple(float(v) for v in z["bbox"])
        if pack.dem is None and not pack.ways:
            raise FileNotFoundError(f"{folder} has no dem.npz or osm.json")
        return pack

    def summary(self) -> str:
        kinds: dict[str, int] = {}
        for w in self.ways:
            kinds[w["kind"]] = kinds.get(w["kind"], 0) + 1
        s, w, n, e = self.bbox
        dem = (f"elevation {self.dem.shape[1]}x{self.dem.shape[0]} px (zoom {self.dem_zoom}, "
               f"{float(np.nanmin(self.dem)):.0f}-{float(np.nanmax(self.dem)):.0f} m)") if self.dem is not None \
            else "no elevation"
        feats = ", ".join(f"{v} {k}" for k, v in sorted(kinds.items())) or "no map features"
        return f"bbox {s:.4f},{w:.4f} to {n:.4f},{e:.4f}; {dem}; {feats}"


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------


def _get(url: str, data: bytes | None = None, timeout: float = 60) -> bytes:
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_dem(bbox, zoom: int = 12, log=print) -> tuple[np.ndarray, tuple[float, float]]:
    s, w, n, e = bbox
    x0, y0 = lonlat_to_tile_px(w, n, zoom)
    x1, y1 = lonlat_to_tile_px(e, s, zoom)
    tx0, ty0, tx1, ty1 = int(x0 // 256), int(y0 // 256), int(x1 // 256), int(y1 // 256)
    count = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
    if count > 400:
        raise ValueError(f"area needs {count} elevation tiles at zoom {zoom}; use a smaller area or --zoom {zoom - 1}")
    log(f"  elevation: {count} tile(s) at zoom {zoom}")
    mosaic = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256), np.float32)
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            png = _get(TERRARIUM_URL.format(z=zoom, x=tx, y=ty))
            rgb = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
            mosaic[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256] = decode_terrarium(rgb)
    return mosaic, (tx0 * 256.0, ty0 * 256.0)


def overpass_query(bbox) -> str:
    s, w, n, e = bbox
    b = f"({s},{w},{n},{e})"
    return (f'[out:json][timeout:90];(way["highway"]{b};way["waterway"~"^(river|stream|canal)$"]{b};'
            f'way["natural"="water"]{b};way["natural"="cliff"]{b};way["water"]{b};);out geom;')


def classify(tags: dict) -> str | None:
    hw = tags.get("highway")
    if hw:
        if hw in ("path", "footway", "track", "bridleway", "steps", "cycleway", "pedestrian"):
            return "path"
        return "road"
    ww = tags.get("waterway")
    if ww in ("river", "canal"):
        return "river"
    if ww == "stream":
        return "stream"
    if tags.get("natural") == "water" or "water" in tags:
        return "water"
    if tags.get("natural") == "cliff":
        return "cliff"
    return None


def parse_overpass(js: dict) -> list[dict]:
    ways = []
    for el in js.get("elements", []):
        if el.get("type") != "way" or "geometry" not in el:
            continue
        tags = el.get("tags", {})
        kind = classify(tags)
        if not kind:
            continue
        coords = [[g["lat"], g["lon"]] for g in el["geometry"]]
        if len(coords) < 2:
            continue
        ways.append({"kind": kind, "tags": {k: tags[k] for k in ("highway", "waterway", "natural", "name", "bridge",
                                                                  "maxspeed", "tunnel") if k in tags},
                     "coords": coords, "closed": coords[0] == coords[-1]})
    return ways


def parse_geojson(js: dict) -> list[dict]:
    ways = []
    for f in js.get("features", []):
        g, tags = f.get("geometry") or {}, f.get("properties") or {}
        kind = classify(tags)
        if not kind:
            continue
        if g.get("type") == "LineString":
            lines, closed = [g["coordinates"]], False
        elif g.get("type") == "MultiLineString":
            lines, closed = g["coordinates"], False
        elif g.get("type") == "Polygon":
            lines, closed = [g["coordinates"][0]], True
        elif g.get("type") == "MultiPolygon":
            lines, closed = [p[0] for p in g["coordinates"]], True
        else:
            continue
        for line in lines:
            ways.append({"kind": kind, "tags": {k: v for k, v in tags.items() if isinstance(v, (str, int, float))},
                         "coords": [[c[1], c[0]] for c in line], "closed": closed})
    return ways


def parse_osm_xml(text: str) -> list[dict]:
    root = ET.fromstring(text)
    nodes = {n.get("id"): (float(n.get("lat")), float(n.get("lon"))) for n in root.iter("node")}
    ways = []
    for w in root.iter("way"):
        tags = {t.get("k"): t.get("v") for t in w.iter("tag")}
        kind = classify(tags)
        if not kind:
            continue
        coords = [list(nodes[nd.get("ref")]) for nd in w.iter("nd") if nd.get("ref") in nodes]
        if len(coords) >= 2:
            ways.append({"kind": kind, "tags": tags, "coords": coords, "closed": coords[0] == coords[-1]})
    return ways


def load_osm_file(path: Path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if text.lstrip().startswith("<"):
        return parse_osm_xml(text)
    js = json.loads(text)
    return parse_overpass(js) if "elements" in js else parse_geojson(js)


def fetch_area(bbox, zoom: int = 12, osm: bool = True, osm_file: Path | None = None,
               overpass_url: str = OVERPASS_URL, log=print) -> AreaPack:
    pack = AreaPack(bbox=tuple(bbox))
    try:
        pack.dem, pack.dem_origin = fetch_dem(bbox, zoom, log)
        pack.dem_zoom = zoom
        pack.source["dem"] = "AWS Terrain Tiles (Terrarium)"
    except Exception as e:
        log(f"  ! elevation download failed ({e}); terrain will be treated as flat")
    if osm_file:
        pack.ways = load_osm_file(osm_file)
        pack.source["osm"] = str(osm_file)
        log(f"  map features: {len(pack.ways)} from {osm_file}")
    elif osm:
        try:
            body = urllib.parse.urlencode({"data": overpass_query(bbox)}).encode()
            pack.ways = parse_overpass(json.loads(_get(overpass_url, body, timeout=120)))
            pack.source["osm"] = overpass_url
            log(f"  map features: {len(pack.ways)} trails/roads/water ways from OpenStreetMap")
        except Exception as e:
            log(f"  ! OpenStreetMap download failed ({e}); export the area from overpass-turbo.eu and pass "
                "--osm-file, or continue without trails/roads")
    return pack
