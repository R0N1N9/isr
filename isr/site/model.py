"""The site model: what the assessor draws over the survey photos.

Everything is stored in local metres, ``[east, north]`` from ``origin``
(a lat/lon), so the geometry is easy to reason about and edit by hand.

``site.json``::

    {"version": 1, "name": ..., "origin": {"lat": .., "lon": ..},
     "bounds": [xmin, ymin, xmax, ymax], "basemap": {"file": .., "bounds": [..]} | null,
     "settings": {...}, "features": [...], "photos": [...]}

Feature kinds:

=========== ======== ==========================================================
kind        geometry meaning
=========== ======== ==========================================================
boundary    polygon  the property line; analysis is limited to its inside
building    polygon  blocks sight and movement; cameras mount on its walls
barrier     line     wall / fence / see-through fence / hedge (``subtype``)
vegetation  polygon  trees and shrubs: can be walked into, cannot be seen into
street      line     public way: where a passer-by can stand and look in
access      point    door, window, gate, driveway, garage (``subtype``)
camera      point    existing or planned camera (bearing, hfov, range, height)
=========== ======== ==========================================================
"""

from __future__ import annotations

import json
import math
from pathlib import Path

VERSION = 1

# Branding of the editor and the report. "Prepared by" on a report defaults to BRAND and can be changed per project.
BRAND = "Ronin Sec"
APP = "Survey"
BRAND_LOGO = ('<svg class="logo" width="26" height="26" viewBox="0 0 32 32" aria-hidden="true">'
              '<rect width="32" height="32" rx="9" fill="#0d1117"/>'
              '<path d="M11 24V8h6.2a4.6 4.6 0 0 1 1.5 8.95L22.5 24M11 17h6.2" fill="none" stroke="#fff" '
              'stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>'
              '<circle cx="23.5" cy="9" r="2.2" fill="#e5484d"/></svg>')

DEFAULT_SETTINGS = {
    "cell_m": 0.5,  # analysis grid
    "camera_height_m": 2.7,  # mounting height for recommended cameras
    "camera_hfov": 90.0,  # horizontal field of view of recommended cameras, degrees
    "camera_range_m": 18.0,  # distance at which a person is still usefully recorded
    "max_new_cameras": 4,
    "target_height_m": 1.5,  # a camera "sees" a spot if it sees this height above it (chest / head)
    "eye_height_m": 1.6,  # passers-by and occupants
    "street_range_m": 50.0,
    "window_range_m": 30.0,
    "min_blind_m2": 6.0,  # smaller unobserved pockets are not listed
}

BARRIER_TYPES = {
    "wall": {"label": "Solid wall", "height": 1.8, "opaque": True},
    "fence": {"label": "Solid fence", "height": 1.8, "opaque": True},
    "open_fence": {"label": "See-through fence", "height": 1.5, "opaque": False},
    "hedge": {"label": "Hedge", "height": 1.8, "opaque": True},
}

# label, weight used when ranking (doors matter more than windows)
ACCESS_TYPES = {
    "door": ("Door", 3),
    "garage": ("Garage door", 3),
    "gate": ("Pedestrian gate", 3),
    "vehicle_gate": ("Driveway / vehicle gate", 3),
    "window": ("Ground-floor window", 2),
    "other": ("Other opening", 2),
}

KIND_LABEL = {"boundary": "Boundary", "building": "Building", "barrier": "Barrier", "vegetation": "Vegetation",
              "street": "Public way", "access": "Access point", "camera": "Camera"}
GEOMETRY = {"boundary": "polygon", "building": "polygon", "vegetation": "polygon", "barrier": "line",
            "street": "line", "access": "point", "camera": "point"}
DEFAULT_HEIGHT = {"building": 6.0, "vegetation": 3.0}


def new_model(name: str, origin: tuple[float, float], bounds, basemap=None, photos=None) -> dict:
    return {"version": VERSION, "name": name, "assessor": "", "company": BRAND, "notes": "",
            "origin": {"lat": origin[0], "lon": origin[1]}, "bounds": [round(float(b), 2) for b in bounds],
            "basemap": basemap, "settings": dict(DEFAULT_SETTINGS), "features": [], "photos": photos or []}


def feature_label(f: dict) -> str:
    if f["kind"] == "barrier":
        return BARRIER_TYPES.get(f.get("subtype"), BARRIER_TYPES["fence"])["label"]
    if f["kind"] == "access":
        return ACCESS_TYPES.get(f.get("subtype"), ACCESS_TYPES["other"])[0]
    return KIND_LABEL.get(f["kind"], f["kind"])


def normalise(model: dict) -> dict:
    """Fill in defaults so the rest of the code can rely on every field. Edits ``model`` in place."""
    model.setdefault("version", VERSION)
    model.setdefault("name", "Site")
    for k in ("assessor", "client", "address", "company", "reference"):  # shown on the report's front matter
        model[k] = str(model.get(k) or "")[:200]
    model.setdefault("notes", "")
    model.setdefault("photos", [])
    model.setdefault("basemap", None)
    s = dict(DEFAULT_SETTINGS)
    for k, v in (model.get("settings") or {}).items():
        if k in s and isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            s[k] = v
    s["cell_m"] = min(2.0, max(0.25, s["cell_m"]))
    s["camera_hfov"] = min(360.0, max(10.0, s["camera_hfov"]))
    s["camera_range_m"] = min(100.0, max(3.0, s["camera_range_m"]))
    s["max_new_cameras"] = int(min(12, max(0, s["max_new_cameras"])))
    model["settings"] = s

    feats, used, counters, serial = [], set(), {}, 0
    for f in model.get("features") or []:
        kind = f.get("kind")
        if kind not in GEOMETRY:
            continue
        pts = [[float(p[0]), float(p[1])] for p in f.get("pts") or []
               if len(p) >= 2 and all(isinstance(c, (int, float)) and math.isfinite(c) for c in p[:2])]
        need = {"polygon": 3, "line": 2, "point": 1}[GEOMETRY[kind]]
        if len(pts) < need:
            continue
        f["pts"] = pts[:1] if GEOMETRY[kind] == "point" else pts
        if kind == "barrier":
            if f.get("subtype") not in BARRIER_TYPES:
                f["subtype"] = "fence"
            f["height"] = _num(f.get("height"), BARRIER_TYPES[f["subtype"]]["height"], 0.2, 12)
        elif kind in DEFAULT_HEIGHT:
            f["height"] = _num(f.get("height"), DEFAULT_HEIGHT[kind], 0.3, 80)
        elif kind == "access":
            if f.get("subtype") not in ACCESS_TYPES:
                f["subtype"] = "door"
        elif kind == "camera":
            f["height"] = _num(f.get("height"), s["camera_height_m"], 0.5, 30)
            f["bearing"] = _num(f.get("bearing"), 0.0, -360, 720) % 360
            f["hfov"] = _num(f.get("hfov"), s["camera_hfov"], 5, 360)
            f["range"] = _num(f.get("range"), s["camera_range_m"], 2, 150)
            if f.get("status") not in ("existing", "planned"):
                f["status"] = "existing"
        fid = str(f.get("id") or "")
        while not fid or fid in used:
            serial += 1
            fid = f"f{serial}"
        f["id"] = fid
        used.add(fid)
        label = feature_label(f)
        counters[label] = counters.get(label, 0) + 1
        if not str(f.get("name") or "").strip():
            f["name"] = f"{label} {counters[label]}"
        f["name"] = str(f["name"])[:80]
        f["notes"] = str(f.get("notes") or "")[:2000]
        feats.append(f)
    model["features"] = feats
    return model


def _num(v, default, lo, hi) -> float:
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
        return float(default)
    return float(min(hi, max(lo, v)))


def load(path: Path) -> dict:
    return normalise(json.loads(Path(path).read_text()))


def save(model: dict, path: Path) -> None:
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(model, indent=1))
    tmp.replace(path)


# ---------------------------------------------------------------------------
# small geometry helpers (local metres)
# ---------------------------------------------------------------------------


def point_in_poly(x: float, y: float, pts) -> bool:
    inside = False
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def closest_on_path(x: float, y: float, pts, closed: bool):
    """Nearest point on a polyline/ring: (distance, px, py, segment index)."""
    best = (float("inf"), x, y, -1)
    n = len(pts)
    for i in range(n if closed else n - 1):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / L2))
        px, py = ax + t * dx, ay + t * dy
        d = math.hypot(x - px, y - py)
        if d < best[0]:
            best = (d, px, py, i)
    return best


def path_length(pts, closed: bool = False) -> float:
    n = len(pts)
    return sum(math.dist(pts[i], pts[(i + 1) % n]) for i in range(n if closed else n - 1))


def centroid(pts) -> tuple[float, float]:
    return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)


def bearing_deg(dx_east: float, dy_north: float) -> float:
    return math.degrees(math.atan2(dx_east, dy_north)) % 360


_COMPASS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]
_COMPASS_SHORT = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def compass(bearing: float, short: bool = False) -> str:
    i = int(((bearing % 360) + 22.5) // 45) % 8
    return (_COMPASS_SHORT if short else _COMPASS)[i]
