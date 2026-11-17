"""Turn a folder of survey photos into a site project.

* **Drone photos looking down at the ground** are warped onto it using the
  drone's GPS position, height and gimbal angles, and stitched into a metric
  base map (north up) to draw the site plan on. No mapping mission is needed:
  hand-flown shots work. A straight-down frame is best, but angled shots
  taken while circling a building are used too, each supplying the ground it
  looks at most steeply.
* **Every drone and phone photo that is not straight down** is also placed on
  the map as a view cone (phones record the compass direction the camera was
  pointing), and later attached as evidence to whatever it looks at.

Nothing is written to the source folder.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ..camera import PhotoMeta, pixel_to_ground, read_photo_meta
from ..geo import to_local
from . import model as M

EXTS = {".jpg", ".jpeg"}
NADIR_PITCH = -70.0  # steeper than this counts as "straight down"
MAP_PITCH = -35.0  # shallower than this shows too little ground to be worth putting on the map
MAX_OFF_NADIR = 62.0  # ground seen at a flatter angle than this is too smeared to trace from


def find_photos(src: Path, exclude: Path | None = None) -> list[Path]:
    src = Path(src)
    if src.is_file():
        return [src]
    gen = (exclude.resolve() / "photos") if exclude else None  # our own previews, if the project is inside src
    out = []
    for p in sorted(src.rglob("*")):
        if p.suffix.lower() not in EXTS or p.name.startswith("."):
            continue
        rp = p.resolve()
        if gen and (gen in rp.parents or rp == exclude.resolve() / "basemap.jpg"):
            continue
        out.append(p)
    return out


def classify(meta: PhotoMeta) -> str:
    if not meta.has_position:
        return "unplaced"
    if meta.rel_alt_m is not None and meta.rel_alt_m >= 5:
        if meta.can_geolocate and meta.gimbal_pitch <= NADIR_PITCH:
            return "nadir"
        return "oblique"
    return "ground"


def _coeffs(dst, src):
    """PIL perspective coefficients mapping output pixels ``dst`` back to source pixels ``src``."""
    A, B = [], []
    for (X, Y), (u, v) in zip(dst, src):
        A.append([X, Y, 1, 0, 0, 0, -u * X, -u * Y])
        B.append(u)
        A.append([0, 0, 0, X, Y, 1, -v * X, -v * Y])
        B.append(v)
    return np.linalg.solve(np.array(A, float), np.array(B, float))


def ground_quad(meta: PhotoMeta, lat0: float, lon0: float, ground_offset_m: float = 0.0):
    """The part of a photo that looks down steeply enough to map.

    Returns (pixel corners, the same corners on the ground in local metres), or
    None. For a straight-down photo that is the whole frame; for an angled one
    the top of the frame (far ground, horizon, sky) is cut off.
    """
    if not meta.can_geolocate or meta.gimbal_pitch > MAP_PITCH:
        return None
    w, h = meta.width, meta.height
    # image row v looks down at (-pitch + atan((v - h/2) / f)) degrees below the horizon
    v0 = h / 2 + meta.focal_px() * math.tan(math.radians(90 - MAX_OFF_NADIR + meta.gimbal_pitch))
    v0 = max(0.0, v0)
    if v0 > 0.8 * h:
        return None
    px = [(0, v0), (w, v0), (w, h), (0, h)]
    ll = [pixel_to_ground(meta, u, v, ground_offset_m) for u, v in px]
    if any(p is None for p in ll):
        return None
    return px, [list(to_local(lat0, lon0, *p)) for p in ll]


def build_basemap(shots, bounds, out_path: Path, max_px: int = 3000, log=print) -> dict | None:
    """Stitch drone photos into one north-up image.

    ``shots`` = [{"path", "px" (corners in the photo), "fp" (the same corners on
    the ground), "pos" (drone x, y), "height"}]. Where photos overlap, each
    piece of ground is taken from the photo that looks at it most steeply:
    that is the view in which walls and trees lean over the least.
    """
    if not shots:
        return None
    xmin, ymin, xmax, ymax = bounds
    ppm = min(20.0, max_px / max(xmax - xmin, ymax - ymin))
    W, H = int(round((xmax - xmin) * ppm)), int(round((ymax - ymin) * ppm))
    canvas = np.full((H, W, 3), (24, 26, 31), np.uint8)  # unphotographed ground stays dark grey
    best = np.zeros((H, W), np.float32)
    floor = math.cos(math.radians(MAX_OFF_NADIR + 6))
    for sh in shots:
        path = sh["path"]
        with Image.open(path) as im:
            fw, fh = im.size
            im.draft("RGB", (2800, 2800))
            im = im.convert("RGB")
            if max(im.size) > 2800:
                im.thumbnail((2800, 2800), Image.BILINEAR)
            k = im.size[0] / fw
            out_xy = [((x - xmin) * ppm, (ymax - y) * ppm) for x, y in sh["fp"]]
            x0 = max(0, int(math.floor(min(p[0] for p in out_xy))))
            y0 = max(0, int(math.floor(min(p[1] for p in out_xy))))
            x1 = min(W, int(math.ceil(max(p[0] for p in out_xy))))
            y1 = min(H, int(math.ceil(max(p[1] for p in out_xy))))
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            try:
                co = _coeffs([(px - x0, py - y0) for px, py in out_xy], [(u * k, v * k) for u, v in sh["px"]])
            except np.linalg.LinAlgError:
                continue
            size = (x1 - x0, y1 - y0)
            warped = np.asarray(im.transform(size, Image.PERSPECTIVE, tuple(co), Image.BICUBIC))
            inside = np.asarray(Image.new("L", im.size, 255).transform(size, Image.PERSPECTIVE, tuple(co))) > 127
            # weight = cosine of the angle away from straight down, as seen from this drone position
            gx = xmin + (np.arange(x0, x1) + 0.5) / ppm - sh["pos"][0]
            gy = ymax - (np.arange(y0, y1) + 0.5) / ppm - sh["pos"][1]
            wt = (sh["height"] / np.sqrt(sh["height"] ** 2 + gx[None, :] ** 2 + gy[:, None] ** 2)).astype(np.float32)
            wt[~inside | (wt < floor)] = 0
            if sh["pitch"] <= NADIR_PITCH:
                wt[wt > 0] += 1.0  # a straight-down frame always beats an angled one: no seams of leaning walls
            sub = best[y0:y1, x0:x1]
            take = wt > sub
            sub[take] = wt[take]
            canvas[y0:y1, x0:x1][take] = warped[take]
        log(f"  mapped {path.name} ({fw}x{fh}, camera {-sh['pitch']:.0f} deg below the horizon)")
    Image.fromarray(canvas).save(out_path, "JPEG", quality=86)
    steep = sum(s["pitch"] <= NADIR_PITCH for s in shots)
    return {"file": out_path.name, "bounds": [round(b, 3) for b in bounds], "px_per_m": round(ppm, 3),
            "width": W, "height": H, "straight_down": steep, "angled": len(shots) - steep}


def run_survey(src: Path, out: Path, name: str | None = None, ground_offset_m: float = 0.0,
               max_px: int = 3000, log=print) -> dict:
    """Read the photos, build the base map and write ``site.json``. Returns the model.

    Re-running on an existing project refreshes photos and base map but keeps
    everything already drawn.
    """
    t0 = time.time()
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "photos").mkdir(exist_ok=True)
    files = find_photos(src, exclude=out)
    metas = []
    for p in files:
        try:
            metas.append(read_photo_meta(p))
        except Exception as e:  # unreadable file: skip, keep going
            log(f"  skipped {p.name}: {e}")
    kinds = [classify(m) for m in metas]
    log(f"{len(metas)} photo(s): {kinds.count('nadir')} straight-down aerial, {kinds.count('oblique')} angled "
        f"aerial, {kinds.count('ground')} ground, {kinds.count('unplaced')} without GPS")

    existing = M.load(out / "site.json") if (out / "site.json").exists() else None
    placed = [m for m, k in zip(metas, kinds) if k != "unplaced"]
    anchor = [m for m, k in zip(metas, kinds) if k == "nadir"] or \
        [m for m, k in zip(metas, kinds) if k == "oblique"] or placed
    if existing:
        lat0, lon0 = existing["origin"]["lat"], existing["origin"]["lon"]
    elif anchor:
        lat0 = sum(m.lat for m in anchor) / len(anchor)
        lon0 = sum(m.lon for m in anchor) / len(anchor)
    else:
        lat0 = lon0 = 0.0

    photos, shots, ext = [], [], []
    for n, (meta, kind) in enumerate(zip(metas, kinds), 1):
        pid = f"p{n:03d}"
        x = y = None
        if kind != "unplaced":
            x, y = to_local(lat0, lon0, meta.lat, meta.lon)
        fp = None
        if kind in ("nadir", "oblique"):
            quad = ground_quad(meta, lat0, lon0, ground_offset_m)
            if quad:
                fp = quad[1]
                shots.append({"path": meta.path, "px": quad[0], "fp": fp, "pos": (x, y), "pitch": meta.gimbal_pitch,
                              "height": meta.height_agl(ground_offset_m)})
            elif kind == "nadir":
                kind = "oblique"
        if kind in ("ground", "oblique") and math.hypot(x, y) < 250:
            ext.append([x, y])
        try:
            with Image.open(meta.path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                w, h = im.size
                big = im.copy()
                big.thumbnail((1400, 1400))
                big.save(out / "photos" / f"{pid}.jpg", "JPEG", quality=82)
                big.thumbnail((360, 360))
                big.save(out / "photos" / f"{pid}_t.jpg", "JPEG", quality=78)
        except Exception as e:
            log(f"  skipped {meta.path.name}: {e}")
            continue
        aerial = kind in ("nadir", "oblique")
        f35 = meta.focal_35mm or (24.0 if aerial else 26.0)
        try:
            rel = str(meta.path.relative_to(src)) if src.is_dir() else meta.path.name
        except ValueError:
            rel = meta.path.name
        photos.append({
            "id": pid, "name": rel, "kind": kind, "x": _r(x), "y": _r(y),
            "lat": meta.lat, "lon": meta.lon,
            "heading": _r(meta.gimbal_yaw % 360 if aerial and meta.gimbal_yaw is not None else meta.img_direction, 1),
            "hfov": round(math.degrees(2 * math.atan((36.0 if w >= h else 24.0) / 2 / f35)), 1),
            "alt_m": _r(meta.rel_alt_m, 1), "pitch": _r(meta.gimbal_pitch, 1),
            "taken": meta.taken_at.isoformat(sep=" ") if meta.taken_at else None,
            "file": f"photos/{pid}.jpg", "thumb": f"photos/{pid}_t.jpg",
            "footprint": [[_r(a), _r(b)] for a, b in fp] if fp else None,
        })

    # the map covers what the straight-down photos show; angled shots only set its extent when there are none
    steep = [p for sh in shots if sh["pitch"] <= NADIR_PITCH for p in sh["fp"]]
    cover = steep or [p for sh in shots for p in sh["fp"]]
    if cover:
        xs, ys = [p[0] for p in cover], [p[1] for p in cover]
        bounds = [min(xs), min(ys), max(xs), max(ys)]
    elif ext:
        xs, ys = [p[0] for p in ext], [p[1] for p in ext]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        hw, hh = max(30.0, (max(xs) - min(xs)) / 2 + 25), max(30.0, (max(ys) - min(ys)) / 2 + 25)
        bounds = [cx - hw, cy - hh, cx + hw, cy + hh]
    else:
        bounds = [-40.0, -40.0, 40.0, 40.0]
    basemap = build_basemap(shots, bounds, out / "basemap.jpg", max_px, log)
    if existing:
        model = existing
        model["photos"], model["basemap"] = photos, basemap
        model["bounds"] = [round(b, 2) for b in bounds]
        if name:
            model["name"] = name
    else:
        model = M.new_model(name or (src.name if src.is_dir() else src.parent.name) or "Site", (lat0, lon0),
                            bounds, basemap, photos)
    model["survey"] = {"source": str(src), "date": time.strftime("%Y-%m-%d %H:%M"),
                       "georeferenced": bool(anchor), "seconds": round(time.time() - t0, 1),
                       "counts": {k: sum(p["kind"] == k for p in photos)
                                  for k in ("nadir", "oblique", "ground", "unplaced")}}
    M.normalise(model)
    M.save(model, out / "site.json")
    return model


def _r(v, nd=2):
    return None if v is None else round(float(v), nd)
