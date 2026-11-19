"""Bring a photogrammetry result into a survey: a true orthophoto and measured heights.

ISRStudio does not do photogrammetry itself. OpenDroneMap (free, open source)
turns overlapping drone photos into an orthophoto and elevation models; this
module reads those GeoTIFFs and resamples them onto the survey's own metric,
north-up grid:

* the **orthophoto** replaces the quick stitched base map (no leaning walls);
* the **surface model** (DSM) minus the ground (the DTM, or a ground level
  estimated from the surface itself) gives the height of everything standing
  on the site: buildings, walls, trees. Sightlines are then worked out from
  what is really there, not only from what was traced and typed in.

Heights are kept in ``heights.png`` in the project folder as centimetres above
ground, packed into the red and green channels so a browser can read them
exactly (the 3-D view uses the same file); blue marks where there was data.
"""

from __future__ import annotations

import math
import subprocess
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, TiffImagePlugin
from scipy import ndimage

from ..geo import to_local

Image.MAX_IMAGE_PIXELS = 400_000_000  # orthophotos are large; anything bigger is refused below
A_WGS, F_WGS, K0 = 6378137.0, 1 / 298.257223563, 0.9996
_N = F_WGS / (2 - F_WGS)
_A = A_WGS / (1 + _N) * (1 + _N ** 2 / 4 + _N ** 4 / 64)
_ALPHA = (_N / 2 - 2 * _N ** 2 / 3 + 5 * _N ** 3 / 16, 13 * _N ** 2 / 48 - 3 * _N ** 3 / 5, 61 * _N ** 3 / 240)
_BETA = (_N / 2 - 2 * _N ** 2 / 3 + 37 * _N ** 3 / 96, _N ** 2 / 48 + _N ** 3 / 15, 17 * _N ** 3 / 480)
_DELTA = (2 * _N - 2 * _N ** 2 / 3 - 2 * _N ** 3, 7 * _N ** 2 / 3 - 8 * _N ** 3 / 5, 56 * _N ** 3 / 15)


def utm_zone(lon: float) -> int:
    return int((lon + 180) // 6) % 60 + 1


def latlon_to_utm(lat: float, lon: float, zone: int | None = None):
    """WGS84 -> (easting, northing, zone, northern hemisphere?). Krüger series, good to millimetres."""
    zone = zone or utm_zone(lon)
    lam = math.radians(lon - (zone * 6 - 183))
    phi = math.radians(lat)
    c = 2 * math.sqrt(_N) / (1 + _N)
    t = math.sinh(math.atanh(math.sin(phi)) - c * math.atanh(c * math.sin(phi)))
    xi = math.atan2(t, math.cos(lam))
    eta = math.atanh(math.sin(lam) / math.sqrt(1 + t * t))
    e = eta + sum(a * math.cos(2 * j * xi) * math.sinh(2 * j * eta) for j, a in enumerate(_ALPHA, 1))
    n = xi + sum(a * math.sin(2 * j * xi) * math.cosh(2 * j * eta) for j, a in enumerate(_ALPHA, 1))
    return 500000 + K0 * _A * e, K0 * _A * n + (0 if lat >= 0 else 10_000_000), zone, lat >= 0


def utm_to_latlon(e: float, n: float, zone: int, north: bool = True):
    xi = (n - (0 if north else 10_000_000)) / (K0 * _A)
    eta = (e - 500000) / (K0 * _A)
    xi_p = xi - sum(b * math.sin(2 * j * xi) * math.cosh(2 * j * eta) for j, b in enumerate(_BETA, 1))
    eta_p = eta - sum(b * math.cos(2 * j * xi) * math.sinh(2 * j * eta) for j, b in enumerate(_BETA, 1))
    chi = math.asin(math.sin(xi_p) / math.cosh(eta_p))
    phi = chi + sum(d * math.sin(2 * j * chi) for j, d in enumerate(_DELTA, 1))
    lam = math.atan2(math.sinh(eta_p), math.cos(xi_p))
    return math.degrees(phi), zone * 6 - 183 + math.degrees(lam)


# ---------------------------------------------------------------------------
# GeoTIFF: just enough to place a north-up raster in UTM or in degrees
# ---------------------------------------------------------------------------


class Georef:
    """Where a raster sits: world coordinates of pixel corners, and how to get lat/lon from them."""

    def __init__(self, im: Image.Image, path: Path):
        tags = im.tag_v2
        scale, tie = tags.get(33550), tags.get(33922)
        if not scale or not tie or len(tie) < 6:
            raise ValueError(f"{path.name} has no position information (not a GeoTIFF from a mapping tool)")
        self.sx, self.sy = float(scale[0]), float(scale[1])
        self.x0 = float(tie[3]) - float(tie[0]) * self.sx
        self.y0 = float(tie[4]) + float(tie[1]) * self.sy
        keys = tags.get(34735) or ()
        geo = {keys[i]: keys[i + 3] for i in range(4, len(keys) - 3, 4)}
        epsg = geo.get(3072)
        self.zone = self.north = None
        if epsg and (32601 <= epsg <= 32660 or 32701 <= epsg <= 32760):
            self.zone, self.north = epsg % 100, epsg < 32700
        elif epsg or geo.get(1024) not in (None, 2):
            raise ValueError(f"{path.name} uses map projection EPSG:{epsg}; only UTM (WGS84) and plain latitude/"
                             f"longitude are understood. Re-export it in UTM.")
        nodata = tags.get(42113)
        try:
            self.nodata = float(str(nodata).strip("\x00 ")) if nodata is not None else None
        except ValueError:
            self.nodata = None

    def latlon(self, i: float, j: float):
        x, y = self.x0 + i * self.sx, self.y0 - j * self.sy
        return utm_to_latlon(x, y, self.zone, self.north) if self.zone else (y, x)


def _open(path: Path):
    im = Image.open(path)
    if im.width * im.height > 250_000_000:
        raise ValueError(f"{path.name} is {im.width}x{im.height} pixels, too large to load. Export it at a coarser "
                         "resolution (5 cm a pixel is plenty).")
    return im, Georef(im, path)


def _to_grid(im: Image.Image, g: Georef, origin, bounds, ppm: float, resample):
    """Resample a georeferenced raster onto the survey grid (true north up, metres from ``origin``)."""
    w, h = im.size
    corners = [(0, 0), (w, 0), (0, h)]
    loc = [to_local(origin[0], origin[1], *g.latlon(i, j)) for i, j in corners]
    # pixel -> local is affine to well under a pixel across a site; invert it for PIL (output px -> source px)
    M = np.array([[loc[1][0] - loc[0][0], loc[2][0] - loc[0][0]], [loc[1][1] - loc[0][1], loc[2][1] - loc[0][1]]]) / \
        np.array([[w, h], [w, h]], float)
    Mi = np.linalg.inv(M)
    x0, y0, x1, y1 = bounds
    W, H = max(2, int(round((x1 - x0) * ppm))), max(2, int(round((y1 - y0) * ppm)))
    # local(u, v) = (x0 + u/ppm, y1 - v/ppm);  src = Mi @ (local - loc0)
    a, b = Mi[0, 0] / ppm, -Mi[0, 1] / ppm
    c = Mi[0, 0] * (x0 - loc[0][0]) + Mi[0, 1] * (y1 - loc[0][1])
    d, e = Mi[1, 0] / ppm, -Mi[1, 1] / ppm
    f = Mi[1, 0] * (x0 - loc[0][0]) + Mi[1, 1] * (y1 - loc[0][1])
    return im.transform((W, H), Image.AFFINE, (a, b, c, d, e, f), resample)


def raster_bounds(g: Georef, size, origin) -> list[float]:
    w, h = size
    pts = [to_local(origin[0], origin[1], *g.latlon(i, j)) for i, j in ((0, 0), (w, 0), (w, h), (0, h))]
    return [min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)]


def _heights(path: Path, origin, bounds, ppm: float):
    """An elevation GeoTIFF on the survey grid: (metres as float32, where it has data)."""
    im, g = _open(path)
    a = np.asarray(im, np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    bad = ~np.isfinite(a) | (a < -1000) | (a > 9000)
    if g.nodata is not None:
        bad |= a == np.float32(g.nodata)
    if bad.all():
        raise ValueError(f"{path.name} holds no heights")
    a = np.where(bad, float(np.median(a[~bad])), a).astype(np.float32)
    z = np.asarray(_to_grid(Image.fromarray(a, "F"), g, origin, bounds, ppm, Image.BILINEAR), np.float32)
    ok = np.asarray(_to_grid(Image.fromarray((~bad).astype(np.uint8) * 255), g, origin, bounds, ppm, Image.NEAREST)) > 127
    ok = ndimage.binary_erosion(ok, iterations=1)  # the outermost pixels are blended with the blank surround
    if ok.any() and not ok.all():  # carry the nearest real height outward so the blank surround is not a cliff
        idx = ndimage.distance_transform_edt(~ok, return_distances=False, return_indices=True)
        z = z[tuple(idx)]
    return z.astype(np.float32), ok


def find_odm(folder: Path) -> dict | None:
    """The orthophoto and elevation models of an OpenDroneMap / WebODM project in or under ``folder``."""
    folder = Path(folder)
    for root in (folder, folder / "code", folder.parent):
        o = root / "odm_orthophoto" / "odm_orthophoto.tif"
        if o.exists():
            d = {"ortho": o}
            for k in ("dsm", "dtm"):
                p = root / "odm_dem" / f"{k}.tif"
                if p.exists():
                    d[k] = p
            return d
    return None


def ortho_centre(ortho: Path):
    im, g = _open(Path(ortho))
    return g.latlon(im.width / 2, im.height / 2)


def import_terrain(out: Path, origin, ortho: Path, dsm: Path | None = None, dtm: Path | None = None,
                   max_px: int = 3600, log=print):
    """Write basemap.jpg (from the orthophoto) and heights.png into the project. Returns (basemap, terrain)."""
    out = Path(out)
    im, g = _open(Path(ortho))
    bounds = raster_bounds(g, im.size, origin)
    span = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    ppm = min(20.0, max_px / span, 1.0 / max(g.sx if g.zone else g.sx * 111_000, 1e-6))
    rgba = im.convert("RGBA")
    grid = _to_grid(rgba, g, origin, bounds, ppm, Image.BICUBIC)
    canvas = Image.new("RGB", grid.size, (24, 26, 31))
    canvas.paste(grid.convert("RGB"), mask=grid.getchannel("A"))
    canvas.save(out / "basemap.jpg", "JPEG", quality=88)
    log(f"  base map from the orthophoto {Path(ortho).name} ({im.width}x{im.height}, {1 / ppm * 100:.0f} cm a pixel)")
    basemap = {"file": "basemap.jpg", "bounds": [round(b, 3) for b in bounds], "px_per_m": round(ppm, 3),
               "width": grid.width, "height": grid.height, "straight_down": 0, "angled": 0, "ortho": True}
    if not dsm:
        return basemap, None
    hp = min(4.0, ppm)  # 25 cm cells are ample for sightlines
    z, ok = _heights(Path(dsm), origin, bounds, hp)
    if dtm:
        ground, gok = _heights(Path(dtm), origin, bounds, hp)
        ok &= gok
    else:  # no terrain model: the ground is the low envelope of the surface, smoothed
        # An opening follows slopes but cannot climb onto anything narrower than its window (40 m here),
        # so buildings and trees are left standing proud of it. Very wide buildings need a real DTM.
        k = max(3, int(round(40 * hp)) | 1)
        ks = max(3, k // 4 | 1)
        pad = min(ks, min(z[::4, ::4].shape) - 1)
        small = np.pad(z[::4, ::4], pad, mode="reflect", reflect_type="odd")  # continue the slope past the edge
        ground = ndimage.gaussian_filter(ndimage.grey_opening(small, size=ks), 1.5)[pad:-pad, pad:-pad]
        ground = np.asarray(Image.fromarray(ground, "F").resize((z.shape[1], z.shape[0]), Image.BILINEAR), np.float32)
    nd = np.clip(z - ground, 0, 600)
    nd[~ok] = 0
    nd[nd < 0.25] = 0  # reconstruction noise on bare ground
    cm = np.round(nd * 100).astype(np.uint16)
    Image.fromarray(np.dstack([cm >> 8, cm & 255, ok.astype(np.uint8) * 255]).astype(np.uint8), "RGB").save(
        out / "heights.png", optimize=True)
    log(f"  measured heights from {Path(dsm).name}" + (f" and {Path(dtm).name}" if dtm else " (ground level estimated)")
        + f": tallest {float(nd.max()):.1f} m")
    terrain = {"file": "heights.png", "bounds": basemap["bounds"], "px_per_m": round(hp, 3),
               "width": int(nd.shape[1]), "height": int(nd.shape[0]), "has_dtm": bool(dtm),
               "max_m": round(float(nd.max()), 1), "source": "OpenDroneMap elevation model"}
    return basemap, terrain


def load_heights(project: Path, terrain: dict):
    """heights.png -> (metres above ground as float32, valid mask), or None if the file is missing."""
    p = Path(project) / terrain["file"]
    if not p.exists():
        return None
    a = np.asarray(Image.open(p).convert("RGB"), np.uint16)
    return ((a[..., 0] << 8) | a[..., 1]).astype(np.float32) / 100.0, a[..., 2] > 127


def write_geotiff(path: Path, array: np.ndarray, e0: float, n0: float, res: float, zone: int, north: bool = True,
                  nodata: float | None = None) -> Path:
    """A minimal UTM GeoTIFF (float32 heights or 8-bit colour). Used by the demo and the tests."""
    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    ifd[33550] = (float(res), float(res), 0.0)
    ifd.tagtype[33550] = 12
    ifd[33922] = (0.0, 0.0, 0.0, float(e0), float(n0), 0.0)
    ifd.tagtype[33922] = 12
    ifd[34735] = (1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, (32600 if north else 32700) + zone)
    ifd.tagtype[34735] = 3
    if nodata is not None:
        ifd[42113] = str(nodata)
        ifd.tagtype[42113] = 2
    im = Image.fromarray(array.astype(np.float32), "F") if array.ndim == 2 else Image.fromarray(array.astype(np.uint8))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    im.save(path, format="TIFF", tiffinfo=ifd, compression="tiff_adobe_deflate")
    return Path(path)


# ---------------------------------------------------------------------------
# running OpenDroneMap
# ---------------------------------------------------------------------------

ODM_IMAGE = "opendronemap/odm"


def odm_available() -> str | None:
    """The container tool that can run OpenDroneMap here, if any."""
    return shutil.which("docker") or shutil.which("podman")


def run_odm(photos: list[Path], work: Path, log=print, quality: str = "low") -> dict:
    """Build an orthophoto and elevation models from overlapping drone photos with OpenDroneMap.

    Needs Docker (or Podman) and, the first time, an internet connection to
    fetch the OpenDroneMap image (several GB). It is slow and memory-hungry:
    allow tens of minutes and 8 GB or more of RAM for a few dozen photos.
    """
    tool = odm_available()
    if not tool:
        raise RuntimeError("Building a 3-D map needs Docker with OpenDroneMap, and Docker was not found. Install "
                           "Docker, or build the map in WebODM and choose its project folder instead.")
    work = Path(work)
    images = work / "code" / "images"
    images.mkdir(parents=True, exist_ok=True)
    for p in photos:
        if not (images / p.name).exists():
            shutil.copy2(p, images / p.name)
    cmd = [tool, "run", "--rm", "-v", f"{work.resolve()}:/datasets", ODM_IMAGE, "--project-path", "/datasets", "code",
           "--dsm", "--dtm", "--orthophoto-resolution", "4", "--dem-resolution", "8", "--pc-quality", quality,
           "--feature-quality", "medium", "--skip-3dmodel"]
    log(f"Running OpenDroneMap on {len(photos)} photos. This takes a while...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    last = ""
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            last = line
            if "[INFO]" in line or "[WARNING]" in line or "[ERROR]" in line:
                log("  " + line[-160:])
    if proc.wait() != 0:
        raise RuntimeError(f"OpenDroneMap stopped with an error: {last[-300:] or 'no output'}")
    found = find_odm(work)
    if not found:
        raise RuntimeError("OpenDroneMap finished but wrote no orthophoto. It usually means too few photos or too "
                           "little overlap between them (aim for 70% or more).")
    return found
