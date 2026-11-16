"""Read DJI photo metadata and project image pixels onto the ground.

DJI drones write two blocks of metadata into every JPEG:

* standard EXIF: GPS position, focal length, capture time;
* a DJI XMP block (``drone-dji:`` namespace) with the gimbal attitude
  (yaw/pitch/roll) and the altitude relative to the take-off point.

With the camera position, attitude and lens we can cast a ray through any
pixel and intersect it with a flat ground plane, which turns "a red blob at
pixel (3120, 1877)" into a latitude/longitude a ground team can walk to.

Accuracy caveats (also printed in the report):

* ``RelativeAltitude`` is height above the *take-off point*, not above the
  ground under the drone. Over sloping terrain the projected position is off
  by roughly (terrain height difference) x tan(view angle). For nadir shots
  that error is small; for oblique shots it grows quickly. Use
  ``ground_offset_m`` when you know the search area sits above/below take-off.
* Consumer Mavic 3 GPS is good to a few metres; there is no RTK.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PIL import ExifTags, Image

from .geo import offset_latlon

# 35 mm film diagonal, used to turn "35 mm equivalent focal length" into pixels.
FULL_FRAME_DIAGONAL_MM = math.hypot(36.0, 24.0)

_XMP_ATTR = re.compile(rb'drone-dji:(\w+)="([^"]*)"')
_XMP_ELEM = re.compile(rb"<drone-dji:(\w+)>([^<]*)</drone-dji:\1>")


@dataclass
class PhotoMeta:
    path: Path
    width: int
    height: int
    lat: float | None = None
    lon: float | None = None
    rel_alt_m: float | None = None  # above take-off point
    abs_alt_m: float | None = None
    gimbal_yaw: float | None = None  # deg, clockwise from true north
    gimbal_pitch: float | None = None  # deg, -90 = straight down
    gimbal_roll: float = 0.0
    flight_yaw: float | None = None
    focal_35mm: float | None = None
    focal_mm: float | None = None
    taken_at: datetime | None = None
    camera_model: str | None = None
    raw_xmp: dict[str, str] = field(default_factory=dict)

    @property
    def has_position(self) -> bool:
        return self.lat is not None and self.lon is not None

    @property
    def can_geolocate(self) -> bool:
        return self.has_position and self.rel_alt_m is not None and self.gimbal_pitch is not None

    def focal_px(self) -> float:
        """Focal length in pixels, derived from the 35 mm-equivalent focal length."""
        f35 = self.focal_35mm or 24.0  # Mavic 3 Hasselblad main camera is 24 mm eq.
        return f35 / FULL_FRAME_DIAGONAL_MM * math.hypot(self.width, self.height)

    def gsd_m(self, ground_offset_m: float = 0.0) -> float | None:
        """Approximate ground sample distance (metres per pixel) at image centre."""
        h = self.height_agl(ground_offset_m)
        if h is None or h <= 0:
            return None
        pitch = self.gimbal_pitch if self.gimbal_pitch is not None else -90.0
        # slant range to the centre of the image
        s = math.sin(math.radians(-pitch))
        if s < 0.05:
            return None
        return (h / s) / self.focal_px()

    def height_agl(self, ground_offset_m: float = 0.0) -> float | None:
        if self.rel_alt_m is None:
            return None
        return self.rel_alt_m - ground_offset_m


def _rational(v) -> float:
    try:
        return float(v)
    except TypeError:
        num, den = v
        return num / den if den else 0.0


def _dms_to_deg(dms, ref) -> float:
    d, m, s = (_rational(x) for x in dms)
    deg = d + m / 60 + s / 3600
    if ref in ("S", "W", b"S", b"W"):
        deg = -deg
    return deg


def _parse_xmp(data: bytes) -> dict[str, str]:
    start = data.find(b"<x:xmpmeta")
    if start < 0:
        return {}
    end = data.find(b"</x:xmpmeta>", start)
    block = data[start : end if end > 0 else start + 65536]
    out: dict[str, str] = {}
    for rx in (_XMP_ATTR, _XMP_ELEM):
        for k, v in rx.findall(block):
            out[k.decode()] = v.decode(errors="replace").strip()
    return out


def _f(d: dict[str, str], key: str) -> float | None:
    v = d.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def read_photo_meta(path: str | Path) -> PhotoMeta:
    path = Path(path)
    with Image.open(path) as im:
        width, height = im.size
        exif = im.getexif()
        info_xmp = im.info.get("xmp")
    meta = PhotoMeta(path=path, width=width, height=height)

    meta.camera_model = exif.get(ExifTags.Base.Model)
    sub = exif.get_ifd(ExifTags.IFD.Exif)
    if ExifTags.Base.FocalLengthIn35mmFilm in sub:
        meta.focal_35mm = float(sub[ExifTags.Base.FocalLengthIn35mmFilm]) or None
    if ExifTags.Base.FocalLength in sub:
        meta.focal_mm = _rational(sub[ExifTags.Base.FocalLength])
    dt = sub.get(ExifTags.Base.DateTimeOriginal) or exif.get(ExifTags.Base.DateTime)
    if dt:
        try:
            meta.taken_at = datetime.strptime(str(dt).strip("\x00 "), "%Y:%m:%d %H:%M:%S")
        except ValueError:
            pass

    gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
    try:
        if 2 in gps and 4 in gps:
            meta.lat = _dms_to_deg(gps[2], gps.get(1, "N"))
            meta.lon = _dms_to_deg(gps[4], gps.get(3, "E"))
        if 6 in gps:
            alt = _rational(gps[6])
            if gps.get(5) in (1, b"\x01"):
                alt = -alt
            meta.abs_alt_m = alt
    except (TypeError, ValueError, ZeroDivisionError):
        pass

    # XMP: Pillow exposes it on some versions; otherwise scan the first 256 KB.
    xmp_bytes = info_xmp if isinstance(info_xmp, bytes) else (info_xmp or "").encode()
    if b"drone-dji" not in xmp_bytes:
        with open(path, "rb") as fh:
            xmp_bytes = fh.read(262144)
    x = _parse_xmp(xmp_bytes)
    meta.raw_xmp = x
    if x:
        lat, lon = _f(x, "GpsLatitude"), _f(x, "GpsLongitude") or _f(x, "GpsLongtitude")
        if lat is not None and lon is not None and (lat, lon) != (0.0, 0.0):
            meta.lat, meta.lon = lat, lon
        meta.rel_alt_m = _f(x, "RelativeAltitude")
        meta.abs_alt_m = _f(x, "AbsoluteAltitude") if _f(x, "AbsoluteAltitude") is not None else meta.abs_alt_m
        meta.gimbal_yaw = _f(x, "GimbalYawDegree")
        meta.gimbal_pitch = _f(x, "GimbalPitchDegree")
        meta.gimbal_roll = _f(x, "GimbalRollDegree") or 0.0
        meta.flight_yaw = _f(x, "FlightYawDegree")
    if meta.gimbal_yaw is None and meta.flight_yaw is not None:
        meta.gimbal_yaw = meta.flight_yaw
    return meta


# --------------------------------------------------------------------------
# Pixel -> ground projection
# --------------------------------------------------------------------------


def _camera_axes(yaw_deg: float, pitch_deg: float, roll_deg: float):
    """Camera axes in a local North-East-Down frame.

    Returns (forward, right, down_in_image) unit vectors. Pitch is DJI style:
    0 = horizon, -90 = nadir. Yaw is clockwise from north.
    """
    y, p, r = (math.radians(a) for a in (yaw_deg, pitch_deg, roll_deg))
    fwd = (math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), -math.sin(p))
    right = (-math.sin(y), math.cos(y), 0.0)
    # down_in_image = fwd x right
    down = (
        fwd[1] * right[2] - fwd[2] * right[1],
        fwd[2] * right[0] - fwd[0] * right[2],
        fwd[0] * right[1] - fwd[1] * right[0],
    )
    if r:
        cr, sr = math.cos(r), math.sin(r)
        right, down = (
            tuple(cr * a + sr * b for a, b in zip(right, down)),
            tuple(-sr * a + cr * b for a, b in zip(right, down)),
        )
    return fwd, right, down


def pixel_to_ground(
    meta: PhotoMeta, u: float, v: float, ground_offset_m: float = 0.0
) -> tuple[float, float] | None:
    """Project pixel (u, v) onto flat ground. Returns (lat, lon) or None.

    None means the ray does not hit the ground (pixel at/above the horizon) or
    the photo lacks the metadata needed.
    """
    if not meta.can_geolocate:
        return None
    h = meta.height_agl(ground_offset_m)
    if h is None or h <= 0:
        return None
    yaw = meta.gimbal_yaw or 0.0
    fwd, right, down = _camera_axes(yaw, meta.gimbal_pitch, meta.gimbal_roll)
    f = meta.focal_px()
    cx, cy = meta.width / 2, meta.height / 2
    a, b = (u - cx) / f, (v - cy) / f
    d = [fwd[i] + a * right[i] + b * down[i] for i in range(3)]
    if d[2] <= 1e-3:  # pointing at or above the horizon
        return None
    t = h / d[2]
    north, east = t * d[0], t * d[1]
    # refuse absurd ranges (near-horizon rays): > 20x altitude or > 3 km
    rng = math.hypot(north, east)
    if rng > 3000.0 or rng > 20 * h:
        return None
    return offset_latlon(meta.lat, meta.lon, north, east)


def photo_footprint(meta: PhotoMeta, ground_offset_m: float = 0.0) -> list[tuple[float, float]] | None:
    """Ground polygon (lat, lon) of the image corners, or None if not projectable."""
    w, hgt = meta.width, meta.height
    pts = []
    for u, v in ((0, 0), (w, 0), (w, hgt), (0, hgt)):
        p = pixel_to_ground(meta, u, v, ground_offset_m)
        if p is None:
            # oblique shot: clip top corners down until they hit ground
            for frac in (0.25, 0.5, 0.75):
                p = pixel_to_ground(meta, u, hgt * frac if v == 0 else v, ground_offset_m)
                if p is not None:
                    break
        if p is None:
            return None
        pts.append(p)
    return pts
