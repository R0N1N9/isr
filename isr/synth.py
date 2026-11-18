"""Synthetic drone photos with real metadata and known ground truth, for tests and the app's demos."""

from __future__ import annotations

import math
from PIL.TiffImagePlugin import IFDRational as Fraction
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

XMP_TEMPLATE = """<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="DJI Meta Data"
    xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/"
   drone-dji:GpsLatitude="{lat:+.8f}"
   drone-dji:GpsLongitude="{lon:+.8f}"
   drone-dji:AbsoluteAltitude="{abs_alt:+.2f}"
   drone-dji:RelativeAltitude="{rel_alt:+.2f}"
   drone-dji:GimbalRollDegree="+0.00"
   drone-dji:GimbalYawDegree="{yaw:+.2f}"
   drone-dji:GimbalPitchDegree="{pitch:+.2f}"
   drone-dji:FlightRollDegree="+0.00"
   drone-dji:FlightYawDegree="{yaw:+.2f}"
   drone-dji:FlightPitchDegree="+0.00"/>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end='w'?>"""


def _dms(v: float):
    v = abs(v)
    d = int(v)
    m = int((v - d) * 60)
    s = (v - d - m / 60) * 3600
    return (Fraction(d), Fraction(m), Fraction(int(round(s * 10000)), 10000))


def terrain(w: int, h: int, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    # low-frequency colour variation (grass / dry grass / soil)
    lo = rng.random((h // 64 + 2, w // 64 + 2, 3)).astype(np.float32)
    lo = np.asarray(Image.fromarray((lo * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC), np.float32) / 255
    base = np.array([78, 104, 52], np.float32)
    var = np.array([40, 34, 22], np.float32)
    img = base + (lo - 0.5) * var * 2
    img += rng.normal(0, 9, (h, w, 1)).astype(np.float32)  # texture
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    # trees with shadows
    for _ in range(int(w * h / 90000)):
        x, y, r = rng.integers(0, w), rng.integers(0, h), rng.integers(25, 90)
        d.ellipse((x - r + r // 3, y - r + r // 3, x + r + r // 3, y + r + r // 3), fill=(28, 36, 22))
        g = int(rng.integers(40, 70))
        d.ellipse((x - r, y - r, x + r, y + r), fill=(g - 10, g + 20, g - 25))
    # dirt road
    y0 = int(h * 0.8)
    d.line([(0, y0), (w, y0 - h // 6)], fill=(150, 134, 108), width=max(30, w // 70))
    return im.filter(ImageFilter.GaussianBlur(0.8))


def draw_person(im: Image.Image, cx: float, cy: float, gsd_m: float, jacket=(200, 25, 30), pants=(30, 50, 140),
                lying: bool = False) -> tuple[float, float, float, float]:
    """Top-down person: shoulders ~0.5 m, height 1.7 m when lying, torso blob when standing."""
    d = ImageDraw.Draw(im)
    px = lambda m: m / gsd_m  # noqa: E731
    if lying:
        L, W = px(1.7), px(0.45)
        d.rectangle((cx - L / 2, cy - W / 2, cx, cy + W / 2), fill=jacket)
        d.rectangle((cx, cy - W / 2.4, cx + L / 2, cy + W / 2.4), fill=pants)
        d.ellipse((cx - L / 2 - px(0.22), cy - px(0.11), cx - L / 2, cy + px(0.11)), fill=(170, 130, 100))
        return cx - L / 2 - px(0.22), cy - W / 2, cx + L / 2, cy + W / 2
    sw, dep = px(0.5), px(0.3)
    d.ellipse((cx - sw / 2, cy - dep / 2, cx + sw / 2, cy + dep / 2), fill=jacket)
    d.ellipse((cx - px(0.1), cy - px(0.1), cx + px(0.1), cy + px(0.1)), fill=(60, 40, 30))  # hair
    d.ellipse((cx - sw / 2 + px(0.3), cy - dep / 2 + px(0.3), cx + sw / 2 + px(0.3), cy + dep / 2 + px(0.3)),
              fill=(30, 38, 24))  # shadow
    return cx - sw / 2, cy - dep / 2, cx + sw / 2, cy + dep / 2


def save_dji_jpeg(im: Image.Image, path: Path, lat: float, lon: float, rel_alt: float, yaw: float = 0.0,
                  pitch: float = -90.0, when: str = "2026:11:16 10:00:00") -> Path:
    exif = Image.Exif()
    exif[0x010F] = "DJI"
    exif[0x0110] = "L2D-20c"
    ex = exif.get_ifd(0x8769)
    ex[0x920A] = Fraction(1229, 100)  # FocalLength 12.29 mm
    ex[0xA405] = 24  # FocalLengthIn35mmFilm
    ex[0x9003] = when
    gps = exif.get_ifd(0x8825)
    gps[1] = "N" if lat >= 0 else "S"
    gps[2] = _dms(lat)
    gps[3] = "E" if lon >= 0 else "W"
    gps[4] = _dms(lon)
    gps[5] = 0
    gps[6] = Fraction(int((rel_alt + 1600) * 100), 100)
    xmp = XMP_TEMPLATE.format(lat=lat, lon=lon, abs_alt=rel_alt + 1600, rel_alt=rel_alt, yaw=yaw, pitch=pitch)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, "JPEG", quality=90, exif=exif, xmp=xmp.encode())
    return path


MAVIC3_W, MAVIC3_H = 5280, 3956


def mavic3_gsd(alt_m: float) -> float:
    f_px = 24 / math.hypot(36, 24) * math.hypot(MAVIC3_W, MAVIC3_H)
    return alt_m / f_px
