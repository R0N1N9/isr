"""A made-up property, for trying the tool and for the tests.

``build_demo`` paints a straight-down drone photo of a house, garage, hedges
and street with real DJI metadata, four angled shots circling the house as a
hand-flown survey would, and a few phone-style ground photos with
GPS and compass headings, runs the normal survey on them, and then draws the
site plan a user would draw by hand. The layout is in metres, east/north of
the middle of the lot.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from PIL.TiffImagePlugin import IFDRational as Fraction

from ..geo import offset_latlon
from . import model as M
from .survey import run_survey

LAT0, LON0 = 51.752, -1.2577  # middle of the lot (an arbitrary spot; the property does not exist)
DRONE = (3.0, -4.0)  # where the drone hovered, east/north of the lot centre
ALT, YAW = 70.0, 12.0
IMG_W, IMG_H = 2640, 1978

HOUSE = [(-10, -6), (6, -6), (6, 5), (0, 5), (0, 9), (-10, 9)]
GARAGE = [(11, -8), (19, -8), (19, 0), (11, 0)]
SHED = [(-19.5, 13), (-16.5, 13), (-16.5, 16), (-19.5, 16)]
LOT = [(-22, -18), (22, -18), (22, 18), (-22, 18)]
SHRUBS = [(-16.5, -3), (-11.5, -3), (-11.2, 6.5), (-16, 7)]
TREES = [(9, 9), (14, 7.5), (19, 10), (20, 15.5), (14, 17), (9.5, 14.5)]

FEATURES = [
    ("boundary", None, "Property line", LOT, {}),
    ("building", None, "House", HOUSE, {"height": 6.5}),
    ("building", None, "Garage", GARAGE, {"height": 3.2}),
    ("building", None, "Shed", SHED, {"height": 2.4}),
    ("barrier", "hedge", "West hedge", [(-22, -18), (-22, 18)], {"height": 2.2}),
    ("barrier", "hedge", "Front hedge", [(-22, -18), (-7, -18)], {"height": 1.8}),
    ("barrier", "wall", "Front wall", [(-7, -18), (11, -18)], {"height": 0.9}),
    ("barrier", "wall", "Front wall east", [(19, -18), (22, -18)], {"height": 0.9}),
    ("barrier", "fence", "Rear fence", [(-22, 18), (22, 18)], {"height": 1.8}),
    ("barrier", "open_fence", "East fence north", [(22, 18), (22, 7.5)], {"height": 1.5}),
    ("barrier", "open_fence", "East fence south", [(22, 2), (22, -18)], {"height": 1.5}),
    ("vegetation", None, "Side shrubs", SHRUBS, {"height": 2.5}),
    ("vegetation", None, "Back trees", TREES, {"height": 7.0}),
    ("street", None, "Street", [(-48, -21), (48, -21)], {}),
    ("access", "door", "Front door", [(-2, -6)], {}),
    ("access", "door", "Side door", [(-10, 2)], {}),
    ("access", "door", "Back door", [(-5, 9)], {}),
    ("access", "window", "Front window", [(3, -6)], {}),
    ("access", "window", "East window", [(6, 0)], {}),
    ("access", "window", "Kitchen window", [(3, 5)], {}),
    ("access", "garage", "Garage door", [(15, -8)], {}),
    ("access", "door", "Garage side door", [(11, -3)], {}),
    ("access", "vehicle_gate", "Driveway", [(15, -18)], {}),
    ("access", "gate", "Front gate", [(-3, -18)], {}),
    ("camera", None, "Doorbell camera", [(-2.6, -6)], {"height": 1.3, "bearing": 180, "hfov": 120, "range": 8,
                                                         "status": "existing"}),
]

GROUND = [  # name, east, north, heading, caption
    ("walk_01_front.jpg", -2, -20.5, 0, "Front of house from the pavement"),
    ("walk_02_driveway.jpg", 15, -20, 0, "Driveway and garage"),
    ("walk_03_side_passage.jpg", -10.8, -11, 0, "West side passage towards the side door"),
    ("walk_04_back_garden.jpg", -4, 15, 180, "Back of house from the lawn"),
    ("walk_05_east_fence.jpg", 19, 4.5, 90, "Missing panel in the east fence"),
    ("walk_06_back_trees.jpg", 5, 13, 60, "Trees in the back corner"),
]


def _gsd() -> float:
    f_px = 24 / math.hypot(36, 24) * math.hypot(IMG_W, IMG_H)
    return ALT / f_px


def _to_px(e: float, n: float):
    """Lot coordinates -> pixel in the drone photo (the inverse of isr.camera's projection)."""
    g, y = _gsd(), math.radians(YAW)
    e, n = e - DRONE[0], n - DRONE[1]
    return (IMG_W / 2 + (e * math.cos(y) - n * math.sin(y)) / g,
            IMG_H / 2 + (-n * math.cos(y) - e * math.sin(y)) / g)


def _pp(pts):
    return [_to_px(*p) for p in pts]


def _shift(pts, dx, dy):
    return [(x + dx, y + dy) for x, y in pts]


def render_aerial(seed: int = 3) -> Image.Image:
    rng = np.random.default_rng(seed)
    g = _gsd()
    lo = rng.random((IMG_H // 90 + 2, IMG_W // 90 + 2, 3)).astype(np.float32)
    lo = np.asarray(Image.fromarray((lo * 255).astype(np.uint8)).resize((IMG_W, IMG_H), Image.BICUBIC), np.float32) / 255
    img = np.array([96, 128, 66], np.float32) + (lo - 0.5) * np.array([26, 26, 16], np.float32)
    img += rng.normal(0, 6, (IMG_H, IMG_W, 1)).astype(np.float32)
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    w = lambda m: max(1, int(round(m / g)))  # noqa: E731

    d.polygon(_pp([(-60, -31), (60, -31), (60, -22.5), (-60, -22.5)]), fill=(72, 74, 78))  # road
    for x in range(-58, 58, 6):
        d.line(_pp([(x, -26.8), (x + 3, -26.8)]), fill=(206, 206, 196), width=w(0.15))
    d.polygon(_pp([(-60, -22.5), (60, -22.5), (60, -19.5), (-60, -19.5)]), fill=(168, 166, 160))  # pavement
    d.polygon(_pp([(12, -19.5), (18, -19.5), (18, -8), (12, -8)]), fill=(150, 148, 142))  # driveway
    d.polygon(_pp([(-3.7, -18), (-2.3, -18), (-1.3, -6), (-2.7, -6)]), fill=(176, 160, 138))  # path
    d.polygon(_pp([(-10, 9), (0, 9), (0, 13), (-10, 13)]), fill=(170, 150, 128))  # patio

    sh = (-0.34, 0.30)  # shadow offset per metre of height (sun in the south-east)
    for pts, h in ((HOUSE, 6.5), (GARAGE, 3.2), (SHED, 2.4), (TREES, 5.0), (SHRUBS, 2.0)):
        d.polygon(_pp(_shift(pts, sh[0] * h, sh[1] * h)), fill=(46, 62, 38))
    for line, h in (([(-22, -18), (-22, 18)], 2.2), ([(-22, -18), (-7, -18)], 1.8), ([(-22, 18), (22, 18)], 1.8)):
        d.line(_pp(_shift(line, sh[0] * h, sh[1] * h)), fill=(46, 62, 38), width=w(1.0))

    d.line(_pp([(-22, -18), (-22, 18)]), fill=(38, 78, 40), width=w(1.3))  # hedges
    d.line(_pp([(-22, -18), (-7, -18)]), fill=(40, 82, 42), width=w(1.2))
    d.line(_pp([(-7, -18), (11, -18)]), fill=(196, 188, 176), width=w(0.35))  # low wall
    d.line(_pp([(19, -18), (22, -18)]), fill=(196, 188, 176), width=w(0.35))
    d.line(_pp([(-22, 18), (22, 18)]), fill=(122, 92, 62), width=w(0.25))  # timber fence
    for a, b in (((22, 18), (22, 7.5)), ((22, 2), (22, -18))):
        d.line(_pp([a, b]), fill=(150, 152, 150), width=w(0.12))

    for pts in (SHRUBS, TREES):  # planting as overlapping crowns
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        big = pts is TREES
        for _ in range(60 if big else 34):
            x, y = rng.uniform(min(xs), max(xs)), rng.uniform(min(ys), max(ys))
            if not M.point_in_poly(x, y, pts):
                continue
            r = rng.uniform(1.4, 2.6) if big else rng.uniform(0.6, 1.2)
            c = int(rng.integers(0, 26))
            px, py = _to_px(x, y)
            d.ellipse((px - r / g, py - r / g, px + r / g, py + r / g), fill=(30 + c, 70 + c, 32 + c // 2))

    d.polygon(_pp(HOUSE), fill=(150, 78, 62))  # roofs, with a lit and a shaded pitch
    d.polygon(_pp([(-10, 1.5), (6, 1.5), (6, 5), (0, 5), (0, 9), (-10, 9)]), fill=(128, 64, 52))
    d.line(_pp([(-10, 1.5), (6, 1.5)]), fill=(98, 50, 42), width=w(0.25))
    d.polygon(_pp(GARAGE), fill=(108, 110, 116))
    d.polygon(_pp([(11, -4), (19, -4), (19, 0), (11, 0)]), fill=(92, 94, 100))
    d.polygon(_pp(SHED), fill=(112, 96, 74))
    d.polygon(_pp([(13.2, -16.5), (15.1, -16.5), (15.1, -12.2), (13.2, -12.2)]), fill=(34, 60, 122))  # car
    d.polygon(_pp([(13.4, -15.4), (14.9, -15.4), (14.9, -13.2), (13.4, -13.2)]), fill=(24, 30, 44))
    return im.filter(ImageFilter.GaussianBlur(0.9))


XMP = """<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="DJI Meta Data" xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/"
   drone-dji:GpsLatitude="{lat:+.8f}" drone-dji:GpsLongitude="{lon:+.8f}"
   drone-dji:AbsoluteAltitude="{abs_alt:+.2f}" drone-dji:RelativeAltitude="{alt:+.2f}"
   drone-dji:GimbalRollDegree="+0.00" drone-dji:GimbalYawDegree="{yaw:+.2f}"
   drone-dji:GimbalPitchDegree="{pitch:+.2f}" drone-dji:FlightYawDegree="{yaw:+.2f}"/>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end='w'?>"""


def _dms(v: float):
    v = abs(v)
    d = int(v)
    m = int((v - d) * 60)
    return Fraction(d), Fraction(m), Fraction(int(round((v - d - m / 60) * 3600 * 10000)), 10000)


def _exif(lat, lon, f35, when, heading=None, model="Demo camera"):
    exif = Image.Exif()
    exif[0x010F] = "Demo"
    exif[0x0110] = model
    ex = exif.get_ifd(0x8769)
    ex[0xA405] = f35
    ex[0x9003] = when
    gps = exif.get_ifd(0x8825)
    gps[1], gps[2] = ("N" if lat >= 0 else "S"), _dms(lat)
    gps[3], gps[4] = ("E" if lon >= 0 else "W"), _dms(lon)
    if heading is not None:
        gps[16], gps[17] = "T", Fraction(int(round(heading * 100)), 100)
    return exif


def render_ground(caption: str, heading: float, seed: int) -> Image.Image:
    """A stand-in for a phone photo: clearly a placeholder, but with real metadata."""
    rng = np.random.default_rng(seed)
    w, h = 1600, 1200
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    sky = np.array([150, 186, 224], np.float32) + yy * np.array([70, 44, 20], np.float32)
    ground = np.array([92, 122, 64], np.float32) + (yy - 0.55) * np.array([-40, -40, -30], np.float32)
    img = np.where(yy < 0.55, sky, ground) + rng.normal(0, 4, (h, w, 1)).astype(np.float32)
    im = Image.fromarray(np.clip(np.broadcast_to(img, (h, w, 3)), 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.rectangle((w * 0.28, h * 0.30, w * 0.72, h * 0.62), fill=(196, 180, 160))
    d.polygon([(w * 0.25, h * 0.30), (w * 0.5, h * 0.16), (w * 0.75, h * 0.30)], fill=(150, 78, 62))
    d.rectangle((w * 0.46, h * 0.44, w * 0.54, h * 0.62), fill=(70, 56, 48))
    d.rectangle((40, h - 150, w - 40, h - 40), fill=(20, 22, 28))
    d.text((70, h - 132), "DEMO PLACEHOLDER PHOTO", fill=(255, 196, 60), font_size=30)
    d.text((70, h - 90), f"{caption} (camera heading {heading:.0f} deg)", fill=(240, 240, 240), font_size=30)
    return im


ORBIT = [  # hand-flown angled shots circling the house: east, north, yaw (towards the house), pitch, height
    (2, -44, 0, -48, 38), (46, 0, 270, -50, 40), (-2, 44, 180, -48, 38), (-46, 2, 90, -52, 42)]


def write_obliques(folder: Path, aerial: Image.Image | None = None) -> list[Path]:
    """Angled drone photos of the same scene, made by re-projecting the flat ground of the aerial render."""
    from ..camera import PhotoMeta, pixel_to_ground
    from ..geo import to_local
    from .survey import _coeffs

    folder.mkdir(parents=True, exist_ok=True)
    aerial = aerial or render_aerial()
    w, h = IMG_W // 2, IMG_H // 2
    out = []
    for i, (e, n, yaw, pitch, alt) in enumerate(ORBIT, 2):
        lat, lon = offset_latlon(LAT0, LON0, n, e)
        p = folder / f"DJI_{i:04d}.JPG"
        meta = PhotoMeta(p, w, h, lat=lat, lon=lon, rel_alt_m=alt, gimbal_yaw=yaw, gimbal_pitch=pitch, focal_35mm=24)
        corners = [(0, 0), (w, 0), (w, h), (0, h)]
        src = [_to_px(*to_local(LAT0, LON0, *pixel_to_ground(meta, u, v))) for u, v in corners]
        im = aerial.transform((w, h), Image.PERSPECTIVE, tuple(_coeffs(corners, src)), Image.BICUBIC,
                              fillcolor=(96, 128, 66))
        im.save(p, "JPEG", quality=88, exif=_exif(lat, lon, 24, f"2026:11:16 10:1{i}:00", model="L2D-20c"),
                xmp=XMP.format(lat=lat, lon=lon, alt=alt, abs_alt=alt + 80, yaw=yaw, pitch=pitch).encode())
        out.append(p)
    return out


def write_photos(folder: Path) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    lat, lon = offset_latlon(LAT0, LON0, DRONE[1], DRONE[0])
    exif = _exif(lat, lon, 24, "2026:11:16 10:02:00", model="L2D-20c")
    p = folder / "DJI_0001.JPG"
    render_aerial().save(p, "JPEG", quality=90, exif=exif,
                         xmp=XMP.format(lat=lat, lon=lon, alt=ALT, abs_alt=ALT + 80, yaw=YAW, pitch=-90).encode())
    out.append(p)
    out += write_obliques(folder)
    for i, (name, e, n, hdg, cap) in enumerate(GROUND):
        lat, lon = offset_latlon(LAT0, LON0, n, e)
        p = folder / name
        render_ground(cap, hdg, i).save(p, "JPEG", quality=85,
                                        exif=_exif(lat, lon, 26, f"2026:11:16 10:2{i}:00", heading=hdg))
        out.append(p)
    return out


def demo_features() -> list[dict]:
    """The site plan, in coordinates relative to the drone photo (which becomes the project origin)."""
    feats = []
    for i, (kind, subtype, name, pts, extra) in enumerate(FEATURES, 1):
        f = {"id": f"d{i}", "kind": kind, "name": name,
             "pts": [[round(x - DRONE[0], 2), round(y - DRONE[1], 2)] for x, y in pts], **extra}
        if subtype:
            f["subtype"] = subtype
        feats.append(f)
    return feats


def build_demo(out: Path, log=print) -> dict:
    out = Path(out)
    write_photos(out / "source_photos")
    model = run_survey(out / "source_photos", out, name="Demo property (synthetic)", log=log)
    model["features"] = demo_features()
    model["assessor"] = "A. Assessor"
    model["company"] = M.BRAND
    model["client"] = "Example client"
    model["address"] = "1 Example Lane (synthetic demo site)"
    model["reference"] = "DEMO-001"
    model["notes"] = ("Synthetic example. The aerial photo and the ground photos are generated, "
                      "and the plan was drawn from known positions.")
    M.normalise(model)
    M.save(model, out / "site.json")
    return model
