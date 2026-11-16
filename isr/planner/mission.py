"""Build photo missions (mapping grids and SAR search patterns) and export them."""

from __future__ import annotations

import json
import math
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

from ..geo import from_local, polygon_area_m2, to_local
from .patterns import CAMERAS, CameraSpec, densify, expanding_square, lawnmower, path_length, sector_search

# Usable flight time per battery we plan against (Mavic 3 is rated ~40+ min in
# still air; leave margin for wind, climb, transit and a safe RTH reserve).
PLAN_MINUTES_PER_BATTERY = 25.0
STOP_SECONDS_PER_PHOTO = 2.5
MAX_ALT_M = 120.0  # FAA Part 107 / EU open category; override with --allow-high


@dataclass
class Waypoint:
    lat: float
    lon: float
    alt_m: float
    photo: bool = True
    gimbal_pitch: float = -90.0


@dataclass
class Mission:
    name: str
    kind: str
    camera: CameraSpec
    alt_m: float
    speed_ms: float
    waypoints: list[Waypoint]
    aoi: list[tuple[float, float]] | None = None  # lat, lon
    notes: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def summary(self) -> str:
        s = self.stats
        lines = [
            f"Mission: {self.name} ({self.kind})",
            f"  Camera: {self.camera.name}",
            f"  Altitude: {self.alt_m:.0f} m above take-off   GSD: {self.camera.gsd_cm(self.alt_m):.2f} cm/px",
            f"  Photo footprint: {s['footprint_w_m']:.0f} x {s['footprint_h_m']:.0f} m",
            f"  Track spacing: {s['track_spacing_m']:.1f} m   Photo spacing: {s['photo_spacing_m']:.1f} m",
            f"  Waypoints/photos: {len(self.waypoints)}   Path: {s['path_m'] / 1000:.2f} km",
            f"  Est. flight time: {s['minutes']:.1f} min (~{s['batteries']} battery(s) at "
            f"{PLAN_MINUTES_PER_BATTERY:.0f} min planned each)",
        ]
        if s.get("area_ha"):
            lines.insert(4, f"  Area: {s['area_ha']:.2f} ha")
        lines += [f"  ! {n}" for n in self.notes]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------


def _spacings(cam: CameraSpec, alt: float, front: float, side: float) -> tuple[float, float, float, float]:
    fw, fh = cam.footprint_m(alt)
    return fw, fh, fw * (1 - side), fh * (1 - front)


def _finish(m: Mission, path_en: list[tuple[float, float]], origin, fw, fh, track, photo, area_m2=None) -> Mission:
    length = path_length(path_en)
    secs = length / m.speed_ms + len(m.waypoints) * STOP_SECONDS_PER_PHOTO + 2 * m.alt_m / 4.0
    m.stats = {
        "footprint_w_m": fw,
        "footprint_h_m": fh,
        "track_spacing_m": track,
        "photo_spacing_m": photo,
        "path_m": length,
        "minutes": secs / 60,
        "batteries": max(1, math.ceil(secs / 60 / PLAN_MINUTES_PER_BATTERY)),
        "area_ha": area_m2 / 10_000 if area_m2 else None,
        "origin": origin,
    }
    if m.alt_m > MAX_ALT_M:
        m.notes.append(f"altitude {m.alt_m:.0f} m exceeds {MAX_ALT_M:.0f} m AGL limit in most jurisdictions")
    blur_px = m.speed_ms * (1 / 1000) / (m.camera.gsd_cm(m.alt_m) / 100)
    if blur_px > 0.7:
        m.notes.append(f"at {m.speed_ms} m/s, 1/1000 s shutter blurs ~{blur_px:.1f} px if shooting while moving")
    return m


def _wps(path_en, origin, alt, pitch=-90.0) -> list[Waypoint]:
    out = []
    for e, n in path_en:
        lat, lon = from_local(origin[0], origin[1], e, n)
        out.append(Waypoint(lat, lon, alt, True, pitch))
    return out


def plan_grid(
    aoi: list[tuple[float, float]],
    alt_m: float | None = None,
    gsd_cm: float | None = None,
    front: float = 0.75,
    side: float = 0.70,
    angle_deg: float | None = None,
    crosshatch: bool = False,
    speed_ms: float = 8.0,
    camera: str = "mavic3",
    name: str = "grid",
    kind: str = "mapping grid",
    inset_edges: bool = False,
) -> Mission:
    cam = CAMERAS[camera]
    if alt_m is None:
        alt_m = cam.alt_for_gsd(gsd_cm) if gsd_cm else 80.0
    origin = (sum(p[0] for p in aoi) / len(aoi), sum(p[1] for p in aoi) / len(aoi))
    poly = [to_local(origin[0], origin[1], la, lo) for la, lo in aoi]
    if angle_deg is None:
        angle_deg = _best_angle(poly)
    fw, fh, track, photo = _spacings(cam, alt_m, front, side)
    angles = [angle_deg, angle_deg + 90] if crosshatch else [angle_deg]
    path: list[tuple[float, float]] = []
    for a in angles:
        for seg in lawnmower(poly, track, a, overshoot=photo / 2, swath=fw if inset_edges else 0.0):
            path += densify(seg, photo)
    m = Mission(name, kind + (" (crosshatch)" if crosshatch else ""), cam, alt_m, speed_ms, _wps(path, origin, alt_m),
                aoi=list(aoi))
    if crosshatch:
        m.notes.append("crosshatch: consider a -60..-70 deg gimbal for better 3D models of buildings/terrain")
        for w in m.waypoints:
            w.gimbal_pitch = -90.0
    return _finish(m, path, origin, fw, fh, track, photo, polygon_area_m2(poly))


def plan_search_grid(aoi, alt_m=60.0, front=0.30, side=0.20, angle_deg=None, speed_ms=8.0, camera="mavic3",
                     name="parallel-track") -> Mission:
    """Parallel-track (creeping line) photo search over an area."""
    return plan_grid(aoi, alt_m=alt_m, front=front, side=side, angle_deg=angle_deg, speed_ms=speed_ms,
                     camera=camera, name=name, kind="SAR parallel track", inset_edges=True)


def plan_expanding_square(datum, radius_m, alt_m=60.0, front=0.30, side=0.20, heading_deg=0.0, speed_ms=8.0,
                          camera="mavic3", name="expanding-square") -> Mission:
    cam = CAMERAS[camera]
    fw, fh, track, photo = _spacings(cam, alt_m, front, side)
    path = densify(expanding_square(track, radius_m, heading_deg), photo)
    m = Mission(name, "SAR expanding square", cam, alt_m, speed_ms, _wps(path, datum, alt_m))
    return _finish(m, path, datum, fw, fh, track, photo, (2 * radius_m) ** 2)


def plan_sector(datum, radius_m, alt_m=60.0, front=0.30, heading_deg=0.0, speed_ms=8.0, camera="mavic3",
                second_pass=True, name="sector") -> Mission:
    cam = CAMERAS[camera]
    fw, fh, track, photo = _spacings(cam, alt_m, front, 0.0)
    pts = sector_search(radius_m, heading_deg, second_pass)
    # drop repeated datum visits so we don't take duplicate photos there
    path = densify(pts, photo)
    dedup = [path[0]]
    for p in path[1:]:
        if math.hypot(p[0] - dedup[-1][0], p[1] - dedup[-1][1]) > 0.5:
            dedup.append(p)
    m = Mission(name, "SAR sector search", cam, alt_m, speed_ms, _wps(dedup, datum, alt_m))
    return _finish(m, dedup, datum, fw, fh, radius_m * math.pi / 6, photo, math.pi * radius_m**2)


def _best_angle(poly) -> float:
    """Fly parallel to the polygon's longest edge (fewest turns)."""
    best, ang = -1.0, 0.0
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        d = math.hypot(x2 - x1, y2 - y1)
        if d > best:
            best, ang = d, math.degrees(math.atan2(x2 - x1, y2 - y1)) % 180
    return ang


def split(m: Mission, max_waypoints: int) -> list[Mission]:
    if len(m.waypoints) <= max_waypoints:
        return [m]
    parts = []
    n = math.ceil(len(m.waypoints) / max_waypoints)
    size = math.ceil(len(m.waypoints) / n)
    for i in range(n):
        wps = m.waypoints[i * size : (i + 1) * size]
        pm = Mission(f"{m.name}-part{i + 1}of{n}", m.kind, m.camera, m.alt_m, m.speed_ms, wps, m.aoi, list(m.notes))
        origin = m.stats["origin"]
        path = [to_local(origin[0], origin[1], w.lat, w.lon) for w in wps]
        _finish(pm, path, origin, m.stats["footprint_w_m"], m.stats["footprint_h_m"], m.stats["track_spacing_m"],
                m.stats["photo_spacing_m"])
        pm.notes = list(dict.fromkeys(pm.notes))
        parts.append(pm)
    return parts


# --------------------------------------------------------------------------
# AOI input
# --------------------------------------------------------------------------


def rectangle(center: tuple[float, float], width_m: float, height_m: float, rotation_deg: float = 0.0):
    r = math.radians(rotation_deg)
    pts = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        x, y = sx * width_m / 2, sy * height_m / 2
        e = x * math.cos(r) + y * math.sin(r)
        n = -x * math.sin(r) + y * math.cos(r)
        pts.append(from_local(center[0], center[1], e, n))
    return pts


def load_aoi(path: str | Path) -> list[tuple[float, float]]:
    """First polygon in a KML / GeoJSON / GPX file (e.g. exported from CalTopo or Google Earth)."""
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    suf = p.suffix.lower()
    pts: list[tuple[float, float]] = []
    if suf in (".geojson", ".json"):
        gj = json.loads(text)
        feats = gj.get("features", [gj])
        for f in feats:
            g = f.get("geometry", f)
            if g.get("type") == "Polygon":
                pts = [(c[1], c[0]) for c in g["coordinates"][0]]
            elif g.get("type") == "MultiPolygon":
                pts = [(c[1], c[0]) for c in g["coordinates"][0][0]]
            elif g.get("type") == "LineString":
                pts = [(c[1], c[0]) for c in g["coordinates"]]
            if pts:
                break
    elif suf == ".kml":
        m = re.search(r"<Polygon>.*?<coordinates>(.*?)</coordinates>", text, re.S) or re.search(
            r"<coordinates>(.*?)</coordinates>", text, re.S)
        if m:
            for tok in m.group(1).split():
                c = tok.split(",")
                pts.append((float(c[1]), float(c[0])))
    elif suf == ".gpx":
        root = ET.fromstring(text)
        for el in root.iter():
            if el.tag.endswith(("trkpt", "rtept", "wpt")):
                pts.append((float(el.get("lat")), float(el.get("lon"))))
    else:
        raise ValueError(f"unsupported AOI file type {suf}; use .kml, .geojson or .gpx")
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        raise ValueError(f"no polygon with 3+ points found in {path}")
    return pts


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


def write_geojson(m: Mission, path: Path) -> None:
    feats = [{
        "type": "Feature",
        "properties": {"title": m.name, "kind": m.kind, "stroke": "#d1147a"},
        "geometry": {"type": "LineString", "coordinates": [[w.lon, w.lat] for w in m.waypoints]},
    }]
    if m.aoi:
        ring = [[lo, la] for la, lo in m.aoi] + [[m.aoi[0][1], m.aoi[0][0]]]
        feats.append({"type": "Feature", "properties": {"title": f"{m.name} AOI", "fill-opacity": 0.1},
                      "geometry": {"type": "Polygon", "coordinates": [ring]}})
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))


def write_kml(m: Mission, path: Path) -> None:
    coords = " ".join(f"{w.lon:.7f},{w.lat:.7f},{w.alt_m:.1f}" for w in m.waypoints)
    parts = ['<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
             f"<name>{escape(m.name)}</name>",
             f"<description>{escape(m.summary())}</description>",
             '<Style id="r"><LineStyle><color>ff7a14d1</color><width>3</width></LineStyle></Style>',
             f"<Placemark><name>route</name><styleUrl>#r</styleUrl><LineString><altitudeMode>relativeToGround"
             f"</altitudeMode><coordinates>{coords}</coordinates></LineString></Placemark>"]
    if m.aoi:
        ring = " ".join(f"{lo:.7f},{la:.7f},0" for la, lo in m.aoi + [m.aoi[0]])
        parts.append(f"<Placemark><name>AOI</name><Polygon><outerBoundaryIs><LinearRing><coordinates>{ring}"
                     "</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>")
    parts.append("</Document></kml>")
    path.write_text("".join(parts))


def write_gpx(m: Mission, path: Path) -> None:
    pts = "".join(f'<rtept lat="{w.lat:.7f}" lon="{w.lon:.7f}"><ele>{w.alt_m:.1f}</ele></rtept>' for w in m.waypoints)
    path.write_text('<?xml version="1.0" encoding="UTF-8"?><gpx version="1.1" creator="isr-tools" '
                    f'xmlns="http://www.topografix.com/GPX/1/1"><rte><name>{escape(m.name)}</name>{pts}</rte></gpx>')


# ---- DJI WPML (.kmz) ------------------------------------------------------

WPML_NS = "http://www.dji.com/wpmz/1.0.2"

# Default mission config when no DJI Fly template is given. The drone enum
# values below are DJI's published values for the Mavic 3 Enterprise series;
# DJI Fly on consumer Mavic 3 models writes its own values, which is why
# passing --template (a mission saved by your own DJI Fly) is strongly advised.
DEFAULT_MISSION_CONFIG = """<wpml:missionConfig>
      <wpml:flyToWaylineMode>safely</wpml:flyToWaylineMode>
      <wpml:finishAction>goHome</wpml:finishAction>
      <wpml:exitOnRCLost>executeLostAction</wpml:exitOnRCLost>
      <wpml:executeRCLostAction>goBack</wpml:executeRCLostAction>
      <wpml:takeOffSecurityHeight>20</wpml:takeOffSecurityHeight>
      <wpml:globalTransitionalSpeed>{speed}</wpml:globalTransitionalSpeed>
      <wpml:droneInfo>
        <wpml:droneEnumValue>77</wpml:droneEnumValue>
        <wpml:droneSubEnumValue>0</wpml:droneSubEnumValue>
      </wpml:droneInfo>
    </wpml:missionConfig>"""


def _action_group(idx: int, pitch: float, photo: bool) -> str:
    actions = [f"""<wpml:action>
          <wpml:actionId>0</wpml:actionId>
          <wpml:actionActuatorFunc>gimbalRotate</wpml:actionActuatorFunc>
          <wpml:actionActuatorFuncParam>
            <wpml:gimbalHeadingYawBase>aircraft</wpml:gimbalHeadingYawBase>
            <wpml:gimbalRotateMode>absoluteAngle</wpml:gimbalRotateMode>
            <wpml:gimbalPitchRotateEnable>1</wpml:gimbalPitchRotateEnable>
            <wpml:gimbalPitchRotateAngle>{pitch:g}</wpml:gimbalPitchRotateAngle>
            <wpml:gimbalRollRotateEnable>0</wpml:gimbalRollRotateEnable>
            <wpml:gimbalRollRotateAngle>0</wpml:gimbalRollRotateAngle>
            <wpml:gimbalYawRotateEnable>0</wpml:gimbalYawRotateEnable>
            <wpml:gimbalYawRotateAngle>0</wpml:gimbalYawRotateAngle>
            <wpml:gimbalRotateTimeEnable>0</wpml:gimbalRotateTimeEnable>
            <wpml:gimbalRotateTime>0</wpml:gimbalRotateTime>
            <wpml:payloadPositionIndex>0</wpml:payloadPositionIndex>
          </wpml:actionActuatorFuncParam>
        </wpml:action>"""]
    if photo:
        actions.append("""<wpml:action>
          <wpml:actionId>1</wpml:actionId>
          <wpml:actionActuatorFunc>takePhoto</wpml:actionActuatorFunc>
          <wpml:actionActuatorFuncParam>
            <wpml:payloadPositionIndex>0</wpml:payloadPositionIndex>
            <wpml:useGlobalPayloadLensIndex>0</wpml:useGlobalPayloadLensIndex>
          </wpml:actionActuatorFuncParam>
        </wpml:action>""")
    return f"""<wpml:actionGroup>
        <wpml:actionGroupId>{idx}</wpml:actionGroupId>
        <wpml:actionGroupStartIndex>{idx}</wpml:actionGroupStartIndex>
        <wpml:actionGroupEndIndex>{idx}</wpml:actionGroupEndIndex>
        <wpml:actionGroupMode>sequence</wpml:actionGroupMode>
        <wpml:actionTrigger><wpml:actionTriggerType>reachPoint</wpml:actionTriggerType></wpml:actionTrigger>
        {"".join(actions)}
      </wpml:actionGroup>"""


_TURN = """<wpml:waypointTurnParam>
        <wpml:waypointTurnMode>toPointAndStopWithDiscontinuityCurvature</wpml:waypointTurnMode>
        <wpml:waypointTurnDampingDist>0</wpml:waypointTurnDampingDist>
      </wpml:waypointTurnParam>"""
_HEADING = """<wpml:waypointHeadingParam>
        <wpml:waypointHeadingMode>followWayline</wpml:waypointHeadingMode>
        <wpml:waypointHeadingAngle>0</wpml:waypointHeadingAngle>
        <wpml:waypointPoiPoint>0.000000,0.000000,0.000000</wpml:waypointPoiPoint>
        <wpml:waypointHeadingPathMode>followBadArc</wpml:waypointHeadingPathMode>
      </wpml:waypointHeadingParam>"""


def _read_template(template: Path) -> tuple[str, str]:
    """Pull the WPML namespace and missionConfig (drone/payload info, safety
    settings) from a mission .kmz saved by DJI Fly on your own controller."""
    with zipfile.ZipFile(template) as z:
        names = z.namelist()
        tk = next((n for n in names if n.endswith("template.kml")), None)
        wl = next((n for n in names if n.endswith("waylines.wpml")), None)
        src = z.read(tk or wl).decode("utf-8", "replace")
    ns = re.search(r'xmlns:wpml="([^"]+)"', src)
    mc = re.search(r"<wpml:missionConfig>.*?</wpml:missionConfig>", src, re.S)
    if not mc:
        raise ValueError(f"{template} does not look like a DJI waypoint mission (no missionConfig)")
    return (ns.group(1) if ns else WPML_NS), mc.group(0)


def write_wpml_kmz(m: Mission, path: Path, template: Path | None = None) -> None:
    ns, mission_cfg = WPML_NS, DEFAULT_MISSION_CONFIG.format(speed=m.speed_ms)
    if template:
        ns, mission_cfg = _read_template(template)
        # keep the template's safety settings but use our transit speed
        mission_cfg = re.sub(r"<wpml:globalTransitionalSpeed>[^<]*<", f"<wpml:globalTransitionalSpeed>{m.speed_ms}<",
                             mission_cfg)
    now = int(time.time() * 1000)

    def placemarks(full: bool) -> str:
        out = []
        for i, w in enumerate(m.waypoints):
            if full:  # template.kml
                body = f"""<wpml:index>{i}</wpml:index>
      <wpml:ellipsoidHeight>{w.alt_m:g}</wpml:ellipsoidHeight>
      <wpml:height>{w.alt_m:g}</wpml:height>
      <wpml:useGlobalHeight>1</wpml:useGlobalHeight>
      <wpml:useGlobalSpeed>1</wpml:useGlobalSpeed>
      <wpml:useGlobalHeadingParam>1</wpml:useGlobalHeadingParam>
      <wpml:useGlobalTurnParam>1</wpml:useGlobalTurnParam>
      <wpml:useStraightLine>1</wpml:useStraightLine>
      <wpml:gimbalPitchAngle>{w.gimbal_pitch:g}</wpml:gimbalPitchAngle>"""
            else:  # waylines.wpml
                body = f"""<wpml:index>{i}</wpml:index>
      <wpml:executeHeight>{w.alt_m:g}</wpml:executeHeight>
      <wpml:waypointSpeed>{m.speed_ms:g}</wpml:waypointSpeed>
      {_HEADING}
      {_TURN}
      <wpml:useStraightLine>1</wpml:useStraightLine>"""
            out.append(f"""    <Placemark>
      <Point><coordinates>{w.lon:.8f},{w.lat:.8f}</coordinates></Point>
      {body}
      {_action_group(i, w.gimbal_pitch, w.photo)}
    </Placemark>""")
        return "\n".join(out)

    template_kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:wpml="{ns}">
  <Document>
    <wpml:author>isr-tools</wpml:author>
    <wpml:createTime>{now}</wpml:createTime>
    <wpml:updateTime>{now}</wpml:updateTime>
    {mission_cfg}
    <Folder>
      <wpml:templateType>waypoint</wpml:templateType>
      <wpml:templateId>0</wpml:templateId>
      <wpml:waylineCoordinateSysParam>
        <wpml:coordinateMode>WGS84</wpml:coordinateMode>
        <wpml:heightMode>relativeToStartPoint</wpml:heightMode>
      </wpml:waylineCoordinateSysParam>
      <wpml:autoFlightSpeed>{m.speed_ms:g}</wpml:autoFlightSpeed>
      <wpml:globalHeight>{m.alt_m:g}</wpml:globalHeight>
      <wpml:caliFlightEnable>0</wpml:caliFlightEnable>
      <wpml:gimbalPitchMode>usePointSetting</wpml:gimbalPitchMode>
      <wpml:globalWaypointHeadingParam>
        <wpml:waypointHeadingMode>followWayline</wpml:waypointHeadingMode>
        <wpml:waypointHeadingAngle>0</wpml:waypointHeadingAngle>
        <wpml:waypointPoiPoint>0.000000,0.000000,0.000000</wpml:waypointPoiPoint>
        <wpml:waypointHeadingPathMode>followBadArc</wpml:waypointHeadingPathMode>
      </wpml:globalWaypointHeadingParam>
      <wpml:globalWaypointTurnMode>toPointAndStopWithDiscontinuityCurvature</wpml:globalWaypointTurnMode>
      <wpml:globalUseStraightLine>1</wpml:globalUseStraightLine>
{placemarks(True)}
    </Folder>
  </Document>
</kml>
"""
    waylines = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:wpml="{ns}">
  <Document>
    {mission_cfg}
    <Folder>
      <wpml:templateId>0</wpml:templateId>
      <wpml:executeHeightMode>relativeToStartPoint</wpml:executeHeightMode>
      <wpml:waylineId>0</wpml:waylineId>
      <wpml:distance>{m.stats.get('path_m', 0):.1f}</wpml:distance>
      <wpml:duration>{m.stats.get('minutes', 0) * 60:.1f}</wpml:duration>
      <wpml:autoFlightSpeed>{m.speed_ms:g}</wpml:autoFlightSpeed>
{placemarks(False)}
    </Folder>
  </Document>
</kml>
"""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("wpmz/template.kml", template_kml)
        z.writestr("wpmz/waylines.wpml", waylines)


def export_all(m: Mission, out_dir: Path, template: Path | None = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / m.name
    paths = [base.with_suffix(".kmz"), base.with_suffix(".kml"), base.with_suffix(".geojson"),
             base.with_suffix(".gpx"), base.with_suffix(".txt")]
    write_wpml_kmz(m, paths[0], template)
    write_kml(m, paths[1])
    write_geojson(m, paths[2])
    write_gpx(m, paths[3])
    paths[4].write_text(m.summary() + "\n")
    return paths
