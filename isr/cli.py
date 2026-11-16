"""Command line entry point: ``isr triage``, ``isr plan``, ``isr info``."""

from __future__ import annotations

import argparse
import sys
import time
import webbrowser
from pathlib import Path


def _latlon(s: str) -> tuple[float, float]:
    try:
        a, b = (float(x) for x in s.replace(" ", "").split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("expected 'lat,lon', e.g. 39.7392,-104.9903")
    if not (-90 <= a <= 90 and -180 <= b <= 180):
        raise argparse.ArgumentTypeError("lat must be -90..90 and lon -180..180")
    return a, b


def _size(s: str) -> tuple[float, float]:
    try:
        w, h = (float(x) for x in s.lower().split("x"))
        return w, h
    except ValueError:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT in metres, e.g. 400x250")


def _pct(s: str) -> float:
    v = float(s)
    v = v / 100 if v > 1 else v
    if not 0 <= v < 1:
        raise argparse.ArgumentTypeError("overlap must be 0-99 %")
    return v


# --------------------------------------------------------------------------
# triage
# --------------------------------------------------------------------------


def cmd_triage(a) -> int:
    from .triage.export import write_all
    from .triage.pipeline import TriageConfig, run_triage
    from .triage.report import write_report
    from .triage.detectors import parse_color

    src = Path(a.source).expanduser()
    if not src.exists():
        print(f"error: {src} does not exist", file=sys.stderr)
        return 2
    for c in a.color:
        try:
            parse_color(c)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    out = Path(a.output).expanduser() if a.output else Path.cwd() / f"triage_{time.strftime('%Y%m%d_%H%M%S')}"
    cfg = TriageConfig(
        colors=a.color,
        anomaly=not a.no_anomaly,
        sensitivity=a.sensitivity,
        ai=a.ai,
        ai_model=a.ai_model,
        ai_conf=a.ai_conf,
        work_px=a.work_px,
        min_area_m2=a.min_area,
        max_area_m2=a.max_area,
        ground_offset_m=a.ground_offset,
        min_score=a.min_score,
        max_per_photo=a.max_per_photo,
        workers=a.workers,
    )
    if not cfg.anomaly and not cfg.colors and not cfg.ai:
        print("error: all detectors disabled; add --color, --ai, or drop --no-anomaly", file=sys.stderr)
        return 2
    data = run_triage(src, out, cfg)
    write_all(data, out)
    report = write_report(data, out)
    s = data["summary"]
    print()
    print(f"Done in {s['seconds']} s: {s['photos']} photos, {s['photos_geolocatable']} geolocated, "
          f"{s['detections']} candidate(s) in {s['photos_with_hits']} photo(s).")
    if s["errors"]:
        print(f"  {s['errors']} photo(s) could not be fully processed (see results.json)")
    if s["photos"] and not s["photos_with_gps"]:
        print("  ! no GPS found in these photos; candidates are not geolocated")
    print(f"Review: {report}")
    print(f"Exports: {out}/detections.(csv|geojson|kml|gpx)")
    if a.open:
        webbrowser.open(report.resolve().as_uri())
    return 0


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------


def _aoi(a):
    from .planner.mission import load_aoi, rectangle

    if a.aoi:
        return load_aoi(a.aoi)
    if a.center and a.size:
        return rectangle(a.center, a.size[0], a.size[1], a.rotation)
    raise SystemExit("error: give --aoi FILE or --center LAT,LON --size WxH")


def _emit(m, a) -> int:
    from .planner.mission import export_all, split

    out = Path(a.output).expanduser()
    tmpl = Path(a.template).expanduser() if a.template else None
    parts = split(m, a.max_waypoints)
    for p in parts:
        print(p.summary())
        files = export_all(p, out, tmpl)
        print(f"  -> {files[0]}  (+ .kml .geojson .gpx .txt)\n")
    if len(parts) > 1:
        print(f"Split into {len(parts)} missions of <= {a.max_waypoints} waypoints each.")
    if not tmpl:
        print("Note: no --template given. For DJI Fly, save a small waypoint mission on your RC first and pass its\n"
              "      .kmz with --template so the drone/payload IDs match your aircraft. See README.")
    return 0


def cmd_plan(a) -> int:
    from .planner import mission as M
    from .planner.patterns import CAMERAS

    if a.camera not in CAMERAS:
        raise SystemExit(f"unknown camera {a.camera}; choose from {', '.join(CAMERAS)}")
    if a.pattern == "calc":
        cam = CAMERAS[a.camera]
        print(f"{cam.name}")
        print(f"{'alt m':>6} {'GSD cm/px':>10} {'footprint m':>14} {'person px*':>11}")
        for alt in a.alts or [30, 40, 50, 60, 80, 100, 120]:
            fw, fh = cam.footprint_m(alt)
            g = cam.gsd_cm(alt)
            print(f"{alt:>6.0f} {g:>10.2f} {fw:>6.0f} x {fh:<5.0f} {50 / g:>11.0f}")
        print("* approx. pixels across a person's shoulders (~50 cm) seen from directly above")
        return 0

    if a.alt and a.alt > M.MAX_ALT_M and not a.allow_high:
        raise SystemExit(f"error: --alt {a.alt} exceeds {M.MAX_ALT_M:.0f} m; pass --allow-high if you are authorised")
    common = dict(speed_ms=a.speed, camera=a.camera)
    if a.pattern == "grid":
        m = M.plan_grid(_aoi(a), alt_m=a.alt, gsd_cm=a.gsd, front=a.front or 0.75, side=a.side or 0.70,
                        angle_deg=a.angle, crosshatch=a.crosshatch, name=a.name or "grid", **common)
    elif a.pattern == "search":
        m = M.plan_search_grid(_aoi(a), alt_m=a.alt or 60, front=a.front or 0.30, side=a.side or 0.20,
                               angle_deg=a.angle, name=a.name or "parallel-track", **common)
    elif a.pattern == "square":
        m = M.plan_expanding_square(a.datum, a.radius, alt_m=a.alt or 60, front=a.front or 0.30,
                                    side=a.side or 0.20, heading_deg=a.heading, name=a.name or "expanding-square",
                                    **common)
    elif a.pattern == "sector":
        m = M.plan_sector(a.datum, a.radius, alt_m=a.alt or 60, front=a.front or 0.30, heading_deg=a.heading,
                          name=a.name or "sector", **common)
    else:  # pragma: no cover
        raise SystemExit("unknown pattern")
    if m.alt_m > M.MAX_ALT_M and not a.allow_high:
        raise SystemExit(f"error: computed altitude {m.alt_m:.0f} m exceeds {M.MAX_ALT_M:.0f} m; "
                         "use a coarser --gsd, set --alt, or pass --allow-high if authorised")
    return _emit(m, a)


# --------------------------------------------------------------------------
# info
# --------------------------------------------------------------------------


def cmd_info(a) -> int:
    from .camera import photo_footprint, read_photo_meta

    for p in a.photos:
        m = read_photo_meta(p)
        print(f"{p}")
        print(f"  camera: {m.camera_model}  size: {m.width}x{m.height}  focal35: {m.focal_35mm}")
        print(f"  position: {m.lat}, {m.lon}  rel alt: {m.rel_alt_m} m  abs alt: {m.abs_alt_m} m")
        print(f"  gimbal yaw/pitch/roll: {m.gimbal_yaw} / {m.gimbal_pitch} / {m.gimbal_roll}")
        g = m.gsd_m()
        print(f"  GSD: {g * 100:.2f} cm/px" if g else "  GSD: unknown")
        fp = photo_footprint(m)
        print(f"  footprint: {[(round(la, 6), round(lo, 6)) for la, lo in fp]}" if fp else "  footprint: n/a")
        if not m.raw_xmp:
            print("  ! no DJI XMP metadata found (not a DJI photo, or metadata stripped)")
    return 0


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="isr", description="Drone ISR tools: photo triage and mission planning.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("triage", help="scan a folder / SD card for possible people and objects",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    t.add_argument("source", help="folder of photos, e.g. the SD card's DCIM folder (searched recursively)")
    t.add_argument("-o", "--output", help="output folder (default: ./triage_<timestamp>; never written to the card)")
    t.add_argument("-c", "--color", action="append", default=[], metavar="COLOR",
                   help="known clothing/gear colour to look for; repeatable. Names (red, orange, yellow, blue, "
                        "purple, pink, cyan, white, green) or #rrggbb")
    t.add_argument("--no-anomaly", action="store_true", help="disable the colour-anomaly detector")
    t.add_argument("--sensitivity", type=float, default=1.0, help="anomaly sensitivity; >1 flags more, <1 fewer")
    t.add_argument("--ai", action="store_true", help="also run the YOLO person/object detector (needs ultralytics)")
    t.add_argument("--ai-model", default="yolov8n.pt", help="YOLO weights file (e.g. a SAR-trained model)")
    t.add_argument("--ai-conf", type=float, default=0.20, help="AI confidence threshold")
    t.add_argument("--min-score", type=float, default=0.35, help="drop candidates below this score")
    t.add_argument("--max-per-photo", type=int, default=8, help="keep at most N candidates per photo")
    t.add_argument("--min-area", type=float, default=0.04, help="smallest object to flag, m²")
    t.add_argument("--max-area", type=float, default=12.0, help="largest object to flag, m²")
    t.add_argument("--ground-offset", type=float, default=0.0,
                   help="search ground height minus take-off height, metres (e.g. -30 if the valley floor is "
                        "30 m below where you launched)")
    t.add_argument("--work-px", type=int, default=2000, help="analysis resolution (long side, px)")
    t.add_argument("--workers", type=int, default=0, help="parallel processes (0 = CPU count - 1)")
    t.add_argument("--open", action="store_true", help="open the report in a browser when done")
    t.set_defaults(func=cmd_triage)

    p = sub.add_parser("plan", help="generate mapping grids and search patterns")
    psub = p.add_subparsers(dest="pattern", required=True)

    def common(sp, area=False, datum=False):
        sp.add_argument("--camera", default="mavic3", help="camera profile (mavic3, mavic3pro-70mm, m3e, m3t-wide)")
        sp.add_argument("--alt", type=float, help="altitude above take-off, m")
        sp.add_argument("--front", type=_pct, help="front (along-track) overlap, %%")
        sp.add_argument("--side", type=_pct, help="side (between tracks) overlap, %%")
        sp.add_argument("--speed", type=float, default=8.0, help="flight speed m/s (default 8)")
        sp.add_argument("--name", help="mission name / file stem")
        sp.add_argument("-o", "--output", default="missions", help="output folder (default ./missions)")
        sp.add_argument("--template", help="a DJI Fly waypoint .kmz from your controller to copy drone IDs from")
        sp.add_argument("--max-waypoints", type=int, default=200, help="split missions larger than this")
        sp.add_argument("--allow-high", action="store_true", help="permit altitudes above 120 m")
        if area:
            sp.add_argument("--aoi", help="area polygon: .kml, .geojson or .gpx (e.g. a CalTopo export)")
            sp.add_argument("--center", type=_latlon, help="rectangle centre 'lat,lon' (instead of --aoi)")
            sp.add_argument("--size", type=_size, help="rectangle size WxH metres, e.g. 400x250")
            sp.add_argument("--rotation", type=float, default=0.0, help="rectangle rotation, deg clockwise")
            sp.add_argument("--angle", type=float, help="flight line direction, deg from north (default: auto)")
        if datum:
            sp.add_argument("--datum", type=_latlon, required=True, help="last known point 'lat,lon'")
            sp.add_argument("--radius", type=float, required=True, help="search radius, m")
            sp.add_argument("--heading", type=float, default=0.0, help="first leg heading, deg (e.g. into wind)")
        sp.set_defaults(func=cmd_plan)

    g = psub.add_parser("grid", help="photogrammetry mapping grid")
    common(g, area=True)
    g.add_argument("--gsd", type=float, help="target ground resolution cm/px (sets altitude)")
    g.add_argument("--crosshatch", action="store_true", help="fly a second grid at 90 deg (for 3D models)")
    common(psub.add_parser("search", help="SAR parallel-track (lawnmower) photo search over an area"), area=True)
    common(psub.add_parser("square", help="SAR expanding square around a datum"), datum=True)
    common(psub.add_parser("sector", help="SAR sector search around a datum"), datum=True)
    c = psub.add_parser("calc", help="altitude / GSD / footprint table")
    c.add_argument("--camera", default="mavic3")
    c.add_argument("alts", nargs="*", type=float, help="altitudes to tabulate")
    c.set_defaults(func=cmd_plan)

    i = sub.add_parser("info", help="show the metadata the tools read from a photo")
    i.add_argument("photos", nargs="+")
    i.set_defaults(func=cmd_info)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a) or 0


if __name__ == "__main__":
    sys.exit(main())
