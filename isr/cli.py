"""Command line entry point: ``isr triage | plan | track | studio | enhance | site | model | dataset | info``."""

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
    ref = None
    if a.reference:
        from dataclasses import asdict

        from .triage.reference import TARGET_SIZES, build_profile

        box = tuple(int(float(v)) for v in a.ref_box.split(",")) if a.ref_box else None
        out.mkdir(parents=True, exist_ok=True)
        prof = build_profile(Path(a.reference).expanduser(), a.target, box, save_crop=out / "reference.jpg")
        ref = asdict(prof)
        sw = ", ".join(f"{c['name']} {c['hex'] if c['hex'] != 'white' else ''} ({c['share']:.0%})".replace("  ", " ")
                       for c in prof.colors) or "none"
        print(f"Reference {a.target}: distinctive colours {sw}")
        for n in prof.notes:
            print(f"  ! {n}")
        a.color = list(dict.fromkeys(a.color + prof.color_specs()))
        lo, hi = TARGET_SIZES[a.target]
        if a.min_area == 0.04 and a.max_area == 12.0:  # user didn't override the size window
            a.min_area, a.max_area = lo, hi
    cfg = TriageConfig(
        colors=a.color,
        anomaly=not a.no_anomaly,
        sensitivity=a.sensitivity,
        ai=a.ai,
        ai_model=a.ai_model,
        ai_conf=a.ai_conf,
        ai_tile=a.ai_tile,
        work_px=a.work_px,
        min_area_m2=a.min_area,
        max_area_m2=a.max_area,
        ground_offset_m=a.ground_offset,
        min_score=a.min_score,
        max_per_photo=a.max_per_photo,
        workers=a.workers,
        reference=ref,
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
# studio / enhance
# --------------------------------------------------------------------------


def cmd_studio(a) -> int:
    from .studio.gallery import write_gallery
    from .studio.pipeline import StudioConfig, run_studio

    src = Path(a.source).expanduser()
    if not src.exists():
        print(f"error: {src} does not exist", file=sys.stderr)
        return 2
    out = Path(a.output).expanduser() if a.output else Path.cwd() / f"studio_{time.strftime('%Y%m%d_%H%M%S')}"
    cfg = StudioConfig(top=a.top, enhance=not a.no_enhance, strength=a.strength, straighten=not a.no_straighten,
                       dup_bits=a.similarity, workers=a.workers)
    data = run_studio(src, out, cfg)
    gallery = write_gallery(data, out)
    s = data["summary"]
    print()
    print(f"Done in {s['seconds']} s: {s['photos']} photos, {s['groups']} distinct moments, "
          f"top {s['top']} suggested, {s['edited']} auto-edited.")
    if s["errors"]:
        print(f"  {s['errors']} photo(s) could not be read (see studio.json)")
    print(f"Gallery: {gallery}")
    if s["edited"]:
        print(f"Edits:   {out / 'edited'}")
    if a.open:
        webbrowser.open(gallery.resolve().as_uri())
    return 0


def _gather_inputs(a) -> list[Path]:
    from .triage.pipeline import IMAGE_EXTS, find_photos

    files: list[Path] = []
    for x in a.inputs:
        p = Path(x).expanduser()
        if p.is_dir():
            files += find_photos(p)
        elif p.suffix.lower() in IMAGE_EXTS and p.exists():
            files.append(p)
        else:
            print(f"  ! skipping {p} (not a JPEG or folder)", file=sys.stderr)
    if a.list:
        for line in Path(a.list).expanduser().read_text(encoding="utf-8").splitlines():
            line = line.strip().strip('"')
            if line and not line.startswith("#"):
                files.append(Path(line))
    seen, out = set(), []
    for f in files:
        if f.resolve() not in seen:
            seen.add(f.resolve())
            out.append(f)
    return out


def cmd_enhance(a) -> int:
    from .studio.enhance import enhance_file

    files = _gather_inputs(a)
    if not files:
        print("error: no photos given (pass files, folders, or --list picks.txt)", file=sys.stderr)
        return 2
    out = Path(a.output).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    print(f"Enhancing {len(files)} photo(s) at strength {a.strength} -> {out}")
    failed = 0
    for n, f in enumerate(files, 1):
        dst = out / f"{f.stem}{a.suffix}.jpg"
        if dst.exists() and not a.overwrite:
            dst = out / f"{f.stem}{a.suffix}_{n}.jpg"
        t = time.time()
        try:
            rep = enhance_file(f, dst, a.strength, auto_straighten=not a.no_straighten, quality=a.quality)
            print(f"  [{n}/{len(files)}] {f.name} -> {dst.name}  ({rep.summary()}; {time.time() - t:.1f} s)")
            for note in rep.notes:
                print(f"           {note}")
        except FileNotFoundError:
            failed += 1
            print(f"  [{n}/{len(files)}] {f}: file not found (is the card plugged in?)", file=sys.stderr)
        except Exception as e:
            failed += 1
            print(f"  [{n}/{len(files)}] {f.name}: failed ({e})", file=sys.stderr)
    print(f"Done: {len(files) - failed} written to {out}" + (f", {failed} failed" if failed else ""))
    return 1 if failed else 0


# --------------------------------------------------------------------------
# track
# --------------------------------------------------------------------------


def _bbox(a) -> tuple[float, float, float, float]:
    from .geo import from_local

    if a.bbox:
        s, w, n, e = (float(x) for x in a.bbox.split(","))
        return min(s, n), min(w, e), max(s, n), max(w, e)
    if a.center:
        r = a.radius
        s, w = from_local(a.center[0], a.center[1], -r, -r)
        n, e = from_local(a.center[0], a.center[1], r, r)
        return s, w, n, e
    raise SystemExit("error: give --center LAT,LON --radius M or --bbox S,W,N,E")


def cmd_track_fetch(a) -> int:
    from .track.terrain import fetch_area

    bbox = _bbox(a)
    out = Path(a.output).expanduser()
    print(f"Fetching area pack for {bbox[0]:.4f},{bbox[1]:.4f} to {bbox[2]:.4f},{bbox[3]:.4f} ...")
    pack = fetch_area(bbox, zoom=a.zoom, osm=not a.no_osm, osm_file=Path(a.osm_file) if a.osm_file else None,
                      overpass_url=a.overpass)
    pack.save(out)
    print(f"Saved {out}: {pack.summary()}")
    return 0


def cmd_track_predict(a) -> int:
    import json as _json

    from .track.pipeline import TrackConfig, run_track
    from .track.report import write_track_report
    from .track.sightings import load_file, parse_point, parse_time
    from .track.terrain import AreaPack

    sights = []
    for f in a.sightings:
        got = load_file(Path(f).expanduser())
        print(f"  {len(got)} sighting(s) from {f}")
        sights += got
    if a.lkp:
        sights.append(parse_point(a.lkp, "Last known point"))
    for k, pt in enumerate(a.point, 1):
        sights.append(parse_point(pt, f"Point {k}"))
    if not sights:
        print("error: no sightings (use -s FILE, --lkp, or --point)", file=sys.stderr)
        return 2
    pack = AreaPack.load(Path(a.area).expanduser()) if a.area else None
    coverage = []
    for f in a.coverage:
        js = _json.loads(Path(f).expanduser().read_text())
        coverage += [p["footprint"] for p in js.get("photos", []) if p.get("footprint")]
    if a.coverage:
        print(f"  {len(coverage)} photo footprint(s) of searched ground")
    cfg = TrackConfig(mode=a.mode, profile=a.profile, speed_factor=a.speed_factor, offroad_kmh=a.offroad_kmh,
                      at=parse_time(a.at) if a.at else None, hours=a.hours, heading_weight=a.heading_weight,
                      trail_bias=a.trail_bias, pod=a.pod, segment_m=a.segment_m, top_segments=a.segments,
                      res_m=a.res)
    out = Path(a.output).expanduser() if a.output else Path.cwd() / f"track_{time.strftime('%Y%m%d_%H%M%S')}"
    data = run_track(sights, pack, cfg, out, coverage)
    report = write_track_report(data, out)
    s = data["summary"]
    print()
    print(f"Predicted for {s['at']}: {s['elapsed_min']:.0f} min since '{s['last']['label']}'. "
          f"Reachable: {s['reach_km2']} km², up to {s['reach_max_km']} km away.")
    for k, leg in enumerate(data["legs"], 1):
        print(f"  leg {k}: {leg['from']} -> {leg['to']}: {leg['verdict']}")
    for sg in data["segments"][:5]:
        print(f"  segment {sg['rank']}: {sg['p_remaining']:.1%} of remaining probability -> "
              f"{out / 'segments' / ('segment_%02d.kml' % sg['rank'])}")
    for n in s["notes"]:
        print(f"  ! {n}")
    print(f"Map: {report}")
    if a.open:
        webbrowser.open(report.resolve().as_uri())
    return 0


# --------------------------------------------------------------------------
# site
# --------------------------------------------------------------------------


def _project(a) -> Path | None:
    d = Path(a.project).expanduser()
    if not (d / "site.json").exists():
        print(f"error: {d} is not a survey project (no site.json). Create one with: survey new PHOTOS -o {d}",
              file=sys.stderr)
        return None
    return d


def cmd_site(a) -> int:
    from .site import model as SM

    if a.action in ("new", "survey"):
        from .site.survey import run_survey

        src = Path(a.source).expanduser()
        if not src.exists():
            print(f"error: {src} does not exist", file=sys.stderr)
            return 2
        out = Path(a.output).expanduser() if a.output else Path.cwd() / f"site_{time.strftime('%Y%m%d_%H%M%S')}"
        cfg = None
        if not a.no_detect:
            from .triage.detectors import parse_color
            from .triage.pipeline import TriageConfig

            for c in a.color:
                try:
                    parse_color(c)
                except ValueError as e:
                    print(f"error: {e}", file=sys.stderr)
                    return 2
            cfg = TriageConfig(colors=a.color, sensitivity=a.sensitivity, ai=a.ai, ai_model=a.ai_model,
                               ground_offset_m=a.ground_offset)
        model = run_survey(src, out, name=a.name, ground_offset_m=a.ground_offset, detect=not a.no_detect,
                           detect_cfg=cfg)
        if not model["photos"]:
            print("  ! no photos found; the plan can still be sketched on a blank grid")
        elif not model["basemap"]:
            print("  ! no drone photos looking down at the ground, so there is no base map. Sketch the plan on\n"
                  "    the grid, using the photo positions as a guide.")
        elif not model["basemap"]["straight_down"]:
            print("  ! base map built from angled shots only. The ground is placed correctly, but roofs and\n"
                  "    treetops lean away from the drone by roughly their own height: trace buildings where the\n"
                  "    walls meet the ground. One photo taken straight down from above the site avoids this.")
        if model.get("detections"):
            print(f"  {len(model['detections'])} possible people, vehicles or objects found in the drone photos: "
                  "confirm or dismiss each one in the editor.")
        print(f"Project: {out}")
        if a.no_edit:
            print(f"Next: survey edit {out}")
            return 0
        a.project = str(out)
        a.action = "edit"

    if a.action == "demo":
        from .site.demo import build_demo
        from .site.report import write_report

        out = Path(a.output).expanduser()
        model = build_demo(out)
        report = write_report(model, out)
        print(f"Demo project: {out}\nReport: {report}")
        if a.open:
            webbrowser.open(report.resolve().as_uri())
        if not a.edit:
            print(f"Open it in the editor with: survey edit {out}")
            return 0
        a.project, a.no_browser = str(out), False

    d = _project(a)
    if d is None:
        return 2
    if a.action == "report":
        from .site.report import write_report

        report = write_report(SM.load(d / "site.json"), d)
        print(f"Report: {report}\nExports: {d}/site.(geojson|kml), assessment.(csv|json)")
        if a.open:
            webbrowser.open(report.resolve().as_uri())
        return 0
    from .site.server import serve

    serve(d, port=a.port, open_browser=not a.no_browser)
    return 0


# --------------------------------------------------------------------------
# model / dataset
# --------------------------------------------------------------------------


def cmd_model(a) -> int:
    from .ml.onnx_detector import ModelCard, OnnxDetector

    path = Path(a.model).expanduser()
    if a.model_cmd == "card":
        card = ModelCard.load(path)
        if a.format:
            card.format = a.format
        if a.classes:
            card.classes = [c.strip() for c in a.classes.split(",") if c.strip()]
        for k in ("input_size", "normalize", "tile", "conf"):
            if getattr(a, k) is not None:
                setattr(card, k, getattr(a, k))
        if a.notes:
            card.notes = a.notes
        print(f"Wrote {card.save(path)}")
    det = OnnxDetector(path)
    info = det.describe()
    print(f"{path.name}")
    for k, v in info.items():
        if k == "classes" and isinstance(v, list) and len(v) > 12:
            v = f"{len(v)} classes: {', '.join(v[:10])}, ..."
        if k != "file":
            print(f"  {k:11} {v}")
    if getattr(a, "photo", None):
        from PIL import Image

        t = det.benchmark(a.photo)
        with Image.open(a.photo) as im:
            dets = det.detect_image(im)
        print(f"  speed      {t:.2f} s per photo ({Path(a.photo).name}, {len(dets)} detection(s) at conf {det.conf})")
        for x0, y0, x1, y1, sc, name in dets[:10]:
            print(f"             {name:12} {sc:.2f}  box {x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}")
    return 0


def cmd_dataset(a) -> int:
    from .ml.dataset import Dataset, add_from_triage, export

    if a.dataset_cmd == "add":
        res = add_from_triage(Path(a.triage).expanduser(), Path(a.labels).expanduser(), Path(a.dataset).expanduser(),
                              tile=a.tile, pad=a.pad, force=a.force)
        ad, st = res["added"], res["dataset"]
        print(f"Added {ad['positive']} positive and {ad['negative']} hard-negative tile(s)"
              + (f"; {ad['skipped_existing']} already in the dataset" if ad["skipped_existing"] else ""))
        print(f"Dataset: {st['tiles']} tiles ({st['train']} train / {st['val']} val) from {st['runs']} run(s); "
              f"boxes: {st['boxes'] or 'none'}")
    elif a.dataset_cmd == "export":
        cfg = export(Path(a.dataset).expanduser(), Path(a.output).expanduser(), a.format,
                     [c.strip() for c in a.classes.split(",")] if a.classes else None)
        print(f"Exported {a.format.upper()} dataset -> {cfg}")
    else:
        st = Dataset(Path(a.dataset).expanduser()).stats()
        for k, v in st.items():
            print(f"  {k:15} {v}")
        if st["tiles"] and st["val"] == 0:
            print("  ! no validation tiles yet: add more flights so some photos land in the validation split")
    return 0


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
    ap = argparse.ArgumentParser(prog="isr", description="Drone tools: SAR photo triage, mission planning, and a portfolio studio.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("triage", help="scan a folder / SD card for possible people and objects",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    t.add_argument("source", help="folder of photos, e.g. the SD card's DCIM folder (searched recursively)")
    t.add_argument("-o", "--output", help="output folder (default: ./triage_<timestamp>; never written to the card)")
    t.add_argument("-c", "--color", action="append", default=[], metavar="COLOR",
                   help="known clothing/gear colour to look for; repeatable. Names (red, orange, yellow, blue, "
                        "purple, pink, cyan, white, green) or #rrggbb")
    t.add_argument("--reference", metavar="PHOTO",
                   help="photo of the missing person / their vehicle / animal: its colours and size class are "
                        "used to search and rank (no face recognition)")
    t.add_argument("--target", choices=["person", "vehicle", "animal"], default="person",
                   help="what the reference photo shows")
    t.add_argument("--ref-box", metavar="X0,Y0,X1,Y1", help="pixel box around the subject in the reference photo")
    t.add_argument("--no-anomaly", action="store_true", help="disable the colour-anomaly detector")
    t.add_argument("--sensitivity", type=float, default=1.0, help="anomaly sensitivity; >1 flags more, <1 fewer")
    t.add_argument("--ai", action="store_true", help="also run the YOLO person/object detector (needs ultralytics)")
    t.add_argument("--ai-model", default="yolov8n.pt",
                   help="model file: an exported .onnx (runs without PyTorch) or Ultralytics .pt weights")
    t.add_argument("--ai-tile", type=int, help="ONNX models: tile size in full-resolution pixels (smaller = finds "
                                               "smaller people, slower)")
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

    st = sub.add_parser("studio", help="find your most portfolio-worthy photos and auto-edit the best",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    st.add_argument("source", help="folder or SD card to scan (searched recursively; never modified)")
    st.add_argument("-o", "--output", help="output folder (default: ./studio_<timestamp>)")
    st.add_argument("--top", type=int, default=30, help="how many best moments to suggest (and auto-edit)")
    st.add_argument("--no-enhance", action="store_true", help="score and rank only; don't create edited copies")
    st.add_argument("--strength", type=float, default=0.6, help="edit strength 0 (none) .. 1 (strong)")
    st.add_argument("--no-straighten", action="store_true", help="never rotate to level the horizon")
    st.add_argument("--similarity", type=int, default=30,
                    help="how alike frames must be to group as one moment (0-256 differing bits; higher groups more)")
    st.add_argument("--workers", type=int, default=0, help="parallel processes (0 = CPU count - 1)")
    st.add_argument("--open", action="store_true", help="open the gallery in a browser when done")
    st.set_defaults(func=cmd_studio)

    en = sub.add_parser("enhance", help="light automatic touch-up of photos into new files",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    en.add_argument("inputs", nargs="*", help="photos and/or folders")
    en.add_argument("--list", help="text file with one photo path per line (e.g. picks.txt from the Studio gallery)")
    en.add_argument("-o", "--output", default="edited", help="output folder")
    en.add_argument("--strength", type=float, default=0.6, help="edit strength 0 (none) .. 1 (strong)")
    en.add_argument("--no-straighten", action="store_true", help="never rotate to level the horizon")
    en.add_argument("--quality", type=int, default=95, help="JPEG quality of the edited files")
    en.add_argument("--suffix", default="_edit", help="added to each output file name")
    en.add_argument("--overwrite", action="store_true", help="replace existing files in the output folder")
    en.set_defaults(func=cmd_enhance)

    tr = sub.add_parser("track", help="movement analysis: where could the subject be now?")
    trs = tr.add_subparsers(dest="track_cmd", required=True)
    tf = trs.add_parser("fetch", help="download elevation + trails/roads/water for an area (run while online)",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    tf.add_argument("--center", type=_latlon, help="area centre 'lat,lon'")
    tf.add_argument("--radius", type=float, default=8000, help="half-width of the area, m")
    tf.add_argument("--bbox", help="S,W,N,E instead of --center/--radius")
    tf.add_argument("-o", "--output", default="area", help="area pack folder")
    tf.add_argument("--zoom", type=int, default=12, help="elevation zoom (12 = ~30 m, 13 = ~15 m)")
    tf.add_argument("--no-osm", action="store_true", help="elevation only")
    tf.add_argument("--osm-file", help="use an exported OSM file (Overpass JSON, GeoJSON or .osm) instead of downloading")
    tf.add_argument("--overpass", default="https://overpass-api.de/api/interpreter", help="Overpass API URL")
    tf.set_defaults(func=cmd_track_fetch)

    tp = trs.add_parser("predict", help="reach, corridors, probability map and ranked search segments",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    tp.add_argument("-s", "--sightings", action="append", default=[], metavar="FILE",
                    help="CSV (e.g. confirmed.csv from the triage report) or GPX of timed sightings; repeatable")
    tp.add_argument("--lkp", help="last known point 'lat,lon,time' (e.g. 39.70,-105.20,2026-11-16T09:30)")
    tp.add_argument("--point", action="append", default=[], help="extra sighting 'lat,lon,time'; repeatable")
    tp.add_argument("--mode", choices=["foot", "vehicle"], default="foot")
    tp.add_argument("--profile", choices=["hiker", "runner", "child", "elderly", "dementia", "injured"],
                    default="hiker", help="walking-speed profile (foot mode)")
    tp.add_argument("--speed-factor", type=float, default=1.0, help="multiply all speeds (e.g. 0.8 at night)")
    tp.add_argument("--offroad-kmh", type=float, default=3.0, help="vehicle speed off the road network")
    tp.add_argument("--at", help="predict for this time (default: now), e.g. 2026-11-16T14:00")
    tp.add_argument("--hours", type=float, help="hours since the last sighting (instead of --at)")
    tp.add_argument("--area", help="area pack folder from `isr track fetch`")
    tp.add_argument("--coverage", action="append", default=[], metavar="RESULTS_JSON",
                    help="triage results.json whose photo footprints count as searched; repeatable")
    tp.add_argument("--pod", type=float, default=0.7, help="probability a photo search would have spotted them")
    tp.add_argument("--heading-weight", type=float, default=1.0, help="how strongly to favour the direction of travel")
    tp.add_argument("--trail-bias", type=float, default=1.5, help="extra weight near trails/roads (foot)")
    tp.add_argument("--segment-m", type=float, default=0, help="search segment size, m (0 = 250 foot / 1000 vehicle)")
    tp.add_argument("--segments", type=int, default=8, help="how many ranked segments to export")
    tp.add_argument("--res", type=float, default=0, help="model cell size, m (0 = auto)")
    tp.add_argument("-o", "--output", help="output folder (default: ./track_<timestamp>)")
    tp.add_argument("--open", action="store_true", help="open the map in a browser when done")
    tp.set_defaults(func=cmd_track_predict)

    si = sub.add_parser("site", help="site security assessment: survey photos -> site plan -> report")
    ss = si.add_subparsers(dest="action", required=True)
    sv = ss.add_parser("new", aliases=["survey"], help="read hand-flown drone and walk-round photos, build the base map and open the editor",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sv.add_argument("source", help="folder of photos (searched recursively)")
    sv.add_argument("-o", "--output", help="project folder (default ./site_<timestamp>); re-run to add photos")
    sv.add_argument("--name", help="site name for the report")
    sv.add_argument("--ground-offset", type=float, default=0.0,
                    help="height of the site above (+) or below (-) the drone's take-off point, metres")
    sv.add_argument("--no-detect", action="store_true",
                    help="do not look for people, vehicles and objects in the drone photos")
    sv.add_argument("-c", "--color", action="append", default=[], metavar="COLOUR",
                    help="also look for this colour (clothing, a vehicle): a name such as red, or #rrggbb; repeatable")
    sv.add_argument("--sensitivity", type=float, default=1.0, help="detector sensitivity: 1.5 flags more, 0.7 fewer")
    sv.add_argument("--ai", action="store_true", help="also run the AI person/object detector (needs the [ai] extra)")
    sv.add_argument("--ai-model", default="yolov8n.pt", help="weights for --ai")
    sv.add_argument("--no-edit", action="store_true", help="just build the project, do not open the editor")
    se = ss.add_parser("edit", help="open a project in the editor (local web page)",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sr = ss.add_parser("report", help="re-run the analysis and write report.html and the exports")
    sr.add_argument("--open", action="store_true", help="open the report in a browser when done")
    for sp in (se, sr):
        sp.add_argument("project", help="project folder (contains site.json)")
    sd = ss.add_parser("demo", help="build a made-up example property to try the tool",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sd.add_argument("-o", "--output", default="site_demo", help="project folder")
    sd.add_argument("--open", action="store_true", help="open the report in a browser")
    sd.add_argument("--edit", action="store_true", help="then open the editor")
    for sp in (sv, se, sd):
        sp.add_argument("--port", type=int, default=8765, help="local port for the editor")
        sp.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    si.set_defaults(func=cmd_site)

    mo = sub.add_parser("model", help="inspect, describe and benchmark an exported .onnx detector")
    mos = mo.add_subparsers(dest="model_cmd", required=True)
    mi = mos.add_parser("info", help="show what the app reads from a model; optionally time it on a photo")
    mi.add_argument("model")
    mi.add_argument("--photo", help="run the model on this photo and report speed and detections")
    mi.set_defaults(func=cmd_model)
    mc = mos.add_parser("card", help="write/update the model card (model.json) next to the .onnx")
    mc.add_argument("model")
    mc.add_argument("--format", choices=["yolo", "yolo_e2e", "detr", "dfine"])
    mc.add_argument("--classes", help="comma-separated class names in the model's order")
    mc.add_argument("--input-size", type=int)
    mc.add_argument("--normalize", choices=["none", "imagenet"])
    mc.add_argument("--tile", type=int, help="default full-resolution tile size")
    mc.add_argument("--conf", type=float, help="default confidence threshold")
    mc.add_argument("--notes", help="free text: training data, date, scorecard...")
    mc.set_defaults(func=cmd_model, photo=None)

    ds = sub.add_parser("dataset", help="build training data from your triage reviews")
    dss = ds.add_subparsers(dest="dataset_cmd", required=True)
    da = dss.add_parser("add", help="add a reviewed triage run (results + labels.json from the report)",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    da.add_argument("triage", help="triage output folder (with results.json)")
    da.add_argument("--labels", required=True, help="labels.json exported from that run's report")
    da.add_argument("-d", "--dataset", default="sar_dataset", help="dataset folder (grows with every run)")
    da.add_argument("--tile", type=int, default=640, help="tile size in full-resolution pixels")
    da.add_argument("--pad", type=float, default=0.15, help="grow candidate boxes by this share on each side")
    da.add_argument("--force", action="store_true", help="accept labels exported from a different run")
    da.set_defaults(func=cmd_dataset)
    de = dss.add_parser("export", help="write the dataset in YOLO or COCO layout for training")
    de.add_argument("-d", "--dataset", default="sar_dataset")
    de.add_argument("--format", choices=["yolo", "coco"], default="coco", help="coco for RF-DETR / D-FINE")
    de.add_argument("--classes", help="class order, e.g. person,vehicle (default: order first seen)")
    de.add_argument("-o", "--output", required=True)
    de.set_defaults(func=cmd_dataset)
    dst = dss.add_parser("stats", help="summarise the dataset")
    dst.add_argument("-d", "--dataset", default="sar_dataset")
    dst.set_defaults(func=cmd_dataset)

    i = sub.add_parser("info", help="show the metadata the tools read from a photo")
    i.add_argument("photos", nargs="+")
    i.set_defaults(func=cmd_info)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a) or 0


def survey_main(argv=None) -> int:
    """The ``survey`` command: Ronin Sec Survey on its own, the same as ``isr site ...``."""
    return main(["site", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    sys.exit(main())
