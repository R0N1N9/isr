"""Run the movement analysis and write map layers + exports."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image

from .model import (FOOT_PROFILES, ROAD_KMH, Grid, Model, bearing, build_graph, build_surface, corridor,
                    coverage_mask, hillshade, probability, segments)
from .sightings import Sighting, check_leg, order
from .terrain import AreaPack


@dataclass
class TrackConfig:
    mode: str = "foot"  # foot | vehicle
    profile: str = "hiker"  # foot only
    speed_factor: float = 1.0
    offroad_kmh: float = 3.0  # vehicle only
    at: datetime | None = None  # time to predict for (default: now)
    hours: float | None = None  # elapsed time if sightings have no times
    heading_weight: float = 1.0
    trail_bias: float = 1.5
    pod: float = 0.7  # probability that a photo search of a cell would have found the subject
    segment_m: float = 0.0  # 0 = auto (250 m foot, 1000 m vehicle)
    top_segments: int = 8
    res_m: float = 0.0  # 0 = auto
    max_cells: int = 640
    notes: list[str] = field(default_factory=list)


def _colormap(v: np.ndarray, stops) -> np.ndarray:
    """v in 0..1 -> RGB via linear interpolation of colour stops [(pos, (r,g,b)), ...]."""
    pos = np.array([s[0] for s in stops])
    out = np.zeros(v.shape + (3,), np.float32)
    for c in range(3):
        out[..., c] = np.interp(v, pos, [s[1][c] for s in stops])
    return out


HEAT = [(0, (255, 214, 102)), (0.35, (255, 140, 40)), (0.7, (235, 50, 80)), (1, (170, 20, 160))]


def _rgba_png(rgb: np.ndarray, alpha: np.ndarray, path: Path) -> None:
    a = np.dstack([np.clip(rgb, 0, 255), np.clip(alpha, 0, 1) * 255]).astype(np.uint8)
    Image.fromarray(a, "RGBA").save(path, optimize=True)


def run_track(sightings: list[Sighting], pack: AreaPack | None, cfg: TrackConfig, out_dir: Path,
              coverage: list[list[tuple[float, float]]] | None = None, log=print) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "layers").mkdir(parents=True, exist_ok=True)
    (out_dir / "segments").mkdir(parents=True, exist_ok=True)
    sights = order(sightings)
    if not sights:
        raise SystemExit("no sightings given")
    timed = [s for s in sights if s.time]
    last = timed[-1] if timed else sights[-1]
    at = cfg.at or datetime.now().replace(microsecond=0)
    if cfg.hours is not None:
        elapsed = cfg.hours * 3600
        at = (last.time + timedelta(seconds=elapsed)) if last.time else at
    elif last.time:
        elapsed = (at - last.time).total_seconds()
    else:
        raise SystemExit("sightings have no times: pass --hours (time since the last sighting)")
    if elapsed <= 0:
        raise SystemExit(f"prediction time {at} is not after the last sighting ({last.time}); set --at")

    # ---- grid sized to everywhere the subject could physically reach
    if cfg.mode == "foot":
        factor = FOOT_PROFILES.get(cfg.profile, 1.0) * cfg.speed_factor
        vmax = 6.0 / 3.6 * factor
        base_res = cfg.res_m or 25.0
    else:
        factor = cfg.speed_factor
        vmax = max(ROAD_KMH.values()) / 3.6 * factor
        base_res = cfg.res_m or 50.0
    legs_t = [(b.time - a.time).total_seconds() for a, b in zip(timed, timed[1:])]
    margin = max([vmax * elapsed] + [vmax * t / 2 for t in legs_t if t > 0]) * 1.05
    cap = 25_000 if cfg.mode == "foot" else 120_000
    if margin > cap:
        cfg.notes.append(f"reach limited to {cap / 1000:.0f} km from the sightings (modelled area cap)")
        margin = cap
    grid = Grid.around([s.pos for s in sights], margin, base_res, cfg.max_cells)
    log(f"Modelling {grid.nx * grid.res / 1000:.1f} x {grid.ny * grid.res / 1000:.1f} km at {grid.res:.0f} m cells "
        f"({cfg.mode}{', ' + cfg.profile if cfg.mode == 'foot' else ''})...")
    if pack is None:
        cfg.notes.append("no area pack: flat terrain, no trails/roads/water (run `isr track fetch` for real terrain)")
        if cfg.mode == "vehicle":
            cfg.notes.append("vehicle mode without a road map treats everywhere as off-road")
    else:
        s, w, n, e = grid.bounds()
        ps, pw, pn, pe = pack.bbox
        if s < ps or w < pw or n > pn or e > pe:
            cfg.notes.append("the reachable area extends beyond the area pack; terrain outside it is approximate "
                             "(fetch a larger area)")
        if not pack.ways:
            cfg.notes.append("area pack has no trails/roads/water")
        if pack.dem is None:
            cfg.notes.append("area pack has no elevation: terrain treated as flat")

    surf = build_surface(grid, pack, cfg.mode, cfg.offroad_kmh)
    model = Model(surf, build_graph(surf, factor if cfg.mode == "foot" else cfg.speed_factor))

    # ---- legs between timed sightings: plausibility + corridors
    legs, paths = [], []
    corr = np.full((grid.ny, grid.nx), np.nan)
    for a, b in zip(timed, timed[1:]):
        dt = (b.time - a.time).total_seconds()
        if dt > 0:
            slack, min_t, best = corridor(model, a.pos, b.pos, dt)
            frac = slack / dt
            corr = np.fmax(corr, frac)
            paths.append([[la, lo] for la, lo in best])
        else:
            min_t = None
            paths.append([])
        legs.append(check_leg(a, b, cfg.mode, min_t))

    # ---- where now
    heading = None
    if len(timed) >= 2:
        prev = next((s for s in reversed(timed[:-1]) if math.dist(s.pos, last.pos) > 2e-4), None)
        if prev is not None:
            heading = bearing(prev.pos, last.pos)
    p, t = probability(model, last.pos, elapsed, heading, cfg.heading_weight, cfg.trail_bias)
    covered = coverage_mask(grid, coverage or [])
    p_rem = p * (1 - cfg.pod * covered)
    seg_m = cfg.segment_m or (250.0 if cfg.mode == "foot" else 1000.0)
    segs = segments(grid, p_rem, p, max(seg_m, grid.res), cfg.top_segments)
    reach = np.isfinite(t) & (t <= elapsed)
    area_km2 = float(reach.sum()) * grid.res**2 / 1e6
    pos_reach = np.argwhere(reach)
    far = 0.0
    if len(pos_reach):
        li, lj = grid.cell(*last.pos)
        far = float(np.max(np.hypot(pos_reach[:, 0] - li, pos_reach[:, 1] - lj))) * grid.res

    # ---- layers
    layers = {}
    if surf.elev is not None:
        hs = hillshade(surf.elev, grid.res)
        Image.fromarray((hs * 150 + 35).astype(np.uint8), "L").save(out_dir / "layers/hillshade.png", optimize=True)
        layers["hillshade"] = "layers/hillshade.png"
    if p.max() > 0:
        v = np.clip(p / np.percentile(p[p > 0], 99), 0, 1)
        _rgba_png(_colormap(v, HEAT), np.where(p > 0, 0.18 + 0.72 * v, 0), out_dir / "layers/probability.png")
        layers["probability"] = "layers/probability.png"
    if reach.any():
        mins = np.where(reach, t / 60, np.nan)
        step = _nice_step(elapsed / 60 / 5)
        band = np.floor(np.nan_to_num(mins, nan=-1) / step)
        rgb = np.zeros(t.shape + (3,), np.float32)
        rgb[...] = (60, 130, 255)
        alpha = np.where(reach, np.where(band % 2 == 0, 0.07, 0.0), 0)
        edge = reach & (np.abs(np.diff(np.pad(band, ((0, 0), (0, 1)), mode="edge"), axis=1)) +
                        np.abs(np.diff(np.pad(band, ((0, 1), (0, 0)), mode="edge"), axis=0)) > 0)
        alpha = np.where(edge, 0.7, alpha)
        _rgba_png(rgb, alpha, out_dir / "layers/reach.png")
        layers["reach"] = "layers/reach.png"
        layers["reach_step_min"] = step
    if np.isfinite(corr).any():
        c = np.nan_to_num(corr, nan=-1)
        _rgba_png(_colormap(np.clip(c, 0, 1), [(0, (20, 200, 170)), (1, (20, 200, 170))]),
                  np.where(c >= 0, 0.12 + 0.3 * np.clip(c, 0, 1), 0), out_dir / "layers/corridor.png")
        layers["corridor"] = "layers/corridor.png"
    if covered.any():
        _rgba_png(np.full(t.shape + (3,), 255, np.float32), np.where(covered, 0.22, 0), out_dir / "layers/coverage.png")
        layers["coverage"] = "layers/coverage.png"

    # ---- vectors for the page (grid pixel space)
    def px(la, lo):
        x, y = grid.px(la, lo)
        return [round(x, 2), round(y, 2)]

    ways_px = []
    for wy in (pack.ways if pack else []):
        pts = [px(la, lo) for la, lo in wy["coords"]]
        if any(-5 <= x <= grid.nx + 5 and -5 <= y <= grid.ny + 5 for x, y in pts):
            ways_px.append({"kind": wy["kind"], "name": wy["tags"].get("name", ""), "pts": pts[:: max(1, len(pts) // 400)]})

    for sg in segs:
        sg["px"] = [px(*c) for c in sg["corners"]]
    # initial map view: the reachable area plus all sightings, with a margin
    pts = [px(s.lat, s.lon) for s in sights]
    if len(pos_reach):
        pts += [[float(pos_reach[:, 1].min()), float(pos_reach[:, 0].min())],
                [float(pos_reach[:, 1].max()) + 1, float(pos_reach[:, 0].max()) + 1]]
    fx0, fy0 = min(p_[0] for p_ in pts), min(p_[1] for p_ in pts)
    fx1, fy1 = max(p_[0] for p_ in pts), max(p_[1] for p_ in pts)
    fw, fh = max(fx1 - fx0, 20) * 1.15, max(fy1 - fy0, 20) * 1.15
    if fw / fh < grid.nx / grid.ny:
        fw = fh * grid.nx / grid.ny
    else:
        fh = fw * grid.ny / grid.nx
    focus = [round((fx0 + fx1 - fw) / 2, 1), round((fy0 + fy1 - fh) / 2, 1), round(fw, 1), round(fh, 1)]
    summary = {
        "mode": cfg.mode,
        "profile": cfg.profile if cfg.mode == "foot" else None,
        "speed_factor": cfg.speed_factor,
        "at": at.isoformat(sep=" "),
        "last": {"label": last.label, "time": last.time.isoformat(sep=" ") if last.time else None,
                 "lat": last.lat, "lon": last.lon},
        "elapsed_min": round(elapsed / 60, 1),
        "heading_deg": round(math.degrees(heading) % 360, 1) if heading is not None else None,
        "reach_km2": round(area_km2, 2),
        "reach_max_km": round(far / 1000, 2),
        "covered_share": round(float((p * covered).sum()), 3),
        "pod": cfg.pod,
        "grid": {"nx": grid.nx, "ny": grid.ny, "res_m": round(grid.res, 1), "bounds": grid.bounds()},
        "focus": focus,
        "pack": pack.summary() if pack else None,
        "notes": cfg.notes,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    data = {
        "summary": summary,
        "sightings": [{**asdict(s), "time": s.time.isoformat(sep=" ") if s.time else None, "px": px(s.lat, s.lon)}
                      for s in sights],
        "legs": legs,
        "paths": [[px(la, lo) for la, lo in path] for path in paths],
        "paths_latlon": paths,
        "segments": segs,
        "ways": ways_px,
        "layers": layers,
        "config": {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in asdict(cfg).items()},
    }
    _exports(data, out_dir)
    (out_dir / "track.json").write_text(json.dumps(data, indent=1, default=str))
    return data


def _nice_step(x: float) -> float:
    for s in (5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720):
        if s >= x:
            return s
    return 1440


def _kml_poly(name: str, corners, color: str, desc: str = "") -> str:
    ring = " ".join(f"{lo:.7f},{la:.7f},0" for la, lo in list(corners) + [corners[0]])
    return (f"<Placemark><name>{escape(name)}</name><description>{escape(desc)}</description>"
            f"<Style><LineStyle><color>{color}</color><width>2</width></LineStyle><PolyStyle><color>30{color[2:]}</color>"
            f"</PolyStyle></Style><Polygon><outerBoundaryIs><LinearRing><coordinates>{ring}</coordinates>"
            "</LinearRing></outerBoundaryIs></Polygon></Placemark>")


def _exports(data: dict, out_dir: Path) -> None:
    S = data["summary"]
    parts = ['<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
             f"<name>Movement analysis ({S['mode']}) for {escape(S['at'])}</name><Folder><name>Sightings</name>"]
    for s in data["sightings"]:
        parts.append(f"<Placemark><name>{escape(s['label'])}</name><description>{escape(str(s['time'] or ''))}"
                     f"</description><Point><coordinates>{s['lon']:.7f},{s['lat']:.7f},0</coordinates></Point></Placemark>")
    parts.append("</Folder><Folder><name>Fastest routes between sightings</name>")
    for k, path in enumerate(data["paths_latlon"]):
        if path:
            coords = " ".join(f"{lo:.7f},{la:.7f},0" for la, lo in path)
            parts.append(f"<Placemark><name>Leg {k + 1}</name><Style><LineStyle><color>ffaac814</color><width>3</width>"
                         f"</LineStyle></Style><LineString><coordinates>{coords}</coordinates></LineString></Placemark>")
    parts.append("</Folder><Folder><name>Search segments (ranked)</name>")
    for sg in data["segments"]:
        desc = f"{sg['p_remaining']:.1%} of remaining probability; {sg['size_m'][0]}x{sg['size_m'][1]} m"
        parts.append(_kml_poly(f"Segment {sg['rank']}", sg["corners"], "ff3c50eb", desc))
        seg_kml = ('<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
                   + _kml_poly(f"Segment {sg['rank']}", sg["corners"], "ff3c50eb", desc) + "</Document></kml>")
        (out_dir / "segments" / f"segment_{sg['rank']:02d}.kml").write_text(seg_kml)
    parts.append("</Folder></Document></kml>")
    (out_dir / "track.kml").write_text("".join(parts))
    gpx = ['<?xml version="1.0" encoding="UTF-8"?><gpx version="1.1" creator="isr-tools" '
           'xmlns="http://www.topografix.com/GPX/1/1">']
    for s in data["sightings"]:
        t = f"<desc>{escape(str(s['time']))}</desc>" if s["time"] else ""
        gpx.append(f'<wpt lat="{s["lat"]:.7f}" lon="{s["lon"]:.7f}"><name>{escape(s["label"])}</name>{t}</wpt>')
    for sg in data["segments"]:
        la = sum(c[0] for c in sg["corners"]) / 4
        lo = sum(c[1] for c in sg["corners"]) / 4
        gpx.append(f'<wpt lat="{la:.7f}" lon="{lo:.7f}"><name>SEG{sg["rank"]:02d}</name>'
                   f"<desc>{sg['p_remaining']:.1%} remaining probability</desc></wpt>")
    gpx.append("</gpx>")
    (out_dir / "track.gpx").write_text("\n".join(gpx))
