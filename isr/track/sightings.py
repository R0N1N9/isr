"""Sightings: timed positions of the subject, and checks on the legs between them."""

from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..geo import haversine_m

TIME_COLUMNS = ("taken_at", "time", "timestamp", "datetime", "date_time", "when")
LABEL_COLUMNS = ("label", "name", "id", "title")


@dataclass
class Sighting:
    lat: float
    lon: float
    time: datetime | None
    label: str
    source: str = ""

    @property
    def pos(self) -> tuple[float, float]:
        return self.lat, self.lon


def parse_time(s: str | None) -> datetime | None:
    """ISO 8601, DJI EXIF ('2026:11:16 10:04:00') or 'YYYY-MM-DD HH:MM'. Aware times become local naive.

    DJI photo times are local camera time without a time zone; GPX times are
    usually UTC. Converting everything to the laptop's local time keeps them
    comparable as long as the laptop is set to the search area's time zone.
    """
    if not s:
        return None
    s = str(s).strip()
    if len(s) >= 19 and s[4] == ":" and s[7] == ":":
        s = s[:4] + "-" + s[5:7] + "-" + s[8:]
    s = s.replace("Z", "+00:00")
    for fmt in (None, "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
        try:
            dt = datetime.fromisoformat(s) if fmt is None else datetime.strptime(s, fmt)
            break
        except ValueError:
            dt = None
    if dt is None:
        raise ValueError(f"unrecognised time {s!r}; use e.g. 2026-11-16T14:30")
    if dt.tzinfo is not None:
        dt = dt.astimezone().replace(tzinfo=None)
    return dt


def parse_point(spec: str, label: str) -> Sighting:
    """'lat,lon[,time]' from the command line."""
    parts = [p.strip() for p in spec.split(",", 2)]
    if len(parts) < 2:
        raise ValueError(f"expected 'lat,lon[,time]', got {spec!r}")
    return Sighting(float(parts[0]), float(parts[1]), parse_time(parts[2]) if len(parts) > 2 else None, label,
                    "command line")


def load_file(path: Path) -> list[Sighting]:
    """Sightings from a CSV (triage detections.csv / confirmed.csv, or any lat/lon/time table) or GPX."""
    path = Path(path)
    out: list[Sighting] = []
    if path.suffix.lower() == ".gpx":
        root = ET.fromstring(path.read_text(encoding="utf-8", errors="replace"))
        for el in root.iter():
            if el.tag.split("}")[-1] in ("wpt", "trkpt", "rtept"):
                kids = {c.tag.split("}")[-1]: (c.text or "") for c in el}
                out.append(Sighting(float(el.get("lat")), float(el.get("lon")), parse_time(kids.get("time")),
                                    kids.get("name") or f"{path.stem} {len(out) + 1}", path.name))
        return out
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    for n, r in enumerate(rows, 1):
        low = {k.lower().strip(): v for k, v in r.items() if k}
        lat, lon = low.get("lat") or low.get("latitude"), low.get("lon") or low.get("lng") or low.get("longitude")
        if not lat or not lon:
            continue
        t = next((low[c] for c in TIME_COLUMNS if low.get(c)), None)
        lab = next((low[c] for c in LABEL_COLUMNS if low.get(c)), f"{path.stem} {n}")
        out.append(Sighting(float(lat), float(lon), parse_time(t), str(lab), path.name))
    return out


def order(sightings: list[Sighting]) -> list[Sighting]:
    timed = sorted((s for s in sightings if s.time), key=lambda s: s.time)
    return timed + [s for s in sightings if not s.time]


def check_leg(a: Sighting, b: Sighting, mode: str, model_min_s: float | None) -> dict:
    """Distance, time and plausibility of travelling from sighting a to b."""
    dist = haversine_m(a.lat, a.lon, b.lat, b.lon)
    leg = {"from": a.label, "to": b.label, "distance_m": round(dist, 1), "dt_s": None, "speed_kmh": None,
           "model_min_s": round(model_min_s, 1) if model_min_s is not None and model_min_s < float("inf") else None,
           "verdict": "no times", "level": "info"}
    if not (a.time and b.time):
        return leg
    dt = (b.time - a.time).total_seconds()
    leg["dt_s"] = dt
    if dt <= 0:
        leg.update(verdict="same time: probably the same moment from two photos" if dist < 30 else
                   "same time, different places: two different subjects or a position error", level="warn")
        return leg
    v = dist / dt * 3.6
    leg["speed_kmh"] = round(v, 2)
    vmax = 8.0 if mode == "foot" else 140.0
    if v > vmax:
        leg.update(verdict=f"implausible: {v:.1f} km/h straight-line exceeds {vmax:.0f} km/h for {mode} travel. "
                           "Different subject, wrong time, or a position error?", level="bad")
    elif model_min_s is None or model_min_s == float("inf"):
        leg.update(verdict="unreachable in the terrain model (barrier between them?): check the map", level="warn")
    elif model_min_s > dt * 1.05:
        leg.update(verdict=f"faster than terrain allows: the quickest modelled route takes "
                           f"{model_min_s / 60:.0f} min but only {dt / 60:.0f} min passed. Faster than assumed "
                           "(vehicle? runner?) or a time/position error.", level="warn")
    else:
        leg.update(verdict=f"plausible: the quickest route uses {model_min_s / dt:.0%} of the available time",
                   level="good")
    return leg
