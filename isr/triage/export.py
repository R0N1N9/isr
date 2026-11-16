"""Write triage results in formats search teams already use.

* ``detections.csv``     spreadsheet / incident log
* ``detections.geojson`` CalTopo / SARTopo import, QGIS
* ``detections.kml``     Google Earth, CalTopo, ATAK
* ``detections.gpx``     Garmin / handheld GPS waypoints for ground teams
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape


def _rows(data: dict, min_score: float = 0.0):
    for p in data["photos"]:
        for d in p["detections"]:
            if d["score"] >= min_score:
                yield p, d


def write_csv(data: dict, path: Path) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "score", "label", "detectors", "lat", "lon", "position_source", "area_m2",
                    "photo", "taken_at", "drone_alt_m", "gimbal_pitch", "gsd_cm", "box_x0", "box_y0", "box_x1",
                    "box_y1", "crop"])
        for p, d in sorted(_rows(data), key=lambda t: -t[1]["score"]):
            w.writerow([d["id"], d["score"], d["label"], "|".join(d["kinds"]), d["lat"], d["lon"],
                        d["position_source"], d["area_m2"], p["rel"], p["taken_at"], p["rel_alt_m"],
                        p["gimbal_pitch"], p["gsd_cm"], *d["box"], d.get("crop")])


def write_geojson(data: dict, path: Path) -> None:
    feats = []
    for p, d in _rows(data):
        if d["lat"] is None:
            continue
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [d["lon"], d["lat"]]},
            "properties": {
                "title": f"{d['id']} ({d['score']:.2f})",
                "description": f"{d['label']} — photo {p['rel']}",
                "score": d["score"],
                "detectors": d["kinds"],
                "photo": p["rel"],
                "position_source": d["position_source"],
                "marker-color": _hex(d["score"]),
            },
        })
    for p in data["photos"]:
        if p["footprint"]:
            ring = [[lon, lat] for lat, lon in p["footprint"]]
            ring.append(ring[0])
            feats.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {"title": p["rel"], "kind": "photo_footprint", "stroke": "#888888",
                               "fill-opacity": 0.05, "hits": len(p["detections"])},
            })
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))


def _hex(score: float) -> str:
    if score >= 0.75:
        return "#e8174a"
    if score >= 0.55:
        return "#f58a07"
    return "#f2c600"


def _kml_color(score: float) -> str:
    h = _hex(score)[1:]
    return "ff" + h[4:6] + h[2:4] + h[0:2]  # KML is aabbggrr


def write_kml(data: dict, path: Path) -> None:
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
           f"<name>Drone triage — {escape(data['summary']['generated'])}</name>"]
    out.append("<Folder><name>Candidates</name>")
    for p, d in sorted(_rows(data), key=lambda t: -t[1]["score"]):
        if d["lat"] is None:
            continue
        out.append(
            "<Placemark>"
            f"<name>{escape(d['id'])} ({d['score']:.2f})</name>"
            f"<description>{escape(d['label'])}&lt;br/&gt;Photo: {escape(p['rel'])}&lt;br/&gt;"
            f"Position: {escape(str(d['position_source']))}</description>"
            f"<Style><IconStyle><color>{_kml_color(d['score'])}</color></IconStyle></Style>"
            f"<Point><coordinates>{d['lon']:.7f},{d['lat']:.7f},0</coordinates></Point>"
            "</Placemark>"
        )
    out.append("</Folder><Folder><name>Photo footprints</name><visibility>0</visibility>")
    for p in data["photos"]:
        if not p["footprint"]:
            continue
        coords = " ".join(f"{lon:.7f},{lat:.7f},0" for lat, lon in p["footprint"] + [p["footprint"][0]])
        out.append(
            f"<Placemark><name>{escape(p['rel'])}</name><visibility>0</visibility>"
            "<Style><LineStyle><color>ff888888</color></LineStyle><PolyStyle><color>18ffffff</color>"
            "</PolyStyle></Style>"
            f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{coords}</coordinates></LinearRing>"
            "</outerBoundaryIs></Polygon></Placemark>"
        )
    out.append("</Folder></Document></kml>")
    path.write_text("\n".join(out))


def write_gpx(data: dict, path: Path, min_score: float = 0.0) -> None:
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<gpx version="1.1" creator="isr-tools" xmlns="http://www.topografix.com/GPX/1/1">']
    for p, d in sorted(_rows(data, min_score), key=lambda t: -t[1]["score"]):
        if d["lat"] is None:
            continue
        out.append(f'<wpt lat="{d["lat"]:.7f}" lon="{d["lon"]:.7f}"><name>{escape(d["id"])}</name>'
                   f"<desc>{escape(d['label'])} score {d['score']:.2f}</desc><sym>Flag, Red</sym></wpt>")
    out.append("</gpx>")
    path.write_text("\n".join(out))


def write_all(data: dict, out_dir: Path) -> list[Path]:
    paths = [out_dir / n for n in ("detections.csv", "detections.geojson", "detections.kml", "detections.gpx")]
    write_csv(data, paths[0])
    write_geojson(data, paths[1])
    write_kml(data, paths[2])
    write_gpx(data, paths[3])
    return paths
