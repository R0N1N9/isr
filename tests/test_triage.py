import json

import pytest
from PIL import ImageDraw

from isr.camera import read_photo_meta
from isr.geo import haversine_m, offset_latlon
from isr.triage.detectors import (
    ColorTargetDetector,
    RXAnomalyDetector,
    SizeFilter,
    WorkImage,
    merge_detections,
    parse_color,
)
from isr.triage.export import write_all
from isr.triage.pipeline import TriageConfig, find_photos, run_triage
from isr.triage.report import write_report

from synth import MAVIC3_W, MAVIC3_H, draw_person, mavic3_gsd, save_dji_jpeg, terrain

ALT = 60.0
LAT, LON = 39.70, -105.20


@pytest.fixture(scope="module")
def card(tmp_path_factory):
    """A fake SD card: 4 photos, a red-jacket person in one, an orange lying person + blue tarp in another."""
    root = tmp_path_factory.mktemp("card")
    d = root / "DCIM" / "DJI_001"
    g = mavic3_gsd(ALT)
    truth = []
    for i in range(4):
        lat, lon = offset_latlon(LAT, LON, 0, i * 90)
        im = terrain(MAVIC3_W, MAVIC3_H, seed=i + 10)
        if i == 1:
            truth.append(("red", d / "DJI_0002.JPG", draw_person(im, 3100, 1800, g)))
        if i == 2:
            truth.append(("orange", d / "DJI_0003.JPG",
                          draw_person(im, 1200, 2600, g, jacket=(240, 120, 10), lying=True)))
            box = (4000, 900, 4000 + 2 / g, 900 + 3 / g)
            ImageDraw.Draw(im).rectangle(box, fill=(25, 80, 200))
            truth.append(("tarp", d / "DJI_0003.JPG", box))
        save_dji_jpeg(im, d / f"DJI_{i + 1:04d}.JPG", lat, lon, ALT, yaw=0.0, when=f"2026:11:16 10:0{i}:00")
    (root / "DCIM" / "notes.txt").write_text("not a photo")
    return root, truth


def _hit(dets, box):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(d.x0 - 5 <= cx <= d.x1 + 5 and d.y0 - 5 <= cy <= d.y1 + 5 for d in dets)


def test_find_photos(card):
    root, _ = card
    assert len(find_photos(root)) == 4


def test_rx_finds_all_targets_without_false_alarms(card):
    root, truth = card
    sizes = SizeFilter()
    for path in sorted(find_photos(root)):
        m = read_photo_meta(path)
        dets = RXAnomalyDetector().detect(WorkImage.load(path, 2000, m.gsd_m()), sizes)
        expected = [t for t in truth if t[1] == path]
        for _, _, box in expected:
            assert _hit(dets, box), f"missed target in {path.name}"
        strong_extra = [d for d in dets if d.score > 0.5 and not any(_hit([d], b) for _, _, b in expected)]
        assert len(strong_extra) <= 1, f"{len(strong_extra)} strong false alarms in {path.name}"


def test_color_target_only_matches_requested_colour(card):
    root, truth = card
    path = next(t[1] for t in truth if t[0] == "red")
    m = read_photo_meta(path)
    w = WorkImage.load(path, 2000, m.gsd_m())
    red = ColorTargetDetector(["red"]).detect(w, SizeFilter())
    blue = ColorTargetDetector(["blue"]).detect(w, SizeFilter())
    assert _hit(red, truth[0][2])
    assert not _hit(blue, truth[0][2])


def test_parse_color():
    assert parse_color("RED")[0] == "red"
    assert parse_color("#1e90ff")[0] == "#1e90ff"
    with pytest.raises(ValueError):
        parse_color("#808080")
    with pytest.raises(ValueError):
        parse_color("chartreuse-ish")


def test_merge_combines_detectors():
    from isr.triage.detectors import Detection
    a = Detection(10, 10, 20, 20, 0.6, ["anomaly"], "a")
    b = Detection(12, 12, 22, 22, 0.7, ["color:red"], "b")
    c = Detection(500, 500, 510, 510, 0.5, ["anomaly"], "c")
    m = merge_detections([a, b, c])
    assert len(m) == 2
    top = max(m, key=lambda d: d.score)
    assert set(top.kinds) == {"anomaly", "color:red"}
    assert top.score > 0.7


def test_end_to_end(card, tmp_path):
    root, truth = card
    out = tmp_path / "out"
    data = run_triage(root, out, TriageConfig(colors=["red"], workers=2), log=lambda *a: None)
    write_all(data, out)
    report = write_report(data, out)

    s = data["summary"]
    assert s["photos"] == 4 and s["photos_geolocatable"] == 4
    dets = [(p, d) for p in data["photos"] for d in p["detections"]]
    assert 3 <= len(dets) <= 6

    # red-jacket person: top ranked, flagged by both detectors, located within 1.5 m
    p, top = max(dets, key=lambda t: t[1]["score"])
    assert p["rel"].endswith("DJI_0002.JPG")
    assert "color:red" in top["kinds"] and "anomaly" in top["kinds"]
    g = mavic3_gsd(ALT)
    x0, y0, x1, y1 = truth[0][2]
    e = ((x0 + x1) / 2 - MAVIC3_W / 2) * g
    n = -((y0 + y1) / 2 - MAVIC3_H / 2) * g
    exp_lat, exp_lon = offset_latlon(*offset_latlon(LAT, LON, 0, 90), n, e)
    assert haversine_m(top["lat"], top["lon"], exp_lat, exp_lon) < 1.5

    for _, d in dets:
        assert (out / d["crop"]).exists()
    for f in ("results.json", "detections.csv", "detections.geojson", "detections.kml", "detections.gpx"):
        assert (out / f).stat().st_size > 0
    html = report.read_text()
    assert "window.DATA = " in html and "</script>" in html
    gj = json.loads((out / "detections.geojson").read_text())
    assert any(f["geometry"]["type"] == "Point" for f in gj["features"])
    # nothing written to the card
    assert len(list(root.rglob("*"))) == 7  # 2 folders, 4 photos, notes.txt
