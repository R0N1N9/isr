import io
import json
import math
from datetime import datetime

import numpy as np
import pytest
from PIL import Image

from isr.cli import main
from isr.geo import haversine_m
from isr.track import terrain
from isr.track.model import Grid, Model, build_graph, build_surface, corridor, coverage_mask, probability, segments
from isr.track.pipeline import TrackConfig, run_track
from isr.track.report import write_track_report
from isr.track.sightings import Sighting, check_leg, load_file, parse_point, parse_time
from isr.track.terrain import AreaPack, decode_terrarium

from synth_area import at, make_pack


def _model(pack=None, mode="foot", half=4000, res=25):
    g = Grid.around([at(0, 0)], half, res)
    s = build_surface(g, pack, mode)
    return g, Model(s, build_graph(s))


def _t(g, T, e, n):
    i, j = g.cell(*at(e, n))
    return T[i, j]


# ------------------------------------------------------------------ terrain

def test_terrarium_roundtrip():
    elev = np.array([[0.0, 1234.5], [-50.25, 4321.0]])
    v = elev + 32768
    rgb = np.stack([np.floor(v / 256), np.floor(v % 256), np.round((v % 1) * 256)], -1).astype(np.uint8)
    assert decode_terrarium(rgb) == pytest.approx(elev, abs=0.01)


def test_parsers():
    js = {"elements": [{"type": "way", "tags": {"highway": "path", "name": "T"},
                        "geometry": [{"lat": 1, "lon": 2}, {"lat": 1.1, "lon": 2.1}]},
                       {"type": "way", "tags": {"building": "yes"}, "geometry": [{"lat": 1, "lon": 2}] * 3}]}
    ways = terrain.parse_overpass(js)
    assert len(ways) == 1 and ways[0]["kind"] == "path" and ways[0]["coords"][1] == [1.1, 2.1]
    gj = {"features": [{"type": "Feature", "properties": {"natural": "water"},
                        "geometry": {"type": "Polygon", "coordinates": [[[2, 1], [2.1, 1], [2.1, 1.1], [2, 1]]]}}]}
    w = terrain.parse_geojson(gj)
    assert w[0]["kind"] == "water" and w[0]["closed"]
    xml = ('<osm><node id="1" lat="1" lon="2"/><node id="2" lat="1.1" lon="2.2"/>'
           '<way id="9"><nd ref="1"/><nd ref="2"/><tag k="waterway" v="river"/></way></osm>')
    assert terrain.parse_osm_xml(xml)[0]["kind"] == "river"


def test_pack_roundtrip(tmp_path):
    pack = make_pack()
    pack.save(tmp_path / "a")
    p2 = AreaPack.load(tmp_path / "a")
    assert len(p2.ways) == len(pack.ways) and p2.dem.shape == pack.dem.shape
    lat, lon = at(1500, 0)
    assert p2.elevation(np.array([lat]), np.array([lon]))[0] == pytest.approx(2400, abs=15)


def test_fetch_dem_mosaics_tiles(monkeypatch):
    def fake_get(url, data=None, timeout=60):
        v = np.full((256, 256), 1500.0 + 32768)
        rgb = np.stack([np.floor(v / 256), v % 256, np.zeros_like(v)], -1).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, "PNG")
        return buf.getvalue()
    monkeypatch.setattr(terrain, "_get", fake_get)
    dem, origin = terrain.fetch_dem((39.69, -105.21, 39.71, -105.19), 12, log=lambda *a: None)
    assert dem.shape[0] % 256 == 0 and float(dem.mean()) == pytest.approx(1500)


# ------------------------------------------------------------------ model

def test_flat_offtrail_walking_time():
    g, m = _model()
    T, _ = m.times_from(*at(0, 0))
    # Tobler on flat ground (5.04 km/h) x 0.6 off-trail
    assert _t(g, T, 0, -1000) / 60 == pytest.approx(1000 / (5.04 * 0.6 / 3.6) / 60, rel=0.03)
    assert _t(g, T, 707, 707) / 60 == pytest.approx(1000 / (5.04 * 0.6 / 3.6) / 60, rel=0.09)


def test_river_is_a_barrier_except_at_bridge():
    g, m = _model(make_pack())
    T, _ = m.times_from(*at(-500, 1500))  # east bank, north of the bridge road
    near_other_bank = _t(g, T, -1500, 1500)  # 1 km straight west, across the river
    # must walk ~1.5 km south to the bridge and back: far longer than 1 km off-trail (19.8 min)
    assert near_other_bank / 60 > 40


def test_uphill_slower_than_downhill():
    g, m = _model(make_pack(ways=False))
    up, _ = m.times_from(*at(800, -2000))
    down, _ = m.times_from(*at(1500, -2000))
    assert _t(g, up, 1500, -2000) > _t(g, down, 800, -2000) * 1.15


def test_vehicle_uses_roads():
    g, m = _model(make_pack(), mode="vehicle")
    T, _ = m.times_from(*at(0, 0))
    on_road = _t(g, T, 3000, 3000)  # 3 km east + 3 km north on roads
    assert on_road / 60 < 10
    off_road = _t(g, T, -500, 3000)  # nowhere near a road (west of the trail, east of the river)
    assert off_road > on_road


def test_corridor_prism():
    g, m = _model(make_pack())
    a, b = at(0, -500), at(600, 1800)
    slack, min_t, best = corridor(m, a, b, 70 * 60)
    assert 0 < min_t < 70 * 60
    assert len(best) > 10
    for la, lo in best[1:-1]:
        i, j = g.cell(la, lo)
        assert np.isfinite(slack[i, j])
    i, j = g.cell(*at(-3500, 3500))  # far away: cannot be in the prism
    assert np.isnan(slack[i, j])


def test_probability_properties():
    g, m = _model(make_pack(ways=False), half=5000, res=40)
    heading = 0.0  # north
    p, t = probability(m, at(0, 0), 45 * 60, heading=heading, heading_weight=1.5)
    assert p.sum() == pytest.approx(1.0)
    assert (p[~(np.isfinite(t) & (t <= 45 * 60))] == 0).all()
    ii, jj = np.mgrid[0:g.ny, 0:g.nx]
    ci, cj = g.cell(*at(0, 0))
    assert p[ii < ci].sum() > 2 * p[ii > ci].sum()  # more mass ahead (north) than behind


def test_coverage_moves_segments():
    g, m = _model(None, half=3000, res=40)
    p, _ = probability(m, at(0, 0), 30 * 60)
    segs = segments(g, p, p, 250, top=3)
    cov = coverage_mask(g, [segs[0]["corners"]])
    segs2 = segments(g, p * (1 - 0.9 * cov), p, 250, top=3)
    assert segs2[0]["corners"] != segs[0]["corners"]


# ------------------------------------------------------------------ sightings

def test_parse_time_formats():
    assert parse_time("2026:11:16 10:04:00") == datetime(2026, 11, 16, 10, 4)
    assert parse_time("2026-11-16T10:04") == datetime(2026, 11, 16, 10, 4)
    assert parse_time("2026-11-16T10:04:00Z").tzinfo is None
    with pytest.raises(ValueError):
        parse_time("yesterday")


def test_load_csv_and_gpx(tmp_path):
    (tmp_path / "c.csv").write_text("id,score,label,lat,lon,photo,taken_at\n"
                                    "a-1,0.9,red,39.7,-105.2,x.jpg,2026-11-16T10:00:00\n")
    s = load_file(tmp_path / "c.csv")
    assert s[0].lat == 39.7 and s[0].time == datetime(2026, 11, 16, 10)
    (tmp_path / "w.gpx").write_text('<gpx xmlns="http://www.topografix.com/GPX/1/1"><wpt lat="39.7" lon="-105.2">'
                                    '<name>seen</name><time>2026-11-16T10:00:00</time></wpt></gpx>')
    g = load_file(tmp_path / "w.gpx")
    assert g[0].label == "seen" and g[0].time == datetime(2026, 11, 16, 10)


def test_leg_checks():
    a = Sighting(*at(0, 0), datetime(2026, 1, 1, 10), "A")
    far = Sighting(*at(0, 10000), datetime(2026, 1, 1, 10, 10), "B")
    assert check_leg(a, far, "foot", 3600)["level"] == "bad"
    assert check_leg(a, far, "vehicle", 400)["level"] == "good"
    near = Sighting(*at(0, 1000), datetime(2026, 1, 1, 10, 15), "C")
    assert check_leg(a, near, "foot", 20 * 60)["level"] == "warn"  # 20 min needed, 15 available
    assert check_leg(a, near, "foot", 10 * 60)["level"] == "good"


# ------------------------------------------------------------------ end to end

def test_run_track_and_report(tmp_path):
    sights = [parse_point(f"{at(0, -500)[0]},{at(0, -500)[1]},2026-11-16T09:00", "LKP"),
              parse_point(f"{at(600, 1800)[0]},{at(600, 1800)[1]},2026-11-16T10:10", "Sighting")]
    cfg = TrackConfig(at=datetime(2026, 11, 16, 11, 0), max_cells=320)
    data = run_track(sights, make_pack(), cfg, tmp_path, log=lambda *a: None)
    report = write_track_report(data, tmp_path)
    s = data["summary"]
    assert s["elapsed_min"] == 50 and s["heading_deg"] is not None
    assert data["legs"][0]["level"] == "good"
    assert len(data["segments"]) == 8 and data["segments"][0]["rank"] == 1
    # the most likely segment is within reach of the last sighting and roughly ahead of it
    c = data["segments"][0]["corners"]
    la, lo = sum(x[0] for x in c) / 4, sum(x[1] for x in c) / 4
    assert haversine_m(la, lo, *at(600, 1800)) < 50 / 60 * 6000
    for f in ("track.html", "track.kml", "track.gpx", "track.json", "layers/probability.png", "layers/reach.png",
              "layers/corridor.png", "layers/hillshade.png", "segments/segment_01.kml"):
        assert (tmp_path / f).exists(), f
    assert "window.DATA = " in report.read_text()


def test_cli_track(tmp_path):
    make_pack().save(tmp_path / "area")
    p = at(0, 0)
    assert main(["track", "predict", "--area", str(tmp_path / "area"), "--lkp", f"{p[0]},{p[1]},2026-11-16T09:00",
                 "--hours", "0.5", "--mode", "vehicle", "-o", str(tmp_path / "out")]) == 0
    js = json.loads((tmp_path / "out" / "track.json").read_text())
    assert js["summary"]["mode"] == "vehicle" and js["summary"]["reach_max_km"] > 5
