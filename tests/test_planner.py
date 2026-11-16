import json
import math
import re
import xml.dom.minidom
import zipfile

import pytest

from isr.cli import main
from isr.geo import haversine_m, to_local
from isr.planner import mission as M
from isr.planner.patterns import CAMERAS, expanding_square, lawnmower, sector_search

C = (39.70, -105.20)


def _covered(m, pt):
    """Is ground point (lat, lon) inside any nadir photo footprint of mission m?

    Each photo is a rectangle (footprint_w across track x footprint_h along
    track) centred on its waypoint, oriented along the direction of travel.
    """
    fw, fh = m.stats["footprint_w_m"], m.stats["footprint_h_m"]
    wps = m.waypoints
    for i, w in enumerate(wps):
        nxt = wps[i + 1] if i + 1 < len(wps) else wps[i - 1]
        de, dn = to_local(w.lat, w.lon, nxt.lat, nxt.lon)
        hd = math.atan2(de, dn) if (de or dn) else 0.0
        pe, pn = to_local(w.lat, w.lon, *pt)
        along = pe * math.sin(hd) + pn * math.cos(hd)
        across = pe * math.cos(hd) - pn * math.sin(hd)
        if abs(along) <= fh / 2 and abs(across) <= fw / 2:
            return True
    return False


def test_camera_numbers():
    cam = CAMERAS["mavic3"]
    assert cam.gsd_cm(100) == pytest.approx(2.67, abs=0.01)
    fw, fh = cam.footprint_m(100)
    assert fw == pytest.approx(140.8, abs=0.2) and fh == pytest.approx(105.8, abs=0.2)
    assert cam.alt_for_gsd(cam.gsd_cm(73)) == pytest.approx(73)


def test_lawnmower_square():
    poly = [(0, 0), (100, 0), (100, 100), (0, 100)]
    lines = lawnmower(poly, 20)
    assert len(lines) == 5
    assert lines[0][0][1] < lines[0][1][1] and lines[1][0][1] > lines[1][1][1]  # serpentine
    inset = lawnmower(poly, 20, swath=60)
    assert len(inset) == 3
    assert min(seg[0][0] for seg in inset) == pytest.approx(30)


def test_lawnmower_concave():
    u = [(0, 0), (90, 0), (90, 90), (60, 90), (60, 30), (30, 30), (30, 90), (0, 90)]
    lines = lawnmower(u, 10, angle_deg=90)  # east-west lines cross the U's arms
    assert any(abs(seg[0][1] - seg[1][1]) < 1e-6 for seg in lines)
    assert len(lines) > 9  # the arms split some scan lines into two segments


def test_grid_covers_aoi():
    aoi = M.rectangle(C, 300, 200, 25)
    m = M.plan_grid(aoi, alt_m=80)
    assert m.stats["area_ha"] == pytest.approx(6.0, rel=0.01)
    lat0, lon0 = C
    for fx in (0.05, 0.5, 0.95):
        for fy in (0.05, 0.5, 0.95):
            la = aoi[0][0] + (aoi[2][0] - aoi[0][0]) * fx
            lo = aoi[0][1] + (aoi[2][1] - aoi[0][1]) * fy
            assert _covered(m, (la, lo))


def test_search_grid_covers_edges_with_inset_lines():
    aoi = M.rectangle(C, 700, 500)
    m = M.plan_search_grid(aoi, alt_m=60)
    for fx in (0.01, 0.5, 0.99):
        for fy in (0.01, 0.5, 0.99):
            la = aoi[0][0] + (aoi[2][0] - aoi[0][0]) * fx
            lo = aoi[0][1] + (aoi[2][1] - aoi[0][1]) * fy
            assert _covered(m, (la, lo))
    # lines run along the long (700 m, east-west) side; the outer ones are inset
    # from the 500 m sides by roughly half a footprint instead of on the boundary
    n = [to_local(*C, w.lat, w.lon)[1] for w in m.waypoints]
    assert -250 + 10 < min(n) and max(n) < 250 - 10


def test_expanding_square_legs():
    pts = expanding_square(10, 25)
    legs = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])]
    assert legs[:6] == pytest.approx([10, 10, 20, 20, 30, 30])
    assert pts[1] == pytest.approx((0, 10))  # first leg north


def test_sector_returns_to_datum():
    pts = sector_search(100, second_pass=True)
    assert sum(1 for p in pts if math.hypot(*p) < 1e-6) == 7
    assert max(math.hypot(*p) for p in pts) == pytest.approx(100)


def test_split():
    m = M.plan_grid(M.rectangle(C, 1000, 1000), alt_m=60)
    parts = M.split(m, 150)
    assert len(parts) > 1
    assert sum(len(p.waypoints) for p in parts) == len(m.waypoints)
    assert all(len(p.waypoints) <= 150 for p in parts)


def test_load_aoi_formats(tmp_path):
    ring = [(39.7, -105.2), (39.7, -105.19), (39.71, -105.19), (39.71, -105.2)]
    gj = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": {
        "type": "Polygon", "coordinates": [[[lo, la] for la, lo in ring + ring[:1]]]}}]}
    (tmp_path / "a.geojson").write_text(json.dumps(gj))
    coords = " ".join(f"{lo},{la},0" for la, lo in ring + ring[:1])
    (tmp_path / "a.kml").write_text(f'<kml><Placemark><Polygon><outerBoundaryIs><LinearRing><coordinates>{coords}'
                                    "</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark></kml>")
    for f in ("a.geojson", "a.kml"):
        got = M.load_aoi(tmp_path / f)
        assert [c for p in got for c in p] == pytest.approx([c for p in ring for c in p])


def _xml(z, name):
    return xml.dom.minidom.parseString(z.read(name))


def test_wpml_kmz(tmp_path):
    m = M.plan_expanding_square(C, 120, alt_m=50)
    M.write_wpml_kmz(m, tmp_path / "m.kmz")
    with zipfile.ZipFile(tmp_path / "m.kmz") as z:
        assert set(z.namelist()) == {"wpmz/template.kml", "wpmz/waylines.wpml"}
        t = _xml(z, "wpmz/template.kml")
        w = _xml(z, "wpmz/waylines.wpml")
    pms = t.getElementsByTagName("Placemark")
    assert len(pms) == len(m.waypoints) == len(w.getElementsByTagName("Placemark"))
    assert len(t.getElementsByTagName("wpml:actionActuatorFunc")) == 2 * len(m.waypoints)
    first = pms[0].getElementsByTagName("coordinates")[0].firstChild.data.split(",")
    assert float(first[1]) == pytest.approx(C[0]) and float(first[0]) == pytest.approx(C[1])


def test_wpml_template_copies_drone_info(tmp_path):
    tmpl = tmp_path / "djifly.kmz"
    cfg = ("<wpml:missionConfig><wpml:flyToWaylineMode>safely</wpml:flyToWaylineMode>"
           "<wpml:finishAction>goHome</wpml:finishAction><wpml:globalTransitionalSpeed>5</wpml:globalTransitionalSpeed>"
           "<wpml:droneInfo><wpml:droneEnumValue>12345</wpml:droneEnumValue>"
           "<wpml:droneSubEnumValue>7</wpml:droneSubEnumValue></wpml:droneInfo></wpml:missionConfig>")
    with zipfile.ZipFile(tmpl, "w") as z:
        z.writestr("wpmz/template.kml", f'<kml xmlns="http://www.opengis.net/kml/2.2" '
                                        f'xmlns:wpml="http://www.dji.com/wpmz/1.0.6"><Document>{cfg}</Document></kml>')
    m = M.plan_sector(C, 80)
    M.write_wpml_kmz(m, tmp_path / "out.kmz", template=tmpl)
    with zipfile.ZipFile(tmp_path / "out.kmz") as z:
        t = z.read("wpmz/template.kml").decode()
        _xml(z, "wpmz/waylines.wpml")
    assert "12345" in t and "wpmz/1.0.6" in t
    assert re.search(r"<wpml:globalTransitionalSpeed>8.0<", t)


def test_cli_plan_and_altitude_guard(tmp_path, capsys):
    assert main(["plan", "grid", "--center", "39.7,-105.2", "--size", "200x150", "--gsd", "2", "-o",
                 str(tmp_path)]) == 0
    assert (tmp_path / "grid.kmz").exists()
    with pytest.raises(SystemExit):
        main(["plan", "grid", "--center", "39.7,-105.2", "--size", "200x150", "--gsd", "5", "-o", str(tmp_path)])
    assert main(["plan", "calc", "50"]) == 0
    assert "1.33" in capsys.readouterr().out
