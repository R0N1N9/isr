"""Site assessment: visibility geometry, the demo property end to end, survey georeferencing, report and server."""

import json
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from isr.camera import read_photo_meta
from isr.site import demo
from isr.site import model as M
from isr.site.analysis import Scene, analyse
from isr.site.report import write_report
from isr.site.server import make_server
from isr.site.survey import classify, run_survey


def _model(features, bounds=(-30, -30, 30, 30)):
    m = M.new_model("t", (51.0, -1.0), bounds)
    m["features"] = features
    return M.normalise(m)


def _box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


LOT = {"kind": "boundary", "pts": _box(-20, -20, 20, 20)}


def _seen(sc, mask, x, y):
    j, i = sc.cell_of(x, y)
    return bool(mask[j, i])


# -- geometry ---------------------------------------------------------------


def test_wall_blocks_eye_level_but_camera_sees_over_low_wall():
    wall = {"kind": "barrier", "subtype": "wall", "height": 1.2, "pts": [[-15, 0], [15, 0]]}
    sc = Scene(_model([LOT, wall]))
    # a camera 3 m up, 4 m south of a 1.2 m wall, looking for someone's chest (1.5 m)
    cam = sc.viewshed(0, -4, 3.0, 30, h_tgt=1.5)
    assert _seen(sc, cam, 0, 10)  # well beyond the wall: sees over it
    # the same wall at 2.5 m hides the ground just behind it but not far beyond
    wall["height"] = 2.5
    sc = Scene(_model([LOT, wall]))
    cam = sc.viewshed(0, -4, 3.0, 30, h_tgt=1.5)
    assert not _seen(sc, cam, 0, 3)
    # sight line drops 0.5 m over 4 m to clear the wall top, so it reaches 1.5 m height 12 m out: y = 8
    assert not _seen(sc, cam, 0, 6.5)
    assert _seen(sc, cam, 0, 10)
    # a person at eye level (1.6 m) behind a 1.8 m hedge is hidden; in front is visible
    hedge = {"kind": "barrier", "subtype": "hedge", "height": 1.8, "pts": [[-15, 0], [15, 0]]}
    sc = Scene(_model([LOT, hedge]))
    eye = sc.viewshed(0, -6, 1.6, 30, h_tgt=1.2)
    assert _seen(sc, eye, 3, -2) and not _seen(sc, eye, 3, 5) and not _seen(sc, eye, -10, 12)


def test_see_through_fence_blocks_movement_not_sight():
    fence = {"kind": "barrier", "subtype": "open_fence", "pts": [[-15, 0], [15, 0]]}
    sc = Scene(_model([LOT, fence]))
    assert _seen(sc, sc.viewshed(0, -6, 1.6, 30), 2, 8)
    j, i = sc.cell_of(0, 0)
    assert sc.solid[j, i] and not sc.domain[j, i]


def test_camera_sector_and_range():
    cam = {"kind": "camera", "pts": [[0, 0]], "bearing": 90, "hfov": 60, "range": 10, "height": 3}
    an = analyse(_model([LOT, cam]))
    sc, mk = an.sc, an.cam_now
    assert _seen(sc, mk, 6, 0) and _seen(sc, mk, 6, 2.5)  # inside the wedge
    assert not _seen(sc, mk, -6, 0) and not _seen(sc, mk, 0, 6)  # behind / to the side
    assert not _seen(sc, mk, 12, 0)  # beyond range
    # area of a 60 degree, 10 m sector is 52 m2
    assert an.result["cameras"][0]["covers_m2"] == pytest.approx(52.4, rel=0.08)


def test_wall_mounted_camera_cannot_see_through_its_building():
    house = {"kind": "building", "pts": _box(-5, -4, 5, 4)}
    cam = {"kind": "camera", "pts": [[0, -4]], "bearing": 180, "hfov": 360, "range": 15}
    an = analyse(_model([LOT, house, cam]))
    assert _seen(an.sc, an.cam_now, 0, -9) and _seen(an.sc, an.cam_now, 8, -8)
    assert not _seen(an.sc, an.cam_now, 0, 9)


def test_perimeter_gap_found():
    fence = {"kind": "barrier", "subtype": "fence",
             "pts": [[-10, -20], [-20, -20], [-20, 20], [20, 20], [20, -20], [-4, -20]]}  # 6 m opening on the south
    per = analyse(_model([LOT, fence])).result["perimeter"]
    gaps = [g for g in per["gaps"] if not g["declared"]]
    assert len(gaps) == 1 and gaps[0]["length_m"] == pytest.approx(6, abs=1.6) and gaps[0]["side"] == "south"
    assert per["length_m"] == pytest.approx(160, abs=0.5)
    gate = {"kind": "access", "subtype": "gate", "pts": [[-7, -20]]}
    per = analyse(_model([LOT, fence, gate])).result["perimeter"]
    assert per["gaps"][0]["declared"] == "Pedestrian gate 1"


def test_empty_and_partial_models_do_not_crash():
    for feats in ([], [LOT], [{"kind": "access", "subtype": "door", "pts": [[0, 0]]}],
                  [{"kind": "street", "pts": [[-20, -25], [20, -25]]}],
                  [{"kind": "camera", "pts": [[0, 0]]}, {"kind": "bogus", "pts": [[0, 0]]},
                   {"kind": "building", "pts": [[0, 0], [1, 1]]}]):
        r = analyse(_model(feats)).result
        json.dumps(r)
        assert "stats" in r


# -- the demo property ---------------------------------------------------------


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    out = tmp_path_factory.mktemp("site")
    return out, demo.build_demo(out, log=lambda *a: None)


@pytest.fixture(scope="module")
def result(project):
    return analyse(project[1]).result


def test_survey_classifies_and_places_photos(project):
    out, model = project
    kinds = [p["kind"] for p in model["photos"]]
    assert kinds.count("nadir") == 1 and kinds.count("ground") == 6
    front = next(p for p in model["photos"] if "front" in p["name"])
    # walk_01 was taken at (-2, -20.5) on the lot, i.e. (-5, -16.5) from the drone, facing north
    assert (front["x"], front["y"]) == pytest.approx((-5.0, -16.5), abs=0.3)
    assert front["heading"] == pytest.approx(0, abs=0.1)
    assert all((out / p["file"]).exists() and (out / p["thumb"]).exists() for p in model["photos"])
    meta = read_photo_meta(out / "source_photos" / "walk_04_back_garden.jpg")
    assert meta.img_direction == pytest.approx(180) and classify(meta) == "ground"


def test_basemap_is_georeferenced(project):
    """The blue car painted at a known spot on the lot must land on the same spot of the stitched map."""
    out, model = project
    bm = model["basemap"]
    im = np.asarray(Image.open(out / bm["file"]).convert("RGB"), np.int32)
    x0, y0, x1, y1 = bm["bounds"]
    blue = (im[..., 2] > im[..., 0] + 40) & (im[..., 2] > im[..., 1] + 30)
    jj, ii = np.nonzero(blue)
    assert len(jj) > 200
    cx = x0 + ii.mean() / bm["px_per_m"]
    cy = y1 - jj.mean() / bm["px_per_m"]
    car = (14.15 - demo.DRONE[0], -14.35 - demo.DRONE[1])  # centre of the car, relative to the drone
    assert (cx, cy) == pytest.approx(car, abs=0.6)


def test_hidden_doors_rank_highest(result):
    acc = {a["name"]: a for a in result["access"]}
    assert acc["Side door"]["priority"] == "high" and acc["Back door"]["priority"] == "high"
    assert not acc["Side door"]["street"] and acc["Side door"]["street_blocker"] == "Front hedge"
    assert acc["Front door"]["camera_now"] and acc["Front door"]["priority"] == "low"
    assert acc["Front door"]["cameras"] == ["Doorbell camera"]
    assert acc["Driveway"]["street"] and not acc["Driveway"]["camera_now"]
    assert [a["code"] for a in result["access"]][:2] == ["A1", "A2"]
    assert {result["access"][0]["name"], result["access"][1]["name"]} == {"Side door", "Back door"}


def test_recommended_cameras_fix_the_worst_first(result):
    st = result["stats"]
    recs = result["recommended"]
    assert 1 <= len(recs) <= 4
    covered = {n for p in recs for n in p["covers"]}
    assert {"Side door", "Back door"} <= covered
    assert st["camera_pct_recommended"] > st["camera_pct_now"] + 25
    assert st["blind_pct_recommended"] < st["blind_pct"]
    assert all(("House" in p["mount"]) or ("Garage" in p["mount"]) or ("Shed" in p["mount"]) for p in recs)


def test_adopting_a_recommendation_delivers_what_it_promised(project, result):
    model = json.loads(json.dumps(project[1]))
    p = next(p for p in result["recommended"] if "Side door" in p["covers"])
    model["features"].append({"kind": "camera", "pts": [[p["x"], p["y"]]], "bearing": p["bearing"], "hfov": p["hfov"],
                              "range": p["range"], "height": p["height"], "status": "planned", "name": "New"})
    acc = {a["name"]: a for a in analyse(model).result["access"]}
    assert acc["Side door"]["camera_planned"] and not acc["Side door"]["camera_now"]


def test_blind_spots_and_blockers(result):
    spots = result["blind_spots"]
    assert spots and spots[0]["severity"] == "high"
    assert {"Side door", "Back door"} & set(spots[0]["near_access"])
    assert "Side shrubs" in spots[0]["hidden_by"]
    assert any("Back trees" in b["hidden_by"] for b in spots)
    hedge = next(b for b in result["blockers"] if b["name"] == "Front hedge")
    assert "Side door" in hedge["hides_access"] and hedge["hides_m2"] > 30
    assert 15 < result["stats"]["blind_pct"] < 45


def test_removing_the_hedge_opens_the_view(project, result):
    model = json.loads(json.dumps(project[1]))
    model["features"] = [f for f in model["features"] if f["name"] not in ("Front hedge", "Side shrubs")]
    r = analyse(model).result
    assert r["stats"]["street_pct"] > result["stats"]["street_pct"] + 5
    assert next(a for a in r["access"] if a["name"] == "Side door")["street"]


def test_perimeter_of_demo(result):
    per = result["perimeter"]
    unmarked = [g for g in per["gaps"] if not g["declared"]]
    assert len(unmarked) == 1 and unmarked[0]["side"] == "east" and 3.5 <= unmarked[0]["length_m"] <= 6
    assert any(g["declared"] == "Driveway" for g in per["gaps"])
    assert any(f["category"] == "Perimeter" and "gap" in f["title"] for f in result["findings"])


def test_photos_are_linked_to_what_they_show(result):
    acc = {a["name"]: a for a in result["access"]}
    assert acc["Back door"]["photos"] and acc["Side door"]["photos"] and acc["Front door"]["photos"]


def test_report_and_exports(project):
    out, model = project
    report = write_report(model, out)
    html = report.read_text()
    assert "<script src" not in html and "http://" not in html.replace("http://www.w3.org", "")
    for needle in ("Priority findings", "Side door", "Camera placement", "basemap.jpg", "report_assets/blind.png"):
        assert needle in html
    gj = json.loads((out / "site.geojson").read_text())
    kinds = {f["properties"]["kind"] for f in gj["features"]}
    assert {"building", "access", "camera", "blind_spot"} <= kinds
    door = next(f for f in gj["features"] if f["properties"]["title"] == "Front door")
    lat, lon = door["geometry"]["coordinates"][1], door["geometry"]["coordinates"][0]
    assert abs(lat - demo.LAT0) < 0.0005 and abs(lon - demo.LON0) < 0.0008
    assert "<kml" in (out / "site.kml").read_text()
    rows = (out / "assessment.csv").read_text().splitlines()
    assert rows[0].startswith("ref,item") and len(rows) > 12
    assert all(Image.open(out / "report_assets" / f"{n}.png").mode == "RGBA" for n in ("cameras", "blind", "proposed"))


def test_resurvey_keeps_the_plan(project, tmp_path):
    out, model = project
    n = len(model["features"])
    again = run_survey(out / "source_photos", out, log=lambda *a: None)
    assert len(again["features"]) == n and again["origin"] == model["origin"] and len(again["photos"]) == 7


def test_walk_only_survey_has_no_basemap(tmp_path):
    src = tmp_path / "walk"
    demo.write_photos(src)
    (src / "DJI_0001.JPG").unlink()
    m = run_survey(src, tmp_path / "proj", log=lambda *a: None)
    assert m["basemap"] is None and len(m["photos"]) == 6
    assert m["bounds"][2] - m["bounds"][0] >= 60
    write_report(m, tmp_path / "proj")  # nothing drawn yet: still produces a page
    assert "Nothing has been drawn" in (tmp_path / "proj" / "report.html").read_text()


def test_editor_server(project):
    out, model = project
    srv = make_server(out, port=0)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    def call(path, body=None, headers=None):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    try:
        code, page = call("/")
        assert code == 200 and b"Site editor" in page
        code, body = call("/api/site")
        site = json.loads(body)
        assert code == 200 and len(site["features"]) == len(model["features"])
        site["features"] = [f for f in site["features"] if f["name"] != "Doorbell camera"]
        code, body = call("/api/analyse", site)
        res = json.loads(body)
        assert code == 200 and res["result"]["stats"]["cameras_existing"] == 0
        assert res["overlays"]["blind_plan"].startswith("data:image/png;base64,")
        assert len(M.load(out / "site.json")["features"]) == len(site["features"])  # saved
        assert call("/files/basemap.jpg")[0] == 200
        assert call("/files/../../etc/passwd")[0] == 404
        assert call("/api/site", headers={"Host": "evil.example"})[0] == 403
        assert call("/api/analyse", site, headers={"Origin": "http://evil.example"})[0] == 403
        assert call("/api/analyse", {"features": []})[0] == 400
        code, body = call("/api/report", site)
        assert code == 200 and (out / "report.html").exists()
    finally:
        srv.shutdown()
        srv.server_close()
        M.save(model, out / "site.json")
