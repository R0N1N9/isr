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


def test_survey_command_and_default_branding(tmp_path, capsys):
    from isr.cli import survey_main

    assert M.new_model("x", (0, 0), (0, 0, 1, 1))["company"] == "Ronin Sec"
    demo.write_obliques(tmp_path / "shots")
    assert survey_main(["new", str(tmp_path / "shots"), "-o", str(tmp_path / "p"), "--no-edit", "--no-detect"]) == 0
    assert M.load(tmp_path / "p" / "site.json")["detections"] == []
    assert survey_main(["new", str(tmp_path / "shots"), "-o", str(tmp_path / "p"), "--no-edit", "-c", "notacolour"]) == 2
    assert survey_main(["report", str(tmp_path / "p")]) == 0
    assert "Ronin Sec" in (tmp_path / "p" / "report.html").read_text()
    assert survey_main(["report", str(tmp_path / "missing")]) == 2


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
    assert kinds.count("nadir") == 1 and kinds.count("oblique") == 4 and kinds.count("ground") == 6
    assert model["basemap"]["straight_down"] == 1 and model["basemap"]["angled"] == 4
    front = next(p for p in model["photos"] if "front" in p["name"])
    # walk_01 was taken at (-2, -20.5) on the lot, i.e. (-5, -16.5) from the drone, facing north
    assert (front["x"], front["y"]) == pytest.approx((-5.0, -16.5), abs=0.3)
    assert front["heading"] == pytest.approx(0, abs=0.1)
    assert all((out / p["file"]).exists() and (out / p["thumb"]).exists() for p in model["photos"])
    meta = read_photo_meta(out / "source_photos" / "walk_04_back_garden.jpg")
    assert meta.img_direction == pytest.approx(180) and classify(meta) == "ground"


def _car_on_map(out, bm):
    im = np.asarray(Image.open(out / bm["file"]).convert("RGB"), np.int32)
    x0, y0, x1, y1 = bm["bounds"]
    blue = (im[..., 2] > im[..., 0] + 40) & (im[..., 2] > im[..., 1] + 30)
    jj, ii = np.nonzero(blue)
    assert len(jj) > 200
    cx = x0 + ii.mean() / bm["px_per_m"]
    return cx, y1 - jj.mean() / bm["px_per_m"]


def test_basemap_is_georeferenced(project):
    """The blue car painted at a known spot on the lot must land on the same spot of the stitched map."""
    out, model = project
    car = (14.15 - demo.DRONE[0], -14.35 - demo.DRONE[1])  # centre of the car, relative to the drone
    assert _car_on_map(out, model["basemap"]) == pytest.approx(car, abs=0.6)


def test_basemap_from_hand_flown_angled_shots_only(tmp_path):
    """No straight-down frame at all: four angled shots circling the house still give a usable, placed map."""
    demo.write_obliques(tmp_path / "orbit")
    out = tmp_path / "proj"
    m = run_survey(tmp_path / "orbit", out, log=lambda *a: None)
    bm = m["basemap"]
    assert bm and bm["straight_down"] == 0 and bm["angled"] == 4
    assert [p["kind"] for p in m["photos"]] == ["oblique"] * 4 and all(p["footprint"] for p in m["photos"])
    # the project origin is the middle of the four drone positions: (0, 0.5) on the lot
    assert _car_on_map(out, bm) == pytest.approx((14.15 - 0.0, -14.35 - 0.5), abs=1.0)
    im = np.asarray(Image.open(out / bm["file"]).convert("RGB"), np.int32)
    j, i = int((bm["bounds"][3] - 0) * bm["px_per_m"]), int((0 - bm["bounds"][0]) * bm["px_per_m"])
    assert tuple(im[j, i]) != (24, 26, 31)  # the house in the middle is covered, not left blank
    # a shot aimed near the horizon shows too little ground and is placed but not mapped
    from isr.camera import PhotoMeta
    from isr.site.survey import ground_quad
    flat = PhotoMeta(Path("x"), 1320, 989, lat=51.0, lon=-1.0, rel_alt_m=30, gimbal_yaw=0, gimbal_pitch=-15)
    assert ground_quad(flat, 51.0, -1.0) is None


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


def test_detector_finds_the_person_and_places_them_on_the_plan(project, result):
    """SAR triage inside the survey: the red jacket in the back corner, located and put in site context."""
    out, model = project
    (px, py), _ = demo.PEOPLE[0]
    want = (px - demo.DRONE[0], py - demo.DRONE[1])
    d = min(result["detections"], key=lambda d: (d["x"] - want[0]) ** 2 + (d["y"] - want[1]) ** 2)
    assert (d["x"], d["y"]) == pytest.approx(want, abs=1.5)
    assert d["code"] == "D1" and d["status"] == "open" and d["seen"] >= 3  # same person in the angled shots: one hit
    assert d["inside"] and d["unseen"] and not d["on_camera"] and "nobody can see" in d["context"]
    assert (out / d["crop"]).exists()
    assert not any(f["category"] == "Seen in photos" for f in result["findings"])  # nothing confirmed yet

    m = json.loads(json.dumps(model))
    for x in m["detections"]:
        x.update(status="confirmed", type="person", note="by the shed") if x["code"] == "D1" else x.update(status="dismissed")
    an = analyse(m)
    assert [x["code"] for x in an.result["detections"]] == ["D1"]
    f = next(f for f in an.result["findings"] if f["category"] == "Seen in photos")
    assert f["severity"] == "medium" and "person" in f["title"] and "by the shed" in f["detail"]
    html = write_report(m, out, an).read_text()
    assert "Seen during the survey" in html and "by the shed" in html and d["crop"] in html
    assert "seen during survey" in (out / "assessment.csv").read_text()
    # a re-survey keeps what the reviewer decided
    M.save(m, out / "site.json")
    again = run_survey(out / "source_photos", out, log=lambda *a: None, detect=True)
    kept = next(x for x in again["detections"] if x["code"] == "D1")
    assert kept["status"] == "confirmed" and kept["type"] == "person" and kept["note"] == "by the shed"
    assert sum(x["status"] == "dismissed" for x in again["detections"]) == len(again["detections"]) - 1
    M.save(model, out / "site.json")


def test_object_against_a_fence_is_flagged_when_confirmed():
    fence = {"kind": "barrier", "subtype": "fence", "pts": [[-20, 20], [20, 20]]}
    m = _model([LOT, fence])
    m["detections"] = [{"x": 5, "y": 19.2, "lat": 0, "lon": 0, "status": "confirmed", "type": "object", "code": "D1",
                        "label": "x", "score": 0.8, "seen": 1, "crop": "c.jpg"},
                       {"x": 0, "y": 0, "status": "dismissed"}, {"x": "bad"}]
    r = analyse(m).result
    assert len(r["detections"]) == 1 and r["detections"][0]["at_barrier"]
    f = next(f for f in r["findings"] if f["category"] == "Seen in photos")
    assert f["severity"] == "medium" and "climb" in f["action"]


def test_guard_plan_for_each_headcount(project, result):
    out, model = project
    g = result["guards"]
    assert [sc["guards"] for sc in g["scenarios"]] == [1, 2, 3] and not g["unreachable"]
    posts = {p["id"]: p for p in g["posts"]}
    # the best post overlooks the front: it sees the driveway and front door but not the hidden doors
    p1 = posts["P1"]
    assert {"Driveway", "Front door"} <= set(p1["sees"]) and {"Side door", "Back door"} <= set(p1["misses"])
    assert p1["pros"] and p1["cons"] and p1["suggestion"] and p1["sees_pct"] >= 40
    assert {"Side door", "Back door"} <= set(posts["P2"]["sees"])  # the second post covers what the first cannot
    one, two, three = (sc["options"] for sc in g["scenarios"])
    # one officer: walk, visiting every door, with a static post offered as the alternative
    assert one[0]["recommended"] and one[0]["rovers"] == 1 and not one[0]["statics"]
    names = [st["name"] for st in one[0]["route"]["stops"]]
    assert {"Side door", "Back door", "Front door", "Driveway"} <= set(names) and len(names) == len(set(names))
    assert one[1]["statics"] == ["P1"] and one[1]["route"] is None and not one[1]["recommended"]
    # two: hold the front, patrol the back; the route only needs to visit what the post cannot see
    assert two[0]["statics"] == ["P1"] and two[0]["rovers"] == 1
    assert two[0]["route"]["length_m"] < one[0]["route"]["length_m"]
    assert "Driveway" not in [st["name"] for st in two[0]["route"]["stops"]]
    assert two[1]["rovers"] == 2 and two[1]["route"]["revisit_min"] == pytest.approx(one[0]["route"]["revisit_min"] / 2, abs=0.1)
    # three: a second post earns its keep, and more of the grounds is watched all the time
    assert three[0]["statics"] == ["P1", "P2"] and three[0]["guard_pct"] > two[0]["guard_pct"] + 15
    for n, opts in ((1, one), (2, two), (3, three)):
        for o in opts:
            assert o["pros"] and o["cons"] and len(o["statics"]) + o["rovers"] == n
    # the loop is walkable: it never cuts through a building
    an = analyse(json.loads(json.dumps(model)))
    for x, y in one[0]["route"]["line"]:
        j, i = an.sc.cell_of(x, y)
        assert not an.sc.building_mask[j, i]
    assert one[0]["route"]["lap_min"] == pytest.approx(
        one[0]["route"]["length_m"] / 1.2 / 60 + 20 * len(names) / 60, abs=0.1)


def test_guard_plan_is_off_by_default_and_respects_barriers():
    house = {"kind": "building", "pts": _box(-6, -4, 6, 4)}
    wall = {"kind": "barrier", "subtype": "wall", "pts": [[-20, 8], [20, 8]]}  # cuts the lot in two, no gate
    doors = [{"kind": "access", "subtype": "door", "name": "Front", "pts": [[0, -4]]},
             {"kind": "access", "subtype": "door", "name": "Back", "pts": [[0, 4]]},
             {"kind": "access", "subtype": "door", "name": "Far side", "pts": [[0, 14]]}]
    m = _model([LOT, house, wall, *doors])
    assert analyse(m).result["guards"] is None
    m["settings"]["guards"] = 2
    g = analyse(m).result["guards"]
    # no way through the wall: what lies beyond it is left off the route and reported
    assert "Far side" in g["unreachable"] and not {"Front", "Back"} & set(g["unreachable"])
    stops = [s["name"] for s in g["scenarios"][0]["options"][0]["route"]["stops"]]
    assert {"Front", "Back"} <= set(stops) and "Far side" not in stops
    # put a gate in the wall and the patrol goes through it
    m["features"].append({"kind": "access", "subtype": "gate", "name": "Wall gate", "pts": [[10, 8]]})
    g = analyse(m).result["guards"]
    assert not g["unreachable"]
    assert "Far side" in [s["name"] for s in g["scenarios"][0]["options"][0]["route"]["stops"]]
    json.dumps(g)


# -- 3-D map: photogrammetry import, measured heights, watch positions ---------------


def test_utm_against_known_values():
    from isr.site import terrain as T

    e, n, zone, north = T.latlon_to_utm(40.0, -105.0)  # on zone 13's central meridian
    assert (zone, north) == (13, True) and e == pytest.approx(500000, abs=1e-3) and n == pytest.approx(4427757.22, abs=0.02)
    assert T.latlon_to_utm(0.0, 3.0)[:2] == pytest.approx((500000, 0), abs=1e-6)
    e, n, zone, north = T.latlon_to_utm(-33.9, 151.2)
    assert zone == 56 and not north and 6_200_000 < n < 6_300_000
    for lat, lon in ((51.752, -1.2577), (-33.9, 151.2), (64.1, -21.9), (1.3, 103.8)):
        assert T.utm_to_latlon(*T.latlon_to_utm(lat, lon)) == pytest.approx((lat, lon), abs=1e-7)


def test_geotiff_import_places_and_measures(tmp_path):
    from isr.site import terrain as T

    lat, lon = 51.752, -1.2577
    E, N, Z, _ = T.latlon_to_utm(lat, lon)
    yy, xx = np.mgrid[0:200, 0:240]
    ground = (100 + 0.02 * xx + 0.01 * yy).astype(np.float32)  # a 4.5% slope
    dsm = ground.copy()
    dsm[80:120, 100:160] += 6.5  # a 30 x 20 m block, 10 m west to 20 m east of the centre
    dsm[:4, :4] = -9999
    T.write_geotiff(tmp_path / "dsm.tif", dsm, E - 60, N + 50, 0.5, Z, nodata=-9999)
    T.write_geotiff(tmp_path / "dtm.tif", ground, E - 60, N + 50, 0.5, Z)
    rgb = np.zeros((400, 480, 3), np.uint8)
    rgb[160:240, 200:320] = (220, 30, 30)
    T.write_geotiff(tmp_path / "ortho.tif", rgb, E - 60, N + 50, 0.25, Z)
    for dtm, tol in ((tmp_path / "dtm.tif", 0.05), (None, 0.4)):  # with a terrain model, and with the ground estimated
        bm, tr = T.import_terrain(tmp_path, (lat, lon), tmp_path / "ortho.tif", tmp_path / "dsm.tif", dtm, log=lambda *a: None)
        h, ok = T.load_heights(tmp_path, tr)
        jj, ii = np.nonzero(h > 3)
        b, p = bm["bounds"], tr["px_per_m"]
        assert (b[0] + ii.min() / p, b[0] + (ii.max() + 1) / p) == pytest.approx((-10, 20), abs=0.6)
        assert (b[3] - (jj.max() + 1) / p, b[3] - jj.min() / p) == pytest.approx((-10, 10), abs=0.7)
        assert float(np.median(h[h > 3])) == pytest.approx(6.5, abs=tol)
        assert float((h[ok & (h < 3)] > 0.6).mean()) < 0.01  # bare sloping ground reads as ground
    im = np.asarray(Image.open(tmp_path / bm["file"]).convert("RGB"), np.int32)
    jj, ii = np.nonzero(im[..., 0] > 150)
    assert (b[0] + ii.mean() / bm["px_per_m"], b[3] - jj.mean() / bm["px_per_m"]) == pytest.approx((5, 0), abs=0.5)
    plain = tmp_path / "plain.tif"
    Image.fromarray(rgb).save(plain)
    with pytest.raises(ValueError, match="no position information"):
        T.import_terrain(tmp_path, (lat, lon), plain)


def test_survey_uses_the_photogrammetry_result(project, result):
    """The demo ships a synthetic OpenDroneMap result: its orthophoto is the base map and its heights are used."""
    out, model = project
    assert model["basemap"]["ortho"] and model["terrain"]["has_dtm"] and model["terrain"]["max_m"] == pytest.approx(7, abs=0.3)
    assert not result["stats"]["measured"] and result["stats"]["has_terrain"]  # no project folder given: plan only
    an = analyse(json.loads(json.dumps(model)), out)
    assert an.result["stats"]["measured"]
    house = next(f for f in model["features"] if f["name"] == "House")
    assert an.result["measured_heights"][house["id"]] == pytest.approx(6.3, abs=0.4)
    drawn = analyse(json.loads(json.dumps(model)), out, use_measured=False)
    assert not drawn.result["stats"]["measured"] and drawn.result["measured_heights"] == {}
    assert drawn.result["stats"]["blind_pct"] == result["stats"]["blind_pct"] != an.result["stats"]["blind_pct"]
    # nobody drew the van on the lawn, but it is in the surface model and blocks sight at its own height
    vx, vy = -16 - demo.DRONE[0], -10.5 - demo.DRONE[1]
    j, i = an.sc.cell_of(vx, vy)
    assert an.sc.H[j, i] == pytest.approx(1.6, abs=0.2) and drawn.sc.H[j, i] == 0
    # switching the setting off is the same as not having it
    m = json.loads(json.dumps(model))
    m["settings"]["use_measured"] = 0
    assert not analyse(m, out).result["stats"]["measured"]
    html = write_report(model, out).read_text()
    assert "From measured heights" in html and "From the plan as drawn" in html
    assert an.overlays_b64()["height"].startswith("data:image/png;base64,")


def test_measured_structure_nobody_drew_blocks_sight(tmp_path):
    from isr.site import terrain as T

    lat, lon = 51.0, -1.0
    E, N, Z, _ = T.latlon_to_utm(lat, lon)
    z = np.full((240, 240), 50, np.float32)
    z[100:104, 60:180] += 2.5  # a wall across the lot, 1 m thick, in the data but not on the plan
    T.write_geotiff(tmp_path / "odm_dem" / "dsm.tif", z, E - 30, N + 30, 0.25, Z)
    T.write_geotiff(tmp_path / "odm_orthophoto" / "odm_orthophoto.tif", np.full((240, 240, 3), 90, np.uint8), E - 30, N + 30, 0.25, Z)
    found = T.find_odm(tmp_path)
    assert set(found) == {"ortho", "dsm"}
    m = _model([LOT, {"kind": "vantage", "name": "Lookout", "pts": [[0, -12]]},
                {"kind": "access", "subtype": "door", "name": "Far door", "pts": [[0, 15]]}])
    m["origin"] = {"lat": lat, "lon": lon}
    m["basemap"], m["terrain"] = T.import_terrain(tmp_path, (lat, lon), found["ortho"], found["dsm"], log=lambda *a: None)
    with_data, plan_only = analyse(m, tmp_path).result["vantages"][0], analyse(m, tmp_path, use_measured=False).result["vantages"][0]
    assert plan_only["sees"] == ["Far door"] and with_data["sees"] == [] and with_data["sees_pct"] < plan_only["sees_pct"] - 20
    # from a rooftop the same lookout sees over it
    m["features"][1]["height"] = 12
    assert analyse(m, tmp_path).result["vantages"][0]["sees"] == ["Far door"]


def test_watch_position_reports_what_it_sees(project):
    out, model = project
    m = json.loads(json.dumps(model))
    m["features"].append({"kind": "vantage", "name": "Back lawn", "pts": [[-9 - demo.DRONE[0] + 3, 18 - demo.DRONE[1] - 4]]})
    an = analyse(m, out)
    v = an.result["vantages"][0]
    assert v["code"] == "V1" and v["inside"] and "Back door" in v["sees"] and "Front door" in v["misses"] and 10 < v["sees_pct"] < 60
    assert an.layers["vantage"].any()
    assert analyse(model, out).result["stats"]["blind_pct"] == an.result["stats"]["blind_pct"]  # it does not change findings
    assert "Watch positions" in write_report(m, out, an).read_text()
    write_report(model, out)


def test_building_a_map_runs_opendronemap(tmp_path, monkeypatch):
    """The container tool is faked: this checks what is asked of it and that its output is picked up."""
    import os
    import stat
    import sys
    from isr.site import terrain as T

    fake = tmp_path / "bin" / "docker"
    fake.parent.mkdir()
    fake.write_text(f"""#!{sys.executable}
import sys, json
from pathlib import Path
args = sys.argv[1:]
work = Path(args[args.index("-v") + 1].split(":")[0])
(work / "args.json").write_text(json.dumps(args))
print("[INFO] pretending to reconstruct", len(list((work / "code" / "images").glob("*"))), "images")
if "--fail" in open(work / "mode").read():
    print("[ERROR] not enough overlap"); sys.exit(1)
from isr.site import demo
demo.write_map(work / "code")
""")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{fake.parent}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).parent.parent))
    src = tmp_path / "photos"
    demo.write_obliques(src)
    (src / "extra").mkdir()
    for i in range(2):
        (src / "extra" / f"copy{i}.JPG").write_bytes((src / "DJI_0002.JPG").read_bytes())
    out = tmp_path / "proj"
    (out / "odm").mkdir(parents=True)
    (out / "odm" / "mode").write_text("ok")
    log = []
    m = run_survey(src, out, log=log.append, build_map=True)
    args = json.loads((out / "odm" / "args.json").read_text())
    assert args[0] == "run" and "opendronemap/odm" in args and {"--dsm", "--dtm"} <= set(args)
    assert m["terrain"] and m["basemap"]["ortho"] and any("pretending" in line for line in log)
    assert len(list((out / "odm" / "code" / "images").iterdir())) == 6
    assert len(run_survey(src, out, log=lambda *a: None)["photos"]) == 6  # ODM's folders are not rescanned as photos
    (out / "odm" / "mode").write_text("--fail")
    with pytest.raises(RuntimeError, match="not enough overlap"):
        T.run_odm(sorted(src.glob("*.JPG")), out / "odm", log=lambda *a: None)
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    with pytest.raises(RuntimeError, match="Docker was not found"):
        T.run_odm([], tmp_path / "x")
    with pytest.raises(SystemExit, match="overlapping drone photos"):
        run_survey(tmp_path / "nowhere", tmp_path / "p2", log=lambda *a: None, build_map=True)


def test_report_and_exports(project):
    out, model = project
    report = write_report(model, out)
    html = report.read_text()
    assert "<script src" not in html and "http://" not in html.replace("http://www.w3.org", "")
    for needle in ("Priority findings", "Side door", "Camera placement", "basemap.jpg", "report_assets/blind.png",
                   "Prepared for", "Example client", "DEMO-001", "Ronin Sec",
                   "Guard posts and patrols", "With 3 guards", "Recommended", "Alternative", "Open and unmarked", ">G1<"):
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
    assert len(again["features"]) == n and again["origin"] == model["origin"] and len(again["photos"]) == 11


def test_walk_only_survey_has_no_basemap(tmp_path):
    src = tmp_path / "walk"
    demo.write_photos(src)
    for p in src.glob("DJI_*.JPG"):
        p.unlink()
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
        assert code == 200 and b"ISRStudio" in page
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
