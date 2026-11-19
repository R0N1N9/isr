"""ISRStudio, the combined app: three sections, jobs in a workspace, reports for each."""

import json
import threading
import time
import urllib.request
from pathlib import Path

import pytest

import synth
from isr import app as A
from isr.camera import read_photo_meta as read_meta
from isr.site import demo


@pytest.fixture(scope="module")
def studio(tmp_path_factory):
    root = tmp_path_factory.mktemp("studio")
    photos = root / "card"
    for i in range(2):
        im = synth.terrain(1760, 1320, seed=i)
        g = synth.mavic3_gsd(50) * 3
        synth.draw_person(im, 600, 500, g)
        synth.save_dji_jpeg(im, photos / f"DJI_{i:04d}.JPG", 39.70, -105.20 + i * 0.0006, 50)
    demo.write_obliques(root / "site")
    srv = A.make_app(root / "ws", port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def call(path, body=None, headers=None, raw=False):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req) as r:
                data = r.read()
                return r.status, data if raw else json.loads(data)
        except urllib.error.HTTPError as e:
            return e.code, e.read() if raw else json.loads(e.read() or b"{}")

    def run(section, **opts):
        code, t = call("/api/new", {"section": section, **opts})
        assert code == 200, t
        for _ in range(600):
            s = call("/api/task/" + t["task"])[1]
            if s["state"] != "running":
                return s
            time.sleep(0.2)
        raise AssertionError("job did not finish")

    yield {"root": root, "photos": photos, "call": call, "run": run, "base": base}
    srv.shutdown()
    srv.server_close()


def test_home_lists_three_sections(studio):
    code, page = studio["call"]("/", raw=True)
    assert code == 200 and b"ISRStudio" in page
    for word in (b"SAR", b"Survey", b"Art", b"Stakeout", b"New search", b"New survey", b"New stakeout", b"Export all to a folder"):
        assert word in page
    code, home = studio["call"]("/api/home")
    assert set(home["jobs"]) == {"sar", "survey", "stakeout"} and set(home["art"]) == {"items", "busy"}
    assert b"New selection" not in page  # Art is no longer something you start; it fills itself and home["workspace"].endswith("ws")


def test_folder_picker(studio):
    code, d = studio["call"]("/api/ls?path=" + str(studio["photos"]))
    assert d["photos"] == 2 and d["parent"] == str(studio["root"])
    code, d = studio["call"]("/api/ls?path=" + str(studio["root"]))
    assert {"card", "site", "ws"} <= set(d["dirs"])
    assert b'data-default-theme="dark"' in studio["call"]("/", raw=True)[1]
    assert studio["call"]("/api/ls?path=/definitely/not/here")[1]["path"] == str(Path.home().resolve())


def test_dropped_folder_is_found_on_disk(studio):
    """A browser only reveals a dropped folder's name and contents; the server works out where it lives."""
    root, photos = studio["root"], studio["photos"]
    size = (photos / "DJI_0000.JPG").stat().st_size
    assert A.locate("card", [{"path": "DJI_0000.JPG", "size": size}], roots=[root]) == [str(photos)]
    # a same-named folder elsewhere with different contents is not mistaken for it
    decoy = root / "other" / "card"
    decoy.mkdir(parents=True)
    (decoy / "DJI_0000.JPG").write_bytes(b"not the same photo")
    assert A.locate("card", [{"path": "DJI_0000.JPG", "size": size}], roots=[root]) == [str(photos)]
    assert A.locate("card", [{"path": "DJI_0000.JPG", "size": size + 1}, {"path": "../../etc/passwd", "size": 1}],
                    roots=[root]) == []
    # a whole drive dropped: photos are in a sub-folder; a single dropped photo selects its folder
    drive = root / "mnt" / "DJI_SD"
    (drive / "DCIM" / "100MEDIA").mkdir(parents=True)
    (drive / "DCIM" / "100MEDIA" / "DJI_0001.JPG").write_bytes(b"x" * 321)
    assert A.locate("DJI_SD", [{"path": "DCIM/100MEDIA/DJI_0001.JPG", "size": 321}], roots=[root / "mnt"]) == [str(drive)]
    assert A.locate("DJI_SD", [{"path": "DCIM/100MEDIA/DJI_0001.JPG", "size": 321}], roots=[drive]) == [str(drive)]
    assert A.locate("DJI_0001.JPG", [{"path": "DJI_0001.JPG", "size": 321}], is_dir=False, roots=[root / "mnt"]) == \
        [str(drive / "DCIM" / "100MEDIA")]
    assert A.locate("nowhere", [], roots=[root]) == [] and A.locate("..", [], roots=[root]) == []
    code, r = studio["call"]("/api/locate", {"name": "card", "samples": [{"path": "DJI_0000.JPG", "size": size}],
                                             "near": str(root)})
    assert code == 200 and r["paths"][0] == str(photos)


def test_sar_job_and_search_report(studio):
    s = studio["run"]("sar", source=str(studio["photos"]), name="Ridge search", colors=["red"])
    assert s["state"] == "done" and s["url"] == "/sar/ridge-search/"
    job = studio["root"] / "ws" / "sar" / "ridge-search"
    data = json.loads((job / "results.json").read_text())
    assert data["summary"]["detections"] >= 2 and (job / "detections.gpx").exists()
    code, page = studio["call"]("/sar/ridge-search/report.html", raw=True)
    assert code == 200 and b"isr-rep" in page and b'href="/"' in page  # report button and way back home
    # before review: the report lists candidates; after confirming one: only that find
    assert "Candidates awaiting review" in (job / "search_report.html").read_text()
    first = next(d for p in data["photos"] for d in p["detections"])
    code, r = studio["call"]("/sar/ridge-search/api/report", {"status": {first["id"]: "confirmed", "x": "bogus"}})
    assert code == 200 and r["url"] == "/sar/ridge-search/search_report.html"
    html = (job / "search_report.html").read_text()
    assert "Confirmed finds" in html and f'{first["lat"]:.6f}' in html and html.count('class="find"') == 1
    assert studio["call"]("/sar/ridge-search/" + first["crop"], raw=True)[0] == 200
    jobs = studio["call"]("/api/home")[1]["jobs"]["sar"]
    assert jobs[0]["name"] == "Ridge search" and "candidates" in jobs[0]["summary"]


def test_survey_job_opens_in_the_editor(studio):
    s = studio["run"]("survey", source=str(studio["root"] / "site"), name="Mill Lane", detect=False)
    assert s["state"] == "done" and s["url"] == "/survey/mill-lane/"
    code, page = studio["call"]("/survey/mill-lane/", raw=True)
    assert code == 200 and b'"base": "/survey/mill-lane"' in page
    code, site = studio["call"]("/survey/mill-lane/api/site")
    assert site["name"] == "Mill Lane" and site["basemap"] and site["company"] == "Ronin Sec"
    site["features"] = [{"kind": "boundary", "pts": [[-20, -20], [20, -20], [20, 20], [-20, 20]]}]
    code, res = studio["call"]("/survey/mill-lane/api/analyse", site)
    assert code == 200 and res["result"]["stats"]["has_boundary"]
    code, rep = studio["call"]("/survey/mill-lane/api/report", site)
    assert rep["url"] == "/survey/mill-lane/files/report.html"
    assert studio["call"](rep["url"], raw=True)[0] == 200 and studio["call"]("/survey/mill-lane/files/basemap.jpg", raw=True)[0] == 200
    assert "report written" in studio["call"]("/api/home")[1]["jobs"]["survey"][0]["summary"]


def test_sar_demo_finds_all_three_people(studio):
    s = studio["run"]("sar", demo=True)
    assert s["state"] == "done" and s["url"] == "/sar/demo-search/"
    job = studio["root"] / "ws" / "sar" / "demo-search"
    data = json.loads((job / "results.json").read_text())
    assert data["summary"]["photos"] == 6 and data["summary"]["photos_geolocatable"] == 6
    hit = {p["rel"]: [d["label"] for d in p["detections"]] for p in data["photos"] if p["detections"]}
    for photo, colour in (("DJI_0002.JPG", "red"), ("DJI_0004.JPG", "orange"), ("DJI_0005.JPG", "blue")):
        assert any(colour in label for label in hit.get(photo, [])), (photo, hit)
    assert all(d["lat"] is not None for p in data["photos"] for d in p["detections"])


def test_watch_demo_finds_the_weekly_pattern(studio):
    s = studio["run"]("stakeout", demo=True)
    assert s["state"] == "done" and s["url"] == "/stakeout/demo-gate-one-week/"
    job = studio["root"] / "ws" / "stakeout" / "demo-gate-one-week"
    d = json.loads((job / "activity.json").read_text())
    sm = d["summary"]
    assert d["mode"] == "triggered" and d["watching"] == "Front gate" and 6 <= sm["days"] <= 7
    assert 150 <= sm["events"] <= 230 and sm["with_movement"] < sm["photos"]  # empty trigger frames are not counted
    by_hour = d["by_hour"]
    assert max(range(24), key=lambda h: by_hour[h]) in (8, 17) and by_hour[8] > 4 * by_hour[11]
    assert sum(d["by_day"][5:]) < 0.3 * sum(d["by_day"][:5])  # quiet weekend
    ooh = [e for e in d["events"] if e["out_of_hours"]]
    # three late leavers (Mon, Wed, Fri about 21:00) and three visits in the small hours (Tue, Thu, Sat)
    assert sorted(e["weekday"] for e in ooh if e["hour"] < 5) == [1, 3, 5] and len(ooh) == 6
    assert all(e["night"] for e in ooh if e["hour"] < 5)
    kinds = [a["kind"] for a in d["advice"]]
    assert "peak" in kinds and "out_of_hours" in kinds and "pattern" in kinds
    # two blustery days set the camera off on a hedge about fifty times: the weather history gets those set aside
    windy = lambda e: e["weekday"] in (2, 6) and 10 <= e["hour"] < 16  # noqa: E731
    assert sm["set_aside"] >= 40 and sum(windy(e) for e in d["set_aside"]) >= sm["set_aside"] - 4
    assert sum(windy(e) for e in d["events"]) <= 10
    assert any("moves when it is windy" in w for e in d["set_aside"] for w in e["why"])
    assert d["weather"]["gusty_hours"] == 12 and d["weather"]["wet_hours"] == 2 and d["weather"]["wind_zone_pct"] > 1
    assert d["sun"]["sunrise"] < "07:00" and d["sun"]["sunset"] > "16:30" and 30 <= sm["in_darkness"] <= 60
    assert (job / "scene.jpg").exists() and (job / "heat.png").exists() and (job / ooh[0]["frame"]).exists()
    html = (job / "report.html").read_text()
    assert "What this means for guarding" in html and "Out of hours" in html and "<svg" in html
    assert "Set aside as probably not somebody" in html
    assert "movements over" in studio["call"]("/api/home")[1]["jobs"]["stakeout"][0]["summary"]


def test_stakeout_attached_to_a_survey_times_the_guard_plan(studio):
    from isr.site import model as SM
    from isr.site.analysis import analyse
    from isr.site.report import write_report
    from isr.stakeout.pipeline import compact

    job = studio["root"] / "ws" / "stakeout" / "demo-gate-one-week"
    code, act = studio["call"]("/stakeout/demo-gate-one-week/activity.json")  # what the editor fetches to attach it
    assert code == 200 and act == json.loads((job / "activity.json").read_text())
    out = studio["root"] / "surveyed"
    m = demo.build_demo(out, log=lambda *a: None)
    gate = next(f for f in m["features"] if f["name"] == "Driveway")
    m["stakeouts"] = [{"id": "demo-gate-one-week", "name": act["name"], "access": gate["id"], "data": compact(act)},
                      {"id": "junk", "data": {"nope": 1}}]
    SM.normalise(m)
    assert len(m["stakeouts"]) == 1
    an = analyse(m)
    w = an.result["activity"][0]
    assert w["label"] == "Driveway" and w["out_of_hours"] == 6 and "08:00" in w["busy"]
    assert any(f["category"] == "Activity" and "Driveway" in f["title"] for f in an.result["findings"])
    timing = an.result["guards"]["timing"]
    assert any("Driveway: busiest" in t and "Post P1 sees it" in t for t in timing)
    assert any("while closed" in t for t in timing)
    html = write_report(m, out, an).read_text()
    assert "Activity over time" in html and "When to hold posts and time patrols" in html


def test_without_weather_the_hedge_is_counted(studio, tmp_path):
    """The same week judged on the pictures alone: the windy-day false alarms mostly get through."""
    from isr.stakeout.pipeline import run_watch

    src = studio["root"] / "ws" / "_demo_photos" / "stakeout"
    d = run_watch(src, tmp_path / "o", log=lambda *a: None)
    assert d["summary"]["set_aside"] < 10 and d["summary"]["events"] > 195 and d["weather"] is None and d["sun"] is None
    assert sum(e["weekday"] in (2, 6) and 10 <= e["hour"] < 16 for e in d["events"]) > 40


def test_sun_and_weather_history(monkeypatch):
    import io
    import urllib.error
    from datetime import date, datetime
    from isr.stakeout import environment as E

    assert E.sun_elevation(51.5, -0.13, datetime(2026, 6, 21, 13, 0), 1) == pytest.approx(62, abs=0.7)
    assert E.sun_elevation(0, 0, datetime(2026, 3, 20, 12, 7), 0) > 89
    assert E.sun_elevation(51.5, -0.13, datetime(2026, 12, 21, 0, 0), 0) < -50
    rise, sett = E.sun_times(51.5, -0.13, date(2026, 11, 12), 0)  # almanac: 07:13 and 16:15
    assert "07:08" <= rise <= "07:18" and "16:10" <= sett <= "16:20"
    rise, sett = E.sun_times(40.71, -74.0, date(2026, 7, 4), -4)  # New York on daylight time: 05:29 and 20:30
    assert "05:24" <= rise <= "05:34" and "20:25" <= sett <= "20:35"
    assert E.sun_times(78.2, 15.6, date(2026, 6, 21), 2) == (None, None)  # midnight sun
    assert E.guess_utc_offset(-105.2) == -7 and E.guess_utc_offset(-1.26) == 0

    asked = []

    class Reply(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    def opener(req, timeout=0):
        asked.append(req.full_url)
        if "archive-api" in req.full_url:
            raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, io.BytesIO(b'{"error":true,"reason":"out of range"}'))
        return Reply(json.dumps({"utc_offset_seconds": 3600, "timezone": "Europe/Paris", "hourly": {
            "time": ["2026-11-09T00:00", "2026-11-09T01:00", "2026-11-09T02:00"], "precipitation": [0.0, 2.5, None],
            "snowfall": [0, 0, None], "wind_speed_10m": [5, 20, None], "wind_gusts_10m": [9.0, 61.0, None],
            "cloud_cover": [10, 100, None]}}).encode())

    w = E.fetch_weather(48.85837, 2.29448, date(2026, 11, 9), date(2026, 11, 9), opener=opener)
    assert w["utc_offset_h"] == 1 and w["timezone"] == "Europe/Paris" and set(w["hours"]) == {"2026-11-09T00", "2026-11-09T01"}
    assert w["hours"]["2026-11-09T01"] == {"gust": 61.0, "wind": 20, "rain": 2.5, "snow": 0, "cloud": 100}
    # only a position rounded to about a kilometre, and the dates, leave the machine
    assert "latitude=48.86" in asked[0] and "longitude=2.29" in asked[0] and "48.858" not in asked[0]
    assert "start_date=2026-11-09" in asked[0] and "isrstudio" not in asked[0].split("?")[1]
    with pytest.raises(E.WeatherError, match="out of range"):
        E.fetch_weather(48.86, 2.29, date(1990, 1, 1), date(1990, 1, 2), opener=lambda req, timeout=0: opener(
            type("R", (), {"full_url": req.full_url.replace("api.open-meteo", "archive-api.open-meteo")})()))


def test_stakeout_location_and_old_folder_name(studio, tmp_path):
    assert A.parse_location("51.752, -1.2577") == (51.752, -1.2577) and A.parse_location("  ") is None
    assert A.parse_location("39.7° -105.2°") == (39.7, -105.2)
    for bad in ("somewhere", "95, 10", "10"):
        with pytest.raises(ValueError):
            A.parse_location(bad)
    assert studio["call"]("/api/new", {"section": "stakeout", "source": str(studio["photos"]), "location": "nope"})[0] == 400
    (tmp_path / "ws" / "watch" / "old-job").mkdir(parents=True)  # the section used to be called Watch
    A.Studio(tmp_path / "ws")
    assert (tmp_path / "ws" / "stakeout" / "old-job").is_dir() and not (tmp_path / "ws" / "watch").exists()
    assert list(A.SECTIONS) == ["sar", "survey", "stakeout"]


def test_watch_reads_file_times_and_interval_cameras(tmp_path):
    """No EXIF: fall back to the file's time. A camera shooting every few minutes is recognised as such."""
    import os
    from datetime import datetime
    import numpy as np
    from PIL import Image
    from isr.stakeout.pipeline import run_watch, taken_at

    rng = np.random.default_rng(0)
    base = rng.integers(60, 180, (120, 160, 3)).astype(np.uint8)
    t0 = datetime(2026, 11, 10, 9, 0).timestamp()
    for i in range(40):  # one frame every 10 minutes; a bright block crosses the view in frames 12-14
        a = base.copy()
        if 12 <= i <= 14:
            a[50:90, 20 + (i - 12) * 40: 50 + (i - 12) * 40] = 255
        p = tmp_path / "cam" / f"f{i:03d}.jpg"
        p.parent.mkdir(exist_ok=True)
        Image.fromarray(a).resize((640, 480), Image.NEAREST).save(p, quality=90)
        os.utime(p, (t0 + 600 * i, t0 + 600 * i))
    assert taken_at(tmp_path / "cam" / "f006.jpg") == datetime(2026, 11, 10, 10, 0)
    d = run_watch(tmp_path / "cam", tmp_path / "out", closed_from=11, closed_to=12, log=lambda *a: None)
    assert d["mode"] == "interval" and d["summary"]["photos"] == 40 and d["summary"]["with_movement"] == 3
    assert [e["hour"] for e in d["events"]] == [11, 11, 11] and d["summary"]["out_of_hours"] == 3
    assert sum(map(sum, d["cover"])) == 7  # watched 09:00 to 15:30: seven clock hours
    with pytest.raises(SystemExit):
        run_watch(tmp_path / "out" / "frames" / "nothing", tmp_path / "o2", log=lambda *a: None)


def test_errors_and_safety(studio):
    call = studio["call"]
    (studio["root"] / "empty").mkdir(exist_ok=True)
    s = studio["run"]("sar", source=str(studio["root"] / "empty"))
    assert s["state"] == "error" and "No JPEG" in s["error"]
    assert not (studio["root"] / "ws" / "sar" / "empty").exists()  # a failed job leaves nothing behind
    assert call("/api/new", {"section": "sar", "source": "/nope"})[0] == 400
    assert call("/api/new", {"section": "sar", "source": str(studio["photos"]), "colors": ["notacolour"]})[0] == 400
    assert call("/api/new", {"section": "bogus"})[0] == 400
    assert call("/sar/../../etc/passwd", raw=True)[0] == 404
    assert call("/sar/ridge-search/../../../etc/passwd", raw=True)[0] == 404
    assert call("/api/home", headers={"Host": "evil.example"})[0] == 403
    assert call("/api/delete", {"section": "sar", "job": "../survey"})[0] == 404
    assert call("/api/delete", {"section": "sar", "job": "x"}, headers={"Origin": "http://evil.example"})[0] == 403
    assert call("/api/new", {"section": "art", "source": str(studio["photos"])})[0] == 400  # no longer a section


def _art_settled(studio, want=None):
    for _ in range(400):
        a = studio["call"]("/api/home")[1]["art"]
        if not a["busy"] and (want is None or want(a)):
            return a
        time.sleep(0.25)
    raise AssertionError("art search did not finish")


def test_art_is_gathered_from_other_jobs_and_exported(studio):
    """A set with keepers and rejects goes through SAR; the keepers turn up in the gallery, edited, and export."""
    from PIL import Image
    from isr import demos

    card = studio["root"] / "scenic"
    demos.write_art_demo(card)
    s = studio["run"]("sar", source=str(card), name="Coast flight")
    assert s["state"] == "done"
    a = _art_settled(studio, lambda a: any(i["from"] == "Coast flight" for i in a["items"]))
    mine = [i for i in a["items"] if i["from"] == "Coast flight"]
    names = " ".join(i["name"] for i in mine)
    assert 3 <= len(mine) <= 12 and "crop_rows" in names and "coast" in names
    for reject in ("blurred", "overexposed"):  # faults an edit cannot fix never make the gallery
        assert reject not in names
    assert names.count("coast") == 1  # a burst counts once
    assert all(i["score"] >= 66 and i["section"] == "sar" for i in mine)
    code, img = studio["call"](f"/art/{mine[0]['dir']}/{mine[0]['edited']}", raw=True)
    assert code == 200 and img[:2] == b"\xff\xd8"
    assert studio["call"](f"/art/{mine[0]['dir']}/{mine[0]['preview']}", raw=True)[0] == 200
    edited = studio["root"] / "ws" / "art" / mine[0]["dir"] / mine[0]["edited"]
    assert read_meta(edited).lat == pytest.approx(read_meta(Path(mine[0]["original"])).lat)  # metadata survives the edit
    # remove one, then export the rest
    studio["call"]("/api/art/remove", {"id": mine[-1]["id"]})
    left = [i for i in studio["call"]("/api/home")[1]["art"]["items"]]
    assert mine[-1]["id"] not in [i["id"] for i in left]
    code, r = studio["call"]("/api/art/export", {})
    dest = Path(r["path"])
    assert code == 200 and r["count"] == len(left) and dest.parent.name == "art-export"
    files = sorted(p.name for p in dest.glob("*.jpg"))
    assert len(files) == len(left) == len(set(files))
    rows = (dest / "pictures.csv").read_text().splitlines()
    assert rows[0].startswith("file,score,qualities") and len(rows) == len(left) + 1
    with Image.open(dest / files[0]) as im:
        assert im.width >= 1500  # full size, not a preview
    assert len(list(card.glob("*.JPG"))) == 10  # the card itself is untouched


def test_art_keeps_one_of_each_lookalike(studio, tmp_path):
    """The same view shot again an hour later, and the same card run through a second job: one copy each."""
    import shutil
    from PIL import Image
    from isr import art, demos
    from isr.synth import save_dji_jpeg

    before = _art_settled(studio)["items"]
    card = studio["root"] / "scenic"
    again = studio["root"] / "scenic2"
    shutil.copytree(card, again)
    with Image.open(card / "DJI_0003_crop_rows.JPG") as im:  # the crop rows once more, an hour on, a touch darker
        darker = im.convert("RGB").point(lambda v: int(v * 0.93))
    save_dji_jpeg(darker, again / "DJI_0099_crop_rows_later.JPG", demos.LAT, demos.LON, 80, when="2026:11:16 11:10:00")
    assert studio["run"]("sar", source=str(again), name="Coast flight again")["state"] == "done"
    _art_settled(studio, lambda a: (studio["root"] / "ws" / "art" / "sar--coast-flight-again" / "art.json").exists())
    own = json.loads((studio["root"] / "ws" / "art" / "sar--coast-flight-again" / "art.json").read_text())["items"]
    assert sum("crop_rows" in i["name"] for i in own) == 1  # within the job: the later, darker copy lost
    after = _art_settled(studio)["items"]
    assert len(after) == len(before)  # across jobs: nothing is shown twice
    assert sum("crop_rows" in i["name"] for i in after) <= 1 and sum("coast" in i["name"] for i in after) <= 1
    assert art.alike("f" * 64, "f" * 64) and not art.alike("f" * 64, "0" * 64) and not art.alike(None, "f" * 64)


def test_stakeout_photos_are_sampled_for_art(studio):
    a = _art_settled(studio)
    d = studio["root"] / "ws" / "art" / "stakeout--demo-gate-one-week"
    assert d.is_dir() and json.loads((d / "art.json").read_text())["scored"] <= 40
    assert sum(i["section"] == "stakeout" for i in a["items"]) <= 1  # a week of the same gate is one picture at most
    assert not any(i["from"] == "Demo search" for i in a["items"])  # plain search frames do not clear the bar


def test_delete_job(studio):
    s = studio["run"]("sar", source=str(studio["photos"]), name="Throwaway")
    d = studio["root"] / "ws" / "sar" / "throwaway"
    assert s["state"] == "done" and d.is_dir()
    _art_settled(studio)
    assert studio["call"]("/api/delete", {"section": "sar", "job": "throwaway"})[0] == 200 and not d.exists()
    assert studio["photos"].is_dir() and len(list(studio["photos"].iterdir())) == 2  # source photos untouched


def test_launcher_and_cli(tmp_path, monkeypatch, capsys):
    from isr.cli import build_parser, main

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert main(["app", "--install-launcher"]) == 0
    entry = (tmp_path / "applications" / "isr-studio.desktop").read_text()
    assert "Name=ISRStudio" in entry and "-m isr app" in entry and (tmp_path / "icons" / "isr-studio.svg").exists()
    p = build_parser()
    assert p.parse_args(["studio"]).func is p.parse_args(["app"]).func  # "isr studio" opens the app
    assert p.parse_args(["art", "x"]).func is not p.parse_args(["app"]).func
    assert A.slugify("12 Mill Lane (rear)!") == "12-mill-lane-rear" and A.slugify("***") == "job"
