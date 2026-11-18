"""ISR Studio, the combined app: three sections, jobs in a workspace, reports for each."""

import json
import threading
import time
import urllib.request
from pathlib import Path

import pytest

import synth
from isr import app as A
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
    assert code == 200 and b"ISR Studio" in page
    for word in (b"SAR", b"Survey", b"Art", b"New search", b"New survey", b"New selection"):
        assert word in page
    code, home = studio["call"]("/api/home")
    assert set(home["jobs"]) == {"sar", "survey", "art"} and home["workspace"].endswith("ws")


def test_folder_picker(studio):
    code, d = studio["call"]("/api/ls?path=" + str(studio["photos"]))
    assert d["photos"] == 2 and d["parent"] == str(studio["root"])
    code, d = studio["call"]("/api/ls?path=" + str(studio["root"]))
    assert {"card", "site", "ws"} <= set(d["dirs"])
    assert studio["call"]("/api/ls?path=/definitely/not/here")[1]["path"] == str(Path.home().resolve())


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


def test_art_job(studio):
    s = studio["run"]("art", source=str(studio["photos"]), top=2)
    assert s["state"] == "done" and s["url"] == "/art/card/"
    code, page = studio["call"]("/art/card/gallery.html", raw=True)
    assert code == 200 and b"isrbar" in page
    assert "top 2 suggested" in studio["call"]("/api/home")[1]["jobs"]["art"][0]["summary"]


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


def test_delete_job(studio):
    s = studio["run"]("art", source=str(studio["photos"]), name="Throwaway", top=1, enhance=False)
    d = studio["root"] / "ws" / "art" / "throwaway"
    assert s["state"] == "done" and d.is_dir()
    assert studio["call"]("/api/delete", {"section": "art", "job": "throwaway"})[0] == 200 and not d.exists()
    assert studio["photos"].is_dir() and len(list(studio["photos"].iterdir())) == 2  # source photos untouched


def test_launcher_and_cli(tmp_path, monkeypatch, capsys):
    from isr.cli import build_parser, main

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert main(["app", "--install-launcher"]) == 0
    entry = (tmp_path / "applications" / "isr-studio.desktop").read_text()
    assert "Name=ISR Studio" in entry and "-m isr app" in entry and (tmp_path / "icons" / "isr-studio.svg").exists()
    p = build_parser()
    assert p.parse_args(["studio"]).func is p.parse_args(["app"]).func  # "isr studio" opens the app
    assert p.parse_args(["art", "x"]).func is not p.parse_args(["app"]).func
    assert A.slugify("12 Mill Lane (rear)!") == "12-mill-lane-rear" and A.slugify("***") == "job"
