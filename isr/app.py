"""ISRStudio: one window for all four jobs.

* **SAR**: scan photos for missing persons or targets of interest.
* **Survey**: assess the security posture of a site.
* **Art**: find the photos worth publishing or selling.
* **Watch**: read a stakeout camera's photos for the site's pattern of activity.

It is a local web app: ``isrstudio`` starts a server on 127.0.0.1 and opens
the browser. You pick a folder of photos, it runs the job in the background,
and each job is kept in the workspace folder (``~/ISR-Studio`` by default) as
``<section>/<job>/``, complete with its report, so it can be reopened later or
copied to a USB stick. Nothing leaves the machine.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .site.server import LocalHandler, bind, guarded, project_get, project_post, run
from .ui import APP_NAME, ICON_SVG, LOGO, icon, page, theme_button

SECTIONS = {
    "sar": {"title": "SAR", "long": "Search and rescue", "main": "report.html", "icon": "target", "color": "#12a150",
            "blurb": "Scan every photo for a missing person or a target of interest. Hits are ranked, placed on "
                     "the map, and written up as a search report with coordinates for ground teams.",
            "new": "New search"},
    "survey": {"title": "Survey", "long": "Site security posture", "main": "site.json", "icon": "map",
               "color": "#5b5bf7",
               "blurb": "Trace a site over its aerial photo and get sightlines, blind spots, camera placement, "
                        "access points and perimeter as a report for the client.",
               "new": "New survey"},
    "art": {"title": "Art", "long": "Stock and portfolio picks", "main": "gallery.html", "icon": "image",
            "color": "#e5484d",
            "blurb": "Rank a card of photos for stock libraries, prints or a portfolio, group near-duplicates, and "
                     "auto-edit the best frames.",
            "new": "New selection"},
    "watch": {"title": "Watch", "long": "Stakeout: pattern of activity", "main": "report.html", "icon": "clock",
              "color": "#e08a00",
              "blurb": "Leave a camera on a gate or approach for a day or a week, then drop its photos here. See when "
                       "the place is busy, dead, or visited out of hours, and time guard posts and patrols to match.",
              "new": "New watch"},
}
SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,60}$")
EXTS = (".jpg", ".jpeg")


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40]
    return s or "job"


class Studio:
    """The workspace on disk plus the jobs currently running."""

    def __init__(self, workspace: Path):
        self.ws = Path(workspace).expanduser().resolve()
        for s in SECTIONS:
            (self.ws / s).mkdir(parents=True, exist_ok=True)
        self.tasks: dict[str, dict] = {}

    # -- jobs on disk ----------------------------------------------------------

    def job_dir(self, section: str, job: str) -> Path | None:
        if section not in SECTIONS or not SLUG.match(job or ""):
            return None
        d = self.ws / section / job
        return d if d.is_dir() else None

    def jobs(self, section: str) -> list[dict]:
        out = []
        for d in (self.ws / section).iterdir():
            if not d.is_dir() or not SLUG.match(d.name) or not (d / SECTIONS[section]["main"]).exists():
                continue
            meta = {}
            try:
                meta = json.loads((d / "job.json").read_text())
            except (OSError, ValueError):
                pass
            info = {"job": d.name, "name": meta.get("name") or d.name, "created": meta.get("created") or "",
                    "source": meta.get("source") or "", "summary": "", "url": f"/{section}/{d.name}/"}
            try:
                info["summary"] = self._summary(section, d)
            except Exception:  # a half-written or hand-edited job still gets listed
                pass
            out.append(info)
        return sorted(out, key=lambda j: j["created"], reverse=True)

    @staticmethod
    def _summary(section: str, d: Path) -> str:
        if section == "sar":
            s = json.loads((d / "results.json").read_text())["summary"]
            return f"{s['photos']} photos · {s['detections']} candidates"
        if section == "art":
            s = json.loads((d / "studio.json").read_text())["summary"]
            return f"{s['photos']} photos · top {s['top']} suggested"
        if section == "watch":
            s = json.loads((d / "activity.json").read_text())["summary"]
            return f"{s['photos']} photos · {s['events']} movements over {s['days']:g} days"
        m = json.loads((d / "site.json").read_text())
        n = len(m.get("features") or [])
        return f"{len(m.get('photos') or [])} photos · " + (f"{n} items drawn" if n else "plan not drawn yet") + \
            (" · report written" if (d / "report.html").exists() else "")

    def new_dir(self, section: str, name: str) -> Path:
        base = slugify(name)
        slug, n = base, 2
        while (self.ws / section / slug).exists():
            slug, n = f"{base}-{n}", n + 1
        d = self.ws / section / slug
        d.mkdir(parents=True)
        return d

    def delete(self, section: str, job: str) -> bool:
        d = self.job_dir(section, job)
        if d is None or self.ws not in d.resolve().parents:
            return False
        shutil.rmtree(d)
        return True

    # -- running a job -----------------------------------------------------------

    def start(self, section: str, opts: dict) -> str:
        src = Path(str(opts.get("source") or "")).expanduser()
        demo = bool(opts.get("demo"))
        if not demo and not src.is_dir():
            raise ValueError("Choose a folder of photos first.")
        if section in ("sar", "survey") and not demo:
            _triage_cfg(opts)  # reject a misspelt colour now, not halfway through the job
        name = str(opts.get("name") or "").strip() or (DEMO_NAMES[section] if demo else src.name) or "Job"
        out = self.new_dir(section, name)
        tid = uuid.uuid4().hex[:10]
        task = self.tasks[tid] = {"state": "running", "log": [], "url": None, "error": None, "section": section,
                                  "name": name}

        def log(*a):
            task["log"].append(" ".join(str(x) for x in a))
            del task["log"][:-200]

        def work():
            nonlocal src
            try:
                if demo and section != "survey":  # paint the example photos once, outside the job folder
                    from . import demos

                    src = self.ws / "_demo_photos" / section
                    if not any(src.glob("*.JPG")):
                        log("Painting a made-up set of photos to work on...")
                        {"sar": demos.write_sar_demo, "art": demos.write_art_demo, "watch": demos.write_watch_demo}[section](src)
                RUNNERS[section](src, out, name, opts, log)
                (out / "job.json").write_text(json.dumps({
                    "name": name, "section": section, "source": "" if demo and section == "survey" else str(src),
                    "created": time.strftime("%Y-%m-%d %H:%M:%S")}))
                task.update(state="done", url=f"/{section}/{out.name}/")
            except BaseException as e:  # SystemExit is how the pipelines report "no photos found"
                if not isinstance(e, (Exception, SystemExit)):
                    raise
                if not isinstance(e, (SystemExit, RuntimeError, ValueError)):
                    traceback.print_exc()
                shutil.rmtree(out, ignore_errors=True)
                task.update(state="error", error=str(e) or type(e).__name__)

        threading.Thread(target=work, daemon=True).start()
        return tid


def _triage_cfg(opts: dict):
    from .triage.detectors import parse_color
    from .triage.pipeline import TriageConfig

    colors = [str(c).strip() for c in opts.get("colors") or [] if str(c).strip()]
    for c in colors:
        parse_color(c)  # raises ValueError with a readable message
    sens = opts.get("sensitivity")
    return TriageConfig(colors=colors, sensitivity=float(sens) if sens else 1.0, ai=bool(opts.get("ai")))


def run_sar(src, out, name, opts, log):
    from .triage.export import write_all
    from .triage.pipeline import run_triage
    from .triage.report import write_report
    from .triage.search_report import write_search_report

    data = run_triage(src, out, _triage_cfg(opts), log)
    write_all(data, out)
    write_report(data, out)
    write_search_report(data, {}, out, name)


def run_survey_job(src, out, name, opts, log):
    from .site.survey import run_survey

    if opts.get("demo"):
        from .site.demo import build_demo
        from .site.report import write_report

        log("Painting a made-up property and surveying it...")
        return write_report(build_demo(out, log=log), out)
    detect = opts.get("detect", True)
    run_survey(src, out, name=name, log=log, detect=bool(detect), detect_cfg=_triage_cfg(opts) if detect else None,
               build_map=bool(opts.get("build_map")))


def run_art(src, out, name, opts, log):
    from .studio.gallery import write_gallery
    from .studio.pipeline import StudioConfig, run_studio

    top = int(opts.get("top") or 30)
    data = run_studio(src, out, StudioConfig(top=max(1, min(500, top)), enhance=opts.get("enhance", True) is not False),
                      log)
    write_gallery(data, out)


DEMO_NAMES = {"sar": "Demo search", "survey": "Demo property", "art": "Demo selection", "watch": "Demo gate, one week"}
def run_watch_job(src, out, name, opts, log):
    from .watch.pipeline import run_watch
    from .watch.report import write_report

    c0, c1 = int(opts.get("closed_from", 20)) % 24, int(opts.get("closed_to", 6)) % 24
    if opts.get("demo"):
        opts = {**opts, "watching": "Front gate"}
    write_report(run_watch(src, out, name=name, watching=str(opts.get("watching") or ""), closed_from=c0,
                           closed_to=c1, log=log), out)


RUNNERS = {"sar": run_sar, "survey": run_survey_job, "art": run_art, "watch": run_watch_job}


def list_dir(path: str | None) -> dict:
    """Folders under ``path`` for the folder picker, plus how many photos sit directly in it."""
    home = Path.home()
    p = Path(path).expanduser() if path else home
    if not p.is_dir():
        p = home
    p = p.resolve()
    dirs, photos = [], 0
    try:
        for e in sorted(os.scandir(p), key=lambda e: e.name.lower()):
            if e.name.startswith("."):
                continue
            try:
                if e.is_dir():
                    dirs.append(e.name)
                elif e.name.lower().endswith(EXTS):
                    photos += 1
            except OSError:
                continue
    except OSError as e:
        return {"path": str(p), "parent": str(p.parent), "dirs": [], "photos": 0, "error": e.strerror, "places": []}
    user = os.environ.get("USER") or home.name
    places = [("Home", home), ("Pictures", home / "Pictures"), ("Desktop", home / "Desktop"),
              ("Downloads", home / "Downloads"), ("Cards and drives", Path("/media") / user),
              ("Cards and drives", Path("/run/media") / user), ("Mounted", Path("/mnt")), ("Volumes", Path("/Volumes"))]
    return {"path": str(p), "parent": str(p.parent) if p.parent != p else None, "dirs": dirs[:500], "photos": photos,
            "places": [{"name": n, "path": str(q)} for n, q in places if q.is_dir()]}


def search_roots(near: str | None = None) -> list[Path]:
    home = Path.home()
    user = os.environ.get("USER") or home.name
    roots = [Path(near) if near else None, Path("/media") / user, Path("/run/media") / user, Path("/media"),
             Path("/mnt"), Path("/Volumes"), home]
    out = []
    for r in roots:
        if r and r.is_dir() and r.resolve() not in out:
            out.append(r.resolve())
    return out


def locate(name: str, samples: list, is_dir: bool = True, near: str | None = None, roots=None,
           budget_s: float = 5.0) -> list[str]:
    """Find on disk the folder that was dropped onto the page.

    A browser will not say where a dropped folder lives, only its name and
    what is inside. So look for a folder of that name which really contains
    one of those files at the same size (for a dropped file: the folder that
    holds it). Cards and external drives are searched first, then the home
    folder. Returns the matching paths, best first.
    """
    name = Path(str(name)).name
    want = [(str(s.get("path") or "").lstrip("/"), s.get("size")) for s in samples or [] if isinstance(s, dict)][:6]
    if not name or name in (".", ".."):
        return []

    def matches(d: Path) -> bool:
        if not is_dir:
            f = d / name
            return f.is_file() and (not want or f.stat().st_size == want[0][1])
        if not want:
            return True
        for rel, size in want:
            f = (d / rel).resolve()
            if d.resolve() in f.parents and f.is_file() and f.stat().st_size == size:
                return True
        return False

    found, seen, t0 = [], set(), time.time()
    for root in roots or search_roots(near):
        level = [root]
        for _depth in range(7):
            nxt = []
            for d in level:
                if time.time() - t0 > budget_s or len(found) >= 5:
                    return found
                try:
                    key = d.resolve()
                    if key in seen:
                        continue
                    seen.add(key)
                    hit = (d.name == name or (is_dir and d == root and root.name == name)) if is_dir else True
                    if hit and matches(d) and str(key) not in found:
                        found.append(str(key))
                    nxt += [Path(e.path) for e in os.scandir(d)
                            if not e.name.startswith(".") and e.is_dir(follow_symlinks=False)]
                except OSError:
                    continue
            level = nxt
    return found


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------

CSS = r"""
.wrap{max-width:1560px;margin:0 auto;padding:28px 24px 70px}
.hero h1{font-size:30px;letter-spacing:-.03em;margin:0;font-weight:700}
.hero p{color:var(--muted);margin:6px 0 22px;max-width:70ch}
.cols{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;align-items:start}
@media (max-width:1250px){.cols{grid-template-columns:repeat(2,1fr)}}
@media (max-width:700px){.cols{grid-template-columns:1fr}}
.sect{background:var(--surface);border:1px solid var(--line);border-radius:var(--r-lg);box-shadow:var(--shadow);overflow:hidden}
.sect-h{padding:18px 18px 16px;border-bottom:1px solid var(--line);border-top:3px solid var(--c)}
.sect-h .top{display:flex;align-items:center;gap:10px}
.sect-h .ico{width:36px;height:36px;border-radius:10px;display:grid;place-items:center;color:#fff;background:var(--c)}
.sect-h h2{margin:0;font-size:19px;letter-spacing:-.015em}
.sect .btn.primary{background:var(--c);box-shadow:0 6px 18px -8px var(--c)}
.sect-h .long{color:var(--muted);font-size:12.5px}
.sect-h p{color:var(--ink-2);font-size:13px;margin:12px 0 14px;min-height:96px}
.sect-h .acts{display:flex;gap:8px;flex-wrap:wrap}
.jobs{padding:8px}
.job{display:grid;grid-template-columns:1fr auto;gap:2px 8px;padding:10px 10px;border-radius:10px;text-decoration:none;align-items:center}
.job:hover{background:var(--surface-2)}
.job .n{font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.job .m{grid-column:1;color:var(--muted);font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.job button{grid-row:1/3;grid-column:2;opacity:.0}.job:hover button,.job button:focus{opacity:1}
.none{color:var(--muted);font-size:13px;padding:14px 10px}
.foot{color:var(--muted);font-size:12.5px;margin-top:22px}
dialog{border:1px solid var(--line-2);border-radius:var(--r-lg);background:var(--surface);color:var(--ink);padding:0;
  width:min(640px,94vw);box-shadow:var(--shadow-lg)}
dialog::backdrop{background:rgba(5,6,8,.6);backdrop-filter:blur(3px)}
.dlg-h{padding:16px 20px;border-bottom:1px solid var(--line);font-weight:650;font-size:16px;display:flex;align-items:center;gap:10px}
[hidden]{display:none!important}
.dlg-b{padding:18px 20px;max-height:calc(100vh - 210px);overflow:auto}
.dlg-b>*{margin-bottom:14px}.dlg-b>*:last-child{margin-bottom:0}
.dlg-f{padding:14px 20px;border-top:1px solid var(--line);display:flex;justify-content:flex-end;gap:8px}
.fld label{display:block;font-size:12px;font-weight:600;color:var(--muted);margin-bottom:5px}
.fld input[type=text],.fld input[type=number]{width:100%;height:36px;padding:0 10px;border-radius:9px;border:1px solid var(--line-2);
  background:var(--surface);color:var(--ink);font:500 13.5px var(--font)}
.fld .hint{color:var(--muted);font-size:12px;margin-top:4px}
.check{display:flex;gap:8px;align-items:flex-start;font-size:13.5px}.check input{margin-top:3px}
.dropzone{display:flex;gap:12px;align-items:center;padding:14px 16px;margin-bottom:8px;border:1.5px dashed var(--line-2);
  border-radius:10px;color:var(--muted);transition:border-color .15s,background .15s}
.dropzone b{display:block;color:var(--ink);font-weight:600}.dropzone span{font-size:12.5px}
.dropzone.ok span{color:var(--good)}.dropzone.bad span{color:var(--warn)}
.dragging .dropzone,.sect.over{border-color:var(--accent);background:var(--accent-soft)}
.sect.over{outline:2px dashed var(--accent);outline-offset:3px}
.picker{border:1px solid var(--line-2);border-radius:10px;overflow:hidden}
.picker .places{display:flex;gap:6px;padding:8px;background:var(--surface-2);border-bottom:1px solid var(--line);flex-wrap:wrap;align-items:center}
.picker .path{font:500 12px var(--mono);padding:8px 10px;border-bottom:1px solid var(--line);word-break:break-all}
.picker .list{max-height:210px;overflow:auto}
.picker .row{display:flex;gap:8px;align-items:center;padding:7px 10px;cursor:pointer;font-size:13.5px}
.picker .row:hover{background:var(--surface-2)}
.picker .count{padding:8px 10px;border-top:1px solid var(--line);font-size:12.5px;color:var(--muted)}
.picker .count b{color:var(--ink)}
#log{font:12px/1.5 var(--mono);background:var(--bg-2);border:1px solid var(--line);border-radius:10px;padding:10px 12px;
  height:190px;overflow:auto;white-space:pre-wrap;color:var(--ink-2)}
.spin{width:16px;height:16px;border-radius:50%;border:2px solid var(--line-2);border-top-color:var(--accent);animation:sp .8s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
.err{color:var(--bad);font-weight:550}
"""

JS = r"""
const $ = s => document.querySelector(s), S = DATA.sections;
let cur = null, here = null, poll = null;
async function api(url, body) {
  const r = await fetch(url, body ? {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)} : undefined);
  const j = await r.json(); if (!r.ok) throw new Error(j.error || r.statusText); return j;
}
async function home() {
  const d = await api("/api/home");
  $("#ws").textContent = d.workspace;
  for (const k in S) { const el = $("#jobs-" + k), list = d.jobs[k];
    el.innerHTML = list.length ? list.map(j => `<a class="job" href="${j.url}"><span class="n">${esc(j.name)}</span>
      <button class="btn icon ghost sm" data-del="${k}/${j.job}" data-name="${esc(j.name)}" title="Delete this job" aria-label="Delete">${DATA.trash}</button>
      <span class="m">${esc(j.created.slice(0, 16))}${j.summary ? " · " + esc(j.summary) : ""}</span></a>`).join("")
      : `<div class="none">No jobs yet.</div>`; }
}
document.addEventListener("click", async ev => {
  const del = ev.target.closest("[data-del]");
  if (del) { ev.preventDefault(); if (!confirm(`Delete "${del.dataset.name}" and everything in its folder? This cannot be undone.`)) return;
    const [section, job] = del.dataset.del.split("/"); await api("/api/delete", {section, job}); toast("Deleted"); return home(); }
  const nw = ev.target.closest("[data-new]"); if (nw) return openNew(nw.dataset.new);
  const dm = ev.target.closest("[data-demo]"); if (dm) return begin(dm.dataset.demo, {demo: true, top: 6});
  const row = ev.target.closest(".picker [data-path]"); if (row) return browse(row.dataset.path);
});
function openNew(k, viaDrop) {
  cur = k; $("#drop").className = "dropzone"; $("#drop-msg").textContent = "from your file manager, or browse to it below.";
  $("#dlg-title").innerHTML = `<span class="sect-ico" style="color:${S[k].color}">${S[k].icon}</span>${S[k].new}`;
  document.querySelectorAll("[data-only]").forEach(e => e.hidden = !e.dataset.only.split(" ").includes(k));
  $("#f-name").value = ""; $("#f-colors").value = ""; $("#go").disabled = true; $("#new-err").textContent = "";
  $("#l-colors").textContent = k === "sar" ? "Clothing or equipment colours to look for (optional)" : "Also look for these colours (optional)";
  $("#dlg").showModal(); if (!viaDrop) browse(here);
}
async function browse(path) {
  const d = await api("/api/ls?path=" + encodeURIComponent(path || "")); here = d.path;
  $("#places").innerHTML = d.places.map(p => `<button type="button" class="btn sm" data-path="${esc(p.path)}">${esc(p.name)}</button>`).join("");
  $("#path").textContent = d.path;
  $("#list").innerHTML = (d.parent ? `<div class="row" data-path="${esc(d.parent)}">${DATA.up} <span class="muted">Up one level</span></div>` : "") +
    d.dirs.map(n => `<div class="row" data-path="${esc(d.path.replace(/\/$/, "") + "/" + n)}">${DATA.folder}${esc(n)}</div>`).join("") +
    (d.error ? `<div class="row err">${esc(d.error)}</div>` : !d.dirs.length ? `<div class="row muted">No folders inside.</div>` : "");
  $("#count").innerHTML = `<b>${d.photos}</b> photo${d.photos === 1 ? "" : "s"} directly in this folder; folders inside it are searched too.`;
  $("#go").disabled = false; $("#list").scrollTop = 0;
}
$("#go").addEventListener("click", () => begin(cur, {source: here, name: $("#f-name").value,
  colors: $("#f-colors").value.split(/[,;]/).map(s => s.trim()).filter(Boolean), ai: $("#f-ai").checked,
  detect: $("#f-detect").checked, build_map: $("#f-map").checked, top: +$("#f-top").value || 30,
  watching: $("#f-watching").value, closed_from: +$("#f-c0").value, closed_to: +$("#f-c1").value}));
async function begin(section, opts) {
  let t; try { t = await api("/api/new", {section, ...opts}); } catch (e) { $("#new-err").textContent = e.message; if (!$("#dlg").open) toast(e.message); return; }
  $("#dlg").close(); $("#run-title").textContent = S[section].title + ": working…"; $("#log").textContent = ""; $("#run-err").textContent = "";
  $("#run-spin").hidden = false; $("#run-close").hidden = true; $("#run").showModal();
  clearInterval(poll); poll = setInterval(async () => {
    const s = await api("/api/task/" + t.task); const lg = $("#log");
    lg.textContent = s.log.join("\n"); lg.scrollTop = lg.scrollHeight;
    if (s.state === "running") return;
    clearInterval(poll); $("#run-spin").hidden = true;
    if (s.state === "done") location.href = s.url;
    else { $("#run-title").textContent = "Could not finish"; $("#run-err").textContent = s.error; $("#run-close").hidden = false; home(); }
  }, 700);
}
// ---- drag and drop: a folder, a drive, or photos, onto a section card or the New dialog
function entryFiles(entry, prefix, out, done) {            // first few files under a dropped folder, with sizes
  if (out.length >= 4 || out.depth > 40) return done();
  out.depth = (out.depth || 0) + 1;
  entry.createReader().readEntries(list => {
    const files = list.filter(e => e.isFile && !e.name.startsWith(".")), dirs = list.filter(e => e.isDirectory && !e.name.startsWith("."));
    const jpg = files.filter(f => /\.jpe?g$/i.test(f.name));
    let wait = 1; const step = () => { if (--wait === 0) nextDir(0); };
    const nextDir = i => (i >= dirs.length || out.length >= 4) ? done() : entryFiles(dirs[i], prefix + dirs[i].name + "/", out, () => nextDir(i + 1));
    for (const f of (jpg.length ? jpg : files).slice(0, 3)) { wait++; f.file(x => { out.push({path: prefix + f.name, size: x.size}); step(); }, step); }
    step();
  }, done);
}
async function dropped(dt) {
  const msg = (t, cls) => { $("#drop-msg").textContent = t; $("#drop").className = "dropzone " + (cls || ""); };
  // some browsers hand over the real address of what was dragged
  const uri = (dt.getData("text/uri-list") || dt.getData("text/x-moz-url") || "").split(/\r?\n/).find(u => u.startsWith("file://"));
  const item = [...dt.items].map(i => i.webkitGetAsEntry && i.webkitGetAsEntry()).find(Boolean);
  if (uri) { let p = decodeURIComponent(new URL(uri).pathname); if (item && item.isFile) p = p.replace(/\/[^/]*$/, "") || "/";
    await browse(p); if (here === p) return msg("Selected " + p, "ok"); }
  if (!item) return msg("That was not a folder. Drag a folder from your file manager, or browse below.", "bad");
  msg("Looking for “" + item.name + "” on this computer…");
  const samples = [];
  if (item.isDirectory) await new Promise(r => entryFiles(item, "", samples, r));
  else await new Promise(r => item.file(f => { samples.push({path: item.name, size: f.size}); r(); }, r));
  const d = await api("/api/locate", {name: item.name, samples, dir: item.isDirectory, near: here});
  if (!d.paths.length) return msg("Could not find “" + item.name + "” automatically. Browse to it below.", "bad");
  await browse(d.paths[0]);
  msg((d.paths.length > 1 ? "Found more than one match; using " : "Selected ") + d.paths[0] + (d.paths.length > 1 ? ". Check it is the right one." : ""), d.paths.length > 1 ? "bad" : "ok");
}
let dragN = 0;
addEventListener("dragenter", ev => { if ([...ev.dataTransfer.types].includes("Files")) { dragN++; document.body.classList.add("dragging"); } });
addEventListener("dragleave", () => { if (--dragN <= 0) { dragN = 0; document.body.classList.remove("dragging"); document.querySelectorAll(".sect.over").forEach(e => e.classList.remove("over")); } });
addEventListener("dragover", ev => { ev.preventDefault(); if ($("#dlg").open) return;
  const s = ev.target.closest && ev.target.closest(".sect"); document.querySelectorAll(".sect").forEach(e => e.classList.toggle("over", e === s)); });
addEventListener("drop", async ev => { ev.preventDefault(); dragN = 0; document.body.classList.remove("dragging");
  const sect = ev.target.closest && ev.target.closest(".sect"); document.querySelectorAll(".sect.over").forEach(e => e.classList.remove("over"));
  if ($("#run").open) return;
  const dt = ev.dataTransfer;
  if (!$("#dlg").open) { if (!sect) return toast("Drop onto one of the sections"); openNew(sect.dataset.k, true); }
  try { await dropped(dt); } catch (e) { $("#drop-msg").textContent = e.message; } });
$("#run-close").addEventListener("click", () => $("#run").close());
$("#cancel").addEventListener("click", () => $("#dlg").close());
$("#quit").addEventListener("click", async () => { if (!confirm("Stop ISRStudio? Jobs are saved; start it again any time.")) return;
  try { await api("/api/quit", {}); } catch (e) {} document.body.innerHTML = `<div class="wrap"><h1>ISRStudio has stopped.</h1><p class="muted">You can close this tab.</p></div>`; });
home();
"""


def home_html() -> str:
    cards = []
    for k, s in SECTIONS.items():
        demo = f'<button class="btn" data-demo="{k}">Try the demo</button>'
        cards.append(f'''<section class="sect" data-k="{k}" style="--c:{s["color"]}"><div class="sect-h"><div class="top">
<div class="ico">{icon(s["icon"], 20)}</div><div><h2>{s["title"]}</h2><div class="long">{s["long"]}</div></div></div>
<p>{s["blurb"]}</p><div class="acts"><button class="btn primary" data-new="{k}">{icon("plus")}{s["new"]}</button>{demo}</div></div>
<div class="jobs" id="jobs-{k}"></div></section>''')
    body = f'''<header class="topbar"><div class="brand">{LOGO}<div class="brand-name">{APP_NAME}</div></div><div class="spacer"></div>
<button class="btn ghost" id="quit" title="Stop the app">{icon("power")}<span class="lbl">Quit</span></button>{theme_button()}</header>
<div class="wrap"><div class="hero"><h1>{APP_NAME}</h1>
<p>Turn a card of drone photos into something you can act on. Press New, or drag a folder or drive onto a section. Everything runs on this computer; nothing is uploaded.</p></div>
<div class="cols">{"".join(cards)}</div>
<p class="foot">Jobs are kept in <span class="mono" id="ws"></span>. Each job folder holds its report and exports and can be copied elsewhere.</p></div>
<dialog id="dlg"><div class="dlg-h" id="dlg-title"></div><div class="dlg-b">
<div class="fld"><label>Folder of photos (the SD card's DCIM folder, or any folder of JPEGs)</label>
<div class="dropzone" id="drop">__DROP__<div><b>Drag the folder or drive here</b><span id="drop-msg">from your file manager, or browse to it below.</span></div></div>
<div class="picker"><div class="places" id="places"></div><div class="path" id="path"></div><div class="list" id="list"></div>
<div class="count" id="count"></div></div></div>
<div class="fld"><label for="f-name">Name (optional)</label><input type="text" id="f-name" placeholder="Defaults to the folder name"></div>
<div class="fld" data-only="sar survey"><label for="f-colors" id="l-colors"></label>
<input type="text" id="f-colors" placeholder="e.g. red, blue"><div class="hint">Names: red, orange, yellow, blue, purple, pink, magenta, cyan, white, green. Or a hex colour such as #1e90ff.</div></div>
<label class="check" data-only="survey"><input type="checkbox" id="f-detect" checked><span>Look for people, vehicles and objects in the drone photos</span></label>
<label class="check" data-only="survey"><input type="checkbox" id="f-map"><span>Build a 3-D map from these photos with OpenDroneMap
<span class="muted">(needs Docker and many overlapping photos; can take an hour and wants 8 GB of memory or more. A folder that already holds an OpenDroneMap or WebODM result is used automatically.)</span></span></label>
<label class="check" data-only="sar survey"><input type="checkbox" id="f-ai"><span>Also use the AI detector <span class="muted">(needs the optional AI install; slow on older computers)</span></span></label>
<div class="fld" data-only="watch"><label for="f-watching">What is the camera pointed at? (optional)</label>
<input type="text" id="f-watching" placeholder="e.g. Front gate, loading bay, north fence"></div>
<div class="fld" data-only="watch"><label>Hours the site is meant to be closed</label>
<div style="display:flex;gap:8px;align-items:center"><input type="number" id="f-c0" value="20" min="0" max="23" style="width:90px"><span>:00 to</span>
<input type="number" id="f-c1" value="6" min="0" max="23" style="width:90px"><span>:00</span></div>
<div class="hint">Movement in these hours is listed separately as out of hours.</div></div>
<div class="fld" data-only="art"><label for="f-top">How many photos to suggest</label><input type="number" id="f-top" value="30" min="1" max="500"></div>
<div class="err" id="new-err"></div></div>
<div class="dlg-f"><button class="btn" id="cancel">Cancel</button><button class="btn primary" id="go">Start with this folder</button></div></dialog>
<dialog id="run"><div class="dlg-h"><div class="spin" id="run-spin"></div><span id="run-title"></span></div>
<div class="dlg-b"><div id="log"></div><div class="err" id="run-err"></div></div>
<div class="dlg-f"><button class="btn" id="run-close" hidden>Close</button></div></dialog>'''
    body = body.replace("__DROP__", icon("folder", 26))
    data = {"sections": {k: {"title": s["title"], "new": s["new"], "color": s["color"], "icon": icon(s["icon"], 18)}
                         for k, s in SECTIONS.items()},
            "trash": icon("trash", 15), "folder": icon("folder", 16), "up": icon("chevron-left", 16)}
    return page(APP_NAME, body, CSS + ".sect-ico{display:inline-flex}", JS, data, default_theme="dark")


# A small bar added to the SAR and Art pages when they are opened inside the app.
BAR = """<style>#isrbar{position:fixed;left:16px;bottom:16px;z-index:80;display:flex;gap:8px}
@media print{#isrbar{display:none}}</style><div id="isrbar"><a class="btn" href="/">&larr; ISRStudio</a>%s</div>"""
SAR_BAR = BAR % """<button class="btn primary" id="isr-rep">Search report</button><script>
document.getElementById("isr-rep").onclick = async () => {
  let st = {}; try { st = JSON.parse(localStorage.getItem("isr-triage:" + DATA.summary.source + ":" + DATA.summary.generated) || "{}"); } catch (e) {}
  const win = window.open("about:blank", "_blank");
  const r = await fetch("api/report", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({status: st})});
  const j = await r.json(); if (win) win.location = j.url; else location.href = j.url; };</script>"""


class AppHandler(LocalHandler):
    studio: Studio

    def _job(self, path: str):
        """Split /<section>/<job>/<rest> -> (section, folder, "/rest", prefix)."""
        parts = path.split("/", 3)
        if len(parts) < 3:
            return None
        d = self.studio.job_dir(parts[1], parts[2])
        return (parts[1], d, "/" + (parts[3] if len(parts) > 3 else ""), f"/{parts[1]}/{parts[2]}") if d else None

    @guarded
    def do_GET(self):
        url = urlparse(self.path)
        path = url.path
        st = self.studio
        if path == "/":
            return self.html(home_html())
        if path == "/api/ping":
            return self.json({"app": APP_NAME})
        if path == "/api/home":
            return self.json({"workspace": str(st.ws), "jobs": {s: st.jobs(s) for s in SECTIONS}})
        if path == "/api/ls":
            return self.json(list_dir((parse_qs(url.query).get("path") or [""])[0]))
        if path.startswith("/api/task/"):
            t = st.tasks.get(path.rsplit("/", 1)[1])
            return self.json(t, 200) if t else self.json({"error": "not found"}, 404)
        hit = self._job(path)
        if not hit:
            return self.json({"error": "not found"}, 404)
        section, d, rest, prefix = hit
        if section == "survey":
            return project_get(self, d, rest, prefix)
        if rest == "/":  # relative links in the page need the file name in the address
            self.send_response(302)
            self.send_header("Location", f"{prefix}/{SECTIONS[section]['main']}")
            self.send_header("Content-Length", "0")
            return self.end_headers()
        return self.file(d, rest[1:], inject=SAR_BAR if section == "sar" and rest == "/report.html" else BAR % "")

    @guarded
    def do_POST(self):
        path = self.path.split("?", 1)[0]
        st = self.studio
        hit = self._job(path)
        if hit and hit[0] == "survey":
            return project_post(self, hit[1], hit[2], hit[3])
        body = self.body()
        if body is None:
            return
        if hit and hit[0] == "sar" and hit[2] == "/api/report":
            from .triage.search_report import write_search_report

            data = json.loads((hit[1] / "results.json").read_text())
            name = json.loads((hit[1] / "job.json").read_text()).get("name", "") if (hit[1] / "job.json").exists() else ""
            status = {str(k): v for k, v in (body.get("status") or {}).items() if v in ("confirmed", "dismissed")}
            write_search_report(data, status, hit[1], name)
            return self.json({"url": f"{hit[3]}/search_report.html"})
        if path == "/api/new":
            if body.get("section") not in SECTIONS:
                return self.json({"error": "unknown section"}, 400)
            try:
                return self.json({"task": st.start(body["section"], body)})
            except ValueError as e:
                return self.json({"error": str(e)}, 400)
        if path == "/api/locate":
            return self.json({"paths": locate(body.get("name"), body.get("samples"), body.get("dir", True),
                                              body.get("near"))})
        if path == "/api/delete":
            return self.json({"ok": True}) if st.delete(body.get("section"), body.get("job")) else \
                self.json({"error": "not found"}, 404)
        if path == "/api/quit":
            self.json({"ok": True})
            return threading.Thread(target=self.server.shutdown, daemon=True).start()
        self.json({"error": "not found"}, 404)


def make_app(workspace: Path, port: int = 8765):
    return bind(type("Handler", (AppHandler,), {"studio": Studio(workspace)}), port)


def already_running(port: int) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ping", timeout=0.6) as r:
            return json.loads(r.read()).get("app") == APP_NAME
    except Exception:
        return False


def serve(workspace: Path, port: int = 8765, open_browser: bool = True) -> None:
    if already_running(port):  # a second click on the launcher: just show the window that is already there
        print(f"{APP_NAME} is already running: http://127.0.0.1:{port}/")
        if open_browser:
            import webbrowser
            webbrowser.open(f"http://127.0.0.1:{port}/")
        return
    srv = make_app(workspace, port)
    print(f"Jobs are kept in {Path(workspace).expanduser()}")
    run(srv, APP_NAME, open_browser)


def install_launcher(workspace: Path | None = None) -> Path:
    """Add ISRStudio to the Linux applications menu (a .desktop file in the user's own folder)."""
    data = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    ico = data / "icons" / "isr-studio.svg"
    ico.parent.mkdir(parents=True, exist_ok=True)
    ico.write_text(ICON_SVG)
    ws = f' --workspace "{Path(workspace).expanduser()}"' if workspace else ""
    entry = data / "applications" / "isr-studio.desktop"
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text(f"""[Desktop Entry]
Type=Application
Name={APP_NAME}
Comment=SAR, site survey and art selection from drone photos
Exec="{sys.executable}" -m isr app{ws}
Icon={ico}
Terminal=false
Categories=Graphics;Photography;Utility;
""")
    return entry
