"""The assessment report: one static HTML page plus map-ready exports.

``report.html`` has no scripts and no network dependencies. It sits next to
``basemap.jpg``, ``photos/`` and ``report_assets/`` so the whole project folder
can be zipped, copied to a USB stick, or printed to PDF from the browser.
"""

from __future__ import annotations

import csv
import json
import time
from html import escape as h
from pathlib import Path
from xml.sax.saxutils import escape as x

from ..geo import from_local
from ..ui import APP_NAME, LOGO, icon, page, theme_button
from . import model as M
from .analysis import Analysis

SEV = {"high": "var(--bad)", "medium": "var(--warn)", "low": "var(--good)", "info": "var(--muted)"}

CSS = r"""
.wrap{max-width:1180px;margin:0 auto;padding:26px 24px 70px}
h1{font-size:28px;letter-spacing:-.025em;margin:0 0 4px;font-weight:700}
h2{font-size:19px;letter-spacing:-.015em;margin:44px 0 6px;font-weight:650;display:flex;align-items:baseline;gap:10px}
h2 .n{font:600 12px var(--mono);color:var(--muted)}
h3{font-size:14px;margin:18px 0 6px}
p.lead{color:var(--ink-2);max-width:78ch;margin:6px 0 14px}
.sub{color:var(--muted);font-size:13px}
.cover{background:var(--surface);border:1px solid var(--line);border-radius:var(--r-lg);padding:22px 24px;box-shadow:var(--shadow)}
.cover .label{margin-bottom:8px;color:var(--accent)}
.front{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px 24px;margin:18px 0 0;padding-top:16px;border-top:1px solid var(--line)}
.front dt{font-size:11px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}
.front dd{margin:2px 0 0;font-weight:550}
.foot{margin-top:40px;padding-top:14px;border-top:1px solid var(--line);color:var(--muted);font-size:12px;max-width:90ch}
.summary{font-size:15.5px;line-height:1.6;max-width:80ch;margin:16px 0 18px;color:var(--ink)}
.stats{margin:14px 0}
.stat .v small{font-size:13px;color:var(--muted);font-weight:500;letter-spacing:0}
.stat.bad .v{color:var(--bad)}.stat.good .v{color:var(--good)}
.mapcard{background:var(--surface);border:1px solid var(--line);border-radius:var(--r-lg);overflow:hidden;box-shadow:var(--shadow);margin:12px 0}
.mapcard svg{display:block;width:100%;height:auto;background:#1b1e25}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;padding:10px 14px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.sw.line{height:3px;vertical-align:3px;border-radius:2px}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:var(--r);background:var(--surface);margin:10px 0}
table{border-collapse:collapse;width:100%;font-size:13px}
th{text-align:left;font-size:11px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);
  padding:9px 12px;border-bottom:1px solid var(--line);white-space:nowrap;background:var(--surface-2)}
td{padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:top}
tr:last-child td{border-bottom:0}
td.mono{white-space:nowrap}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.code{display:inline-grid;place-items:center;min-width:26px;height:20px;padding:0 5px;border-radius:6px;font:700 11px var(--mono);
  color:#fff;background:var(--c,var(--muted))}
.pill{display:inline-flex;align-items:center;height:20px;padding:0 8px;border-radius:999px;font-size:11px;font-weight:650;
  text-transform:capitalize;color:var(--c);background:color-mix(in srgb,var(--c) 12%,transparent);
  border:1px solid color-mix(in srgb,var(--c) 30%,transparent);white-space:nowrap}
.yes{color:var(--good);font-weight:600}.no{color:var(--bad);font-weight:600}
.findings{display:grid;gap:10px;margin:12px 0}
.finding{display:grid;grid-template-columns:4px 1fr;background:var(--surface);border:1px solid var(--line);border-radius:var(--r);overflow:hidden}
.finding>i{background:var(--c)}
.finding>div{padding:12px 16px}
.finding .t{font-weight:650;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.finding .d{color:var(--ink-2);margin:4px 0 0;max-width:90ch}
.finding .a{margin:6px 0 0;max-width:90ch}
.finding .a b{font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);margin-right:6px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px;margin:12px 0}
.acard{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);padding:14px 16px;break-inside:avoid}
.acard .t{display:flex;gap:8px;align-items:center;font-weight:650;margin-bottom:6px}
.acard .t .pill{margin-left:auto}
.checks{display:grid;grid-template-columns:1fr auto;gap:3px 12px;font-size:13px;margin:8px 0}
.checks span:nth-child(odd){color:var(--muted)}
ul.pc{list-style:none;padding:0;margin:8px 0}
ul.pc li{padding-left:20px;position:relative;margin:3px 0;font-size:13px}
ul.pc li::before{position:absolute;left:0;font-weight:700}
ul.pc li.pro::before{content:"+";color:var(--good)}ul.pc li.con::before{content:"\2212";color:var(--bad)}
h3.scn{font-size:16px;margin-top:30px}
.cards.opts{grid-template-columns:repeat(auto-fill,minmax(420px,1fr))}
.thumbs{display:flex;gap:6px;margin-top:10px;flex-wrap:wrap}
.thumbs a{display:block;width:96px;height:72px;border-radius:8px;overflow:hidden;border:1px solid var(--line)}
.thumbs img{width:100%;height:100%;object-fit:cover;display:block}
.photos{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px;margin:12px 0}
.photos figure{margin:0;background:var(--surface);border:1px solid var(--line);border-radius:var(--r);overflow:hidden}
.photos img{width:100%;aspect-ratio:4/3;object-fit:cover;display:block}
.photos figcaption{padding:7px 10px;font-size:11.5px;color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.limits li{margin:5px 0;max-width:88ch;color:var(--ink-2)}
.notes{white-space:pre-wrap;background:var(--surface);border:1px solid var(--line);border-radius:var(--r);padding:12px 16px;max-width:88ch}
.empty-note{color:var(--muted);padding:14px 0}
@media print{
  .topbar,#toast{display:none!important} body{background:#fff} body::before{display:none}
  .wrap{padding:0;max-width:none} h2{break-after:avoid;margin-top:26px} .mapcard,.finding,.acard,tr{break-inside:avoid}
  .mapcard,.panel,.stat,.finding,.acard,.tablewrap{box-shadow:none}
}
"""


# ---------------------------------------------------------------------------
# map
# ---------------------------------------------------------------------------


class MapSVG:
    """Draws the site plan as SVG in metre coordinates (y flipped so north is up)."""

    def __init__(self, model: dict, an: Analysis, assets: str = "report_assets"):
        self.m, self.an, self.r, self.assets = model, an, an.result, assets
        sc = an.sc
        self.b = (sc.xmin, sc.ymin, sc.xmax, sc.ymax)
        self.u = max(sc.xmax - sc.xmin, sc.ymax - sc.ymin) / 100.0  # 1% of the map: the unit for strokes and text

    def render(self, layers=(), cameras="current", blind=True, photos=False, access=True, title="Site plan",
               gaps=(), guards=None) -> str:
        xmin, ymin, xmax, ymax = self.b
        u = self.u
        o = [f'<svg viewBox="{xmin:.2f} {-ymax:.2f} {xmax - xmin:.2f} {ymax - ymin:.2f}" role="img" '
             f'aria-label="{h(title)}" font-family="Inter,system-ui,sans-serif">']
        bm = self.m.get("basemap")
        if bm:
            bx0, by0, bx1, by1 = bm["bounds"]
            o.append(f'<image href="{h(bm["file"])}" x="{bx0}" y="{-by1}" width="{bx1 - bx0}" height="{by1 - by0}" '
                     f'preserveAspectRatio="none" opacity="0.92"/>')
        else:
            o.append(f'<rect x="{xmin}" y="{-ymax}" width="{xmax - xmin}" height="{ymax - ymin}" fill="#262a33"/>')
        for name in layers:
            o.append(f'<image href="{self.assets}/{name}.png" x="{xmin}" y="{-ymax}" width="{xmax - xmin}" '
                     f'height="{ymax - ymin}" preserveAspectRatio="none" style="image-rendering:pixelated"/>')
        F = self.m["features"]
        for f in F:
            o.append(self._feature(f))
        if photos:
            for p in self.m["photos"]:
                if p.get("x") is not None and p["kind"] != "nadir":
                    o.append(self._photo(p))
        for c in self.r["cameras"]:
            if cameras == "none" or cameras == "current" and c["status"] != "existing":
                continue
            col = "#3b82f6"
            o.append(self._camera(c["x"], c["y"], c["bearing"], c["hfov"], c["range"], col, c["code"],
                                  dashed=c["status"] == "planned"))
        if cameras == "plan":
            for p in self.r["recommended"]:
                o.append(self._camera(p["x"], p["y"], p["bearing"], p["hfov"], p["range"], "#10a35a", p["id"],
                                      dashed=True))
        stops = []
        if guards:  # one staffing option: the patrol loop, its stops in order, and the posts
            rt = guards.get("route")
            if rt:
                line = self._pts(rt["line"] + rt["line"][:1])
                o.append(f'<polyline points="{line}" fill="none" stroke="#04263a" stroke-opacity=".75" stroke-width="{u * 1.1:.2f}" '
                         f'stroke-linejoin="round"/><polyline points="{line}" fill="none" stroke="#38bdf8" '
                         f'stroke-width="{u * .55:.2f}" stroke-linejoin="round" stroke-dasharray="{u * 1.4:.2f} {u * .9:.2f}"/>')
                for st in rt["stops"]:
                    sx, sy = st["x"] + u * 1.9, -st["y"] - u * 1.9  # beside the door marker, not under it
                    stops.append(f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="{u * 1.15:.2f}" fill="#38bdf8" stroke="#04263a" '
                                 f'stroke-width="{u * .2:.2f}"/><text x="{sx:.2f}" y="{sy + u * .45:.2f}" font-size="{u * 1.25:.2f}" '
                                 f'font-weight="700" text-anchor="middle" fill="#04263a">{st["n"]}</text>')
            for p in (self.r.get("guards") or {}).get("posts", []):
                if p["id"] in guards["statics"]:
                    o.append(self._label(p["x"], p["y"], p["id"], "#0284c7", shape="cam"))
        for g in gaps:  # openings in the perimeter: red where unmarked, amber where a gate or driveway is marked
            col = "#e08a00" if g["declared"] else "#e5484d"
            o.append(f'<polyline points="{self._pts(g["line"])}" fill="none" stroke="#fff" stroke-width="{u * 1.5:.2f}" '
                     f'stroke-linecap="round"/><polyline points="{self._pts(g["line"])}" fill="none" stroke="{col}" '
                     f'stroke-width="{u * .95:.2f}" stroke-linecap="round"/>')
            if not g["declared"]:
                o.append(self._label(g["x"], g["y"], g["code"], col, shape="rect"))
        if blind:
            for b in (self.r["blind_spots"] if blind is True else blind):
                o.append(self._label(b["x"], b["y"], b["id"], "#e5484d", shape="rect"))
        for d in self.r.get("detections", ()):
            if d["status"] == "confirmed":
                o.append(self._label(d["x"], d["y"], d["code"], "#d946ef", shape="cam"))
        if access:
            for a in self.r["access"]:
                col = {"high": "#e5484d", "medium": "#e08a00", "low": "#12a150"}[a["priority"]]
                o.append(self._label(a["x"], a["y"], a["code"], col))
        o += stops
        o.append(self._furniture())
        o.append("</svg>")
        return "".join(o)

    def _pts(self, pts):
        return " ".join(f"{px:.2f},{-py:.2f}" for px, py in pts)

    def _feature(self, f):
        u, k, p = self.u, f["kind"], self._pts(f["pts"])
        if k == "boundary":
            return (f'<polygon points="{p}" fill="none" stroke="#000" stroke-opacity=".55" stroke-width="{u * .7:.2f}"/>'
                    f'<polygon points="{p}" fill="none" stroke="#fff" stroke-width="{u * .35:.2f}" '
                    f'stroke-dasharray="{u * 1.4:.2f} {u:.2f}"/>')
        if k == "building":
            return (f'<polygon points="{p}" fill="#0f172a" fill-opacity=".5" stroke="#fff" stroke-width="{u * .3:.2f}" '
                    f'stroke-linejoin="round"/>')
        if k == "vegetation":
            return (f'<polygon points="{p}" fill="#16a34a" fill-opacity=".28" stroke="#86efac" '
                    f'stroke-width="{u * .22:.2f}" stroke-dasharray="{u * .5:.2f} {u * .5:.2f}"/>')
        if k == "street":
            return (f'<polyline points="{p}" fill="none" stroke="#facc15" stroke-width="{u * .45:.2f}" '
                    f'stroke-dasharray="{u * .2:.2f} {u * 1.1:.2f}" stroke-linecap="round"/>')
        if k == "barrier":
            col, dash = {"wall": ("#e5e7eb", ""), "fence": ("#d6a56b", ""),
                         "open_fence": ("#d6a56b", f'stroke-dasharray="{u * .9:.2f} {u * .6:.2f}"'),
                         "hedge": ("#4ade80", "")}[f["subtype"]]
            return (f'<polyline points="{p}" fill="none" stroke="#000" stroke-opacity=".5" stroke-width="{u * .95:.2f}" '
                    f'stroke-linejoin="round"/><polyline points="{p}" fill="none" stroke="{col}" '
                    f'stroke-width="{u * .55:.2f}" stroke-linejoin="round" {dash}/>')
        return ""

    def _camera(self, cx, cy, bearing, hfov, rng, col, code, dashed=False):
        import math
        u = self.u
        pts = [(cx, cy)]
        n = max(2, int(hfov // 6))
        for i in range(n + 1):
            a = math.radians(bearing - hfov / 2 + hfov * i / n)
            pts.append((cx + math.sin(a) * rng, cy + math.cos(a) * rng))
        dash = f'stroke-dasharray="{u:.2f} {u * .7:.2f}"' if dashed else ""
        return (f'<polygon points="{self._pts(pts)}" fill="{col}" fill-opacity=".07" stroke="{col}" '
                f'stroke-width="{u * .3:.2f}" stroke-linejoin="round" {dash}/>' +
                self._label(cx - math.sin(math.radians(bearing)) * u * 2.4, cy - math.cos(math.radians(bearing)) * u * 2.4,
                            code, col, "cam") +
                f'<circle cx="{cx:.2f}" cy="{-cy:.2f}" r="{u * .6:.2f}" fill="{col}" stroke="#fff" stroke-width="{u * .2:.2f}"/>')

    def _photo(self, p):
        import math
        u = self.u
        if p.get("heading") is None:
            return f'<circle cx="{p["x"]}" cy="{-p["y"]}" r="{u * .6:.2f}" fill="#fff" stroke="#111" stroke-width="{u * .15:.2f}"/>'
        a = math.radians(p["heading"])
        L = u * 3.2
        tip = (p["x"] + math.sin(a) * L, p["y"] + math.cos(a) * L)
        return (f'<line x1="{p["x"]}" y1="{-p["y"]}" x2="{tip[0]:.2f}" y2="{-tip[1]:.2f}" stroke="#fff" '
                f'stroke-width="{u * .35:.2f}" stroke-linecap="round"/>'
                f'<circle cx="{p["x"]}" cy="{-p["y"]}" r="{u * .7:.2f}" fill="#fff" stroke="#111" stroke-width="{u * .15:.2f}"/>')

    def _label(self, px, py, text, col, shape="circle"):
        u = self.u
        fs = u * 1.5
        w = u * (1.1 + 0.95 * len(text))
        if shape == "circle":
            body = f'<circle cx="{px:.2f}" cy="{-py:.2f}" r="{u * 1.55:.2f}" fill="{col}" stroke="#fff" stroke-width="{u * .25:.2f}"/>'
        elif shape == "cam":
            body = (f'<rect x="{px - w / 2:.2f}" y="{-py - u * 1.2:.2f}" width="{w:.2f}" height="{u * 2.4:.2f}" '
                    f'rx="{u * 1.2:.2f}" fill="{col}" stroke="#fff" stroke-width="{u * .25:.2f}"/>')
        else:
            body = (f'<rect x="{px - w / 2:.2f}" y="{-py - u * 1.2:.2f}" width="{w:.2f}" height="{u * 2.4:.2f}" '
                    f'rx="{u * .4:.2f}" fill="#fff" stroke="{col}" stroke-width="{u * .3:.2f}"/>')
        tc = col if shape == "rect" else "#fff"
        return (body + f'<text x="{px:.2f}" y="{-py + fs * .36:.2f}" font-size="{fs:.2f}" font-weight="700" '
                f'text-anchor="middle" fill="{tc}">{h(text)}</text>')

    def _furniture(self):
        """North arrow and scale bar."""
        xmin, ymin, xmax, ymax = self.b
        u = self.u
        span = xmax - xmin
        bar = next(b for b in (50, 20, 10, 5, 2, 1) if b <= span * 0.25)
        x0, y0 = xmin + u * 3, -ymin - u * 3
        nx, ny = xmax - u * 5, -ymax + u * 8
        halo = f'paint-order="stroke" stroke="#000" stroke-opacity=".7" stroke-width="{u * .5:.2f}"'
        return (f'<path d="M{x0} {y0}h{bar}" stroke="#000" stroke-opacity=".7" stroke-width="{u * 1.1:.2f}"/>'
                f'<path d="M{x0} {y0}h{bar}" stroke="#fff" stroke-width="{u * .55:.2f}"/>'
                f'<text x="{x0 + bar / 2}" y="{y0 - u * 1.2:.2f}" font-size="{u * 1.7:.2f}" font-weight="600" fill="#fff" '
                f'text-anchor="middle" {halo}>{bar} m</text>'
                f'<path d="M{nx} {ny - u * 3.2:.2f}l{u * 1.5:.2f} {u * 4:.2f}l{-u * 1.5:.2f} {-u * 1.1:.2f}'
                f'l{-u * 1.5:.2f} {u * 1.1:.2f}z" fill="#fff" stroke="#000" stroke-opacity=".7" stroke-width="{u * .2:.2f}"/>'
                f'<text x="{nx}" y="{ny + u * 3.2:.2f}" font-size="{u * 1.8:.2f}" font-weight="700" fill="#fff" '
                f'text-anchor="middle" {halo}>N</text>')


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------


def _sw(col, label, line=False):
    return f'<span><i class="sw{" line" if line else ""}" style="background:{col}"></i>{h(label)}</span>'


def _yn(v, yes="Yes", no="No"):
    return f'<span class="{"yes" if v else "no"}">{yes if v else no}</span>'


def _pill(sev):
    return f'<span class="pill" style="--c:{SEV[sev]}">{sev}</span>'


def _code(text, col):
    return f'<span class="code" style="--c:{col}">{h(text)}</span>'


def _thumbs(ids, photos):
    out = []
    for pid in ids:
        p = photos.get(pid)
        if p:
            out.append(f'<a href="{h(p["file"])}" title="{h(p["name"])}"><img src="{h(p["thumb"])}" alt="{h(p["name"])}"></a>')
    return f'<div class="thumbs">{"".join(out)}</div>' if out else ""


def summary_text(r: dict) -> str:
    st = r["stats"]
    parts = []
    if st["access_total"]:
        parts.append(f"{st['access_unwatched']} of {st['access_total']} access points are not covered by an existing "
                     f"camera, and {st['access_high']} of them rate high priority because they are also hidden "
                     f"from the street and from windows." if st["access_high"] else
                     f"{st['access_unwatched']} of {st['access_total']} access points are not covered by an "
                     f"existing camera; none rates high priority.")
    parts.append(f"{st['blind_m2']:.0f} m² of the grounds ({st['blind_pct']:.0f}%) cannot be seen by a camera, "
                 f"from the street or from a window.")
    if st["cameras_recommended"]:
        n = st["cameras_recommended"]
        parts.append(f"Adding the {n} recommended camera{'s' if n != 1 else ''} raises camera coverage from "
                     f"{st['camera_pct_planned']:.0f}% to {st['camera_pct_recommended']:.0f}% of the grounds and cuts "
                     f"the unobserved area to {st['blind_pct_recommended']:.0f}%.")
    return " ".join(parts)


def build_html(model: dict, an: Analysis) -> str:
    r, st = an.result, an.result["stats"]
    photos = {p["id"]: p for p in model["photos"]}
    mp = MapSVG(model, an)
    sv = model.get("survey") or {}
    counts = sv.get("counts") or {}
    o = model["origin"]
    drawn = any(f["kind"] != "boundary" for f in model["features"])
    sec = iter(range(1, 20))
    H = lambda t: f'<h2><span class="n">{next(sec):02d}</span>{t}</h2>'  # noqa: E731
    b = []

    b.append(f'<header class="topbar"><div class="brand">{LOGO}<div class="brand-name">{APP_NAME} <span>/ {M.APP}</span></div></div>'
             f'<div class="crumb">{h(model["name"])}</div><div class="spacer"></div>'
             f'<button class="btn" onclick="print()">{icon("download")}<span class="lbl">Print / save PDF</span></button>'
             f'{theme_button()}</header><div class="wrap">')
    brand = h(model.get("company") or M.BRAND)
    b.append(f'<div class="cover"><div class="label">{brand}</div><h1>{h(model["name"])}</h1>'
             f'<div class="sub">Site security assessment: sightlines, blind spots, camera placement, access points '
             f'and perimeter</div><dl class="front">')
    for k, v in (("Prepared for", model.get("client")), ("Site", model.get("address")),
                 ("Prepared by", " · ".join(x for x in (model.get("assessor"), model.get("company")) if x)),
                 ("Reference", model.get("reference")), ("Report date", time.strftime("%d %B %Y")),
                 ("Survey date", (sv.get("date") or "")[:10]), ("Location", f'{o["lat"]:.6f}, {o["lon"]:.6f}')):
        if v:
            b.append(f"<div><dt>{k}</dt><dd>{h(v)}</dd></div>")
    b.append("</dl></div>")
    if not drawn:
        b.append('<p class="summary">Nothing has been drawn on the site plan yet, so there is nothing to assess. '
                 'Open the project with <span class="mono">isr site edit</span>, trace the boundary, buildings, '
                 'fences, planting and street over the base map, mark the doors, windows and gates, then '
                 'generate the report again.</p>')
    else:
        b.append(f'<p class="summary">{h(summary_text(r))}</p>')
    stat = lambda v, k, cls="": f'<div class="stat {cls}"><div class="v">{v}</div><div class="k">{k}</div></div>'  # noqa: E731
    b.append('<div class="stats">' +
             stat(f'{st["access_high"]}<small> / {st["access_total"]}</small>', "High-priority access points",
                  "bad" if st["access_high"] else "") +
             stat(f'{st["blind_pct"]:.0f}%<small> · {st["blind_m2"]:.0f} m²</small>', "Grounds nobody can see",
                  "bad" if st["blind_pct"] >= 20 else "") +
             stat(f'{st["camera_pct_now"]:.0f}%<small> → {st["camera_pct_recommended"]:.0f}%</small>',
                  "Camera coverage, now → recommended") +
             stat(f'{st["street_pct"]:.0f}%', "Visible from the street") +
             stat(f'{st["cameras_existing"]}<small> + {st["cameras_planned"] + st["cameras_recommended"]} proposed</small>',
                  "Cameras") +
             stat(f'{len(model["photos"])}', "Survey photos") + "</div>")

    # 1 findings
    b.append(H("Priority findings"))
    fs = [f for f in r["findings"]]
    if fs:
        b.append('<div class="findings">')
        for f in fs:
            b.append(f'<div class="finding" style="--c:{SEV[f["severity"]]}"><i></i><div><div class="t">{_pill(f["severity"])}'
                     f'<span>{h(f["title"])}</span><span class="sub">{h(f["category"])}</span></div>'
                     f'<p class="d">{h(f["detail"])}</p><p class="a"><b>Action</b>{h(f["action"])}</p></div></div>')
        b.append("</div>")
    else:
        b.append('<p class="empty-note">No findings.</p>')

    # 2 site plan
    b.append(H("Site plan and current coverage"))
    b.append('<p class="lead">The plan as drawn over the survey imagery, with what the existing cameras record '
             'and the ground that nobody can see. Access points are coloured by priority.</p>')
    b.append(f'<div class="mapcard">{mp.render(layers=("cameras_now", "blind"), photos=True)}<div class="legend">' +
             _sw("rgba(59,130,246,.55)", "Recorded by an existing camera") + _sw("rgba(229,72,77,.6)", "Unobserved (blind spot)") +
             _sw("#e5484d", "High priority") + _sw("#e08a00", "Medium") + _sw("#12a150", "Low") +
             _sw("#4ade80", "Hedge", True) + _sw("#d6a56b", "Fence", True) + _sw("#e5e7eb", "Wall", True) +
             _sw("#facc15", "Public way", True) + _sw("#fff;border:1px solid #111", "Survey photo and direction") +
             "</div></div>")

    # 3 sightlines
    b.append(H("Sightlines"))
    if st["has_street"]:
        b.append(f'<p class="lead">{st["natural_pct"]:.0f}% of the grounds can be seen by someone walking past on the public '
                 f'way or by an occupant at a ground-floor window ({st["street_pct"]:.0f}% from the street alone). '
                 f'The darkened ground is hidden from both. Being overlooked deters intruders; hidden ground relies '
                 f'on cameras.</p>')
    else:
        b.append('<p class="lead">No public way has been drawn, so only views from ground-floor windows count. '
                 'The darkened ground cannot be seen from any of them.</p>')
    b.append(f'<div class="mapcard">{mp.render(layers=("hidden",), blind=False)}<div class="legend">' +
             _sw("rgba(10,10,16,.75)", "Hidden from the street and from windows") + _sw("#facc15", "Public way", True) +
             "</div></div>")
    if r["blockers"]:
        b.append('<h3>What screens the grounds from the street</h3><div class="tablewrap"><table><tr><th>Feature</th>'
                 '<th>Type</th><th class="num">Height</th><th class="num">Hides from street</th>'
                 '<th class="num">…of which seen by nobody</th><th>Access points it hides</th></tr>')
        for k in r["blockers"]:
            b.append(f'<tr><td>{h(k["name"])}</td><td>{h(k["type"])}</td><td class="num">{k["height"]:.1f} m</td>'
                     f'<td class="num">{k["hides_m2"]:.0f} m²</td><td class="num">{k["hides_blind_m2"]:.0f} m²</td>'
                     f'<td>{h(", ".join(k["hides_access"])) or "–"}</td></tr>')
        b.append('</table></div><p class="sub">Each row is a what-if: the extra ground a passer-by would see if that one '
                 'feature were removed or cut below eye level. Screening also gives privacy, so this is a trade-off '
                 'to decide, not a defect.</p>')

    # 4 blind spots
    b.append(H("Blind spots"))
    if r["blind_spots"]:
        b.append('<div class="tablewrap"><table><tr><th></th><th>Where</th><th class="num">Area</th><th>Severity</th>'
                 '<th>Screened by</th><th>Reaches</th><th class="num">Fixed by camera plan</th><th>Position</th></tr>')
        for k in r["blind_spots"]:
            b.append(f'<tr><td>{_code(k["id"], "#e5484d")}</td><td>{h(k["where"].capitalize())}</td>'
                     f'<td class="num">{k["area_m2"]:.0f} m²</td><td>{_pill(k["severity"])}</td>'
                     f'<td>{h(", ".join(k["hidden_by"])) or "–"}</td><td>{h(", ".join(k["near_access"])) or "–"}</td>'
                     f'<td class="num">{k["fixed_by_plan"]:.0%}</td><td class="mono">{k["lat"]:.6f}, {k["lon"]:.6f}</td></tr>')
        b.append("</table></div>")
        shots = [(k, _thumbs(k["photos"], photos)) for k in r["blind_spots"]]
        if any(t for _, t in shots):
            b.append('<div class="cards">' + "".join(
                f'<div class="acard"><div class="t">{_code(k["id"], "#e5484d")}Survey photos looking at it</div>{t}</div>'
                for k, t in shots if t) + "</div>")
    else:
        b.append('<p class="empty-note">No unobserved area larger than '
                 f'{model["settings"]["min_blind_m2"]:.0f} m².</p>')

    # 5 camera plan
    b.append(H("Camera placement"))
    b.append(f'<p class="lead">Existing cameras in blue, recommended positions in green. Positions are chosen one at a time '
             f'from building corners and walls, each taking the spot and aim that adds the most: badly exposed access '
             f'points first, then ground nobody can see, then the rest. Assumed lens: {model["settings"]["camera_hfov"]:.0f}° '
             f'wide, useful to {model["settings"]["camera_range_m"]:.0f} m, mounted at '
             f'{model["settings"]["camera_height_m"]:.1f} m.</p>')
    b.append(f'<div class="mapcard">{mp.render(layers=("cameras", "proposed", "blind_after"), cameras="plan", blind=False)}'
             '<div class="legend">' + _sw("rgba(59,130,246,.55)", "Existing / planned coverage") +
             _sw("rgba(16,163,90,.6)", "Added by recommended cameras") +
             _sw("rgba(229,72,77,.6)", "Still unobserved") + "</div></div>")
    rows = []
    for c in r["cameras"]:
        rows.append(f'<tr><td>{_code(c["code"], "#3b82f6")}</td><td>{h(c["name"])}</td><td>{c["status"].capitalize()}</td>'
                    f'<td class="num">{c["height"]:.1f} m</td><td class="num">{c["bearing"]:.0f}° {M.compass(c["bearing"], True)}</td>'
                    f'<td class="num">{c["hfov"]:.0f}° / {c["range"]:.0f} m</td><td class="num">{c["covers_m2"]:.0f} m²</td>'
                    f'<td>{h(", ".join(c["covers"])) or "–"}</td><td class="mono">{c["lat"]:.6f}, {c["lon"]:.6f}</td></tr>')
    for p in r["recommended"]:
        rows.append(f'<tr><td>{_code(p["id"], "#10a35a")}</td><td>{h(p["mount"])}</td><td>Recommended</td>'
                    f'<td class="num">{p["height"]:.1f} m</td><td class="num">{p["bearing"]:.0f}° {M.compass(p["bearing"], True)}</td>'
                    f'<td class="num">{p["hfov"]:.0f}° / {p["range"]:.0f} m</td><td class="num">+{p["adds_m2"]:.0f} m²</td>'
                    f'<td>{h(", ".join(p["covers"])) or "–"}</td><td class="mono">{p["lat"]:.6f}, {p["lon"]:.6f}</td></tr>')
    if rows:
        b.append('<div class="tablewrap"><table><tr><th></th><th>Camera / mounting point</th><th>Status</th>'
                 '<th class="num">Height</th><th class="num">Aim</th><th class="num">Lens / range</th>'
                 '<th class="num">Ground covered</th><th>Access points covered</th><th>Position</th></tr>' +
                 "".join(rows) + "</table></div>")
    else:
        b.append('<p class="empty-note">No cameras, and nowhere to recommend one until buildings or a boundary are drawn.</p>')

    # guard posts and patrols (only when asked for)
    gp = r.get("guards")
    if gp and gp["scenarios"]:
        asm = gp["assumptions"]
        ul = lambda items, cls: "".join(f'<li class="{cls}">{h(x)}</li>' for x in items)  # noqa: E731
        b.append(H("Guard posts and patrols"))
        b.append(f'<p class="lead">Where officers would do most good, for each number you might have on site. A post is '
                 f'judged by what a standing officer sees from it (eye height {asm["eye_height_m"]:.1f} m, out to '
                 f'{asm["range_m"]} m); a patrol by the shortest walking loop round everything the posts cannot see, at '
                 f'{asm["speed_ms"]:.1f} m/s with {asm["stop_s"]:.0f} s at each stop.</p>')
        if gp["posts"]:
            b.append('<h3>Posts</h3><div class="cards">')
            for p in gp["posts"]:
                b.append(f'<div class="acard"><div class="t">{_code(p["id"], "#0284c7")}{h(p["where"].capitalize())}</div>'
                         f'<div class="sub"><span class="mono">{p["lat"]:.6f}, {p["lon"]:.6f}</span></div>'
                         f'<ul class="pc">{ul(p["pros"], "pro")}{ul(p["cons"], "con")}</ul>'
                         f'<div><b>Suggestion:</b> {h(p["suggestion"])}</div></div>')
            b.append("</div>")
        for sc_ in gp["scenarios"]:
            n = sc_["guards"]
            best = sc_["options"][0]
            b.append(f'<h3 class="scn">With {n} guard{"s" if n != 1 else ""}: {h(best["title"])}</h3>'
                     f'<p class="lead">{h(sc_["why"])}</p>')
            b.append(f'<div class="mapcard">{mp.render(layers=("cameras_now",), blind=False, guards=best, title=f"{n} guards")}'
                     '<div class="legend">' + _sw("#0284c7", "Guard post") + _sw("#38bdf8", "Patrol route and stops, in order", True) +
                     _sw("rgba(59,130,246,.55)", "Existing camera coverage") + "</div></div>")
            b.append('<div class="cards opts">')
            for o in sc_["options"]:
                rt = o["route"]
                facts = [f'{o["guard_pct"]}% of grounds always watched' if o["statics"] else "No fixed watch"]
                if rt:
                    facts.append(f'{rt["length_m"]} m loop · every {rt["revisit_min"]:g} min')
                tag = '<span class="pill" style="--c:var(--good)">Recommended</span>' if o["recommended"] else \
                    '<span class="pill" style="--c:var(--muted)">Alternative</span>'
                stops = (f'<div class="sub" style="margin-top:6px">Route: {h(" → ".join(st_["name"] for st_ in rt["stops"]))} → back to start</div>'
                         if rt else "")
                tips = "".join(f"<div><b>Suggestion:</b> {h(x)}</div>" for x in o["suggestions"])
                b.append(f'<div class="acard"><div class="t">{h(o["title"].capitalize())}{tag}</div>'
                         f'<div class="sub">{h(" · ".join(facts))}</div><ul class="pc">{ul(o["pros"], "pro")}{ul(o["cons"], "con")}</ul>'
                         f'{tips}{stops}</div>')
            b.append("</div>")
        if gp["unreachable"]:
            b.append(f'<p class="sub">Not reachable on foot inside the boundary as drawn, so left off the routes: '
                     f'{h(", ".join(gp["unreachable"]))}.</p>')
        b.append('<p class="sub">These are geometric recommendations: what can be seen and how far it is to walk. They do not '
                 'account for lighting, shelter, radio cover, welfare breaks, shift patterns or the specific threat. '
                 'Walk the route before adopting it.</p>')

    # 6 access points
    b.append(H("Access points"))
    if r["access"]:
        b.append('<div class="tablewrap"><table><tr><th></th><th>Access point</th><th>Type</th><th>Priority</th>'
                 '<th>Camera now</th><th>After plan</th><th>From street</th><th>From a window</th>'
                 '<th class="num">Unobserved ground within 8 m</th></tr>')
        for a in r["access"]:
            col = SEV[a["priority"]]
            after = a["camera_planned"] or bool(a["recommended"])
            b.append(f'<tr><td>{_code(a["code"], col)}</td><td>{h(a["name"])}</td><td>{h(a["type"])}</td>'
                     f'<td>{_pill(a["priority"])}</td><td>{_yn(a["camera_now"])}</td><td>{_yn(after)}</td>'
                     f'<td>{_yn(a["street"])}</td><td>{_yn(a["window"])}</td><td class="num">{a["concealment"]:.0%}</td></tr>')
        b.append('</table></div><div class="cards">')
        for a in r["access"]:
            col = SEV[a["priority"]]
            cams = ", ".join(a["cameras"]) or "none"
            why = (a["why"][0][0].upper() + a["why"][0][1:] + "".join("; " + w for w in a["why"][1:]) + ".") if a["why"] else \
                "Watched and overlooked."
            note = f'<p class="sub">Note: {h(a["notes"])}</p>' if a["notes"] else ""
            blk = f'<span>View from street blocked by</span><span>{h(a["street_blocker"])}</span>' if a["street_blocker"] else ""
            b.append(f'<div class="acard"><div class="t">{_code(a["code"], col)}{h(a["name"])}{_pill(a["priority"])}</div>'
                     f'<div class="sub">{h(a["type"])}{" on " + h(a["on"]) if a["on"] else ""} · '
                     f'<span class="mono">{a["lat"]:.6f}, {a["lon"]:.6f}</span></div>'
                     f'<div class="checks"><span>Cameras covering it</span><span>{h(cams)}</span>'
                     f'<span>Recommended camera</span><span>{h(", ".join(a["recommended"])) or "–"}</span>{blk}</div>'
                     f'<div>{h(why)}</div><div style="margin-top:6px"><b>Action:</b> {h(a["action"])}</div>{note}'
                     f'{_thumbs(a["photos"], photos)}</div>')
        b.append("</div>")
    else:
        b.append('<p class="empty-note">No access points marked.</p>')

    # seen during the survey (only when the detector was run)
    if model.get("detections"):
        b.append(H("Seen during the survey"))
        conf = [d for d in r["detections"] if d["status"] == "confirmed"]
        todo = sum(d["status"] == "open" for d in r["detections"])
        b.append('<p class="lead">People, vehicles and loose objects picked out of the drone photos by the detector and '
                 'confirmed by the assessor, placed on the plan (pink markers on the site plan above).</p>')
        if conf:
            b.append('<div class="cards">')
            for d in conf:
                note = f'<div style="margin-top:6px">{h(d["note"])}</div>' if d["note"] else ""
                b.append(f'<div class="acard"><div class="t">{_code(d["code"], "#d946ef")}'
                         f'{h(M.DETECTION_TYPES.get(d["type"], "Item"))}</div>'
                         f'<div class="sub"><span class="mono">{d["lat"]:.6f}, {d["lon"]:.6f}</span> · seen in '
                         f'{d["seen"]} photo{"s" if d["seen"] != 1 else ""}</div><div style="margin-top:6px">{h(d["context"])}</div>'
                         f'{note}<div class="thumbs"><a href="{h(d["crop"])}" style="width:150px;height:150px">'
                         f'<img src="{h(d["crop"])}" alt=""></a></div></div>')
            b.append("</div>")
        else:
            b.append('<p class="empty-note">Nothing confirmed.</p>')
        if todo:
            b.append(f'<p class="sub">{todo} further detection{"s were" if todo != 1 else " was"} not reviewed and '
                     f'{"are" if todo != 1 else "is"} not shown.</p>')

    # 7 perimeter
    b.append(H("Perimeter"))
    per = r["perimeter"]
    if per:
        b.append(f'<p class="lead">The boundary is {per["length_m"]:.0f} m long; {per["closed_pct"]}% of it has a wall, '
                 f'fence, hedge or building on it.</p>')
        open_m = sum(g["length_m"] for g in per["gaps"] if not g["declared"])
        b.append(f'<div class="mapcard">{mp.render(cameras="none", blind=False, gaps=per["gaps"], title="Perimeter")}'
                 '<div class="legend">' + _sw("#e5484d", "Unmarked gap", True) +
                 _sw("#e08a00", "Marked opening (gate, driveway)", True) + _sw("#4ade80", "Hedge", True) +
                 _sw("#d6a56b", "Fence (dashed: see-through)", True) + _sw("#e5e7eb", "Wall", True) + "</div></div>")
        b.append('<div class="stats">' + stat(f'{per["length_m"]:.0f} m', "Boundary length") +
                 stat(f'{per["closed_pct"]}%', "Closed by a barrier or building") +
                 stat(f'{open_m:.0f} m', "Open and unmarked", "bad" if open_m else "") +
                 stat(str(sum(1 for g in per["gaps"] if g["declared"])), "Marked openings") + "</div>")
        if per["gaps"]:
            b.append('<h3>Openings</h3><div class="tablewrap"><table><tr><th></th><th>Opening</th>'
                     '<th class="num">Length</th><th>Side</th><th>Status</th><th>Position</th></tr>')
            for g in per["gaps"]:
                code = "" if g["declared"] else _code(g["code"], "#e5484d")
                b.append(f'<tr><td>{code}</td><td>{h(g["declared"] or "Unmarked gap")}</td><td class="num">{g["length_m"]:.1f} m</td>'
                         f'<td>{g["side"].capitalize()}</td><td>{"Marked access point" if g["declared"] else _yn(False, no="Open, not marked as an access point")}</td>'
                         f'<td class="mono">{g["lat"]:.6f}, {g["lon"]:.6f}</td></tr>')
            b.append("</table></div>")
        if per["barriers"]:
            b.append('<h3>Barriers</h3><div class="tablewrap"><table><tr><th>Barrier</th><th>Type</th><th class="num">Height</th>'
                     '<th class="num">Length</th><th>Blocks view</th><th>Notes</th></tr>')
            for k in per["barriers"]:
                low = ' <span class="no">(low)</span>' if k["low"] else ""
                b.append(f'<tr><td>{h(k["name"])}</td><td>{h(k["type"])}</td><td class="num">{k["height"]:.1f} m{low}</td>'
                         f'<td class="num">{k["length_m"]:.0f} m</td><td>{"No" if k["see_through"] or k["height"] < 1.2 else "Yes"}</td>'
                         f'<td>{h(k.get("notes") or "")}</td></tr>')
            b.append("</table></div>")
    else:
        b.append('<p class="empty-note">No boundary drawn, so the perimeter is not assessed.</p>')

    # 8 survey record
    b.append(H("Survey record"))
    bm = model.get("basemap")
    base = ("no base map (no drone photo looks down at the ground)" if not bm else
            f"base map from {bm.get('straight_down', 0)} straight-down photo(s)" if bm.get("straight_down") else
            f"base map from {bm.get('angled', 0)} angled drone shots, so roofs and treetops are drawn displaced "
            f"from their true position by roughly their height")
    b.append(f'<p class="lead">{len(model["photos"])} photos: {counts.get("nadir", 0)} straight-down aerial, '
             f'{counts.get("oblique", 0)} angled aerial, {counts.get("ground", 0)} ground-level, '
             f'{counts.get("unplaced", 0)} without a GPS position; {base}. Surveyed {h(sv.get("date", "–"))}.</p>')
    unshot = [a["name"] for a in r["access"] if not a["photos"]]
    if unshot and model["photos"]:
        b.append(f'<p class="sub">No ground or oblique photo looks at: {h(", ".join(unshot))}.</p>')
    if model["photos"]:
        b.append('<div class="photos">' + "".join(
            f'<figure><a href="{h(p["file"])}"><img src="{h(p["thumb"])}" alt=""></a>'
            f'<figcaption>{h(p["name"])} · {p["kind"]}</figcaption></figure>' for p in model["photos"]) + "</div>")
    if model.get("notes"):
        b.append(f'<h3>Assessor notes</h3><div class="notes">{h(model["notes"])}</div>')

    # 9 method
    b.append(H("Method and limits"))
    b.append(f"""<ul class="limits">
<li><b>What is measured.</b> The plan is rasterised to a {st["cell_m"]:.2f} m grid. A spot counts as seen when a straight
line from the observer (camera at its mounting height, passer-by or occupant at {model["settings"]["eye_height_m"]:.1f} m)
to a person standing there clears every wall, fence, hedge, building and planting at the heights entered.</li>
<li><b>It is only as good as the plan.</b> Features are traced by hand over imagery placed by consumer GPS, so positions
are good to a few metres and heights are estimates. The ground is treated as flat. Check anything decisive on site.</li>
<li><b>Camera figures are geometric.</b> Coverage means an unobstructed view within the stated range and angle. It does
not account for lens quality, night performance, glare, weather, or whether footage is recorded and reviewed.</li>
<li><b>Not assessed.</b> Lighting, locks, glazing, alarms, upper floors and roofs, climbable features, and how the site
is used at different times. A daytime photo survey cannot show these; they need a walk-through.</li>
<li><b>Priorities.</b> An access point scores for having no camera (2 for doors, gates and garages, 1 for windows),
for being invisible from both the street and windows (2), and for unobserved ground within 8 m (1 above 25%,
2 above 50%). 5 or more is high, 3–4 medium.</li>
</ul>""")
    by = " · ".join(x for x in (model.get("assessor"), model.get("company")) if x)
    b.append(f'<p class="foot">{"Prepared by " + h(by) + (" for " + h(model["client"]) if model.get("client") else "") + ". " if by else ""}'
             'This report describes what could be observed and measured from a photographic survey on the date shown. '
             'It identifies where the site is exposed and how that could be reduced; it is not a guarantee against '
             'intrusion, and the recommendations should be confirmed on site before work is ordered. '
             f'Produced with {APP_NAME}.</p></div>')
    return page(f"Site assessment: {h(model['name'])}", "".join(b), CSS, "", None, default_theme="light")


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------


def _ll(model, pts):
    o = model["origin"]
    return [from_local(o["lat"], o["lon"], px, py) for px, py in pts]


def write_geojson(model: dict, r: dict, path: Path) -> None:
    feats = []
    for f in model["features"]:
        ring = [[round(lon, 7), round(lat, 7)] for lat, lon in _ll(model, f["pts"])]
        g = M.GEOMETRY[f["kind"]]
        geom = {"type": "Point", "coordinates": ring[0]} if g == "point" else \
            {"type": "LineString", "coordinates": ring} if g == "line" else \
            {"type": "Polygon", "coordinates": [ring + [ring[0]]]}
        props = {"title": f["name"], "kind": f["kind"], "type": M.feature_label(f)}
        for k in ("height", "bearing", "hfov", "range", "status"):
            if k in f:
                props[k] = f[k]
        feats.append({"type": "Feature", "geometry": geom, "properties": props})
    a_by = {a["id"]: a for a in r["access"]}
    for f, ft in zip(model["features"], feats):
        a = a_by.get(f["id"])
        if a:
            ft["properties"].update(priority=a["priority"], camera_now=a["camera_now"], seen_from_street=a["street"],
                                    code=a["code"], action=a["action"],
                                    **{"marker-color": {"high": "#e5484d", "medium": "#e08a00", "low": "#12a150"}[a["priority"]]})
    for p in r["recommended"]:
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
                      "properties": {"title": f"{p['id']} recommended camera", "kind": "camera", "status": "recommended",
                                     "mount": p["mount"], "height": p["height"], "bearing": p["bearing"],
                                     "hfov": p["hfov"], "range": p["range"], "marker-color": "#10a35a"}})
    for k in r["blind_spots"]:
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [k["lon"], k["lat"]]},
                      "properties": {"title": f"{k['id']} blind spot", "kind": "blind_spot", "area_m2": k["area_m2"],
                                     "severity": k["severity"], "marker-color": "#e5484d"}})
    for d in r.get("detections", []):
        if d["status"] == "confirmed":
            feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [d["lon"], d["lat"]]},
                          "properties": {"title": f"{d['code']} {M.DETECTION_TYPES.get(d['type'], 'item')}",
                                         "kind": "detection", "context": d["context"], "note": d["note"],
                                         "marker-color": "#d946ef"}})
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))


def write_kml(model: dict, r: dict, path: Path) -> None:
    out = ['<?xml version="1.0" encoding="UTF-8"?>', '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
           f"<name>Site assessment: {x(model['name'])}</name>"]
    pm = lambda name, desc, lat, lon, col: (  # noqa: E731
        f"<Placemark><name>{x(name)}</name><description>{x(desc)}</description><Style><IconStyle><color>{col}</color>"
        f"</IconStyle></Style><Point><coordinates>{lon:.7f},{lat:.7f},0</coordinates></Point></Placemark>")
    col = {"high": "ff4d48e5", "medium": "ff008ae0", "low": "ff50a112"}
    out.append("<Folder><name>Access points</name>")
    out += [pm(f"{a['code']} {a['name']}", f"{a['type']}, {a['priority']} priority. {a['action']}", a["lat"], a["lon"],
               col[a["priority"]]) for a in r["access"]]
    out.append("</Folder><Folder><name>Cameras</name>")
    out += [pm(f"{c['code']} {c['name']}", f"{c['status']}, aim {c['bearing']:.0f} deg, {c['height']} m high",
               c["lat"], c["lon"], "fff6823b") for c in r["cameras"]]
    out += [pm(f"{p['id']} recommended", f"{p['mount']}, aim {p['bearing']} deg, {p['height']} m high",
               p["lat"], p["lon"], "ff5aa310") for p in r["recommended"]]
    out.append("</Folder><Folder><name>Blind spots</name>")
    out += [pm(f"{k['id']} {k['area_m2']:.0f} m2", f"{k['severity']} severity, {k['where']}", k["lat"], k["lon"],
               "ff4d48e5") for k in r["blind_spots"]]
    out.append("</Folder><Folder><name>Site plan</name>")
    for f in model["features"]:
        g = M.GEOMETRY[f["kind"]]
        if g == "point":
            continue
        ll = _ll(model, f["pts"] + ([f["pts"][0]] if g == "polygon" else []))
        coords = " ".join(f"{lon:.7f},{lat:.7f},0" for lat, lon in ll)
        body = (f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{coords}</coordinates></LinearRing>"
                f"</outerBoundaryIs></Polygon>") if g == "polygon" else f"<LineString><coordinates>{coords}</coordinates></LineString>"
        out.append(f"<Placemark><name>{x(f['name'])}</name><Style><LineStyle><color>ffffffff</color><width>2</width>"
                   f"</LineStyle><PolyStyle><color>33ffffff</color></PolyStyle></Style>{body}</Placemark>")
    out.append("</Folder></Document></kml>")
    path.write_text("\n".join(out))


def write_csv(r: dict, path: Path) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ref", "item", "name", "priority", "detail", "action", "lat", "lon"])
        for a in r["access"]:
            w.writerow([a["code"], "access point", a["name"], a["priority"], "; ".join(a["why"]), a["action"],
                        a["lat"], a["lon"]])
        for k in r["blind_spots"]:
            w.writerow([k["id"], "blind spot", k["where"], k["severity"],
                        f"{k['area_m2']:.0f} m2; screened by {', '.join(k['hidden_by']) or 'n/a'}",
                        f"{k['fixed_by_plan']:.0%} covered by camera plan", k["lat"], k["lon"]])
        for c in r["cameras"]:
            w.writerow([c["code"], f"camera ({c['status']})", c["name"], "",
                        f"aim {c['bearing']:.0f} deg, {c['hfov']:.0f} deg lens, {c['range']:.0f} m, {c['height']:.1f} m high",
                        "", c["lat"], c["lon"]])
        for p in r["recommended"]:
            w.writerow([p["id"], "camera (recommended)", p["mount"], "",
                        f"aim {p['bearing']} deg, {p['hfov']} deg lens, {p['range']:.0f} m, {p['height']:.1f} m high",
                        f"covers {', '.join(p['covers']) or 'open ground'}", p["lat"], p["lon"]])
        for d in r.get("detections", []):
            if d["status"] == "confirmed":
                w.writerow([d["code"], "seen during survey", M.DETECTION_TYPES.get(d["type"], "item"), "",
                            d["context"], d["note"], d["lat"], d["lon"]])
        for g in (r["perimeter"] or {}).get("gaps", []):
            w.writerow(["", "perimeter opening", g["declared"] or "unmarked gap", "" if g["declared"] else "medium",
                        f"{g['length_m']:.1f} m on the {g['side']} side", "", g["lat"], g["lon"]])


def write_report(model: dict, out: Path, an: Analysis | None = None) -> Path:
    """Analyse (unless given), then write report.html, overlays and exports into the project folder."""
    out = Path(out)
    an = an or Analysis(model)
    assets = out / "report_assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name in an.layers:
        (assets / f"{name}.png").write_bytes(an.overlay_png(name))
    report = out / "report.html"
    report.write_text(build_html(model, an), encoding="utf-8")
    (out / "assessment.json").write_text(json.dumps(an.result, indent=1))
    write_geojson(model, an.result, out / "site.geojson")
    write_kml(model, an.result, out / "site.kml")
    write_csv(an.result, out / "assessment.csv")
    return report
