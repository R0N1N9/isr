"""A printable search report: what was scanned, and the finds a reviewer confirmed.

The review page (``report.html``) is a working tool; this is the page handed
to whoever is running the search. It lists confirmed finds with coordinates
and close-ups, or, if nothing has been confirmed yet, the open candidates.
"""

from __future__ import annotations

import time
from html import escape as h
from pathlib import Path

from ..geo import to_local
from ..ui import APP_NAME, LOGO, icon, page, theme_button

CSS = r"""
.wrap{max-width:1100px;margin:0 auto;padding:26px 24px 70px}
h1{font-size:28px;letter-spacing:-.025em;margin:0 0 4px;font-weight:700}
h2{font-size:19px;margin:36px 0 8px;font-weight:650}
.sub{color:var(--muted);font-size:13px}
.lead{color:var(--ink-2);max-width:80ch}
.stats{margin:16px 0}
.cover{background:var(--surface);border:1px solid var(--line);border-radius:var(--r-lg);padding:22px 24px;box-shadow:var(--shadow)}
.cover .label{margin-bottom:8px;color:var(--accent)}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}
.find{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);overflow:hidden;break-inside:avoid}
.find img{width:100%;aspect-ratio:1/1;object-fit:cover;display:block;background:#000}
.find div{padding:10px 12px}
.find .t{font-weight:650;display:flex;gap:8px;align-items:center}
.code{display:inline-grid;place-items:center;min-width:26px;height:20px;padding:0 5px;border-radius:3px;font:700 11px var(--mono);color:#fff;background:var(--c,var(--bad))}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:var(--r);background:var(--surface);margin:10px 0}
table{border-collapse:collapse;width:100%;font-size:13px}
th{text-align:left;font-size:11px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);padding:9px 12px;border-bottom:1px solid var(--line);background:var(--surface-2)}
td{padding:9px 12px;border-bottom:1px solid var(--line)}tr:last-child td{border-bottom:0}
.mapcard{background:var(--surface);border:1px solid var(--line);border-radius:var(--r-lg);overflow:hidden;margin:12px 0}
.mapcard svg{display:block;width:100%;height:auto;max-height:520px;background:var(--surface-2)}
.limits li{margin:5px 0;max-width:88ch;color:var(--ink-2)}
@media print{.topbar,#toast{display:none!important}body{background:#fff}body::before{display:none}.wrap{padding:0;max-width:none}}
"""


def _sketch(data: dict, finds: list) -> str:
    """Where the photos were taken (grey) and where the finds are (numbered), north up, to scale."""
    pts = [(p["lat"], p["lon"]) for p in data["photos"] if p.get("lat") is not None]
    pts += [(d["lat"], d["lon"]) for _, d in finds if d.get("lat") is not None]
    if not pts:
        return ""
    lat0, lon0 = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
    loc = lambda la, lo: to_local(lat0, lon0, la, lo)  # noqa: E731
    xy = [loc(*p) for p in pts]
    xs, ys = [p[0] for p in xy], [p[1] for p in xy]
    m = max(15.0, 0.08 * max(max(xs) - min(xs), max(ys) - min(ys)))
    x0, x1, y0, y1 = min(xs) - m, max(xs) + m, min(ys) - m, max(ys) + m
    u = max(x1 - x0, y1 - y0) / 100
    o = [f'<svg viewBox="{x0:.1f} {-y1:.1f} {x1 - x0:.1f} {y1 - y0:.1f}" font-family="system-ui,sans-serif">']
    for p in data["photos"]:
        if p.get("footprint"):
            ring = " ".join("{:.1f},{:.1f}".format(*(lambda q: (q[0], -q[1]))(loc(la, lo))) for la, lo in p["footprint"])
            o.append(f'<polygon points="{ring}" fill="#888" fill-opacity=".08" stroke="#888" stroke-opacity=".35" stroke-width="{u * .15:.2f}"/>')
    for p in data["photos"]:
        if p.get("lat") is not None:
            x, y = loc(p["lat"], p["lon"])
            o.append(f'<circle cx="{x:.1f}" cy="{-y:.1f}" r="{u * .5:.2f}" fill="#888"/>')
    for n, (_, d) in enumerate(finds, 1):
        if d.get("lat") is None:
            continue
        x, y = loc(d["lat"], d["lon"])
        o.append(f'<circle cx="{x:.1f}" cy="{-y:.1f}" r="{u * 1.7:.2f}" fill="#e5484d" stroke="#fff" stroke-width="{u * .3:.2f}"/>'
                 f'<text x="{x:.1f}" y="{-y + u * .6:.1f}" font-size="{u * 1.7:.2f}" font-weight="700" fill="#fff" text-anchor="middle">{n}</text>')
    bar = next(b for b in (1000, 500, 200, 100, 50, 20, 10, 5) if b <= (x1 - x0) * 0.3)
    o.append(f'<path d="M{x0 + u * 3:.1f} {-y0 - u * 3:.1f}h{bar}" stroke="currentColor" stroke-width="{u * .5:.2f}"/>'
             f'<text x="{x0 + u * 3 + bar / 2:.1f}" y="{-y0 - u * 4.2:.1f}" font-size="{u * 2:.2f}" text-anchor="middle" fill="currentColor">{bar} m</text>'
             f'<text x="{x1 - u * 4:.1f}" y="{-y1 + u * 5:.1f}" font-size="{u * 2.4:.2f}" font-weight="700" text-anchor="middle" fill="currentColor">N ↑</text>')
    return '<div class="mapcard">' + "".join(o) + "</svg></div>"


def write_search_report(data: dict, status: dict, out_dir: Path, name: str = "") -> Path:
    """``status`` maps detection id -> "confirmed" / "dismissed" (from the review page)."""
    s = data["summary"]
    rows = [(p, d) for p in data["photos"] for d in p["detections"]]
    rows.sort(key=lambda t: -t[1]["score"])
    confirmed = [t for t in rows if status.get(t[1]["id"]) == "confirmed"]
    open_ = [t for t in rows if not status.get(t[1]["id"])]
    dismissed = len(rows) - len(confirmed) - len(open_)
    finds = confirmed or open_[:24]
    kind = "Confirmed finds" if confirmed else "Candidates awaiting review"
    stat = lambda v, k: f'<div class="stat"><div class="v">{v}</div><div class="k">{k}</div></div>'  # noqa: E731
    b = [f'<header class="topbar"><div class="brand">{LOGO}<div class="brand-name">{APP_NAME} <span>/ SAR</span></div></div>'
         f'<div class="spacer"></div><button class="btn" onclick="print()">{icon("download")}<span class="lbl">Print / save PDF</span></button>'
         f'{theme_button()}</header><div class="wrap">',
         f'<div class="cover"><div class="label">Search report</div><h1>{h(name or Path(s["source"]).name)}</h1>'
         f'<div class="sub">Photos scanned {h(s["generated"])} · report written {time.strftime("%Y-%m-%d %H:%M")} · '
         f'source {h(s["source"])}</div></div>',
         '<div class="stats">' + stat(s["photos"], "Photos scanned") + stat(s["photos_geolocatable"], "Geolocated") +
         stat(len(confirmed), "Confirmed finds") + stat(len(open_), "Not yet reviewed") + stat(dismissed, "Dismissed") +
         (stat(h(", ".join(s["config"]["colors"])), "Target colours") if s["config"]["colors"] else "") + "</div>"]
    if not confirmed:
        b.append('<p class="lead"><b>Nothing has been confirmed yet.</b> The strongest unreviewed candidates are listed '
                 'below; confirm or dismiss them on the review page and write this report again.</p>')
    b.append(f"<h2>{kind}</h2>")
    if finds:
        b.append(_sketch(data, finds))
        b.append('<div class="tablewrap"><table><tr><th>#</th><th>Latitude, longitude</th><th>What</th><th>Score</th>'
                 '<th>Position from</th><th>Photo</th><th>Taken</th></tr>')
        for n, (p, d) in enumerate(finds, 1):
            ll = f'{d["lat"]:.6f}, {d["lon"]:.6f}' if d.get("lat") is not None else "no GPS"
            src = {"projected": "pixel projected to ground", "drone": "drone position only"}.get(d["position_source"], "–")
            b.append(f'<tr><td><span class="code">{n}</span></td><td class="mono">{ll}</td><td>{h(d["label"])}</td>'
                     f'<td>{d["score"]:.2f}</td><td>{src}</td><td>{h(p["rel"])}</td><td>{h(str(p["taken_at"] or "–"))}</td></tr>')
        b.append('</table></div><div class="cards">')
        for n, (p, d) in enumerate(finds, 1):
            img = f'<img src="{h(d["crop"])}" alt="">' if d.get("crop") else ""
            ll = f'{d["lat"]:.6f}, {d["lon"]:.6f}' if d.get("lat") is not None else "no GPS"
            b.append(f'<div class="find">{img}<div><div class="t"><span class="code">{n}</span>{h(d["label"])}</div>'
                     f'<div class="mono">{ll}</div><div class="sub">{h(p["rel"])}</div></div></div>')
        b.append("</div>")
    else:
        b.append('<p class="lead">The detectors flagged nothing in these photos.</p>')
    b.append("""<h2>How to read this</h2><ul class="limits">
<li><b>This narrows a search; it does not clear an area.</b> Canopy, shadow and drab or camouflage clothing hide
people from colour-based detection. Someone still needs to look at the photos of high-probability areas.</li>
<li>Positions come from the aircraft's GPS, height and camera angle projected onto flat ground: typically within a
few metres for straight-down photos, worse for angled ones and on slopes. "Drone position only" means the find could
not be projected and the aircraft's own position is given.</li>
<li>Coordinates are WGS84 decimal degrees. The same finds are in <span class="mono">detections.gpx / .kml / .csv</span>
in the job folder for GPS units and mapping tools.</li></ul></div>""")
    path = out_dir / "search_report.html"
    path.write_text(page(f"Search report: {h(name)}", "".join(b), CSS, "", None, default_theme="light"), encoding="utf-8")
    return path
