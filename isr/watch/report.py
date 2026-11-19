"""The activity report for a stakeout: when things move, where in the view, and what that means for guarding."""

from __future__ import annotations

from html import escape as h
from pathlib import Path

from ..ui import APP_NAME, LOGO, icon, page, theme_button
from .pipeline import DAYS

CSS = r"""
.wrap{max-width:1120px;margin:0 auto;padding:26px 24px 70px}
h1{font-size:28px;letter-spacing:-.025em;margin:0 0 4px;font-weight:700}
h2{font-size:19px;margin:38px 0 8px;font-weight:650}
.sub{color:var(--muted);font-size:13px}.lead{color:var(--ink-2);max-width:80ch}
.cover{background:var(--surface);border:1px solid var(--line);border-radius:var(--r-lg);padding:22px 24px;box-shadow:var(--shadow)}
.cover .label{margin-bottom:8px;color:var(--accent)}
.stats{margin:16px 0}
.card{background:var(--surface);border:1px solid var(--line);border-radius:var(--r-lg);padding:16px 18px;overflow-x:auto}
.grid2{display:grid;grid-template-columns:1.25fr 1fr;gap:16px;align-items:start}
@media (max-width:900px){.grid2{grid-template-columns:1fr}}
.scene{position:relative;border-radius:var(--r);overflow:hidden;border:1px solid var(--line);background:#000}
.scene img{display:block;width:100%}.scene img+img{position:absolute;inset:0;height:100%}
.advice{display:grid;gap:8px}
.adv{display:grid;grid-template-columns:4px 1fr;background:var(--surface);border:1px solid var(--line);border-radius:var(--r);overflow:hidden}
.adv i{background:var(--c)}.adv div{padding:11px 15px;max-width:92ch}
.frames{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px}
.frames figure{margin:0;background:var(--surface);border:1px solid var(--line);border-radius:var(--r);overflow:hidden;break-inside:avoid}
.frames img{width:100%;aspect-ratio:4/3;object-fit:cover;display:block;background:#000}
.frames figcaption{padding:7px 10px;font-size:12px;color:var(--muted)}
.limits li{margin:5px 0;max-width:88ch;color:var(--ink-2)}
@media print{.topbar,#toast,#isrbar{display:none!important}body{background:#fff}body::before{display:none}.wrap{padding:0;max-width:none}}
"""

ADV = {"peak": "var(--accent)", "pattern": "var(--accent-2)", "out_of_hours": "var(--bad)", "quiet": "var(--warn)",
       "info": "var(--muted)"}


def heatmap_svg(rate, cover, closed=None, cell=26) -> str:
    """Seven rows (Mon..Sun) by 24 hours, coloured by movements per hour watched. Grey = not watched."""
    top = max((v for row in rate for v in row), default=0) or 1
    w, hh, lx, ty = cell * 24, cell * 7, 40, 18
    o = [f'<svg viewBox="0 0 {w + lx + 6} {hh + ty + 6}" style="width:100%;max-width:760px;height:auto;display:block" '
         f'font-family="system-ui,sans-serif" role="img" aria-label="Movement by hour and day of the week">']
    for hr in range(0, 24, 3):
        o.append(f'<text x="{lx + hr * cell + 2}" y="12" font-size="10" fill="currentColor" opacity=".6">{hr:02d}</text>')
    for d in range(7):
        o.append(f'<text x="0" y="{ty + d * cell + cell * .65:.0f}" font-size="11" fill="currentColor" opacity=".8">{DAYS[d]}</text>')
        for hr in range(24):
            v, c = rate[d][hr], cover[d][hr]
            x, y = lx + hr * cell, ty + d * cell
            if not c:
                fill, op = "currentColor", 0.05
            else:
                fill, op = "#f97316", 0.07 + 0.93 * (v / top) ** 0.6 if v else 0.0
            base = f'<rect x="{x + 1}" y="{y + 1}" width="{cell - 2}" height="{cell - 2}" rx="4" fill="currentColor" opacity=".06"/>'
            tip = f"{DAYS[d]} {hr:02d}:00: " + (f"{v:g} an hour" if c else "not watched")
            o.append(base + (f'<rect x="{x + 1}" y="{y + 1}" width="{cell - 2}" height="{cell - 2}" rx="4" fill="{fill}" '
                             f'opacity="{op:.2f}"><title>{tip}</title></rect>' if c and v else
                             f'<rect x="{x + 1}" y="{y + 1}" width="{cell - 2}" height="{cell - 2}" rx="4" fill="transparent">'
                             f'<title>{tip}</title></rect>' if c else
                             f'<path d="M{x + 6} {y + cell - 6}L{x + cell - 6} {y + 6}" stroke="currentColor" opacity=".25"><title>{tip}</title></path>'))
    if closed and closed[0] != closed[1]:  # mark the hours the site is meant to be closed
        hr = closed[0]
        while hr != closed[1]:
            o.append(f'<rect x="{lx + hr * cell}" y="{ty + hh + 1}" width="{cell}" height="3" fill="#e5484d" opacity=".7"/>')
            hr = (hr + 1) % 24
    o.append("</svg>")
    return "".join(o)


def advice_html(advice) -> str:
    return '<div class="advice">' + "".join(
        f'<div class="adv" style="--c:{ADV.get(a["kind"], "var(--muted)")}"><i></i><div>{h(a["text"])}</div></div>'
        for a in advice) + "</div>"


def write_report(d: dict, out: Path) -> Path:
    s = d["summary"]
    stat = lambda v, k, cls="": f'<div class="stat {cls}"><div class="v">{v}</div><div class="k">{k}</div></div>'  # noqa: E731
    c0, c1 = d["closed"]
    b = [f'<header class="topbar"><div class="brand">{LOGO}<div class="brand-name">{APP_NAME} <span>/ Watch</span></div></div>'
         f'<div class="spacer"></div><button class="btn" onclick="print()">{icon("download")}<span class="lbl">Print / save PDF</span></button>'
         f'{theme_button()}</header><div class="wrap">',
         f'<div class="cover"><div class="label">Pattern of activity</div><h1>{h(d["name"])}</h1><div class="sub">'
         + (f'Watching: {h(d["watching"])} · ' if d["watching"] else "") +
         f'{h(s["first"])} to {h(s["last"])} · {"camera fired on movement" if d["mode"] == "triggered" else "camera shot at a fixed interval"}'
         f' · site treated as closed {c0:02d}:00–{c1:02d}:00</div></div>',
         '<div class="stats">' + stat(s["photos"], "Photos read") + stat(s["events"], "Movements") +
         stat(f'{s["days"]:g}', "Days watched") + stat(f'{s["per_day"]:g}', "Movements a day") +
         stat(s["out_of_hours"], "Out of hours", "bad" if s["out_of_hours"] else "") + "</div>",
         "<h2>What this means for guarding</h2>", advice_html(d["advice"]),
         "<h2>When things move</h2>",
         '<p class="lead">Each square is one hour of one day of the week; the stronger the colour, the more movements an '
         'hour. Hatched squares were not watched. The red line under the grid marks the hours treated as closed.</p>',
         f'<div class="grid2"><div class="card">{heatmap_svg(d["rate"], d["cover"], d["closed"])}</div>']
    if d["has_heat"]:
        b.append('<div><div class="scene"><img src="scene.jpg" alt="The camera\'s view"><img src="heat.png" alt=""></div>'
                 '<p class="sub" style="margin-top:8px">Where in the view the movement was: stronger colour, more often.</p></div>')
    else:
        b.append('<div><div class="scene"><img src="scene.jpg" alt="The camera\'s view"></div></div>')
    b.append("</div>")
    for title, pick in (("Out of hours", True), ("Typical movement", False)):
        ev = [e for e in d["events"] if e.get("frame") and e["out_of_hours"] == pick]
        if ev:
            b.append(f"<h2>{title}</h2><div class='frames'>" + "".join(
                f'<figure><a href="{h(e["frame"])}"><img src="{h(e["frame"])}" alt=""></a><figcaption>'
                f'{DAYS[e["weekday"]]} {h(e["time"])}{" · night" if e["night"] else ""}</figcaption></figure>' for e in ev)
                + "</div>")
    b.append("""<h2>How to read this</h2><ul class="limits">
<li><b>A movement is a change in the picture</b>, compared with the same view at nearby times: a person, a vehicle,
an animal, a gate swinging. Frames in which the camera fired but nothing differs are ignored. Movements less than a
minute apart count once. Nobody is identified.</li>
<li><b>Wind, rain, headlights and moving shadows cause false movements</b>, and a small or distant figure can be
missed. Look at the frames before acting on a number.</li>
<li><b>One camera sees one view.</b> A quiet gate here says nothing about the back fence.</li>
<li><b>Only watch a site you are engaged to assess</b>, with the owner's agreement, and follow local rules on
recording people (signage, retention, who may see the pictures).</li></ul></div>""")
    path = out / "report.html"
    path.write_text(page(f"Watch: {h(d['name'])}", "".join(b), CSS, "", None, default_theme="light"), encoding="utf-8")
    return path
