"""Offline HTML scorecard: recall vs false alarms per photo, plus the numbers table."""

from __future__ import annotations

from pathlib import Path

from ..ui import LOGO, icon, page, theme_button

CSS = r"""
.wrap{max-width:1200px;margin:0 auto;padding:22px 24px 60px}
.hero h1{font-size:26px;letter-spacing:-.025em;margin:0;font-weight:700}
.hero .sub{color:var(--muted);font-size:13px;margin:2px 0 16px}
.viz{--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--grid:rgba(13,17,23,.08);--axis:rgba(13,17,23,.35)}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz{--s1:#3987e5;--s2:#d95926;--s3:#199e70;
  --grid:rgba(255,255,255,.07);--axis:rgba(255,255,255,.3)}}
:root[data-theme="dark"] .viz{--s1:#3987e5;--s2:#d95926;--s3:#199e70;--grid:rgba(255,255,255,.07);--axis:rgba(255,255,255,.3)}
.chartcard{padding:16px 18px;margin-top:16px}
.chartcard h2{font-size:15px;margin:0 0 2px}
.chartcard .cap{color:var(--muted);font-size:12.5px;margin:0 0 10px}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12.5px;color:var(--ink-2);margin-bottom:6px}
.legend i{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:6px}
.plot{position:relative}
svg.chart{display:block;width:100%;height:auto;overflow:visible}
.tip{position:absolute;pointer-events:none;background:var(--raise);border:1px solid var(--line-2);border-radius:10px;
  padding:8px 10px;font-size:12px;box-shadow:var(--shadow-lg);display:none;min-width:180px;color:var(--ink)}
.tip .row{display:flex;justify-content:space-between;gap:12px}
.tip i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:right;padding:9px 10px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
th{color:var(--muted);font-weight:600;font-size:12px}
th:first-child,td:first-child{text-align:left}
td .sw{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:8px;vertical-align:-1px}
.note{color:var(--muted);font-size:12.5px;margin-top:10px}
"""

JS = r"""
const R = DATA;
// the palette is validated for three series: chart the three best (by AP); the table lists every system
const sys = [...R.systems].sort((a, b) => b.ap - a.ap).slice(0, 3);
const $ = s => document.querySelector(s);
const cols = ["var(--s1)", "var(--s2)", "var(--s3)"];
const pct = v => (v * 100).toFixed(0) + "%";
$("#sub").textContent = `${R.tiles} ${R.split} tiles · ${R.targets} labelled targets · ${R.megapixels} MP (≈ ${(R.megapixels / 20.9).toFixed(1)} Mavic 3 photos) · ${R.generated}`;
const st = (v, k, c = "") => `<div class="stat ${c}"><div class="v">${v}</div><div class="k">${k}</div></div>`;
const best = [...R.systems].sort((a, b) => b.recall_at_fa["1"] - a.recall_at_fa["1"])[0];
$("#stats").innerHTML = st(R.targets, "Labelled targets") + st(R.tiles, "Test tiles") +
  st(best ? pct(best.recall_at_fa["1"]) : "—", `Best recall at 1 false alarm/photo`, "accent") + st(best ? esc(best.name) : "—", "Best system");
$("#legend").innerHTML = sys.map((s, k) => `<span><i style="background:${cols[k]}"></i>${esc(s.name)}</span>`).join("");

// chart: recall (y, 0..1) vs false alarms per photo (x, log 0.1..max)
const W = 800, H = 380, m = {l: 52, r: 190, t: 14, b: 44};
const xmax = Math.max(10, ...sys.flatMap(s => s.curve.map(c => c.fa)));
const xmin = 0.1, lx = v => Math.log10(Math.max(v, xmin));
const X = v => m.l + (lx(v) - lx(xmin)) / (lx(xmax) - lx(xmin)) * (W - m.l - m.r);
const Y = v => m.t + (1 - v) * (H - m.t - m.b);
let g = "";
for (const t of [0, .25, .5, .75, 1]) g += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(t)}" y2="${Y(t)}" stroke="var(--grid)"/>
  <text x="${m.l - 8}" y="${Y(t) + 4}" text-anchor="end" font-size="11" fill="var(--muted)">${t * 100}%</text>`;
for (const t of [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100].filter(v => v <= xmax)) g += `<line x1="${X(t)}" x2="${X(t)}" y1="${m.t}" y2="${H - m.b}" stroke="var(--grid)"/>
  <text x="${X(t)}" y="${H - m.b + 16}" text-anchor="middle" font-size="11" fill="var(--muted)">${t}</text>`;
g += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--axis)"/>
  <text x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle" font-size="12" fill="var(--ink-2)">False alarms per 20 MP photo (log scale)</text>
  <text transform="translate(14 ${(m.t + H - m.b) / 2}) rotate(-90)" text-anchor="middle" font-size="12" fill="var(--ink-2)">Recall (share of targets found)</text>`;
const pts = sys.map(s => [...s.curve].sort((a, b) => a.fa - b.fa || a.recall - b.recall));
const ends = [];
pts.forEach((p, k) => {
  if (!p.length) return;
  // step line: recall achievable at each false-alarm level
  let d = `M${X(xmin)},${Y(0)}`, prevY = Y(0);
  for (const c of p) { const x = X(c.fa), y = Y(c.recall); d += ` L${x},${prevY} L${x},${y}`; prevY = y; }
  d += ` L${X(xmax)},${prevY}`;
  g += `<path d="${d}" fill="none" stroke="${cols[k]}" stroke-width="2" stroke-linejoin="round"/>`;
  ends.push({k, y: prevY});
});
// direct labels at the line ends, nudged apart so they never overprint
ends.sort((a, b) => a.y - b.y);
for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 15);
for (const e of ends) g += `<text x="${W - m.r + 8}" y="${e.y + 4}" font-size="12" fill="var(--ink-2)">${esc(sys[e.k].name.slice(0, 28))}</text>`;
g += `<line id="xh" y1="${m.t}" y2="${H - m.b}" stroke="var(--axis)" stroke-dasharray="3 3" visibility="hidden"/>`;
g += sys.map((s, k) => `<circle id="dot${k}" r="4" fill="${cols[k]}" stroke="var(--surface)" stroke-width="2" visibility="hidden"/>`).join("");
g += `<rect x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent" id="hit"/>`;
$("#chart").setAttribute("viewBox", `0 0 ${W} ${H}`);
$("#chart").innerHTML = g;
const recallAt = (p, fa) => p.filter(c => c.fa <= fa).reduce((a, c) => Math.max(a, c.recall), 0);
const hit = $("#hit"), tip = $("#tip");
hit.addEventListener("pointermove", e => {
  const r = $("#chart").getBoundingClientRect(), sx = (e.clientX - r.left) * W / r.width;
  const fa = Math.pow(10, lx(xmin) + (sx - m.l) / (W - m.l - m.r) * (lx(xmax) - lx(xmin)));
  const xh = $("#xh"); xh.setAttribute("x1", sx); xh.setAttribute("x2", sx); xh.setAttribute("visibility", "visible");
  let rows = "";
  pts.forEach((p, k) => { const rc = recallAt(p, fa); const d = $("#dot" + k);
    d.setAttribute("cx", sx); d.setAttribute("cy", Y(rc)); d.setAttribute("visibility", "visible");
    rows += `<div class="row"><span><i style="background:${cols[k]}"></i>${esc(sys[k].name)}</span><b>${pct(rc)}</b></div>`; });
  tip.innerHTML = `<div class="label" style="margin-bottom:4px">≤ ${fa < 1 ? fa.toFixed(2) : fa.toFixed(1)} false alarms / photo</div>${rows}`;
  tip.style.display = "block";
  const px = (e.clientX - r.left), py = (e.clientY - r.top);
  tip.style.left = Math.min(px + 14, r.width - 200) + "px"; tip.style.top = Math.max(0, py - 60) + "px";
});
hit.addEventListener("pointerleave", () => { tip.style.display = "none"; $("#xh").setAttribute("visibility", "hidden");
  sys.forEach((_, k) => $("#dot" + k).setAttribute("visibility", "hidden")); });

// table (every system, not just the charted three)
$("#table").innerHTML = `<thead><tr><th>System</th><th>AP</th><th>Recall @ 1 FA/photo</th><th>@ 5 FA/photo</th>
  <th>Recall at default</th><th>FA/photo at default</th><th>Max recall</th><th>Small targets (&lt;16 px)</th><th>s / photo</th></tr></thead><tbody>` +
  R.systems.map(s => { const k = sys.indexOf(s); return `<tr><td>${k >= 0 ? `<span class="sw" style="background:${cols[k]}"></span>` : `<span class="sw" style="background:transparent;border:1px solid var(--line-2)"></span>`}${esc(s.name)}</td>
  <td>${s.ap.toFixed(3)}</td><td>${pct(s.recall_at_fa["1"])}</td><td>${pct(s.recall_at_fa["5"])}</td>
  <td>${pct(s.at_default.recall)} <span class="muted">@ ${s.at_default.threshold}</span></td><td>${s.at_default.fa_per_photo.toFixed(1)}</td>
  <td>${pct(s.max_recall)}</td><td>${s.by_size["<16 px"] ? pct(s.by_size["<16 px"].recall) + ` <span class="muted">(${s.by_size["<16 px"].targets})</span>` : "—"}</td>
  <td>${s.seconds_per_photo}</td></tr>`; }).join("") + "</tbody>";
if (R.systems.length > 3) $("#capnote").textContent = `Chart shows the three systems with the highest AP; all ${R.systems.length} are in the table.`;
"""


def _body() -> str:
    return f"""
<header class="topbar">
  <div class="brand">{LOGO}<div class="brand-name">ISR <span>/ Model scorecard</span></div></div>
  <div class="spacer"></div>
  <a class="btn" href="scorecard.json" download>{icon("download")}<span class="lbl">JSON</span></a>
  {theme_button()}
</header>
<div class="wrap viz">
  <section class="hero"><h1>How well does each system find people?</h1><div class="sub" id="sub"></div></section>
  <section class="stats" id="stats"></section>
  <section class="panel chartcard">
    <h2>Recall vs false alarms</h2>
    <p class="cap">Higher and further left is better: more targets found for less noise. Hover to compare systems at any
    false-alarm level. <span id="capnote"></span></p>
    <div class="legend" id="legend"></div>
    <div class="plot"><svg class="chart" id="chart" role="img" aria-label="Recall against false alarms per photo"></svg>
      <div class="tip" id="tip"></div></div>
  </section>
  <section class="panel chartcard">
    <h2>Numbers</h2>
    <table id="table"></table>
    <p class="note">AP = average precision (area under the precision–recall curve). False alarms are normalised to a 20 MP
    Mavic 3 photo. A prediction matches a target if the boxes overlap (IoU ≥ 0.25) or either centre falls in the other box;
    extra boxes on an already-found target are ignored. Speed is measured tile by tile, so it overstates the per-photo
    time of detectors with a fixed cost per image (the colour detectors). Only as trustworthy as the labels and as representative as the
    test tiles: test on your own terrain and altitude.</p>
  </section>
</div>
"""


def write_scorecard(res: dict, out_dir: Path) -> Path:
    path = Path(out_dir) / "scorecard.html"
    path.write_text(page("Model Scorecard", _body(), CSS, JS, res), encoding="utf-8")
    return path
