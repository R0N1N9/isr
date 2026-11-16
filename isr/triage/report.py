"""Self-contained, offline HTML review page for triage results.

No internet needed: everything (data, styles, script, map) is in one file next
to the ``crops/`` and ``previews/`` folders, so the whole output folder can be
copied to a USB stick and opened on any laptop at the command post.
"""

from __future__ import annotations

import json
from pathlib import Path

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Drone Photo Triage</title>
<style>
:root{
  --bg:#f5f6f8;--panel:#fff;--ink:#14181f;--muted:#5d6675;--line:#dde1e7;
  --accent:#d1147a;--hi:#e8174a;--mid:#f58a07;--lo:#d9b100;--ok:#16a34a;--no:#8a93a3;
  --shadow:0 1px 2px rgba(0,0,0,.06),0 2px 8px rgba(0,0,0,.05);
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0f1217;--panel:#171b22;--ink:#e8ebf0;--muted:#98a2b3;--line:#2a313c;
  --accent:#ff4fa8;--shadow:none;
}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:16px 20px;border-bottom:1px solid var(--line);background:var(--panel)}
h1{font-size:19px;margin:0 0 4px}
.sub{color:var(--muted);font-size:13px;word-break:break-all}
.stats{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
.stat{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:6px 10px}
.stat b{font-size:16px;display:block}
.stat span{color:var(--muted);font-size:12px}
details.caveat{margin-top:10px;color:var(--muted);font-size:13px}
.toolbar{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:10px 20px;
  background:var(--panel);border-bottom:1px solid var(--line)}
.toolbar label{display:flex;gap:6px;align-items:center;color:var(--muted);font-size:13px}
select,button,input[type=range]{font:inherit;color:inherit}
select,button{background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:5px 9px;cursor:pointer}
button:hover{border-color:var(--accent)}
.kbd{color:var(--muted);font-size:12px;margin-left:auto}
kbd{border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 4px;font-size:11px}
main{display:grid;grid-template-columns:minmax(280px,38%) 1fr;gap:16px;padding:16px 20px}
@media (max-width:900px){main{grid-template-columns:1fr;padding:12px 16px}.kbd{display:none}}
.mapwrap{position:sticky;top:64px;align-self:start;background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:8px;box-shadow:var(--shadow)}
@media (max-width:900px){.mapwrap{position:static}svg#map{max-height:42vh}}
svg#map{width:100%;height:auto;display:block;touch-action:none}
.maplegend{display:flex;flex-wrap:wrap;gap:10px;color:var(--muted);font-size:12px;padding:6px 4px 2px}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;vertical-align:-1px;margin-right:4px}
#cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px;align-content:start}
.card{background:var(--panel);border:2px solid var(--line);border-radius:10px;overflow:hidden;box-shadow:var(--shadow);
  display:flex;flex-direction:column}
.card.sel{border-color:var(--accent)}
.card.confirmed{border-color:var(--ok)}
.card.dismissed{opacity:.45}
.card img{width:100%;aspect-ratio:1/1;object-fit:cover;background:#000;cursor:zoom-in;display:block}
.cbody{padding:8px 10px;display:flex;flex-direction:column;gap:3px;flex:1}
.row{display:flex;justify-content:space-between;gap:6px;align-items:center}
.badge{font-weight:700;border-radius:5px;padding:1px 7px;color:#fff;font-size:13px}
.lbl{font-weight:600}
.meta{color:var(--muted);font-size:12px;word-break:break-all}
.coord{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;cursor:copy}
.acts{display:flex;gap:6px;margin-top:6px}
.acts button{flex:1}
.acts .c.on{background:var(--ok);color:#fff;border-color:var(--ok)}
.acts .x.on{background:var(--no);color:#fff;border-color:var(--no)}
.empty{color:var(--muted);padding:30px;text-align:center;grid-column:1/-1}
#lb{position:fixed;inset:0;background:rgba(0,0,0,.88);display:none;z-index:20;flex-direction:column;align-items:center;
  justify-content:center;padding:12px}
#lb.open{display:flex}
#lb img{max-width:100%;max-height:calc(100vh - 70px);object-fit:contain}
#lb .cap{color:#ddd;margin-top:8px;font-size:13px;text-align:center}
#toast{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);background:var(--ink);color:var(--bg);padding:6px 14px;
  border-radius:6px;opacity:0;transition:opacity .2s;pointer-events:none;z-index:30}
</style>
</head>
<body>
<header>
  <h1>Drone Photo Triage</h1>
  <div class="sub" id="src"></div>
  <div class="stats" id="stats"></div>
  <details class="caveat"><summary>How to read this / accuracy</summary>
    <p>Candidates are ranked by score (0–1). Colour <b>target</b> matches and hits found by more than one detector rank
    highest. <b>Anomaly</b> hits are anything statistically unusual for that photo: expect false alarms (litter, flowers,
    rocks, vehicles). Every hit needs a human look — the tool narrows thousands of photos to a short list; it does
    not clear an area.</p>
    <p>Positions are projected from the drone GPS, altitude above <i>take-off</i> and gimbal angle onto flat ground.
    Expect a few metres of error straight down, more for oblique photos or terrain above/below the take-off point.
    "drone" position means the photo could not be projected and the drone's own location is shown.</p>
    <p>Not finding anything does <b>not</b> mean nobody is there: canopy, shadow, and camouflage-coloured clothing all
    hide people from colour-based detection.</p>
  </details>
</header>
<div class="toolbar">
  <label>Min score <input type="range" id="minScore" min="0" max="1" step="0.05"><span id="minScoreV"></span></label>
  <label>Detector <select id="kind"><option value="">All</option></select></label>
  <label>Show <select id="status">
    <option value="open">To review</option><option value="all">All</option>
    <option value="confirmed">Confirmed</option><option value="dismissed">Dismissed</option></select></label>
  <button id="expGpx" title="Confirmed candidates as GPS waypoints">Export confirmed GPX</button>
  <button id="expCsv">Export confirmed CSV</button>
  <span class="kbd"><kbd>J</kbd>/<kbd>K</kbd> move · <kbd>C</kbd> confirm · <kbd>X</kbd> dismiss · <kbd>U</kbd> undo ·
    <kbd>Space</kbd> full photo</span>
</div>
<main>
  <div class="mapwrap">
    <svg id="map" viewBox="0 0 600 600" role="img" aria-label="Map of photo footprints and candidates"></svg>
    <div class="maplegend">
      <span><span class="dot" style="background:var(--hi)"></span>score ≥ .75</span>
      <span><span class="dot" style="background:var(--mid)"></span>≥ .55</span>
      <span><span class="dot" style="background:var(--lo)"></span>lower</span>
      <span><span class="dot" style="background:var(--ok)"></span>confirmed</span>
      <span>grey = photo coverage, line = flight path</span>
    </div>
  </div>
  <div id="cards"></div>
</main>
<div id="lb"><img alt=""><div class="cap"></div></div>
<div id="toast"></div>
<script>
const DATA = __DATA__;
const S = DATA.summary;
const KEY = "isr-triage:" + S.source + ":" + S.generated;
let status = {};
try { status = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(status)); } catch (e) {} };
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const col = (s, st) => st === "confirmed" ? "var(--ok)" : s >= .75 ? "var(--hi)" : s >= .55 ? "var(--mid)" : "var(--lo)";

const items = [];
for (const p of DATA.photos) for (const d of p.detections) items.push({...d, photo: p});
items.sort((a, b) => b.score - a.score);

$("#src").textContent = "Source: " + S.source + " · generated " + S.generated;
const stat = (v, l) => `<div class="stat"><b>${v}</b><span>${l}</span></div>`;
$("#stats").innerHTML = stat(S.photos, "photos scanned") + stat(S.photos_geolocatable, "geolocated") +
  stat(S.photos_with_hits, "photos with hits") + stat(S.detections, "candidates") +
  stat(S.seconds + " s", "processing time") +
  (S.config.colors.length ? stat(esc(S.config.colors.join(", ")), "target colours") : "") +
  (S.errors ? stat(S.errors, "unreadable") : "");

const kinds = [...new Set(items.flatMap(i => i.kinds.map(k => k.split(":")[0])))];
for (const k of kinds) $("#kind").insertAdjacentHTML("beforeend", `<option>${esc(k)}</option>`);
const ms = $("#minScore"); ms.value = S.config.min_score; $("#minScoreV").textContent = (+ms.value).toFixed(2);

let visible = [], sel = 0, history = [];
function filtered() {
  const m = +ms.value, k = $("#kind").value, st = $("#status").value;
  return items.filter(i => i.score >= m && (!k || i.kinds.some(x => x.split(":")[0] === k)) &&
    (st === "all" || (st === "open" ? !status[i.id] : status[i.id] === st)));
}
function card(i, idx) {
  const st = status[i.id] || "";
  const ll = i.lat != null ? `${i.lat.toFixed(6)}, ${i.lon.toFixed(6)}` : "no position";
  const t = i.photo.taken_at ? i.photo.taken_at.replace("T", " ") : "";
  return `<div class="card ${st} ${idx === sel ? "sel" : ""}" data-idx="${idx}" id="c-${esc(i.id)}">
    <img loading="lazy" src="${esc(i.crop || "")}" alt="${esc(i.label)}" data-open="${idx}">
    <div class="cbody">
      <div class="row"><span class="lbl">${esc(i.label)}</span>
        <span class="badge" style="background:${col(i.score)}">${i.score.toFixed(2)}</span></div>
      <div class="meta">${esc(i.photo.rel)} · #${i.n}${t ? " · " + esc(t) : ""}</div>
      <div class="meta"><span class="coord" title="Click to copy" data-copy="${esc(ll)}">${esc(ll)}</span>
        ${i.position_source === "drone" ? " (drone pos)" : ""}${i.area_m2 ? " · ~" + i.area_m2 + " m²" : ""}
        ${i.photo.gsd_cm ? " · " + i.photo.gsd_cm + " cm/px" : ""}</div>
      <div class="acts"><button class="c ${st === "confirmed" ? "on" : ""}" data-act="confirmed" data-idx="${idx}">Confirm</button>
        <button class="x ${st === "dismissed" ? "on" : ""}" data-act="dismissed" data-idx="${idx}">Dismiss</button></div>
    </div></div>`;
}
function render(keepSel) {
  visible = filtered();
  if (!keepSel) sel = 0;
  sel = Math.max(0, Math.min(sel, visible.length - 1));
  $("#cards").innerHTML = visible.length ? visible.map(card).join("") :
    `<div class="empty">Nothing to show with these filters.${items.length ? "" : " No candidates were found."}</div>`;
  drawMap();
}
function setStatus(idx, st) {
  const i = visible[idx]; if (!i) return;
  history.push([i.id, status[i.id]]);
  if (status[i.id] === st) delete status[i.id]; else status[i.id] = st;
  save();
  const open = $("#status").value === "open";
  render(true);
  if (!open) select(idx);
  else select(Math.min(idx, visible.length - 1));
}
function select(idx, scroll = true) {
  sel = idx;
  document.querySelectorAll(".card.sel").forEach(c => c.classList.remove("sel"));
  const c = document.querySelector(`.card[data-idx="${idx}"]`);
  if (c) { c.classList.add("sel"); if (scroll) c.scrollIntoView({block: "nearest", behavior: "smooth"}); }
  drawMap();
}
function openLb(idx) {
  const i = visible[idx]; if (!i) return;
  const lb = $("#lb"); lb.classList.add("open");
  lb.querySelector("img").src = i.photo.preview || i.crop;
  lb.querySelector(".cap").textContent = `${i.photo.rel} — box #${i.n}: ${i.label} (${i.score.toFixed(2)}). Original: ${i.photo.file}`;
}
const toast = m => { const t = $("#toast"); t.textContent = m; t.style.opacity = 1; setTimeout(() => t.style.opacity = 0, 1400); };

document.addEventListener("click", e => {
  const t = e.target;
  if (t.dataset.act) return setStatus(+t.dataset.idx, t.dataset.act);
  if (t.dataset.open != null) { select(+t.dataset.open, false); return openLb(+t.dataset.open); }
  if (t.dataset.copy) { navigator.clipboard?.writeText(t.dataset.copy).then(() => toast("Copied " + t.dataset.copy)); return; }
  if (t.closest("#lb")) return $("#lb").classList.remove("open");
  const c = t.closest(".card"); if (c) select(+c.dataset.idx, false);
});
document.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
  const lb = $("#lb").classList.contains("open");
  const k = e.key.toLowerCase();
  if (k === "escape") return $("#lb").classList.remove("open");
  if (k === "j" || k === "arrowright" || k === "arrowdown") { e.preventDefault(); select(Math.min(sel + 1, visible.length - 1)); if (lb) openLb(sel); }
  else if (k === "k" || k === "arrowleft" || k === "arrowup") { e.preventDefault(); select(Math.max(sel - 1, 0)); if (lb) openLb(sel); }
  else if (k === "c") setStatus(sel, "confirmed");
  else if (k === "x") setStatus(sel, "dismissed");
  else if (k === " " || k === "enter") { e.preventDefault(); lb ? $("#lb").classList.remove("open") : openLb(sel); }
  else if (k === "u") {
    const h = history.pop(); if (!h) return;
    if (h[1]) status[h[0]] = h[1]; else delete status[h[0]];
    save(); render(true); toast("Undone");
  }
});
["#minScore", "#kind", "#status"].forEach(s => $(s).addEventListener("input", () => {
  $("#minScoreV").textContent = (+ms.value).toFixed(2); render();
}));

function download(name, text, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], {type}));
  a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
const confirmed = () => items.filter(i => status[i.id] === "confirmed" && i.lat != null);
$("#expGpx").onclick = () => {
  const c = confirmed(); if (!c.length) return toast("No confirmed candidates yet");
  download("confirmed.gpx", `<?xml version="1.0" encoding="UTF-8"?>\n<gpx version="1.1" creator="isr-tools" xmlns="http://www.topografix.com/GPX/1/1">\n` +
    c.map(i => `<wpt lat="${i.lat.toFixed(7)}" lon="${i.lon.toFixed(7)}"><name>${esc(i.id)}</name><desc>${esc(i.label)} — ${esc(i.photo.rel)}</desc><sym>Flag, Red</sym></wpt>`).join("\n") +
    "\n</gpx>\n", "application/gpx+xml");
};
$("#expCsv").onclick = () => {
  const c = confirmed(); if (!c.length) return toast("No confirmed candidates yet");
  const q = v => `"${String(v ?? "").replace(/"/g, '""')}"`;
  download("confirmed.csv", "id,score,label,lat,lon,photo,taken_at\n" +
    c.map(i => [i.id, i.score, i.label, i.lat, i.lon, i.photo.rel, i.photo.taken_at].map(q).join(",")).join("\n") + "\n", "text/csv");
};

// ---- map (local equirectangular metres, no tiles so it works offline) ----
const geo = DATA.photos.filter(p => p.lat != null);
let proj = null;
if (geo.length) {
  const lat0 = geo.reduce((a, p) => a + p.lat, 0) / geo.length, lon0 = geo.reduce((a, p) => a + p.lon, 0) / geo.length;
  const R = 6371008.8, k = Math.cos(lat0 * Math.PI / 180);
  const en = (lat, lon) => [(lon - lon0) * Math.PI / 180 * R * k, (lat - lat0) * Math.PI / 180 * R];
  const pts = [];
  for (const p of geo) { pts.push(en(p.lat, p.lon)); (p.footprint || []).forEach(q => pts.push(en(q[0], q[1]))); }
  items.forEach(i => i.lat != null && pts.push(en(i.lat, i.lon)));
  let [x0, y0, x1, y1] = [Infinity, Infinity, -Infinity, -Infinity];
  for (const [x, y] of pts) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
  const span = Math.max(x1 - x0, y1 - y0, 50), pad = 30, size = 600;
  const s = (size - 2 * pad) / span, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  proj = { en, s, xy: (lat, lon) => { const [x, y] = en(lat, lon); return [size / 2 + (x - cx) * s, size / 2 - (y - cy) * s]; } };
}
function drawMap() {
  const svg = $("#map");
  if (!proj) { svg.innerHTML = `<text x="300" y="300" text-anchor="middle" fill="currentColor" opacity=".6">No GPS data in these photos</text>`; return; }
  let h = "";
  for (const p of geo) if (p.footprint) {
    h += `<polygon points="${p.footprint.map(q => proj.xy(q[0], q[1]).join(",")).join(" ")}" fill="var(--muted)" fill-opacity=".06" stroke="var(--muted)" stroke-opacity=".35" stroke-width=".6"/>`;
  }
  const track = [...geo].sort((a, b) => (a.taken_at || "").localeCompare(b.taken_at || "") || a.rel.localeCompare(b.rel));
  h += `<polyline points="${track.map(p => proj.xy(p.lat, p.lon).join(",")).join(" ")}" fill="none" stroke="var(--muted)" stroke-width="1" stroke-dasharray="3 3"/>`;
  const shown = new Set(visible.map(i => i.id));
  const selId = visible[sel]?.id;
  for (const i of [...items].reverse()) {
    if (i.lat == null || status[i.id] === "dismissed") continue;
    const [x, y] = proj.xy(i.lat, i.lon), on = shown.has(i.id);
    h += `<circle cx="${x}" cy="${y}" r="${i.id === selId ? 8 : 5}" fill="${col(i.score, status[i.id])}" fill-opacity="${on || status[i.id] === "confirmed" ? .95 : .25}" stroke="${i.id === selId ? "var(--ink)" : "var(--panel)"}" stroke-width="${i.id === selId ? 2.5 : 1}" data-id="${esc(i.id)}" style="cursor:pointer"><title>${esc(i.id)} ${i.score.toFixed(2)} ${esc(i.label)}</title></circle>`;
  }
  // scale bar
  const nice = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000].find(m => m * proj.s > 60) || 5000;
  const L = nice * proj.s;
  h += `<g fill="currentColor" font-size="12" opacity=".75"><rect x="20" y="572" width="${L}" height="4"/><text x="20" y="566">${nice >= 1000 ? nice / 1000 + " km" : nice + " m"}</text>
    <text x="575" y="28" text-anchor="middle" font-weight="700">N</text><path d="M575 34 l-6 16 l6 -4 l6 4z"/></g>`;
  svg.innerHTML = h;
}
$("#map").addEventListener("click", e => {
  const id = e.target.dataset?.id; if (!id) return;
  let idx = visible.findIndex(i => i.id === id);
  if (idx < 0) { $("#status").value = "all"; ms.value = 0; $("#kind").value = ""; render(); idx = visible.findIndex(i => i.id === id); }
  select(idx);
});
render();
</script>
</body>
</html>
"""


def write_report(data: dict, out_dir: Path) -> Path:
    blob = json.dumps(data, default=str).replace("</", "<\\/")
    path = out_dir / "report.html"
    path.write_text(TEMPLATE.replace("__DATA__", blob), encoding="utf-8")
    return path
