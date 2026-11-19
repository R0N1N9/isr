"""The site editor page: trace the plan over the base map and watch the findings update.

Served by ``isr.site.server``. All drawing happens in the browser; every
change is posted to the local server, which saves it and returns the analysis.
"""

from __future__ import annotations

import json

from ..ui import APP_NAME, LOGO, page, theme_button
from . import model as M

CSS = r"""
html,body{height:100%}
body{display:flex;flex-direction:column;overflow:hidden}
.topbar{flex:none}
#name{font:600 14px var(--font);color:var(--ink);background:transparent;border:1px solid transparent;border-radius:8px;
  padding:5px 8px;min-width:120px;width:260px;max-width:30vw}
#name:hover,#name:focus{border-color:var(--line-2);background:var(--surface);outline:none}
#status{font-size:12px;color:var(--muted);white-space:nowrap;min-width:84px;text-align:right}
#status.err{color:var(--bad)}
.main{flex:1;min-height:0;display:grid;grid-template-columns:200px 1fr 380px}
.tools{border-right:1px solid var(--line);background:var(--surface);padding:10px 8px;overflow:auto;display:flex;flex-direction:column;gap:3px}
.tool{display:flex;align-items:center;gap:9px;width:100%;height:34px;padding:0 9px;border-radius:9px;border:1px solid transparent;
  background:transparent;color:var(--ink-2);font:500 12.5px var(--font);cursor:pointer;text-align:left}
.tool:hover{background:var(--surface-2)}
.tool[aria-pressed="true"]{background:var(--accent-soft);color:var(--ink);border-color:color-mix(in srgb,var(--accent) 35%,transparent)}
.tool kbd{margin-left:auto}
.tool i{flex:none;width:14px;height:14px;border-radius:4px;display:block}
.tools .label{margin:12px 6px 4px}
.tools .hint{font-size:12px;color:var(--muted);padding:8px 8px 0;line-height:1.45}
.stage{position:relative;min-width:0;min-height:0;background:#15171c;overflow:hidden}
#map{position:absolute;inset:0;width:100%;height:100%;display:block;touch-action:none;cursor:grab;user-select:none;
  font-family:var(--font)}
#map.draw{cursor:crosshair}#map.panning{cursor:grabbing}
.float{position:absolute;background:var(--glass);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
  border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow);font-size:12.5px}
#layers{top:12px;right:12px;padding:8px 12px;display:grid;gap:4px}
#layers label{display:flex;align-items:center;gap:7px;cursor:pointer;white-space:nowrap}
#layers i{width:10px;height:10px;border-radius:3px;display:inline-block}
#zoom{bottom:14px;right:12px;display:flex;flex-direction:column;overflow:hidden}
#zoom button{width:34px;height:32px;border:0;background:transparent;color:var(--ink);font:600 16px var(--font);cursor:pointer}
#zoom button:hover{background:var(--surface-2)}
#tip{left:12px;bottom:14px;padding:8px 12px;max-width:min(520px,60%);color:var(--ink-2)}
#tip:empty{display:none}
.side{border-left:1px solid var(--line);background:var(--surface);display:flex;flex-direction:column;min-height:0}
.side .tabs{padding:10px 12px 8px;border-bottom:1px solid var(--line)}
.side .seg{display:flex}.side .seg button{flex:1;justify-content:center}
.pane{flex:1;overflow:auto;padding:14px 14px 30px;display:none}
.pane.on{display:block}
.mini{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:14px}
.mini .stat{padding:10px 12px}.mini .stat .v{font-size:18px}.mini .stat .k{font-size:11.5px}
.mini .stat.bad .v{color:var(--bad)}
.sec{margin:16px 0 6px;display:flex;align-items:center;gap:8px}
.sec .btn{margin-left:auto}
.item{display:grid;grid-template-columns:auto 1fr;gap:2px 9px;padding:9px 10px;border:1px solid var(--line);border-radius:10px;
  margin-bottom:6px;cursor:pointer;background:var(--surface)}
.item:hover{border-color:var(--line-2);background:var(--surface-2)}
.item .t{font-weight:600;font-size:13px}
.item .d{grid-column:2;color:var(--muted);font-size:12px}
.item .a{grid-column:2;font-size:12px;color:var(--ink-2)}
.item .btn{grid-column:2;justify-self:start;margin-top:5px}
.det{display:grid;grid-template-columns:72px 1fr;gap:10px;padding:9px 10px;border:1px solid var(--line);border-radius:10px;margin-bottom:6px;
  background:var(--surface);cursor:pointer;transition:border-color .2s,box-shadow .2s}
.det:hover{border-color:var(--line-2)}.det.hot{border-color:#d946ef;box-shadow:0 0 0 3px rgba(217,70,239,.25)}
.det img{width:72px;height:72px;object-fit:cover;border-radius:8px;cursor:zoom-in;background:#000}
.det>div{min-width:0}
.det .t{font-weight:600;font-size:13px;display:flex;gap:7px;align-items:center}
.det .d{color:var(--muted);font-size:12px;margin:3px 0 6px}
.det .row{display:flex;gap:6px;margin-top:5px}
.det select,.det input{height:28px;min-width:0;flex:1;padding:0 7px;border-radius:8px;border:1px solid var(--line-2);background:var(--surface);color:var(--ink);font:500 12px var(--font)}
.gopt{padding:9px 10px;border:1px solid var(--line);border-radius:10px;margin-bottom:6px;cursor:pointer;background:var(--surface)}
.gopt:hover{border-color:var(--line-2)}.gopt.on{border-color:#38bdf8;box-shadow:0 0 0 2px rgba(56,189,248,.2)}
.gopt .t{font-weight:600;font-size:13px;display:flex;gap:7px;align-items:center;flex-wrap:wrap}.gopt .t .tag{margin-left:auto}
.gopt .d{color:var(--muted);font-size:12px;margin-top:3px}.gopt .a{font-size:12px;color:var(--ink-2);margin-top:4px}
ul.pc{list-style:none;padding:0;margin:8px 0 4px}ul.pc li{padding-left:16px;position:relative;margin:3px 0;font-size:12.5px}
ul.pc li::before{position:absolute;left:0;font-weight:700}ul.pc li.pro::before{content:"+";color:var(--good)}ul.pc li.con::before{content:"\2212";color:var(--bad)}
.badge{display:inline-grid;place-items:center;min-width:26px;height:19px;padding:0 5px;border-radius:6px;font:700 10.5px var(--mono);
  color:#fff;background:var(--c,var(--muted));align-self:start;margin-top:1px}
.steps{border:1px solid var(--line);border-radius:12px;padding:10px 12px;margin-bottom:14px;background:var(--surface-2)}
.steps div{display:flex;gap:8px;align-items:center;padding:2px 0;color:var(--ink-2);font-size:12.5px}
.steps div.done{color:var(--muted);text-decoration:line-through}
.steps b{width:16px;height:16px;border-radius:50%;border:1.5px solid var(--line-2);display:inline-grid;place-items:center;font-size:10px;flex:none}
.steps .done b{background:var(--good);border-color:var(--good);color:#fff}
.form{display:grid;grid-template-columns:110px 1fr;gap:8px 10px;align-items:center}
.form label{color:var(--muted);font-size:12.5px}
.form input,.form select,.form textarea{width:100%;height:32px;padding:0 9px;border-radius:8px;border:1px solid var(--line-2);
  background:var(--surface);color:var(--ink);font:500 13px var(--font)}
.form textarea{height:70px;padding:7px 9px;resize:vertical;grid-column:1/-1}
.form .wide{grid-column:1/-1}
.form .ro{color:var(--ink-2);font-variant-numeric:tabular-nums}
.help{color:var(--muted);font-size:12.5px;line-height:1.5}
#lb{position:fixed;inset:0;z-index:70;background:rgba(5,6,8,.9);display:none;place-items:center;padding:30px}
#lb.open{display:grid}
#lb img{max-width:100%;max-height:calc(100vh - 110px);border-radius:10px;box-shadow:0 30px 80px rgba(0,0,0,.6)}
#lb div{color:#fff;margin-top:10px;text-align:center;font-size:13px}
@media (max-width:1000px){.main{grid-template-columns:60px 1fr}.side{position:fixed;right:0;top:59px;bottom:0;width:min(380px,92vw);
  transform:translateX(100%);transition:transform .2s;z-index:40;box-shadow:var(--shadow-lg)}.side.open{transform:none}
  .tool span,.tool kbd,.tools .label,.tools .hint{display:none}.tool{justify-content:center;padding:0}#panelBtn{display:inline-flex!important}}
"""

BODY = """
<header class="topbar">
  <a class="brand" href="__HOME__" title="__HOMETIP__" style="text-decoration:none">__LOGO__<div class="brand-name">__BRAND__ <span>__APP__</span></div></a>
  <input id="name" aria-label="Site name" spellcheck="false">
  <div class="spacer"></div>
  <span id="status" role="status"></span>
  <button class="btn sm" id="undo" title="Undo (Ctrl+Z)">Undo</button>
  <button class="btn sm" id="redo" title="Redo (Ctrl+Shift+Z)">Redo</button>
  <button class="btn sm" id="panelBtn" style="display:none">Results</button>
  <button class="btn primary" id="report">Generate report</button>
  __THEME__
</header>
<div class="main">
  <nav class="tools" id="tools" aria-label="Drawing tools"></nav>
  <div class="stage">
    <svg id="map" xmlns="http://www.w3.org/2000/svg"><defs>
      <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
        <path d="M10 0H0V10" fill="none" stroke="rgba(255,255,255,.09)" stroke-width=".08"/></pattern></defs>
      <g id="gBase"></g><g id="gFeat"></g><g id="gUi"></g></svg>
    <div class="float" id="layers"></div>
    <div class="float" id="zoom"><button data-z="in" title="Zoom in">+</button><button data-z="out" title="Zoom out">−</button>
      <button data-z="fit" title="Fit the site" style="font-size:12px">Fit</button></div>
    <div class="float" id="tip"></div>
  </div>
  <aside class="side" id="side">
    <div class="tabs"><div class="seg" id="tabs">
      <button data-t="results" aria-pressed="true">Results</button>
      <button data-t="props" aria-pressed="false">Selected</button>
      <button data-t="settings" aria-pressed="false">Site</button></div></div>
    <div class="pane on" id="p-results"></div>
    <div class="pane" id="p-props"></div>
    <div class="pane" id="p-settings"></div>
  </aside>
</div>
<div id="lb"><div><img alt=""><div></div></div></div>
"""

JS = r"""
const C = DATA;  // kinds, subtypes, defaults
const $ = s => document.querySelector(s);
const svg = $("#map"), gBase = $("#gBase"), gFeat = $("#gFeat"), gUi = $("#gUi");
let model = null, result = null, overlays = {}, sel = null, tool = "select", draft = null, cursor = null;
let drag = null, flash = null, seq = 0, timer = null, spaceDown = false;
const view = {cx: 0, cy: 0, mpp: 0.1};
const layers = {base: true, cameras: true, natural: false, blind: true, proposed: true, photos: true, detections: true, guards: true};
let gsel = null;  // which staffing option is drawn: [scenario index, option index]
const last = {barrier: "fence", access: "door"};
let hist = [], hi = -1;

const TOOLS = [
  ["select", "Select / move", "V", "var(--muted)"], ["boundary", "Boundary", "B", "#fff;border:1.5px dashed #888"],
  ["building", "Building", "H", "#334155"], ["barrier", "Wall, fence, hedge", "W", "#d6a56b"],
  ["vegetation", "Trees, shrubs", "G", "#16a34a"], ["street", "Street, footpath", "S", "#facc15"],
  ["access", "Door, window, gate", "A", "#e08a00;border-radius:50%"], ["camera", "Camera", "C", "#3b82f6;border-radius:50%"]];
const TIPS = {
  select: "Click an item to select it. Drag it, its corners, or the small hollow dots to reshape. Alt-click a corner to remove it. Drag empty space to pan, scroll to zoom.",
  boundary: "Click round the property line. Double-click or press Enter to finish.",
  building: "Click each corner of the building. Double-click or press Enter to finish. Set its height under Selected.",
  barrier: "Click along the wall, fence or hedge. Double-click or press Enter to finish, then pick its type and height.",
  vegetation: "Click round trees or shrubs that block the view. Double-click or press Enter to finish.",
  street: "Click along each street, lane or footpath where the public can walk past. Double-click or Enter to finish.",
  access: "Click each door, ground-floor window, gate or driveway. It snaps to the nearest wall or fence.",
  camera: "Click where an existing camera is fixed, then drag its handle to aim it."};
const GEOM = C.geometry, PRI = {high: "#e5484d", medium: "#e08a00", low: "#12a150", info: "#6b7385"};

// ---------- helpers ----------
const uid = () => "f" + Math.random().toString(36).slice(2, 9);
const feat = id => model.features.find(f => f.id === id);
const label = f => f.kind === "barrier" ? C.barrier[f.subtype].label : f.kind === "access" ? C.access[f.subtype] : C.kind[f.kind];
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const fmt = (v, d = 1) => (+v).toFixed(d);
function pathLen(pts, closed) { let s = 0; for (let i = 0; i < pts.length - (closed ? 0 : 1); i++) s += dist(pts[i], pts[(i + 1) % pts.length]); return s; }
function area(pts) { let s = 0; for (let i = 0; i < pts.length; i++) { const a = pts[i], b = pts[(i + 1) % pts.length]; s += a[0] * b[1] - b[0] * a[1]; } return Math.abs(s) / 2; }
function closest(p, pts, closed) {
  let best = {d: Infinity};
  for (let i = 0; i < pts.length - (closed ? 0 : 1); i++) {
    const a = pts[i], b = pts[(i + 1) % pts.length], dx = b[0] - a[0], dy = b[1] - a[1], L2 = dx * dx + dy * dy;
    const t = L2 ? Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2)) : 0;
    const q = [a[0] + t * dx, a[1] + t * dy], d = dist(p, q);
    if (d < best.d) best = {d, q, i, n: [dy, -dx]};
  }
  return best;
}
function inPoly(p, pts) { let c = false; for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) { const a = pts[i], b = pts[j];
  if ((a[1] > p[1]) !== (b[1] > p[1]) && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]) c = !c; } return c; }
const bearing = (dx, dy) => (Math.atan2(dx, dy) * 180 / Math.PI + 360) % 360;
const P = pts => pts.map(p => p[0].toFixed(2) + "," + (-p[1]).toFixed(2)).join(" ");
const r2 = v => Math.round(v * 100) / 100;

function toWorld(ev) { const r = svg.getBoundingClientRect();
  return [view.cx + (ev.clientX - r.left - r.width / 2) * view.mpp, view.cy - (ev.clientY - r.top - r.height / 2) * view.mpp]; }
function setViewBox() { const r = svg.getBoundingClientRect();
  svg.setAttribute("viewBox", `${view.cx - r.width * view.mpp / 2} ${-view.cy - r.height * view.mpp / 2} ${r.width * view.mpp} ${r.height * view.mpp}`); }
function fit() {
  let b = model.bounds;
  const bd = model.features.filter(f => f.kind === "boundary");
  const pts = (bd.length ? bd : model.features.filter(f => f.kind !== "street")).flatMap(f => f.pts);
  if (pts.length > 2) { const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
    b = [Math.min(...xs) - 14, Math.min(...ys) - 14, Math.max(...xs) + 14, Math.max(...ys) + 14]; }
  const r = svg.getBoundingClientRect();
  view.cx = (b[0] + b[2]) / 2; view.cy = (b[1] + b[3]) / 2;
  view.mpp = Math.max((b[2] - b[0]) / Math.max(50, r.width), (b[3] - b[1]) / Math.max(50, r.height)) * 1.04;
  render();
}
function zoomAt(f, w) { const m = Math.min(2, Math.max(0.004, view.mpp * f)), k = m / view.mpp;
  if (w) { view.cx = w[0] + (view.cx - w[0]) * k; view.cy = w[1] + (view.cy - w[1]) * k; } view.mpp = m; render(); }

// snapping: points stick to walls and fences, drawn corners stick to existing corners
function snapPoint(p, kind) {
  const order = kind === "camera" ? [["building", 1.2]] : [["building", 1.5], ["barrier", 1.5], ["boundary", 1.5]];
  for (const [k, lim] of order) { let best = null;
    for (const f of model.features) if (f.kind === k) { const c = closest(p, f.pts, GEOM[k] === "polygon"); if (c.d <= lim && (!best || c.d < best.d)) best = {...c, f}; }
    if (best) return best; }
  return null;
}
function snapVertex(p, skip) {
  const lim = 9 * view.mpp; let best = null;
  for (const f of model.features) { if (f === skip || !["boundary", "building", "barrier"].includes(f.kind)) continue;
    for (const q of f.pts) { const d = dist(p, q); if (d <= lim && (!best || d < best.d)) best = {d, q}; } }
  return best ? best.q.slice() : p;
}

// ---------- history / saving ----------
function commit() { hist = hist.slice(0, hi + 1); hist.push(JSON.stringify(model.features)); if (hist.length > 120) hist.shift(); hi = hist.length - 1; changed(); }
function restore(i) { if (i < 0 || i >= hist.length) return; hi = i; model.features = JSON.parse(hist[i]); if (sel && !feat(sel)) sel = null; changed(); renderProps(); }
function changed() { render(); schedule(); }
function schedule(delay = 280) { clearTimeout(timer); setStatus("Analysing…"); timer = setTimeout(analyse, delay); }
function setStatus(t, err) { const s = $("#status"); s.textContent = t; s.classList.toggle("err", !!err); }
async function post(url) {
  const res = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(model)});
  const j = await res.json(); if (!res.ok) throw new Error(j.error || res.statusText); return j;
}
async function analyse() {
  const my = ++seq;
  try { const j = await post(C.base + "/api/analyse"); if (my !== seq) return;
    result = j.result; overlays = j.overlays; setStatus("Saved"); renderBase(); render(); renderResults();
  } catch (e) { if (my === seq) setStatus("Not saved: " + e.message, true); }
}

// ---------- drawing the map ----------
function renderBase() {
  let h = `<rect x="-5000" y="-5000" width="10000" height="10000" fill="url(#grid)"/>`;
  const bm = model.basemap;
  if (bm && layers.base) { const [x0, y0, x1, y1] = bm.bounds;
    h += `<image href="${C.base}/files/${esc(bm.file)}" x="${x0}" y="${-y1}" width="${x1 - x0}" height="${y1 - y0}" preserveAspectRatio="none"/>`; }
  if (result) { const [x0, y0, x1, y1] = result.grid.bounds;
    for (const [k, o] of [["natural", "natural"], ["cameras", "cameras"], ["proposed", "proposed"], ["blind", "blind_plan"]]) if (layers[k] && overlays[o])
      h += `<image href="${overlays[o]}" x="${x0}" y="${-y1}" width="${x1 - x0}" height="${y1 - y0}" preserveAspectRatio="none" style="image-rendering:pixelated"/>`; }
  gBase.innerHTML = h;
}
function wedge(c, brg, fov, rng) { const pts = [c], n = Math.max(2, Math.floor(fov / 6));
  for (let i = 0; i <= n; i++) { const a = (brg - fov / 2 + fov * i / n) * Math.PI / 180; pts.push([c[0] + Math.sin(a) * rng, c[1] + Math.cos(a) * rng]); } return pts; }
function tag(p, text, col, shape, u, attrs = "") {
  const fs = 11 * u, w = (10 + 7 * text.length) * u, x = p[0], y = -p[1];
  const body = shape === "circle" ? `<circle cx="${x}" cy="${y}" r="${11 * u}" fill="${col}" stroke="#fff" stroke-width="${1.6 * u}"/>`
    : `<rect x="${x - w / 2}" y="${y - 9 * u}" width="${w}" height="${18 * u}" rx="${(shape === "pill" ? 9 : 3) * u}" fill="${shape === "box" ? "#fff" : col}" stroke="${shape === "box" ? col : "#fff"}" stroke-width="${1.6 * u}"/>`;
  return `<g ${attrs}>${body}<text x="${x}" y="${y + fs * .36}" font-size="${fs}" font-weight="700" text-anchor="middle" fill="${shape === "box" ? col : "#fff"}" pointer-events="none">${esc(text)}</text></g>`;
}
function render() {
  if (!model) return;
  setViewBox();
  const u = view.mpp, acc = {}, cam = {};
  if (result) { for (const a of result.access) acc[a.id] = a; for (const c of result.cameras) cam[c.id] = c; }
  let h = "", top = "";
  const order = ["street", "boundary", "vegetation", "building", "barrier", "camera", "access"];
  for (const f of [...model.features].sort((a, b) => order.indexOf(a.kind) - order.indexOf(b.kind))) {
    const s = f.id === sel, d = `data-id="${f.id}"`, p = P(f.pts), hl = s ? `stroke="var(--accent)"` : "";
    if (f.kind === "boundary")
      h += `<polygon points="${p}" fill="none" stroke="#000" stroke-opacity=".55" stroke-width="${4 * u}"/><polygon points="${p}" fill="none" stroke="${s ? "var(--accent)" : "#fff"}" stroke-width="${2 * u}" stroke-dasharray="${8 * u} ${6 * u}"/><polygon ${d} points="${p}" fill="none" stroke="transparent" stroke-width="${14 * u}"/>`;
    else if (f.kind === "building")
      h += `<polygon ${d} points="${p}" fill="#0f172a" fill-opacity=".5" stroke="${s ? "var(--accent)" : "#fff"}" stroke-width="${2 * u}" stroke-linejoin="round"/>`;
    else if (f.kind === "vegetation")
      h += `<polygon ${d} points="${p}" fill="#16a34a" fill-opacity=".28" stroke="${s ? "var(--accent)" : "#86efac"}" stroke-width="${1.6 * u}" stroke-dasharray="${4 * u} ${4 * u}"/>`;
    else if (f.kind === "street")
      h += `<polyline points="${p}" fill="none" stroke="${s ? "var(--accent)" : "#facc15"}" stroke-width="${3.5 * u}" stroke-dasharray="${1 * u} ${8 * u}" stroke-linecap="round"/><polyline ${d} points="${p}" fill="none" stroke="transparent" stroke-width="${14 * u}"/>`;
    else if (f.kind === "barrier") { const col = {wall: "#e5e7eb", fence: "#d6a56b", open_fence: "#d6a56b", hedge: "#4ade80"}[f.subtype];
      h += `<polyline points="${p}" fill="none" stroke="#000" stroke-opacity=".5" stroke-width="${6 * u}" stroke-linejoin="round"/><polyline points="${p}" fill="none" stroke="${s ? "var(--accent)" : col}" stroke-width="${3.5 * u}" stroke-linejoin="round" ${f.subtype === "open_fence" ? `stroke-dasharray="${7 * u} ${5 * u}"` : ""}/><polyline ${d} points="${p}" fill="none" stroke="transparent" stroke-width="${14 * u}"/>`; }
    else if (f.kind === "camera") { const col = "#3b82f6", c = f.pts[0];
      h += `<polygon points="${P(wedge(c, f.bearing, f.hfov, f.range))}" fill="${col}" fill-opacity="${s ? .16 : .06}" stroke="${s ? "var(--accent)" : col}" stroke-width="${1.6 * u}" stroke-linejoin="round" ${f.status === "planned" ? `stroke-dasharray="${6 * u} ${4 * u}"` : ""} pointer-events="none"/>`;
      top += tag(c, cam[f.id] ? cam[f.id].code : "C", col, "pill", u, d + ` style="cursor:move"`); }
    else if (f.kind === "access") { const a = acc[f.id];
      top += tag(f.pts[0], a ? a.code : "A", a ? PRI[a.priority] : "#6b7385", "circle", u, d + ` style="cursor:move"`); }
    if (s && f.kind === "access") top += `<circle cx="${f.pts[0][0]}" cy="${-f.pts[0][1]}" r="${15 * u}" fill="none" stroke="var(--accent)" stroke-width="${2 * u}" pointer-events="none"/>`;
  }
  if (result && layers.proposed) for (const p of result.recommended) { const c = [p.x, p.y], col = "#10a35a";
    h += `<polygon points="${P(wedge(c, p.bearing, p.hfov, p.range))}" fill="${col}" fill-opacity=".05" stroke="${col}" stroke-width="${1.6 * u}" stroke-dasharray="${6 * u} ${4 * u}" pointer-events="none"/>`;
    top += tag(c, p.id, col, "pill", u, `data-rec="${p.id}" style="cursor:pointer"`); }
  if (result && layers.blind) for (const b of result.blind_spots) if (b.covered_by_planned < .5) top += tag([b.x, b.y], b.id, "#e5484d", "box", u, `pointer-events="none"`);
  if (layers.photos) for (const p of model.photos) { if (p.x == null || p.kind === "nadir") continue;
    if (p.heading != null) { const a = p.heading * Math.PI / 180, L = 20 * u;
      h += `<line x1="${p.x}" y1="${-p.y}" x2="${p.x + Math.sin(a) * L}" y2="${-(p.y + Math.cos(a) * L)}" stroke="#fff" stroke-width="${2.2 * u}" stroke-linecap="round" pointer-events="none"/>`; }
    h += `<circle data-photo="${p.id}" cx="${p.x}" cy="${-p.y}" r="${5 * u}" fill="#fff" stroke="#111" stroke-width="${1 * u}" style="cursor:zoom-in"><title>${esc(p.name)}</title></circle>`; }
  const go = guardOption(); let gstops = "";
  if (go && layers.guards) {
    if (go.route) { const line = P(go.route.line.concat(go.route.line.slice(0, 1)));
      h += `<polyline points="${line}" fill="none" stroke="#04263a" stroke-opacity=".75" stroke-width="${6 * u}" stroke-linejoin="round" pointer-events="none"/><polyline points="${line}" fill="none" stroke="#38bdf8" stroke-width="${3 * u}" stroke-linejoin="round" stroke-dasharray="${9 * u} ${6 * u}" pointer-events="none"/>`;
      for (const st of go.route.stops) gstops += `<g pointer-events="none"><circle cx="${st.x + 13 * u}" cy="${-st.y - 13 * u}" r="${8 * u}" fill="#38bdf8" stroke="#04263a" stroke-width="${1.2 * u}"/><text x="${st.x + 13 * u}" y="${-st.y - 9.7 * u}" font-size="${9 * u}" font-weight="700" text-anchor="middle" fill="#04263a">${st.n}</text></g>`; }
    for (const p of result.guards.posts) if (go.statics.includes(p.id)) top += tag([p.x, p.y], p.id, "#0284c7", "pill", u, `pointer-events="none"`);
  }
  if (result && layers.detections) for (const d of result.detections)
    top += tag([d.x, d.y], d.code, "#d946ef", d.status === "confirmed" ? "pill" : "box", u, `data-det="${d.code}" style="cursor:pointer"`);
  gFeat.innerHTML = h + top + gstops;

  // selection handles, draft, flash
  let ui = "";
  const f = sel && feat(sel);
  if (f && GEOM[f.kind] !== "point" && tool === "select") {
    const closed = GEOM[f.kind] === "polygon";
    for (let i = 0; i < f.pts.length - (closed ? 0 : 1); i++) { const a = f.pts[i], b = f.pts[(i + 1) % f.pts.length];
      ui += `<circle data-mid="${i}" cx="${(a[0] + b[0]) / 2}" cy="${-(a[1] + b[1]) / 2}" r="${4 * u}" fill="#15171c" stroke="var(--accent)" stroke-width="${1.5 * u}" style="cursor:copy"/>`; }
    f.pts.forEach((p, i) => ui += `<circle data-v="${i}" cx="${p[0]}" cy="${-p[1]}" r="${5.5 * u}" fill="#fff" stroke="var(--accent)" stroke-width="${2 * u}" style="cursor:move"/>`);
  }
  if (f && f.kind === "camera" && tool === "select") { const a = f.bearing * Math.PI / 180, L = Math.max(f.range * .6, 30 * u), c = f.pts[0];
    const hp = [c[0] + Math.sin(a) * L, c[1] + Math.cos(a) * L];
    ui += `<line x1="${c[0]}" y1="${-c[1]}" x2="${hp[0]}" y2="${-hp[1]}" stroke="var(--accent)" stroke-width="${1.5 * u}" pointer-events="none"/><circle data-rot="1" cx="${hp[0]}" cy="${-hp[1]}" r="${7 * u}" fill="var(--accent)" stroke="#fff" stroke-width="${2 * u}" style="cursor:crosshair"><title>Drag to aim</title></circle>`; }
  if (draft) { const pts = cursor ? [...draft.pts, cursor] : draft.pts, poly = GEOM[draft.kind] === "polygon";
    ui += `<${poly && pts.length > 2 ? "polygon" : "polyline"} points="${P(pts)}" fill="${poly ? "rgba(124,124,255,.14)" : "none"}" stroke="var(--accent)" stroke-width="${2 * u}" pointer-events="none"/>`;
    draft.pts.forEach(p => ui += `<circle cx="${p[0]}" cy="${-p[1]}" r="${4 * u}" fill="#fff" stroke="var(--accent)" stroke-width="${1.5 * u}" pointer-events="none"/>`);
    if (cursor && draft.pts.length) { const a = draft.pts[draft.pts.length - 1];
      ui += `<text x="${cursor[0] + 12 * u}" y="${-cursor[1] - 10 * u}" font-size="${12 * u}" font-weight="600" fill="#fff" paint-order="stroke" stroke="#000" stroke-width="${3 * u}" pointer-events="none">${fmt(dist(a, cursor))} m${draft.pts.length > 1 ? " · " + fmt(pathLen(pts, false)) + " m" : ""}</text>`; } }
  if (flash) ui += `<circle cx="${flash[0]}" cy="${-flash[1]}" r="${22 * u}" fill="none" stroke="#fff" stroke-width="${3 * u}" pointer-events="none"><animate attributeName="r" from="${8 * u}" to="${34 * u}" dur=".7s" repeatCount="3"/><animate attributeName="opacity" from="1" to="0" dur=".7s" repeatCount="3"/></circle>`;
  gUi.innerHTML = ui;
}

// ---------- tools ----------
function setTool(t) { tool = t; draft = null; cursor = null;
  document.querySelectorAll(".tool").forEach(b => b.setAttribute("aria-pressed", b.dataset.t === t));
  svg.classList.toggle("draw", t !== "select"); $("#tip").textContent = TIPS[t]; render(); }
function nextName(lbl) { let n = 1; const names = new Set(model.features.map(f => f.name)); while (names.has(lbl + " " + n)) n++; return lbl + " " + n; }
function create(kind, pts, extra = {}) {
  const f = {id: uid(), kind, pts: pts.map(p => [r2(p[0]), r2(p[1])]), notes: "", ...extra};
  if (kind === "barrier") { f.subtype = last.barrier; f.height = C.barrier[f.subtype].height; }
  if (kind === "access") f.subtype = last.access;
  if (C.height[kind]) f.height = C.height[kind];
  if (kind === "camera") Object.assign(f, {height: model.settings.camera_height_m, hfov: model.settings.camera_hfov, range: model.settings.camera_range_m, status: "existing", bearing: 0, ...extra});
  f.name = nextName(label(f)); model.features.push(f); sel = f.id; setTool("select"); commit(); showTab("props"); renderProps(); return f;
}
function finishDraft() { if (!draft) return; const need = GEOM[draft.kind] === "polygon" ? 3 : 2, d = draft; draft = null; cursor = null;
  if (d.pts.length >= need) create(d.kind, d.pts); else render(); }
function placePoint(kind, w, ev) {
  const s = ev.altKey ? null : snapPoint(w, kind), p = s ? s.q : w, extra = {};
  if (kind === "camera" && s) { let n = s.n; const L = Math.hypot(n[0], n[1]) || 1; n = [n[0] / L, n[1] / L];
    if (inPoly([p[0] + n[0] * .2, p[1] + n[1] * .2], s.f.pts)) n = [-n[0], -n[1]]; extra.bearing = Math.round(bearing(n[0], n[1])); }
  create(kind, [p], extra);
}
function remove(id) { model.features = model.features.filter(f => f.id !== id); if (sel === id) sel = null; commit(); renderProps(); }

// ---------- pointer ----------
svg.addEventListener("contextmenu", e => e.preventDefault());
svg.addEventListener("pointerdown", ev => {
  const t = ev.target, w = toWorld(ev);
  svg.setPointerCapture(ev.pointerId);
  if (ev.button === 1 || ev.button === 2 && !t.dataset.v || spaceDown) { drag = {type: "pan", x: ev.clientX, y: ev.clientY, cx: view.cx, cy: view.cy}; svg.classList.add("panning"); return; }
  if (tool !== "select") {
    if (GEOM[tool] === "point") return placePoint(tool, w, ev);
    const p = ev.altKey ? w : snapVertex(w);
    if (!draft) draft = {kind: tool, pts: []};
    if (GEOM[tool] === "polygon" && draft.pts.length >= 3 && dist(p, draft.pts[0]) < 10 * view.mpp) return finishDraft();
    const lp = draft.pts[draft.pts.length - 1];
    if (!lp || dist(lp, p) > 2 * view.mpp) draft.pts.push(p);
    render(); return;
  }
  const f = sel && feat(sel);
  if (t.dataset.photo) return openPhoto(t.dataset.photo);
  const dg = t.closest("[data-det]");
  if (dg) { showTab("results"); const it = document.querySelector(`.det[data-code="${dg.dataset.det}"]`); if (it) { it.scrollIntoView({block: "center"}); it.classList.add("hot"); setTimeout(() => it.classList.remove("hot"), 1600); } return; }
  if (t.dataset.v != null && f) { const i = +t.dataset.v, min = GEOM[f.kind] === "polygon" ? 3 : 2;
    if (ev.altKey || ev.button === 2) { if (f.pts.length > min) { f.pts.splice(i, 1); commit(); renderProps(); } return; }
    drag = {type: "vertex", f, i}; return; }
  if (t.dataset.mid != null && f) { const i = +t.dataset.mid; f.pts.splice(i + 1, 0, w.map(r2)); drag = {type: "vertex", f, i: i + 1, moved: true}; render(); return; }
  if (t.dataset.rot && f) { drag = {type: "rot", f}; return; }
  const g = t.closest("[data-id],[data-rec]");
  if (g && g.dataset.rec) return adopt(g.dataset.rec);
  if (g) { const hit = feat(g.dataset.id); if (sel !== hit.id) { sel = hit.id; render(); renderProps(); showTab("props"); }
    drag = {type: "move", f: hit, start: w, orig: hit.pts.map(p => p.slice())}; return; }
  drag = {type: "pan", x: ev.clientX, y: ev.clientY, cx: view.cx, cy: view.cy, deselect: true}; svg.classList.add("panning");
});
svg.addEventListener("pointermove", ev => {
  const w = toWorld(ev);
  if (!drag) { if (draft) { cursor = ev.altKey ? w : snapVertex(w); render(); } return; }
  if (drag.type === "pan") { const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y; if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
    view.cx = drag.cx - dx * view.mpp; view.cy = drag.cy + dy * view.mpp; render(); return; }
  if (drag.type === "vertex") { drag.f.pts[drag.i] = (ev.altKey ? w : snapVertex(w, drag.f)).map(r2); drag.moved = true; render(); }
  else if (drag.type === "rot") { const c = drag.f.pts[0]; drag.f.bearing = Math.round(bearing(w[0] - c[0], w[1] - c[1])); drag.moved = true; render(); }
  else if (drag.type === "move") { const dx = w[0] - drag.start[0], dy = w[1] - drag.start[1];
    if (!drag.moved && Math.hypot(dx, dy) < 4 * view.mpp) return; drag.moved = true;
    if (GEOM[drag.f.kind] === "point") { const s = ev.altKey ? null : snapPoint(w, drag.f.kind); drag.f.pts[0] = (s ? s.q : w).map(r2); }
    else drag.f.pts = drag.orig.map(p => [r2(p[0] + dx), r2(p[1] + dy)]);
    render(); }
});
svg.addEventListener("pointerup", () => { const d = drag; drag = null; svg.classList.remove("panning"); if (!d) return;
  if (d.type === "pan") { if (d.deselect && !d.moved && sel) { sel = null; render(); renderProps(); } return; }
  if (d.moved) { commit(); renderProps(); } });
svg.addEventListener("dblclick", () => { if (draft) finishDraft(); });
svg.addEventListener("wheel", ev => { ev.preventDefault(); zoomAt(Math.exp(ev.deltaY * 0.0016), toWorld(ev)); }, {passive: false});
$("#zoom").addEventListener("click", ev => { const z = ev.target.dataset.z; if (z === "fit") fit(); else if (z) zoomAt(z === "in" ? 0.7 : 1.43); });
addEventListener("resize", render);
addEventListener("keydown", ev => {
  if (ev.target.matches("input,textarea,select")) return;
  const k = ev.key.toLowerCase();
  if ((ev.ctrlKey || ev.metaKey) && k === "z") { ev.preventDefault(); restore(ev.shiftKey ? hi + 1 : hi - 1); return; }
  if ((ev.ctrlKey || ev.metaKey) && k === "y") { ev.preventDefault(); restore(hi + 1); return; }
  if (ev.ctrlKey || ev.metaKey) return;
  if (k === " ") { spaceDown = true; ev.preventDefault(); }
  else if (k === "escape") { if ($("#lb").classList.contains("open")) $("#lb").classList.remove("open"); else if (draft) { draft = null; cursor = null; render(); } else setTool("select"); }
  else if (k === "enter") finishDraft();
  else if (k === "backspace" && draft) { draft.pts.pop(); render(); }
  else if ((k === "delete" || k === "backspace") && sel) remove(sel);
  else { const t = TOOLS.find(t => t[2].toLowerCase() === k); if (t) setTool(t[0]); }
});
addEventListener("keyup", ev => { if (ev.key === " ") spaceDown = false; });

// ---------- side panel ----------
function showTab(t) { document.querySelectorAll("#tabs button").forEach(b => b.setAttribute("aria-pressed", b.dataset.t === t));
  document.querySelectorAll(".pane").forEach(p => p.classList.toggle("on", p.id === "p-" + t)); }
$("#tabs").addEventListener("click", ev => { const b = ev.target.closest("button"); if (b) showTab(b.dataset.t); });
function focusOn(x, y, id) { view.cx = x; view.cy = y; flash = [x, y]; if (id && feat(id)) { sel = id; renderProps(); } render(); setTimeout(() => { flash = null; render(); }, 2200); }
function adopt(id) { const list = id === "all" ? result.recommended : result.recommended.filter(p => p.id === id);
  for (const p of list) model.features.push({id: uid(), kind: "camera", name: nextName("Planned camera"), pts: [[p.x, p.y]], height: p.height, bearing: p.bearing, hfov: p.hfov, range: p.range, status: "planned", notes: "Recommended position: " + p.mount});
  if (list.length) { toast(list.length + " camera" + (list.length > 1 ? "s" : "") + " added to the plan"); commit(); } }

function guardOption() { const g = result && result.guards; if (!g || !g.scenarios.length) return null;
  const [a, b] = gsel || [g.scenarios.length - 1, 0]; const sc = g.scenarios[Math.min(a, g.scenarios.length - 1)]; return sc.options[Math.min(b, sc.options.length - 1)]; }
function renderResults() {
  const el = $("#p-results"), has = k => model.features.some(f => f.kind === k);
  const steps = [["boundary", "Draw the property boundary"], ["building", "Trace each building"], ["barrier", "Add walls, fences and hedges"],
    ["street", "Draw the street or footpath outside"], ["access", "Mark doors, windows, gates, driveways"], ["camera", "Add existing cameras (if any)"]];
  let h = "";
  if (steps.some(s => !has(s[0]) && s[0] !== "camera")) h += `<div class="steps"><div class="label" style="margin-bottom:4px">Build the plan</div>` +
    steps.map((s, i) => `<div class="${has(s[0]) ? "done" : ""}"><b>${has(s[0]) ? "✓" : i + 1}</b>${s[1]}</div>`).join("") + `</div>`;
  if (!result) { el.innerHTML = h + `<p class="help">Results appear here as you draw.</p>`; return; }
  const st = result.stats, stat = (v, k, bad) => `<div class="stat ${bad ? "bad" : ""}"><div class="v">${v}</div><div class="k">${k}</div></div>`;
  h += `<div class="mini">` + stat(st.blind_pct.toFixed(0) + "%", "Grounds nobody can see", st.blind_pct >= 20) +
    stat(st.camera_pct_now.toFixed(0) + "% → " + st.camera_pct_recommended.toFixed(0) + "%", "Camera coverage, now → recommended") +
    stat(st.access_high + " / " + st.access_total, "High-priority access points", st.access_high > 0) +
    stat(st.street_pct.toFixed(0) + "%", "Visible from the street") + `</div>`;
  if (result.recommended.length) { h += `<div class="sec"><span class="label">Recommended cameras</span><button class="btn sm" data-adopt="all">Add all to plan</button></div>`;
    for (const p of result.recommended) h += `<div class="item" data-x="${p.x}" data-y="${p.y}"><span class="badge" style="--c:#10a35a">${p.id}</span><span class="t">${esc(p.mount)}</span>
      <span class="d">Aim ${p.bearing}° ${p.aim} · adds ${p.adds_m2} m²${p.covers.length ? " · covers " + esc(p.covers.join(", ")) : ""}</span><button class="btn sm" data-adopt="${p.id}">Add to plan</button></div>`; }
  const gp = result.guards;
  if (gp && gp.scenarios.length) { const cur = guardOption(), li = (xs, c) => xs.map(x => `<li class="${c}">${esc(x)}</li>`).join("");
    h += `<div class="sec"><span class="label">Guard plan</span></div>`;
    gp.scenarios.forEach((sc, a) => sc.options.forEach((o, b) => { const rt = o.route;
      h += `<div class="gopt ${o === cur ? "on" : ""}" data-g="${a},${b}"><div class="t"><span class="badge" style="--c:#0284c7">${sc.guards}</span>${esc(o.title)}${o.recommended ? `<span class="tag good">Recommended</span>` : ""}</div>
        <div class="d">${o.statics.length ? o.guard_pct + "% always watched" : "No fixed watch"}${rt ? " · " + rt.length_m + " m loop, every " + rt.revisit_min + " min" : ""}</div>
        ${o === cur ? `<ul class="pc">${li(o.pros, "pro")}${li(o.cons, "con")}</ul>${o.suggestions.map(x => `<div class="a">${esc(x)}</div>`).join("")}` : ""}</div>`; }));
    h += `<p class="help">Click an option to draw it on the plan. Posts: ${gp.posts.map(p => p.id + " " + esc(p.where)).join("; ") || "none worth holding"}.</p>`; }
  if (gp && gp.timing && gp.timing.length) h += `<div class="sec"><span class="label">Timing, from stakeouts</span></div>` + gp.timing.map(t => `<div class="item" style="cursor:default;grid-template-columns:1fr"><span class="a" style="grid-column:1">${esc(t)}</span></div>`).join("");
  else if (!gp && !model.settings.guards) h += `<p class="help" style="margin-top:14px">Planning guards? Set how many under <b>Site</b> to get posts, a patrol route and options for each headcount.</p>`;
  const all = model.detections || [], gone = all.filter(d => d.status === "dismissed").length;
  if (all.length) { h += `<div class="sec"><span class="label">Detected in the drone photos</span>${gone ? `<button class="btn sm" data-det-act="restore">Restore ${gone} dismissed</button>` : ""}</div>`;
    if (!result.detections.length) h += `<p class="help">All dismissed.</p>`;
    for (const d of result.detections) { const ok = d.status === "confirmed";
      h += `<div class="det" data-code="${d.code}" data-x="${d.x}" data-y="${d.y}"><img src="${C.base}/files/${esc(d.crop)}" alt="" data-zoom="1">
        <div><div class="t"><span class="badge" style="--c:#d946ef">${d.code}</span>${ok ? esc(C.dtypes[d.type] || "Confirmed") : "Possible " + esc(d.label.replace(/^colour anomaly/, "object"))}</div>
        <div class="d">${esc(d.context)} ${d.seen > 1 ? "In " + d.seen + " photos." : ""}</div>
        ${ok ? `<div class="row"><select data-det-type="${d.code}"><option value="">What is it?</option>${Object.keys(C.dtypes).map(k => `<option value="${k}" ${k === d.type ? "selected" : ""}>${C.dtypes[k]}</option>`).join("")}</select>
          <input data-det-note="${d.code}" placeholder="Note for the report" value="${esc(d.note || "")}"></div>` : ""}
        <div class="row"><button class="btn sm ${ok ? "on-good" : ""}" data-det-act="${ok ? "open" : "confirmed"}" data-code="${d.code}">${ok ? "Confirmed" : "Confirm"}</button>
          <button class="btn sm" data-det-act="dismissed" data-code="${d.code}">Dismiss</button></div></div></div>`; } }
  h += `<div class="sec"><span class="label">Findings</span></div>`;
  if (!result.findings.length) h += `<p class="help">No findings.</p>`;
  const where = {}; for (const a of result.access) where[a.id] = a; for (const b of result.blind_spots) where[b.id] = b;
  for (const d of result.detections) where[d.code] = d;
  for (const f of result.findings) { const w = where[f.ref] || (feat(f.ref) ? {x: feat(f.ref).pts[0][0], y: feat(f.ref).pts[0][1]} : null);
    h += `<div class="item" ${w ? `data-x="${w.x}" data-y="${w.y}" data-ref="${esc(f.ref)}"` : ""}><span class="badge" style="--c:${PRI[f.severity]}">${f.severity[0].toUpperCase()}</span>
      <span class="t">${esc(f.title)}</span><span class="d">${esc(f.detail)}</span><span class="a">${esc(f.action)}</span></div>`; }
  if (result.access.length) { h += `<div class="sec"><span class="label">Access points</span></div>`;
    for (const a of result.access) h += `<div class="item" data-x="${a.x}" data-y="${a.y}" data-ref="${a.id}"><span class="badge" style="--c:${PRI[a.priority]}">${a.code}</span>
      <span class="t">${esc(a.name)}</span><span class="d">${a.camera_now ? "On camera" : a.camera_planned ? "Planned camera" : "No camera"} · ${a.street ? "seen from street" : "hidden from street"}${a.window ? " · seen from a window" : ""}</span></div>`; }
  el.innerHTML = h;
}
function setDet(code, patch) { for (const d of model.detections || []) if (code === "*" ? d.status === "dismissed" : d.code === code) Object.assign(d, patch); schedule(30); }
$("#p-results").addEventListener("change", ev => { const t = ev.target;
  if (t.dataset.detType != null) setDet(t.dataset.detType, {type: t.value}); else if (t.dataset.detNote != null) setDet(t.dataset.detNote, {note: t.value}); });
$("#p-results").addEventListener("click", ev => { const b = ev.target.closest("[data-adopt]"); if (b) return adopt(b.dataset.adopt);
  const go = ev.target.closest("[data-g]"); if (go) { gsel = go.dataset.g.split(",").map(Number); layers.guards = true; const cb = document.querySelector('[data-l="guards"]'); if (cb) cb.checked = true; render(); return renderResults(); }
  const da = ev.target.closest("[data-det-act]");
  if (da) return da.dataset.detAct === "restore" ? setDet("*", {status: "open"}) : setDet(da.dataset.code, {status: da.dataset.detAct});
  if (ev.target.dataset.zoom) return lightbox(ev.target.src, "Detail from the drone photo");
  if (ev.target.closest("select,input")) return;
  const dt = ev.target.closest(".det"); if (dt) return focusOn(+dt.dataset.x, +dt.dataset.y);
  const it = ev.target.closest(".item[data-x]"); if (it) focusOn(+it.dataset.x, +it.dataset.y, it.dataset.ref); });

function renderProps() {
  const el = $("#p-props"), f = sel && feat(sel);
  if (!f) { el.innerHTML = `<p class="help">Nothing selected. Click an item on the plan to edit its name, type and height, or pick a drawing tool on the left.</p>`; return; }
  const row = (lbl, html) => `<label>${lbl}</label>${html}`, num = (k, step, min, max) => `<input type="number" data-k="${k}" value="${f[k]}" step="${step}" min="${min}" max="${max}">`;
  const opt = (obj, cur, lab) => Object.keys(obj).map(k => `<option value="${k}" ${k === cur ? "selected" : ""}>${esc(lab(obj[k]))}</option>`).join("");
  let h = `<div class="form">` + row("Name", `<input data-k="name" value="${esc(f.name)}">`);
  if (f.kind === "barrier") h += row("Type", `<select data-k="subtype">${opt(C.barrier, f.subtype, v => v.label)}</select>`) + row("Height (m)", num("height", .1, .2, 12));
  if (f.kind === "building" || f.kind === "vegetation") h += row("Height (m)", num("height", .5, .3, 80));
  if (f.kind === "access") h += row("Type", `<select data-k="subtype">${opt(C.access, f.subtype, v => v)}</select>`);
  if (f.kind === "camera") h += row("Status", `<select data-k="status"><option value="existing" ${f.status === "existing" ? "selected" : ""}>Existing</option><option value="planned" ${f.status === "planned" ? "selected" : ""}>Planned</option></select>`) +
    row("Height (m)", num("height", .1, .5, 30)) + row("Aim (° from N)", num("bearing", 5, 0, 359)) + row("Lens angle (°)", num("hfov", 5, 5, 360)) + row("Range (m)", num("range", 1, 2, 150));
  if (GEOM[f.kind] === "line") h += row("Length", `<span class="ro">${fmt(pathLen(f.pts, false))} m</span>`);
  if (GEOM[f.kind] === "polygon") h += row("Size", `<span class="ro">${fmt(area(f.pts), 0)} m² · ${fmt(pathLen(f.pts, true), 0)} m round</span>`);
  h += `<label class="wide">Notes (shown in the report)</label><textarea data-k="notes">${esc(f.notes || "")}</textarea>
    <button class="btn wide" id="del" style="justify-content:center;color:var(--bad)">Delete ${esc(label(f).toLowerCase())}</button></div>`;
  const hints = {barrier: "A see-through fence stops people but not the view. Anything under about 1.2 m is treated as a marker only.",
    vegetation: "Planting hides whoever is inside or behind it, up to the height given.", street: "Passers-by are assumed to look in from every few metres along this line.",
    access: "Ground-floor windows also count as places an occupant can look out from.", camera: "Drag the handle on the plan to aim. Planned cameras count towards the plan but not towards current coverage.",
    building: "Cameras are recommended on building corners and walls.", boundary: "Only ground inside the boundary is assessed."};
  el.innerHTML = h + `<p class="help" style="margin-top:14px">${hints[f.kind]}</p>`;
}
$("#p-props").addEventListener("input", ev => { const f = sel && feat(sel), k = ev.target.dataset.k; if (!f || !k) return;
  const was = k === "subtype" && label(f);
  if (ev.target.type === "number") { const v = parseFloat(ev.target.value); if (!isFinite(v)) return; f[k] = v; } else f[k] = ev.target.value;
  if (k === "subtype") { if (f.kind === "barrier") f.height = C.barrier[f.subtype].height; last[f.kind] = f.subtype;
    if (new RegExp("^" + was + " \\d+$").test(f.name)) { f.name = ""; f.name = nextName(label(f)); } }
  render(); });
$("#p-props").addEventListener("change", ev => { if (ev.target.dataset.k) { commit(); if (ev.target.dataset.k === "subtype") renderProps(); } });
$("#p-props").addEventListener("click", ev => { if (ev.target.id === "del" && sel) remove(sel); });

function renderSettings() {
  const s = model.settings, n = (k, lbl, step, min, max) => `<label>${lbl}</label><input type="number" data-s="${k}" value="${s[k]}" step="${step}" min="${min}" max="${max}">`;
  const sv = model.survey || {}, c = sv.counts || {};
  $("#p-settings").innerHTML = `<div class="sec" style="margin-top:0"><span class="label">Report front page</span></div><div class="form">
    <label>Client</label><input data-m="client" placeholder="Who the report is for" value="${esc(model.client || "")}">
    <label>Site address</label><input data-m="address" value="${esc(model.address || "")}">
    <label>Prepared by</label><input data-m="company" placeholder="Your company" value="${esc(model.company || "")}">
    <label>Assessor</label><input data-m="assessor" placeholder="Your name" value="${esc(model.assessor || "")}">
    <label>Reference</label><input data-m="reference" placeholder="Job or report number" value="${esc(model.reference || "")}">
    <label class="wide">Notes (shown in the report)</label><textarea data-m="notes">${esc(model.notes || "")}</textarea></div>
    <div class="sec"><span class="label">Stakeouts</span></div><div id="stk"></div>
    <div class="sec"><span class="label">Guards</span></div><div class="form">
    ${n("guards", "Plan for up to", 1, 0, 6)}${n("guard_range_m", "Sight range (m)", 5, 5, 150)}</div>
    <p class="help" style="margin-top:8px">Set how many officers you might have (0 turns the guard plan off). You get a post-and-patrol option for each headcount up to that number. Updating the plan takes about a second longer while this is on.</p>
    <div class="sec"><span class="label">Cameras to recommend</span></div><div class="form">
    ${n("max_new_cameras", "How many", 1, 0, 12)}${n("camera_hfov", "Lens angle (°)", 5, 10, 360)}${n("camera_range_m", "Useful range (m)", 1, 3, 100)}${n("camera_height_m", "Mount height (m)", .1, .5, 30)}</div>
    <p class="help" style="margin-top:8px">Range is how far away a person is still usefully recorded: roughly 8–10 m to identify a face and 15–25 m to see what someone is doing, for a typical 4 MP wide-angle camera.</p>
    <div class="sec"><span class="label">Survey</span></div>
    <p class="help">${model.photos.length} photos: ${c.nadir || 0} straight-down aerial, ${c.oblique || 0} angled aerial, ${c.ground || 0} ground, ${c.unplaced || 0} without GPS.<br>
    ${!model.basemap ? "No drone photos looking down at the ground, so there is no base map: sketch the plan on the grid (10 m squares) using the photo positions as a guide."
      : model.basemap.straight_down ? "Base map made from the straight-down drone photo" + (model.basemap.straight_down > 1 ? "s" : "") + "; expect it to be a few metres out in absolute position."
      : "Base map made from angled drone shots only. Ground-level detail is in the right place, but roofs and treetops lean away from the drone by roughly their own height, so trace each building where its walls meet the ground. One photo taken straight down from above the site avoids this."}</p>`;
  renderStakeouts();
}
// stakeouts from the Watch section: attach one to the access point its camera was pointed at
let watchJobs = null;
async function renderStakeouts() {
  const el = $("#stk"); if (!el) return;
  const acc = model.features.filter(f => f.kind === "access"), have = model.stakeouts || [];
  let h = have.map((w, i) => `<div class="item" style="cursor:default"><span class="badge" style="--c:#e08a00">W</span><span class="t">${esc(w.name)}</span>
    <span class="d">${w.data.summary.events} movements over ${w.data.summary.days} days · watching ${esc((acc.find(a => a.id === w.access) || {name: "nothing marked"}).name)}</span>
    <button class="btn sm" data-unwatch="${i}">Remove</button></div>`).join("");
  if (!C.base) { el.innerHTML = h + `<p class="help">Open this survey from ISRStudio to attach photos from a stakeout camera (the Watch section).</p>`; return; }
  if (!watchJobs) { try { watchJobs = (await (await fetch("/api/home")).json()).jobs.watch || []; } catch (e) { watchJobs = []; } }
  const free = watchJobs.filter(j => !have.some(w => w.id === j.job));
  h += free.length ? `<div class="form"><label>Stakeout</label><select id="stk-job">${free.map(j => `<option value="${j.job}">${esc(j.name)}</option>`).join("")}</select>
      <label>It watched</label><select id="stk-acc"><option value="">(not an access point)</option>${acc.map(a => `<option value="${a.id}">${esc(a.name)}</option>`).join("")}</select>
      <button class="btn wide" id="stk-add" style="justify-content:center">Attach to this survey</button></div>`
    : `<p class="help">${watchJobs.length ? "Every stakeout is attached." : "No stakeouts yet. Run one from the Watch section on the home screen, then attach it here to add timing to the findings and the guard plan."}</p>`;
  el.innerHTML = h;
}
$("#p-settings").addEventListener("click", async ev => {
  if (ev.target.dataset.unwatch != null) { model.stakeouts.splice(+ev.target.dataset.unwatch, 1); schedule(30); return renderStakeouts(); }
  if (ev.target.id !== "stk-add") return;
  const job = $("#stk-job").value, access = $("#stk-acc").value;
  try { const d = await (await fetch("/watch/" + job + "/activity.json")).json();
    const data = {}; for (const k of ["name", "watching", "mode", "closed", "summary", "counts", "cover", "rate", "by_hour", "by_day", "advice"]) data[k] = d[k];
    data.out_of_hours = d.events.filter(e => e.out_of_hours).slice(0, 60).map(e => ({time: e.time, hour: e.hour}));
    (model.stakeouts = model.stakeouts || []).push({id: job, name: d.name, access, data}); toast("Stakeout attached"); schedule(30); renderStakeouts();
  } catch (e) { toast("Could not read that stakeout"); } });
$("#p-settings").addEventListener("change", ev => { const t = ev.target;
  if (t.dataset.m) model[t.dataset.m] = t.value; else if (t.dataset.s) { const v = parseFloat(t.value); if (isFinite(v)) model.settings[t.dataset.s] = v; } else return; schedule(50); });
$("#name").addEventListener("change", ev => { model.name = ev.target.value.trim() || "Site"; document.title = model.name + " · " + C.app; schedule(50); });

function lightbox(src, caption) { const lb = $("#lb"); lb.querySelector("img").src = src; lb.querySelector("div div").textContent = caption; lb.classList.add("open"); }
function openPhoto(id) { const p = model.photos.find(p => p.id === id); if (!p) return;
  lightbox(C.base + "/files/" + p.file, p.name + (p.heading != null ? " · looking " + Math.round(p.heading) + "°" : "")); }
$("#lb").addEventListener("click", () => $("#lb").classList.remove("open"));
$("#undo").onclick = () => restore(hi - 1); $("#redo").onclick = () => restore(hi + 1);
$("#panelBtn").onclick = () => $("#side").classList.toggle("open");
$("#report").onclick = async ev => { const b = ev.currentTarget; b.disabled = true; setStatus("Writing report…");
  const win = window.open("about:blank", "_blank");
  try { clearTimeout(timer); const j = await post(C.base + "/api/report"); setStatus("Saved"); if (win) win.location = j.url; else location.href = j.url; }
  catch (e) { if (win) win.close(); setStatus("Report failed: " + e.message, true); } b.disabled = false; };

// ---------- start ----------
$("#tools").innerHTML = `<div class="label">Draw</div>` + TOOLS.map(t => `<button class="tool" data-t="${t[0]}" aria-pressed="false"><i style="background:${t[3]}"></i><span>${t[1]}</span><kbd>${t[2]}</kbd></button>`).join("") +
  `<div class="hint">Trace what you see in the aerial image. Heights and types are set under <b>Selected</b>.</div>`;
$("#tools").addEventListener("click", ev => { const b = ev.target.closest(".tool"); if (b) setTool(b.dataset.t); });
const LAYERS = [["base", "Base map", "#888"], ["cameras", "Camera coverage (existing + planned)", "rgba(59,130,246,.7)"], ["natural", "Seen from street / windows", "rgba(234,179,8,.7)"],
  ["blind", "Seen by nobody", "rgba(229,72,77,.75)"], ["proposed", "Recommended cameras", "rgba(16,163,90,.75)"], ["photos", "Survey photos", "#fff"], ["detections", "Detected in photos", "#d946ef"], ["guards", "Guard plan", "#38bdf8"]];
$("#layers").innerHTML = LAYERS.map(l => `<label><input type="checkbox" data-l="${l[0]}" ${layers[l[0]] ? "checked" : ""}><i style="background:${l[2]}"></i>${l[1]}</label>`).join("");
$("#layers").addEventListener("change", ev => { layers[ev.target.dataset.l] = ev.target.checked; renderBase(); render(); });
(async () => {
  try { model = await (await fetch(C.base + "/api/site")).json(); } catch (e) { setStatus("Cannot load the project", true); return; }
  $("#name").value = model.name; document.title = model.name + " · " + C.app;
  hist = [JSON.stringify(model.features)]; hi = 0;
  setTool("select"); renderBase(); fit(); renderResults(); renderProps(); renderSettings(); analyse();
})();
"""


def editor_html(base: str = "") -> str:
    """``base`` is the URL prefix the project is served under ("" when it is the only one)."""
    data = {
        "app": f"{APP_NAME} {M.APP}",
        "base": base,
        "dtypes": M.DETECTION_TYPES,
        "geometry": M.GEOMETRY,
        "kind": M.KIND_LABEL,
        "barrier": M.BARRIER_TYPES,
        "access": {k: v[0] for k, v in M.ACCESS_TYPES.items()},
        "height": M.DEFAULT_HEIGHT,
    }
    body = (BODY.replace("__LOGO__", LOGO).replace("__THEME__", theme_button())
            .replace("__BRAND__", APP_NAME).replace("__APP__", M.APP)
            .replace("__HOME__", "/" if base else "#").replace("__HOMETIP__", "Back to all jobs" if base else APP_NAME))
    return page(f"{APP_NAME} {M.APP}", body, CSS, JS, json.loads(json.dumps(data)), default_theme="dark")
