"""Self-contained, offline HTML review page for triage results.

No internet needed: data, styles, script and map are in one file next to the
``crops/`` and ``previews/`` folders, so the whole output folder can be copied
to a USB stick and opened on any laptop at the command post.
"""

from __future__ import annotations

from pathlib import Path

from ..ui import LOGO, icon, page, theme_button

CSS = r"""
.wrap{max-width:1600px;margin:0 auto;padding:22px 24px 60px}
.hero{display:flex;flex-wrap:wrap;align-items:flex-end;gap:16px 24px;margin-bottom:16px}
.hero h1{font-size:26px;letter-spacing:-.025em;margin:0;font-weight:700}
.hero .sub{color:var(--muted);font-size:13px}
.stats{margin-bottom:12px}
.toolbar{position:sticky;top:59px;z-index:20;display:flex;flex-wrap:wrap;align-items:center;gap:10px 14px;
  padding:10px 12px;margin:14px 0 16px;background:var(--glass);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
  border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow)}
.toolbar .hint{margin-left:auto;color:var(--muted);font-size:12px;display:flex;gap:10px;align-items:center}
@media (max-width:1100px){.toolbar .hint{display:none}}
@media (max-width:700px){.wrap{padding:16px}.toolbar{top:53px}}
.layout{display:grid;grid-template-columns:minmax(300px,36%) 1fr;gap:18px;align-items:start}
@media (max-width:980px){.layout{grid-template-columns:1fr}}
.mapcard{position:sticky;top:132px;overflow:hidden}
@media (max-width:980px){.mapcard{position:static}}
.mapcard #map{display:block;width:100%;height:auto;background:
  linear-gradient(var(--line) 1px,transparent 1px) 0 0/40px 40px,linear-gradient(90deg,var(--line) 1px,transparent 1px) 0 0/40px 40px}
@media (max-width:980px){.mapcard #map{max-height:46vh}}
.legend{display:flex;flex-wrap:wrap;gap:12px;padding:10px 14px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:0}
#cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px}
.card{position:relative;background:var(--surface);border:1px solid var(--line);border-radius:var(--r-lg);overflow:hidden;
  box-shadow:var(--shadow);transition:border-color .15s,transform .15s,box-shadow .15s;display:flex;flex-direction:column}
.card:hover{transform:translateY(-1px)}
.card.sel{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft),var(--shadow)}
.card.confirmed{border-color:color-mix(in srgb,var(--good) 60%,transparent)}
.card.dismissed{opacity:.5}
.thumb{position:relative;aspect-ratio:1/1;background:#000;cursor:zoom-in;overflow:hidden}
.thumb img{width:100%;height:100%;object-fit:cover;display:block;transition:transform .3s}
.card:hover .thumb img{transform:scale(1.03)}
.over{position:absolute;inset:auto 0 0 0;height:60px;background:linear-gradient(transparent,rgba(0,0,0,.55));pointer-events:none}
.rank{position:absolute;top:10px;left:10px;height:24px;padding:0 9px;border-radius:999px;display:flex;align-items:center;
  background:rgba(10,11,14,.6);backdrop-filter:blur(8px);color:#fff;font:600 12px var(--font);border:1px solid rgba(255,255,255,.15)}
.ringpos{position:absolute;top:8px;right:8px;--sz:42px;border-radius:50%;padding:2px;background:rgba(10,11,14,.55);backdrop-filter:blur(8px)}
.ringpos .ring>span{background:rgba(10,11,14,.85);color:#fff}
.status-pill{position:absolute;left:10px;bottom:10px;height:22px;padding:0 9px;border-radius:999px;display:none;align-items:center;gap:5px;
  font:600 11.5px var(--font);color:#fff}
.card.confirmed .status-pill.c{display:inline-flex;background:var(--good)}
.card.dismissed .status-pill.x{display:inline-flex;background:#555}
.cbody{padding:12px 14px 14px;display:flex;flex-direction:column;gap:8px;flex:1}
.ttl{font-weight:600;font-size:14px;letter-spacing:-.005em;line-height:1.3}
.meta{color:var(--muted);font-size:12px;display:flex;flex-wrap:wrap;gap:4px 10px}
.coord{display:inline-flex;align-items:center;gap:6px;font:500 12px var(--mono);color:var(--ink-2);cursor:copy;
  padding:4px 8px;border-radius:7px;background:var(--surface-2);border:1px solid var(--line);width:max-content;max-width:100%}
.coord:hover{border-color:var(--line-2)}
.acts{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:auto;padding-top:4px}
.acts .btn{justify-content:center}
.acts .c.on{background:var(--good);border-color:var(--good);color:#fff}
.acts .x.on{background:var(--surface-2);color:var(--muted)}
/* lightbox */
#lb{position:fixed;inset:0;z-index:60;display:none;background:rgba(5,6,8,.9);backdrop-filter:blur(6px)}
#lb.open{display:grid;grid-template-columns:1fr 340px}
@media (max-width:900px){#lb.open{grid-template-columns:1fr;grid-template-rows:1fr auto}}
#lb .stage{display:grid;place-items:center;padding:24px;min-height:0;position:relative}
#lb .stage img{max-width:100%;max-height:calc(100vh - 48px);object-fit:contain;border-radius:10px;box-shadow:0 30px 80px rgba(0,0,0,.6)}
#lb aside{background:var(--surface);border-left:1px solid var(--line);padding:20px;overflow:auto;color:var(--ink)}
#lb .nav{position:absolute;top:50%;transform:translateY(-50%);background:rgba(255,255,255,.08);color:#fff;border:1px solid rgba(255,255,255,.15)}
#lb .nav.l{left:16px}#lb .nav.r{right:16px}
#lb .close{position:absolute;top:16px;right:16px;background:rgba(255,255,255,.08);color:#fff;border:1px solid rgba(255,255,255,.15)}
.stat.ref{display:flex;gap:12px;align-items:center;padding:10px 12px}
.stat.ref img{width:52px;height:52px;object-fit:cover;border-radius:10px;border:1px solid var(--line)}
.stat.ref .sw{display:flex;gap:5px;margin-bottom:4px}
.stat.ref .sw i{width:20px;height:20px;border-radius:6px;border:1px solid var(--line-2);display:block}
.kv{display:grid;grid-template-columns:auto 1fr;gap:6px 14px;font-size:13px;margin:12px 0}
.kv dt{color:var(--muted)}.kv dd{margin:0;word-break:break-all}
"""

JS = r"""
const S = DATA.summary;
const KEY = "isr-triage:" + S.source + ":" + S.generated;
let status = {};
try { status = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(status)); } catch (e) {} };
const $ = s => document.querySelector(s);
const colv = (s, st) => st === "confirmed" ? "var(--good)" : s >= .75 ? "var(--hi)" : s >= .55 ? "var(--mid)" : "var(--lo)";
const ICON = DATA.icons;

const items = [];
for (const p of DATA.photos) for (const d of p.detections) items.push({...d, photo: p});
items.sort((a, b) => b.score - a.score);
items.forEach((it, i) => it.rank = i + 1);

$("#src").textContent = S.source;
$("#gen").textContent = "Generated " + S.generated + " · " + S.seconds + " s";
const stat = (v, k, cls = "") => `<div class="stat ${cls}"><div class="v">${v}</div><div class="k">${k}</div></div>`;
function renderStats() {
  const conf = items.filter(i => status[i.id] === "confirmed").length;
  $("#stats").innerHTML = stat(S.photos, "Photos scanned") + stat(S.photos_geolocatable, "Geolocated") +
    stat(S.photos_with_hits, "Photos with hits") + stat(S.detections, "Candidates", "accent") +
    stat(conf, "Confirmed") +
    (S.config.reference ? `<div class="stat ref"><img src="reference.jpg" alt="Reference"><div><div class="sw">${S.config.reference.colors.map(c => `<i title="${esc(c.name)}" style="background:${c.hex === "white" ? "#fff" : c.hex}"></i>`).join("") || "—"}</div><div class="k">Reference ${esc(S.config.reference.target)}</div></div></div>` :
     S.config.colors.length ? stat(esc(S.config.colors.join(", ")), "Target colours") : "") +
    (S.errors ? stat(S.errors, "Unreadable") : "");
}

const kinds = [...new Set(items.flatMap(i => i.kinds.map(k => k.split(":")[0])))];
for (const k of kinds) $("#kind").insertAdjacentHTML("beforeend", `<option value="${esc(k)}">${esc(k[0].toUpperCase() + k.slice(1))}</option>`);
const ms = $("#minScore"); ms.value = S.config.min_score; $("#minScoreV").textContent = (+ms.value).toFixed(2);
let view = "open";

let visible = [], sel = 0, history = [];
const matches = (i, st) => st === "all" || (st === "open" ? !status[i.id] : status[i.id] === st);
function filtered() {
  const m = +ms.value, k = $("#kind").value;
  return items.filter(i => i.score >= m && (!k || i.kinds.some(x => x.split(":")[0] === k)) && matches(i, view));
}
function updateCounts() {
  const m = +ms.value, k = $("#kind").value;
  const base = items.filter(i => i.score >= m && (!k || i.kinds.some(x => x.split(":")[0] === k)));
  document.querySelectorAll("#view button").forEach(b => b.querySelector(".count").textContent =
    base.filter(i => matches(i, b.dataset.v)).length);
}
function card(i, idx) {
  const st = status[i.id] || "";
  const ll = i.lat != null ? `${i.lat.toFixed(6)}, ${i.lon.toFixed(6)}` : "No position";
  const t = i.photo.taken_at ? i.photo.taken_at.replace("T", " ").slice(5, 16) : "";
  const pct = Math.round(i.score * 100);
  const tags = i.kinds.map(k => { const [a, b] = k.split(":"); return `<span class="tag ${a === "color" ? "accent" : a === "ref" ? "good" : ""}">${esc(a === "ref" ? "looks like reference" : b ? (b.startsWith("#") ? "ref colour" : b) : a)}</span>`; }).join("");
  return `<article class="card ${st} ${idx === sel ? "sel" : ""}" data-idx="${idx}">
    <div class="thumb" data-open="${idx}">
      <img loading="lazy" src="${esc(i.crop || "")}" alt="${esc(i.label)}" data-open="${idx}">
      <div class="over"></div>
      <span class="rank">#${i.rank}</span>
      <div class="ringpos"><div class="ring" style="--p:${pct};--c:${colv(i.score)}"><span>${pct}</span></div></div>
      <span class="status-pill c">${ICON.check} Confirmed</span><span class="status-pill x">Dismissed</span>
    </div>
    <div class="cbody">
      <div class="ttl">${esc(i.label)}</div>
      <div class="tags">${tags}</div>
      <div class="meta"><span>${esc(i.photo.rel.split(/[\\/]/).pop())} · #${i.n}</span>${t ? `<span>${esc(t)}</span>` : ""}
        ${i.area_m2 ? `<span>~${i.area_m2} m²</span>` : ""}${i.photo.gsd_cm ? `<span>${i.photo.gsd_cm} cm/px</span>` : ""}</div>
      <span class="coord" title="Copy coordinates" data-copy="${esc(ll)}">${ICON.copy}${esc(ll)}${i.position_source === "drone" ? " · drone" : ""}</span>
      <div class="acts">
        <button class="btn sm c ${st === "confirmed" ? "on" : ""}" data-act="confirmed" data-idx="${idx}">${ICON.check} Confirm</button>
        <button class="btn sm x ${st === "dismissed" ? "on" : ""}" data-act="dismissed" data-idx="${idx}">${ICON.x} Dismiss</button>
      </div>
    </div></article>`;
}
function render(keepSel) {
  visible = filtered();
  if (!keepSel) sel = 0;
  sel = Math.max(0, Math.min(sel, visible.length - 1));
  $("#cards").innerHTML = visible.length ? visible.map(card).join("") :
    `<div class="empty">${DATA.icons.target}<div>${items.length ? "Nothing here with these filters." : "No candidates were found in these photos."}</div></div>`;
  updateCounts(); renderStats(); drawMap();
}
function setStatus(idx, st) {
  const i = visible[idx]; if (!i) return;
  history.push([i.id, status[i.id]]);
  if (status[i.id] === st) delete status[i.id]; else status[i.id] = st;
  save(); render(true); select(Math.min(idx, visible.length - 1));
  if ($("#lb").classList.contains("open")) openLb(sel);
}
function select(idx, scroll = true) {
  sel = idx;
  document.querySelectorAll(".card.sel").forEach(c => c.classList.remove("sel"));
  const c = document.querySelector(`.card[data-idx="${idx}"]`);
  if (c) { c.classList.add("sel"); if (scroll) c.scrollIntoView({block: "nearest", behavior: "smooth"}); }
  drawMap();
}
function openLb(idx) {
  const i = visible[idx]; if (!i) return $("#lb").classList.remove("open");
  sel = idx;
  const lb = $("#lb"); lb.classList.add("open");
  lb.querySelector(".stage img").src = i.photo.preview || i.crop;
  const ll = i.lat != null ? `${i.lat.toFixed(6)}, ${i.lon.toFixed(6)}` : "—";
  const st = status[i.id];
  lb.querySelector("aside").innerHTML = `
    <div style="display:flex;gap:12px;align-items:center"><div class="ring" style="--sz:52px;--p:${Math.round(i.score*100)};--c:${colv(i.score)}"><span>${Math.round(i.score*100)}</span></div>
    <div><div class="label">Candidate #${i.rank}</div><div class="ttl" style="font-size:16px">${esc(i.label)}</div></div></div>
    <img src="${esc(i.crop)}" alt="" style="width:100%;border-radius:10px;margin-top:16px;border:1px solid var(--line)">
    <dl class="kv"><dt>Position</dt><dd class="mono">${esc(ll)}</dd><dt>Source</dt><dd>${esc(i.position_source || "—")}</dd>
    <dt>Photo</dt><dd>${esc(i.photo.rel)}</dd><dt>Box</dt><dd>#${i.n} in photo</dd><dt>Taken</dt><dd>${esc((i.photo.taken_at||"—").replace("T"," "))}</dd>
    <dt>Altitude</dt><dd>${i.photo.rel_alt_m ?? "—"} m · ${i.photo.gsd_cm ?? "—"} cm/px</dd><dt>Original</dt><dd class="mono">${esc(i.photo.file)}</dd></dl>
    <div class="acts"><button class="btn c ${st==="confirmed"?"on":""}" data-act="confirmed" data-idx="${idx}">${ICON.check} Confirm</button>
    <button class="btn x ${st==="dismissed"?"on":""}" data-act="dismissed" data-idx="${idx}">${ICON.x} Dismiss</button></div>`;
}

document.addEventListener("click", e => {
  const t = e.target.closest("[data-act],[data-open],[data-copy],[data-v],.nav,.close,.card");
  if (!t) return;
  if (t.dataset.act) return setStatus(+t.dataset.idx, t.dataset.act);
  if (t.dataset.v) { view = t.dataset.v; document.querySelectorAll("#view button").forEach(b => b.setAttribute("aria-pressed", b === t)); return render(); }
  if (t.dataset.open != null) { select(+t.dataset.open, false); return openLb(+t.dataset.open); }
  if (t.dataset.copy) { navigator.clipboard?.writeText(t.dataset.copy).then(() => toast("Copied " + t.dataset.copy)); return; }
  if (t.classList.contains("close")) return $("#lb").classList.remove("open");
  if (t.classList.contains("nav")) { const n = t.classList.contains("l") ? Math.max(0, sel - 1) : Math.min(visible.length - 1, sel + 1); select(n); return openLb(n); }
  if (t.classList.contains("card")) select(+t.dataset.idx, false);
});
document.addEventListener("keydown", e => {
  if (["SELECT", "INPUT"].includes(e.target.tagName)) return;
  const lb = $("#lb").classList.contains("open");
  const k = e.key.toLowerCase();
  if (k === "escape") return $("#lb").classList.remove("open");
  if (["j", "arrowright", "arrowdown"].includes(k)) { e.preventDefault(); select(Math.min(sel + 1, visible.length - 1)); if (lb) openLb(sel); }
  else if (["k", "arrowleft", "arrowup"].includes(k)) { e.preventDefault(); select(Math.max(sel - 1, 0)); if (lb) openLb(sel); }
  else if (k === "c") setStatus(sel, "confirmed");
  else if (k === "x") setStatus(sel, "dismissed");
  else if (k === " " || k === "enter") { e.preventDefault(); lb ? $("#lb").classList.remove("open") : openLb(sel); }
  else if (k === "u") {
    const h = history.pop(); if (!h) return;
    if (h[1]) status[h[0]] = h[1]; else delete status[h[0]];
    save(); render(true); toast("Undone");
  }
});
["#minScore", "#kind"].forEach(s => $(s).addEventListener("input", () => { $("#minScoreV").textContent = (+ms.value).toFixed(2); render(); }));

const confirmed = () => items.filter(i => status[i.id] === "confirmed" && i.lat != null);
$("#expGpx").onclick = () => {
  const c = confirmed(); if (!c.length) return toast("Confirm a candidate first");
  download("confirmed.gpx", `<?xml version="1.0" encoding="UTF-8"?>\n<gpx version="1.1" creator="isr-tools" xmlns="http://www.topografix.com/GPX/1/1">\n` +
    c.map(i => `<wpt lat="${i.lat.toFixed(7)}" lon="${i.lon.toFixed(7)}"><name>${esc(i.id)}</name><desc>${esc(i.label)} — ${esc(i.photo.rel)}</desc><sym>Flag, Red</sym></wpt>`).join("\n") +
    "\n</gpx>\n", "application/gpx+xml");
  toast(`Exported ${c.length} waypoint(s)`);
};
$("#expCsv").onclick = () => {
  const c = confirmed(); if (!c.length) return toast("Confirm a candidate first");
  const q = v => `"${String(v ?? "").replace(/"/g, '""')}"`;
  download("confirmed.csv", "id,score,label,lat,lon,photo,taken_at\n" +
    c.map(i => [i.id, i.score, i.label, i.lat, i.lon, i.photo.rel, i.photo.taken_at].map(q).join(",")).join("\n") + "\n", "text/csv");
};

// ---- map: local metres, no tiles, works offline ----
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
  const span = Math.max(x1 - x0, y1 - y0, 50), pad = 36, size = 600;
  const s = (size - 2 * pad) / span, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  proj = { s, xy: (lat, lon) => { const [x, y] = en(lat, lon); return [size / 2 + (x - cx) * s, size / 2 - (y - cy) * s]; } };
}
function drawMap() {
  const svg = $("#map");
  if (!proj) { svg.innerHTML = `<text x="300" y="300" text-anchor="middle" fill="currentColor" opacity=".5" font-size="14">No GPS data in these photos</text>`; return; }
  let h = "";
  for (const p of geo) if (p.footprint)
    h += `<polygon points="${p.footprint.map(q => proj.xy(q[0], q[1]).join(",")).join(" ")}" fill="var(--accent)" fill-opacity=".05" stroke="var(--accent)" stroke-opacity=".28" stroke-width=".8"/>`;
  const track = [...geo].sort((a, b) => (a.taken_at || "").localeCompare(b.taken_at || "") || a.rel.localeCompare(b.rel));
  h += `<polyline points="${track.map(p => proj.xy(p.lat, p.lon).join(",")).join(" ")}" fill="none" stroke="var(--muted)" stroke-width="1.2" stroke-dasharray="4 4" opacity=".7"/>`;
  const shown = new Set(visible.map(i => i.id)), selId = visible[sel]?.id;
  for (const i of [...items].reverse()) {
    if (i.lat == null || status[i.id] === "dismissed") continue;
    const [x, y] = proj.xy(i.lat, i.lon), on = shown.has(i.id) || status[i.id] === "confirmed", me = i.id === selId;
    if (me) h += `<circle cx="${x}" cy="${y}" r="16" fill="${colv(i.score, status[i.id])}" opacity=".18"/>`;
    h += `<circle cx="${x}" cy="${y}" r="${me ? 8 : 5.5}" fill="${colv(i.score, status[i.id])}" fill-opacity="${on ? 1 : .25}" stroke="var(--surface)" stroke-width="2" data-id="${esc(i.id)}" style="cursor:pointer"><title>#${i.rank} ${esc(i.label)} (${Math.round(i.score*100)})</title></circle>`;
  }
  const nice = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000].find(m => m * proj.s > 70) || 5000;
  h += `<g fill="currentColor" color="var(--ink-2)" font-size="11.5" font-weight="600"><rect x="22" y="568" width="${nice * proj.s}" height="3" rx="1.5"/>
    <text x="22" y="560">${nice >= 1000 ? nice / 1000 + " km" : nice + " m"}</text>
    <circle cx="572" cy="34" r="16" fill="var(--surface)" stroke="var(--line-2)"/><path d="M572 22 l-5 13 l5 -3 l5 3z"/><text x="572" y="46" text-anchor="middle" font-size="9">N</text></g>`;
  svg.innerHTML = h;
}
$("#map").addEventListener("click", e => {
  const id = e.target.dataset?.id; if (!id) return;
  let idx = visible.findIndex(i => i.id === id);
  if (idx < 0) { view = "all"; document.querySelectorAll("#view button").forEach(b => b.setAttribute("aria-pressed", b.dataset.v === "all")); ms.value = 0; $("#kind").value = ""; render(); idx = visible.findIndex(i => i.id === id); }
  select(idx);
});
render();
"""


def _body() -> str:
    seg = "".join(
        f'<button data-v="{v}" aria-pressed="{str(v == "open").lower()}">{label} <span class="count">0</span></button>'
        for v, label in (("open", "To review"), ("confirmed", "Confirmed"), ("dismissed", "Dismissed"), ("all", "All"))
    )
    return f"""
<header class="topbar">
  <div class="brand">{LOGO}<div class="brand-name">ISR <span>/ Search triage</span></div></div>
  <div class="crumb mono" id="src"></div>
  <div class="spacer"></div>
  <button class="btn" id="expCsv" title="Export confirmed as CSV">{icon("download")}<span class="lbl">CSV</span></button>
  <button class="btn primary" id="expGpx" title="Export confirmed as GPX waypoints">{icon("download")}<span class="lbl">Confirmed GPX</span></button>
  {theme_button()}
</header>
<div class="wrap">
  <section class="hero">
    <div><h1>Search triage</h1><div class="sub" id="gen"></div></div>
  </section>
  <section class="stats" id="stats"></section>
  <details class="note"><summary>{icon("info")} How to read this, and accuracy</summary>
    <p>Candidates are ranked by score (0–100). Hits on a <b>target colour</b> and hits found by more than one detector rank
    highest. <b>Anomaly</b> hits are anything statistically unusual for that photo, so expect false alarms (litter, flowers,
    rocks, vehicles). Every hit needs a human look: the tool narrows thousands of photos to a short list; it does not clear an area.</p>
    <p>Positions are projected from the drone GPS, altitude above <i>take-off</i> and gimbal angle onto flat ground: typically a few
    metres straight down, worse for oblique photos or terrain above/below take-off. “Drone” means the hit could not be projected and the
    drone's own position is shown.</p>
    <p>Finding nothing does <b>not</b> mean nobody is there: canopy, shadow and camouflage-coloured clothing all defeat colour-based detection.</p>
  </details>
  <div class="toolbar">
    <div class="seg" id="view" role="group" aria-label="Review status">{seg}</div>
    <label class="field">Detector <select class="input" id="kind"><option value="">All detectors</option></select></label>
    <label class="field">Min score <input type="range" id="minScore" min="0" max="1" step="0.05"><span class="val" id="minScoreV"></span></label>
    <div class="hint">{icon("keyboard")} <span><kbd>J</kbd> <kbd>K</kbd> move</span><span><kbd>C</kbd> confirm</span>
      <span><kbd>X</kbd> dismiss</span><span><kbd>U</kbd> undo</span><span><kbd>Space</kbd> view</span></div>
  </div>
  <div class="layout">
    <aside class="panel mapcard">
      <div class="panel-h">{icon("map")} Coverage &amp; candidates</div>
      <svg id="map" viewBox="0 0 600 600" role="img" aria-label="Map of photo footprints and candidates"></svg>
      <div class="legend"><span><i class="dot" style="background:var(--hi)"></i>≥ 75</span>
        <span><i class="dot" style="background:var(--mid)"></i>≥ 55</span><span><i class="dot" style="background:var(--lo)"></i>lower</span>
        <span><i class="dot" style="background:var(--good)"></i>confirmed</span><span>tinted = photo coverage · dashed = flight path</span></div>
    </aside>
    <section id="cards" aria-live="polite"></section>
  </div>
</div>
<div id="lb" role="dialog" aria-modal="true" aria-label="Candidate detail">
  <div class="stage"><img alt="">
    <button class="btn icon nav l" aria-label="Previous">{icon("chevron-left", 18)}</button>
    <button class="btn icon nav r" aria-label="Next">{icon("chevron-right", 18)}</button>
    <button class="btn icon close" aria-label="Close">{icon("x", 18)}</button></div>
  <aside></aside>
</div>
"""


def write_report(data: dict, out_dir: Path) -> Path:
    payload = dict(data)
    payload["icons"] = {k: icon(k, 14) for k in ("check", "x", "copy", "target")}
    payload["icons"]["target"] = icon("target", 36)
    path = out_dir / "report.html"
    path.write_text(page("Search Triage", _body(), CSS, JS, payload), encoding="utf-8")
    return path
