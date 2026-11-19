"""Offline HTML page for the movement analysis."""

from __future__ import annotations

from pathlib import Path

from ..ui import LOGO, icon, page, theme_button

CSS = r"""
.wrap{max-width:1680px;margin:0 auto;padding:22px 24px 60px}
.hero h1{font-size:26px;letter-spacing:-.025em;margin:0;font-weight:700}
.hero .sub{color:var(--muted);font-size:13px;margin:2px 0 16px}
.layout{display:grid;grid-template-columns:1fr 420px;gap:18px;align-items:start;margin-top:16px}
@media (max-width:1100px){.layout{grid-template-columns:1fr}}
.mapcard{position:sticky;top:76px;overflow:hidden}
@media (max-width:1100px){.mapcard{position:static}}
#map{display:block;width:100%;height:auto;max-height:calc(100vh - 170px);background:var(--bg-2);touch-action:none;cursor:grab}
.layers{display:flex;flex-wrap:wrap;gap:6px;padding:10px 12px;border-bottom:1px solid var(--line)}
.layers button{height:28px;padding:0 10px;border-radius:999px;border:1px solid var(--line-2);background:var(--surface);
  color:var(--muted);font:500 12px var(--font);cursor:pointer;display:inline-flex;align-items:center;gap:6px}
.layers button[aria-pressed="true"]{color:var(--ink);border-color:var(--accent);background:var(--accent-soft)}
.layers i{width:10px;height:10px;border-radius:3px;display:inline-block}
.legend{display:flex;flex-wrap:wrap;gap:12px;padding:10px 14px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}
.side{display:flex;flex-direction:column;gap:14px}
.sec{padding:14px 16px}
.sec h3{margin:0 0 10px;font-size:13px;display:flex;align-items:center;gap:8px}
.leg{border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin-bottom:8px;font-size:12.5px}
.leg .top{display:flex;justify-content:space-between;gap:8px;font-weight:600}
.leg .v{margin-top:6px;color:var(--ink-2)}
.leg.good{border-left:3px solid var(--good)}.leg.warn{border-left:3px solid var(--warn)}.leg.bad{border-left:3px solid var(--bad)}
.leg.info{border-left:3px solid var(--line-2)}
.segrow{display:grid;grid-template-columns:28px 1fr auto;gap:10px;align-items:center;padding:8px 0;border-bottom:1px solid var(--line);font-size:12.5px;cursor:pointer}
.segrow:last-child{border-bottom:0}
.segrow .n{width:26px;height:26px;border-radius:8px;display:grid;place-items:center;font-weight:700;color:#fff;background:#eb503c}
.segrow .bar{margin-top:5px}
.cmd{display:flex;gap:6px;align-items:center;margin-top:10px}
.cmd code{flex:1;font:11.5px var(--mono);background:var(--bg-2);border:1px solid var(--line);border-radius:8px;padding:7px 9px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.notes{margin:0;padding-left:18px;color:var(--warn);font-size:12.5px}
.kv{display:grid;grid-template-columns:auto 1fr;gap:5px 14px;font-size:12.5px;margin:0}
.kv dt{color:var(--muted)}.kv dd{margin:0}
"""

JS = r"""
const D = DATA, S = D.summary, G = S.grid;
const $ = s => document.querySelector(s);
const svg = $("#map");
const on = {hillshade: true, reach: true, probability: true, corridor: true, coverage: true, ways: true, segments: true};
const fmtMin = m => m >= 120 ? (m / 60).toFixed(1) + " h" : Math.round(m) + " min";
const stat = (v, k, c = "") => `<div class="stat ${c}"><div class="v">${v}</div><div class="k">${k}</div></div>`;
$("#stats").innerHTML = stat(S.mode === "foot" ? "On foot" : "Vehicle", S.mode === "foot" ? `Profile: ${S.profile}` : "Road network") +
  stat(fmtMin(S.elapsed_min), "Since last sighting") + stat(S.reach_max_km + " km", "Furthest reachable", "accent") +
  stat(S.reach_km2 + " km²", "Reachable area") + stat(D.sightings.length, "Sightings") +
  stat(S.heading_deg != null ? Math.round(S.heading_deg) + "°" : "—", "Last heading");
$("#sub").textContent = `Predicted for ${S.at} · last seen ${S.last.time || "(no time)"} · ${G.res_m} m cells`;

const home = S.focus || [0, 0, G.nx, G.ny];
let vb = [...home];
function draw() {
  let h = `<rect x="0" y="0" width="${G.nx}" height="${G.ny}" fill="var(--bg-2)"/>`;
  const L = D.layers;
  if (on.hillshade && L.hillshade) h += `<image href="${L.hillshade}" x="0" y="0" width="${G.nx}" height="${G.ny}" preserveAspectRatio="none" style="image-rendering:auto"/>`;
  if (on.coverage && L.coverage) h += `<image href="${L.coverage}" x="0" y="0" width="${G.nx}" height="${G.ny}" preserveAspectRatio="none"/>`;
  if (on.reach && L.reach) h += `<image href="${L.reach}" x="0" y="0" width="${G.nx}" height="${G.ny}" preserveAspectRatio="none"/>`;
  if (on.probability && L.probability) h += `<image href="${L.probability}" x="0" y="0" width="${G.nx}" height="${G.ny}" preserveAspectRatio="none"/>`;
  if (on.corridor && L.corridor) h += `<image href="${L.corridor}" x="0" y="0" width="${G.nx}" height="${G.ny}" preserveAspectRatio="none"/>`;
  const sw = vb[2] / 600;
  if (on.ways) {
    const st = {road: ["#f2f2f2", 2.2, "#555"], path: ["#c8a46e", 1.6, ""], river: ["#3b9cff", 3, ""], stream: ["#3b9cff", 1.2, ""],
      water: ["#3b9cff", 1, ""], cliff: ["#222", 2, ""]};
    for (const w of D.ways) {
      const [c, wd, casing] = st[w.kind] || ["#999", 1, ""];
      const pts = w.pts.map(p => p.join(",")).join(" ");
      if (w.kind === "water") { h += `<polygon points="${pts}" fill="#3b9cff" fill-opacity=".45" stroke="#3b9cff" stroke-width="${sw}"/>`; continue; }
      if (casing) h += `<polyline points="${pts}" fill="none" stroke="${casing}" stroke-width="${(wd + 1.4) * sw}" stroke-linecap="round" stroke-linejoin="round"/>`;
      h += `<polyline points="${pts}" fill="none" stroke="${c}" stroke-width="${wd * sw}" ${w.kind === "path" ? `stroke-dasharray="${4 * sw} ${2.5 * sw}"` : ""} stroke-linecap="round" stroke-linejoin="round"><title>${esc(w.name || w.kind)}</title></polyline>`;
    }
  }
  if (on.segments) D.segments.forEach(s => {
    const pts = s.px.map(p => p.join(",")).join(" ");
    const cx = s.px.reduce((a, p) => a + p[0], 0) / 4, cy = s.px.reduce((a, p) => a + p[1], 0) / 4;
    h += `<polygon points="${pts}" fill="#eb503c" fill-opacity=".06" stroke="#eb503c" stroke-width="${1.6 * sw}" data-seg="${s.rank}"/>
      <circle cx="${cx}" cy="${cy}" r="${9 * sw}" fill="#eb503c"/><text x="${cx}" y="${cy + 3.6 * sw}" text-anchor="middle" font-size="${10 * sw}" font-weight="700" fill="#fff">${s.rank}</text>`;
  });
  D.paths.forEach(p => { if (p.length) h += `<polyline points="${p.map(q => q.join(",")).join(" ")}" fill="none" stroke="#14c8aa" stroke-width="${2.6 * sw}" stroke-linecap="round" stroke-linejoin="round"/>`; });
  if (D.sightings.length > 1) h += `<polyline points="${D.sightings.filter(s => s.time).map(s => s.px.join(",")).join(" ")}" fill="none" stroke="#fff" stroke-width="${1.2 * sw}" stroke-dasharray="${3 * sw} ${3 * sw}" opacity=".8"/>`;
  D.sightings.forEach((s, k) => {
    const last = s.label === S.last.label && s.lat === S.last.lat;
    h += `<g><circle cx="${s.px[0]}" cy="${s.px[1]}" r="${(last ? 11 : 9) * sw}" fill="${last ? "var(--accent)" : "#111"}" stroke="#fff" stroke-width="${2 * sw}"/>
      <text x="${s.px[0]}" y="${s.px[1] + 3.8 * sw}" text-anchor="middle" font-size="${10.5 * sw}" font-weight="700" fill="#fff">${k + 1}</text>
      <title>${esc(s.label)} ${esc(s.time || "")}</title></g>`;
  });
  // scale bar + north
  const nice = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000].find(m => m / G.res_m > vb[2] / 7) || 20000;
  const L2 = nice / G.res_m, x = vb[0] + vb[2] * 0.04, y = vb[1] + vb[3] * 0.95;
  h += `<g fill="#fff" stroke="#000" stroke-width="${0.6 * sw}" paint-order="stroke" font-size="${11 * sw}" font-weight="700">
    <rect x="${x}" y="${y}" width="${L2}" height="${3 * sw}"/><text x="${x}" y="${y - 5 * sw}">${nice >= 1000 ? nice / 1000 + " km" : nice + " m"}</text>
    <text x="${vb[0] + vb[2] * 0.96}" y="${vb[1] + vb[3] * 0.07}" text-anchor="middle">N ↑</text></g>`;
  svg.innerHTML = h;
  svg.setAttribute("viewBox", vb.join(" "));
}
// pan / zoom
let drag = null;
svg.addEventListener("wheel", e => { e.preventDefault(); const r = svg.getBoundingClientRect();
  const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height, k = e.deltaY > 0 ? 1.15 : 1 / 1.15;
  const w = Math.min(G.nx * 1.2, Math.max(20, vb[2] * k)), h = w * G.ny / G.nx;
  vb = [vb[0] + (vb[2] - w) * fx, vb[1] + (vb[3] - h) * fy, w, h]; draw(); }, {passive: false});
svg.addEventListener("pointerdown", e => { drag = [e.clientX, e.clientY, vb[0], vb[1]]; svg.setPointerCapture(e.pointerId); svg.style.cursor = "grabbing"; });
svg.addEventListener("pointermove", e => { if (!drag) return; const r = svg.getBoundingClientRect();
  vb[0] = drag[2] - (e.clientX - drag[0]) * vb[2] / r.width; vb[1] = drag[3] - (e.clientY - drag[1]) * vb[3] / r.height; draw(); });
svg.addEventListener("pointerup", () => { drag = null; svg.style.cursor = "grab"; });
$("#reset").onclick = () => { vb = [...home]; draw(); };
document.querySelectorAll(".layers button[data-l]").forEach(b => {
  if (!D.layers[b.dataset.l] && !["ways", "segments"].includes(b.dataset.l)) b.remove();
  b.setAttribute("aria-pressed", on[b.dataset.l]);
  b.onclick = () => { on[b.dataset.l] = !on[b.dataset.l]; b.setAttribute("aria-pressed", on[b.dataset.l]); draw(); };
});
function focusSeg(r) {
  const s = D.segments.find(x => x.rank === r); if (!s) return;
  const xs = s.px.map(p => p[0]), ys = s.px.map(p => p[1]);
  const w = Math.max(...xs) - Math.min(...xs), cx = (Math.max(...xs) + Math.min(...xs)) / 2, cy = (Math.max(...ys) + Math.min(...ys)) / 2;
  const W = Math.max(w * 5, 60), H = W * G.ny / G.nx; vb = [cx - W / 2, cy - H / 2, W, H]; draw();
}

// side panel
$("#legs").innerHTML = D.legs.length ? D.legs.map((l, k) => `<div class="leg ${l.level}">
  <div class="top"><span>${k + 1} → ${k + 2}</span><span class="muted num">${(l.distance_m / 1000).toFixed(2)} km${l.dt_s != null ? " · " + fmtMin(l.dt_s / 60) : ""}${l.speed_kmh != null ? " · " + l.speed_kmh + " km/h" : ""}</span></div>
  <div class="v">${esc(l.verdict)}</div></div>`).join("") : `<p class="muted" style="margin:0;font-size:12.5px">Add two or more timed sightings to check legs and draw corridors.</p>`;
$("#sightings").innerHTML = D.sightings.map((s, k) => `<dt>${k + 1}. ${esc(s.label)}</dt><dd class="mono">${s.time ? esc(s.time) : "no time"}<br>${s.lat.toFixed(5)}, ${s.lon.toFixed(5)}</dd>`).join("");
const total = D.segments.reduce((a, s) => a + s.p_remaining, 0);
$("#segs").innerHTML = D.segments.length ? D.segments.map(s => `<div class="segrow" data-r="${s.rank}"><span class="n">${s.rank}</span>
  <div><div><b>${(s.p_remaining * 100).toFixed(1)}%</b> <span class="muted">of probability · ${s.size_m[0]}×${s.size_m[1]} m</span></div>
  <div class="bar"><i style="width:${Math.min(100, s.p_remaining / D.segments[0].p_remaining * 100)}%"></i></div></div>
  <span class="muted mono">seg ${String(s.rank).padStart(2, "0")}</span></div>`).join("") +
  `<div class="cmd"><code id="cmd">isr plan search --aoi segments/segment_01.kml --alt 50</code><button class="btn icon sm" id="copy" title="Copy">${D.icons.copy}</button></div>
   <p class="muted" style="font-size:12px;margin:8px 0 0">Top ${D.segments.length} segments hold ${(total * 100).toFixed(0)}% of the remaining probability${S.covered_share ? ` (photo coverage already searched ${(S.covered_share * 100).toFixed(0)}% at POD ${S.pod})` : ""}. Each is saved as <span class="mono">segments/segment_NN.kml</span> for <span class="mono">isr plan</span>.</p>`
  : `<p class="muted" style="margin:0">No reachable area.</p>`;
document.querySelectorAll(".segrow").forEach(r => r.onclick = () => { focusSeg(+r.dataset.r);
  $("#cmd").textContent = `isr plan search --aoi segments/segment_${String(r.dataset.r).padStart(2, "0")}.kml --alt 50`; });
$("#copy")?.addEventListener("click", () => navigator.clipboard?.writeText($("#cmd").textContent).then(() => toast("Command copied")));
$("#notes").innerHTML = (S.notes || []).map(n => `<li>${esc(n)}</li>`).join("");
if (!(S.notes || []).length) $("#notesec").remove();
if (D.layers.reach_step_min) $("#reachleg").textContent = `blue rings every ${fmtMin(D.layers.reach_step_min)} of travel`;
draw();
"""


def _body() -> str:
    lay = "".join(
        f'<button data-l="{k}"><i style="background:{c}"></i>{n}</button>'
        for k, n, c in (("probability", "Probability", "linear-gradient(90deg,#ffd666,#eb3250,#aa14a0)"),
                        ("reach", "Reach (time)", "#3c82ff"), ("corridor", "Corridors", "#14c8aa"),
                        ("coverage", "Searched", "#ffffff"), ("segments", "Segments", "#eb503c"),
                        ("ways", "Trails & roads", "#c8a46e"), ("hillshade", "Terrain", "#888"))
    )
    return f"""
<header class="topbar">
  <div class="brand">{LOGO}<div class="brand-name">ISR <span>/ Movement analysis</span></div></div>
  <div class="spacer"></div>
  <a class="btn" href="track.kml" download>{icon("download")}<span class="lbl">KML</span></a>
  <a class="btn" href="track.gpx" download>{icon("download")}<span class="lbl">GPX</span></a>
  {theme_button()}
</header>
<div class="wrap">
  <section class="hero"><h1>Where could they be now?</h1><div class="sub" id="sub"></div></section>
  <section class="stats" id="stats"></section>
  <details class="note" style="margin-top:12px"><summary>{icon("info")} How this is calculated, and its limits</summary>
    <p><b>Reach</b> is every place the subject could have got to since the last sighting, using terrain-aware travel times:
    walking speed falls with slope (Tobler's hiking function), is 40% slower off-trail, and rivers, lakes and cliffs block
    movement except at trail and road crossings. Vehicles follow the road network.</p>
    <p><b>Corridors</b> between two timed sightings show every place the subject could have passed through and still make the
    second sighting on time (a “space-time prism”). The teal line is the fastest route.</p>
    <p><b>Probability</b> assumes the subject kept moving for an unknown share of the time, then weights cells ahead in the
    direction of travel and near trails. It is a planning aid built on these assumptions, not a prediction of one route.
    Segments are ranked by probability not yet covered by your photo searches.</p>
  </details>
  <div class="layout">
    <div class="panel mapcard">
      <div class="layers">{lay}<button id="reset" style="margin-left:auto">Reset view</button></div>
      <svg id="map" viewBox="0 0 100 100" role="img" aria-label="Map of reach, probability and corridors"></svg>
      <div class="legend"><span>● numbered = sightings (accent = last)</span><span id="reachleg"></span>
        <span>teal line = fastest route between sightings</span><span>scroll to zoom, drag to pan</span></div>
    </div>
    <div class="side">
      <div class="panel sec"><h3>{icon("target")} Search segments</h3><div id="segs"></div></div>
      <div class="panel sec"><h3>{icon("layers")} Legs between sightings</h3><div id="legs"></div></div>
      <div class="panel sec"><h3>{icon("map")} Sightings</h3><dl class="kv" id="sightings"></dl></div>
      <div class="panel sec" id="notesec"><h3>{icon("info")} Notes</h3><ul class="notes" id="notes"></ul></div>
    </div>
  </div>
</div>
"""


def write_track_report(data: dict, out_dir: Path) -> Path:
    payload = {k: v for k, v in data.items() if k != "paths_latlon"}
    payload["icons"] = {"copy": icon("copy", 14)}
    path = Path(out_dir) / "track.html"
    path.write_text(page("Movement Analysis", _body(), CSS, JS, payload, default_theme="dark"), encoding="utf-8")
    return path
