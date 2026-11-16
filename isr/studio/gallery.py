"""Offline gallery for Studio results: browse, compare edits, pick favourites."""

from __future__ import annotations

from pathlib import Path

from ..ui import LOGO, icon, page, theme_button

CSS = r"""
.wrap{max-width:1760px;margin:0 auto;padding:22px 24px 80px}
.hero{display:flex;flex-wrap:wrap;align-items:flex-end;justify-content:space-between;gap:16px;margin-bottom:16px}
.hero h1{font-size:28px;letter-spacing:-.03em;margin:0;font-weight:720}
.hero h1 em{font-style:normal;background:linear-gradient(90deg,var(--accent),var(--accent-2));-webkit-background-clip:text;background-clip:text;color:transparent}
.hero .sub{color:var(--muted);font-size:13px;margin-top:2px}
.toolbar{position:sticky;top:59px;z-index:20;display:flex;flex-wrap:wrap;align-items:center;gap:10px 14px;padding:10px 12px;
  margin:16px 0;background:var(--glass);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);border:1px solid var(--line);
  border-radius:14px;box-shadow:var(--shadow)}
.toolbar .right{margin-left:auto;display:flex;gap:10px;align-items:center}
@media (max-width:700px){.wrap{padding:16px}.toolbar{top:53px}.toolbar .right{margin-left:0}}
@media (max-width:1100px){.kbdhint{display:none!important}}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(var(--tile,300px),1fr));gap:12px}
.tile{position:relative;border-radius:14px;overflow:hidden;background:var(--surface-2);cursor:zoom-in;aspect-ratio:4/3;
  border:1px solid var(--line);box-shadow:var(--shadow);outline:none;transition:transform .2s,box-shadow .2s}
.tile:hover{transform:translateY(-2px);box-shadow:var(--shadow-lg)}
.tile.sel{box-shadow:0 0 0 3px var(--accent),var(--shadow-lg)}
.tile img{width:100%;height:100%;object-fit:cover;display:block;transition:transform .5s cubic-bezier(.2,.7,.2,1),opacity .3s}
.tile:hover img{transform:scale(1.035)}
.tile.rejected img{opacity:.28;filter:grayscale(1)}
.tile .shade{position:absolute;inset:0;background:linear-gradient(180deg,rgba(0,0,0,.42) 0,transparent 30%,transparent 55%,rgba(0,0,0,.72));
  opacity:.0;transition:opacity .2s;pointer-events:none}
.tile:hover .shade,.tile.sel .shade,.tile.picked .shade{opacity:1}
.tl{position:absolute;top:10px;left:10px;display:flex;gap:6px;align-items:center}
.pill{height:24px;padding:0 9px;border-radius:999px;display:inline-flex;align-items:center;gap:5px;font:600 11.5px var(--font);
  color:#fff;background:rgba(10,11,14,.55);backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,.14)}
.pill.top{background:linear-gradient(90deg,var(--accent),var(--accent-2));border-color:transparent}
.tr{position:absolute;top:8px;right:8px;display:flex;gap:6px}
.heart{width:34px;height:34px;border-radius:50%;display:grid;place-items:center;border:1px solid rgba(255,255,255,.18);
  background:rgba(10,11,14,.5);backdrop-filter:blur(8px);color:#fff;cursor:pointer;opacity:0;transition:opacity .2s,transform .1s}
.tile:hover .heart,.tile.picked .heart,.tile.sel .heart{opacity:1}
.heart:active{transform:scale(.92)}
.tile.picked .heart{background:#ff4d6d;border-color:#ff4d6d}
.tile.picked .heart svg{fill:#fff}
.bl{position:absolute;left:12px;right:12px;bottom:10px;display:flex;align-items:flex-end;justify-content:space-between;gap:8px;color:#fff;
  opacity:0;transform:translateY(4px);transition:opacity .2s,transform .2s;pointer-events:none}
.tile:hover .bl,.tile.sel .bl{opacity:1;transform:none}
.bl .t{font-size:12px;opacity:.85;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bl .tg{display:flex;gap:4px;flex-wrap:wrap;margin-top:4px}
.bl .tg span{font-size:10.5px;font-weight:600;padding:2px 7px;border-radius:999px;background:rgba(255,255,255,.16);backdrop-filter:blur(6px)}
.score{font:750 22px var(--font);letter-spacing:-.03em;text-shadow:0 2px 12px rgba(0,0,0,.5)}
.score small{font-size:11px;font-weight:600;opacity:.7;margin-left:2px}
/* viewer */
#viewer{position:fixed;inset:0;z-index:60;display:none;background:#050608}
#viewer.open{display:grid;grid-template-columns:1fr 380px}
@media (max-width:980px){#viewer.open{grid-template-columns:1fr;grid-template-rows:62vh 1fr}}
.stage{position:relative;min-height:0;display:grid;place-items:center;padding:20px;overflow:hidden}
.cmp{position:relative;max-width:100%;max-height:100%;line-height:0;user-select:none;touch-action:none}
.cmp img{max-width:100%;max-height:calc(100vh - 40px);display:block;border-radius:8px}
@media (max-width:980px){.cmp img{max-height:calc(62vh - 40px)}}
.cmp .after{position:absolute;inset:0;width:100%;height:100%;clip-path:inset(0 0 0 var(--x,50%))}
.cmp .handle{position:absolute;top:0;bottom:0;left:var(--x,50%);width:2px;background:#fff;box-shadow:0 0 0 1px rgba(0,0,0,.3);cursor:ew-resize}
.cmp .handle::after{content:"";position:absolute;top:50%;left:50%;width:34px;height:34px;transform:translate(-50%,-50%);border-radius:50%;
  background:rgba(255,255,255,.95) url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%23111' stroke-width='2.2'%3E%3Cpath d='m9 18-6-6 6-6M15 6l6 6-6 6'/%3E%3C/svg%3E") center no-repeat;
  box-shadow:0 4px 18px rgba(0,0,0,.4)}
.cmp .lab{position:absolute;top:12px;font:600 11px var(--font);letter-spacing:.06em;text-transform:uppercase;color:#fff;
  background:rgba(0,0,0,.5);padding:4px 8px;border-radius:6px;line-height:1}
.cmp .lab.b{left:12px}.cmp .lab.a{right:12px}
.cmp.single .after,.cmp.single .handle,.cmp.single .lab{display:none}
.vbtn{position:absolute;background:rgba(255,255,255,.08);color:#fff;border:1px solid rgba(255,255,255,.14)}
.vbtn:hover{background:rgba(255,255,255,.16)}
.vbtn.l{left:16px;top:50%;transform:translateY(-50%)}.vbtn.r{right:16px;top:50%;transform:translateY(-50%)}
.vbtn.c{right:16px;top:16px}
.side{background:var(--surface);border-left:1px solid var(--line);overflow:auto;padding:20px 20px 28px;color:var(--ink)}
.side h2{font-size:17px;margin:0;letter-spacing:-.01em;word-break:break-all}
.side .sec{margin-top:20px}
.side .sec>.label{margin-bottom:10px;display:block}
.metric{display:grid;grid-template-columns:96px 1fr 30px;gap:10px;align-items:center;font-size:12.5px;margin:7px 0}
.metric .n{color:var(--ink-2)}.metric .v{text-align:right;font-variant-numeric:tabular-nums;color:var(--muted)}
.duo{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.duo .stat{padding:10px 12px}.duo .stat .v{font-size:18px}
.vacts{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:16px}
.vacts .btn{justify-content:center;height:38px}
.btn.pick.on{background:#ff4d6d;border-color:#ff4d6d;color:#fff}
.btn.pick.on svg{fill:#fff}
.strip{display:flex;gap:6px;overflow-x:auto;padding-bottom:4px}
.strip img{width:76px;height:57px;object-fit:cover;border-radius:7px;border:2px solid transparent;cursor:pointer;flex:none;opacity:.75}
.strip img.cur{border-color:var(--accent);opacity:1}
.kv{display:grid;grid-template-columns:auto 1fr;gap:6px 14px;font-size:12.5px;margin:0}
.kv dt{color:var(--muted)}.kv dd{margin:0;word-break:break-all}
.applied{display:flex;flex-wrap:wrap;gap:5px}
/* export popover */
#exp{position:fixed;z-index:70;right:24px;top:64px;width:min(440px,calc(100vw - 32px));display:none;padding:16px}
#exp.open{display:block}
#exp pre{margin:8px 0 12px;padding:10px 12px;border-radius:9px;background:var(--bg-2);border:1px solid var(--line);
  font:12px var(--mono);white-space:pre-wrap;word-break:break-all}
"""

JS = r"""
const S = DATA.summary, I = DATA.icons;
const KEY = "isr-studio:" + S.source + ":" + S.generated;
let marks = {};
try { marks = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(marks)); } catch (e) {} };
const $ = s => document.querySelector(s);
const photos = DATA.photos.filter(p => p.thumb);
const byGroup = {};
photos.forEach(p => (byGroup[p.group] ??= []).push(p));
Object.values(byGroup).forEach(g => g.sort((a, b) => b.score - a.score));

let view = S.top ? "top" : "best", sortBy = "score", visible = [], sel = 0;
const views = {
  top: p => p.pick_rank, best: p => p.group_rank === 0, all: p => true,
  picked: p => marks[p.idx] === "pick", rejected: p => marks[p.idx] === "reject",
};
const sorters = {
  score: (a, b) => b.score - a.score,
  time: (a, b) => (a.taken_at || "").localeCompare(b.taken_at || "") || a.name.localeCompare(b.name),
  sharpness: (a, b) => b.parts.sharpness - a.parts.sharpness,
  colour: (a, b) => b.parts.colour - a.parts.colour,
  composition: (a, b) => b.parts.composition - a.parts.composition,
};
function stats() {
  const picks = photos.filter(p => marks[p.idx] === "pick").length;
  const st = (v, k, c = "") => `<div class="stat ${c}"><div class="v">${v}</div><div class="k">${k}</div></div>`;
  $("#stats").innerHTML = st(S.photos, "Photos scanned") + st(S.groups, "Distinct moments") + st(S.top, "Suggested", "accent") +
    st(S.edited, "Auto-edited") + st(picks, "Your picks") + st(S.seconds + " s", "Processing");
  document.querySelectorAll("#view button").forEach(b => b.querySelector(".count").textContent = photos.filter(views[b.dataset.v]).length);
  $("#expCount").textContent = picks;
}
function tile(p, i) {
  const m = marks[p.idx] || "";
  const tags = [...(p.tags || [])].filter(t => t !== "Top-down").slice(0, 3).map(t => `<span>${esc(t)}</span>`).join("");
  return `<div class="tile ${m === "pick" ? "picked" : ""} ${m === "reject" ? "rejected" : ""} ${i === sel ? "sel" : ""}" data-i="${i}" tabindex="-1">
    <img loading="lazy" src="${esc(p.thumb)}" alt="${esc(p.name)}">
    <div class="shade"></div>
    <div class="tl">${p.pick_rank ? `<span class="pill top">${I.sparkles} #${p.pick_rank}</span>` : ""}
      ${p.group_size > 1 ? `<span class="pill" title="Similar frames">${I.layers} ${p.group_size}</span>` : ""}
      ${p.edited ? `<span class="pill" title="Auto-edited version available">Edited</span>` : ""}</div>
    <div class="tr"><button class="heart" data-pick="${i}" aria-label="Pick">${I.heart}</button></div>
    <div class="bl"><div style="min-width:0"><div class="t">${esc(p.name)}</div><div class="tg">${tags}</div></div>
      <div class="score">${Math.round(p.score)}<small>/100</small></div></div>
  </div>`;
}
function render(keep) {
  visible = photos.filter(views[view]).sort(sorters[sortBy]);
  if (!keep) sel = 0;
  sel = Math.max(0, Math.min(sel, visible.length - 1));
  const msg = {picked: "No picks yet. Press P or tap the heart on photos you love.", rejected: "Nothing rejected.",
    top: "No suggestions.", best: "No photos.", all: "No photos."}[view];
  $("#grid").innerHTML = visible.length ? visible.map(tile).join("") : `<div class="empty">${I.image}<div>${msg}</div></div>`;
  stats();
}
function select(i, scroll = true) {
  sel = i;
  document.querySelectorAll(".tile.sel").forEach(t => t.classList.remove("sel"));
  const t = document.querySelector(`.tile[data-i="${i}"]`);
  if (t) { t.classList.add("sel"); if (scroll) t.scrollIntoView({block: "nearest", behavior: "smooth"}); }
}
function mark(i, m) {
  const p = visible[i]; if (!p) return;
  marks[p.idx] = marks[p.idx] === m ? undefined : m;
  if (!marks[p.idx]) delete marks[p.idx];
  save();
  const t = document.querySelector(`.tile[data-i="${i}"]`);
  if (t && view !== "picked" && view !== "rejected") {
    t.classList.toggle("picked", marks[p.idx] === "pick"); t.classList.toggle("rejected", marks[p.idx] === "reject"); stats();
  } else render(true);
  if ($("#viewer").classList.contains("open")) openViewer(sel);
  toast(marks[p.idx] === "pick" ? "Added to picks" : marks[p.idx] === "reject" ? "Rejected" : "Cleared");
}
const bar = (n, v) => `<div class="metric"><span class="n">${n}</span><div class="bar"><i style="width:${Math.round(v * 100)}%"></i></div><span class="v">${Math.round(v * 100)}</span></div>`;
function openViewer(i) {
  const p = visible[i]; if (!p) return;
  sel = i;
  const v = $("#viewer"); v.classList.add("open"); document.body.style.overflow = "hidden";
  const cmp = $("#cmp");
  const before = p.preview || p.thumb;
  cmp.querySelector(".before").src = before;
  if (p.preview_edit) { cmp.classList.remove("single"); cmp.querySelector(".after").src = p.preview_edit; }
  else cmp.classList.add("single");
  const m = marks[p.idx];
  const sim = byGroup[p.group] || [];
  const parts = p.parts || {};
  const t = p.taken_at ? p.taken_at.replace("T", " ") : "—";
  const applied = p.edit ? Object.keys(p.edit.applied).map(k => `<span class="tag accent">${esc(k.replace("_", " "))}</span>`).join("") : "";
  $("#side").innerHTML = `
    <div style="display:flex;gap:14px;align-items:center">
      <div class="ring" style="--sz:58px;--p:${Math.round(p.score)}"><span>${Math.round(p.score)}</span></div>
      <div style="min-width:0"><div class="label">${p.pick_rank ? "Suggested #" + p.pick_rank : "Photo"}${p.group_size > 1 ? " · best of " + p.group_size : ""}</div><h2>${esc(p.name)}</h2></div>
    </div>
    <div class="vacts">
      <button class="btn pick ${m === "pick" ? "on" : ""}" data-pick="${i}">${I.heart} ${m === "pick" ? "Picked" : "Pick"}</button>
      <button class="btn ${m === "reject" ? "on-bad" : ""}" data-reject="${i}">${I.x} Reject</button>
    </div>
    ${(p.tags?.length || p.warnings?.length) ? `<div class="sec"><div class="tags">${(p.tags || []).map(t => `<span class="tag good">${esc(t)}</span>`).join("")}${(p.warnings || []).map(t => `<span class="tag warn">${esc(t)}</span>`).join("")}</div></div>` : ""}
    <div class="sec"><span class="label">Score breakdown</span>
      <div class="duo"><div class="stat"><div class="v">${Math.round(p.technical * 100)}</div><div class="k">Technical</div></div>
      <div class="stat"><div class="v">${Math.round(p.artistic * 100)}</div><div class="k">Artistic</div></div></div>
      ${bar("Sharpness", parts.sharpness)}${bar("Exposure", parts.exposure)}${bar("Contrast", parts.contrast)}${bar("Low noise", parts.noise)}
      ${bar("Colour", parts.colour)}${bar("Composition", parts.composition)}${bar("Simplicity", parts.simplicity)}${bar("Pattern", parts.pattern)}
    </div>
    ${p.edit ? `<div class="sec"><span class="label">Auto-edit</span><div class="applied">${applied}</div>
      ${p.edit.notes?.length ? `<p class="muted" style="font-size:12.5px;margin:8px 0 0">${esc(p.edit.notes.join("; "))}</p>` : ""}
      <p class="muted" style="font-size:12.5px;margin:8px 0 0">Drag the slider on the photo to compare. Full-resolution file:</p>
      <div class="mono" style="word-break:break-all;margin-top:4px">${esc(p.edit_abs)}</div></div>` :
      p.edit_error ? `<div class="sec"><span class="tag warn">Edit failed: ${esc(p.edit_error)}</span></div>` : ""}
    ${sim.length > 1 ? `<div class="sec"><span class="label">Similar frames (${sim.length})</span><div class="strip">${sim.map(q => `<img src="${esc(q.thumb)}" class="${q.idx === p.idx ? "cur" : ""}" data-jump="${q.idx}" title="${esc(q.name)} · ${Math.round(q.score)}">`).join("")}</div></div>` : ""}
    <div class="sec"><span class="label">Details</span><dl class="kv">
      <dt>Taken</dt><dd>${esc(t)}</dd><dt>Camera</dt><dd>${esc(p.camera || "—")}</dd>
      <dt>Size</dt><dd>${p.width ?? "—"} × ${p.height ?? "—"}</dd>
      <dt>Altitude</dt><dd>${p.alt_m != null ? Math.round(p.alt_m) + " m above take-off" : "—"}</dd>
      <dt>Gimbal</dt><dd>${p.pitch != null ? Math.round(p.pitch) + "°" : "—"}</dd>
      <dt>Location</dt><dd class="mono">${p.lat != null ? p.lat.toFixed(5) + ", " + p.lon.toFixed(5) : "—"}</dd>
      <dt>File</dt><dd class="mono">${esc(p.file)}</dd></dl></div>`;
}
function closeViewer() { $("#viewer").classList.remove("open"); document.body.style.overflow = ""; render(true); select(sel); }

// before/after slider
(() => {
  const cmp = $("#cmp"); let drag = false;
  const set = e => { const r = cmp.getBoundingClientRect(); const x = Math.max(0, Math.min(1, ((e.touches ? e.touches[0].clientX : e.clientX) - r.left) / r.width)); cmp.style.setProperty("--x", (x * 100) + "%"); };
  cmp.addEventListener("pointerdown", e => { drag = true; cmp.setPointerCapture(e.pointerId); set(e); });
  cmp.addEventListener("pointermove", e => drag && set(e));
  cmp.addEventListener("pointerup", () => drag = false);
})();

document.addEventListener("click", e => {
  const t = e.target.closest("[data-pick],[data-reject],[data-jump],[data-v],.tile,.vbtn,#expBtn,#expDl,#expCopy");
  if (!t) { if (!e.target.closest("#exp")) $("#exp").classList.remove("open"); return; }
  if (t.id === "expBtn") return $("#exp").classList.toggle("open");
  if (t.id === "expDl") return exportPicks();
  if (t.id === "expCopy") return navigator.clipboard?.writeText($("#expCmd").textContent).then(() => toast("Command copied"));
  if (t.dataset.pick != null) { e.stopPropagation(); return mark(+t.dataset.pick, "pick"); }
  if (t.dataset.reject != null) return mark(+t.dataset.reject, "reject");
  if (t.dataset.jump != null) {
    let i = visible.findIndex(q => q.idx === +t.dataset.jump);
    if (i < 0) { view = "all"; syncSeg(); render(); i = visible.findIndex(q => q.idx === +t.dataset.jump); }
    return openViewer(i);
  }
  if (t.dataset.v) { view = t.dataset.v; syncSeg(); return render(); }
  if (t.classList.contains("vbtn")) {
    if (t.classList.contains("c")) return closeViewer();
    return openViewer(t.classList.contains("l") ? Math.max(0, sel - 1) : Math.min(visible.length - 1, sel + 1));
  }
  if (t.classList.contains("tile")) { select(+t.dataset.i, false); openViewer(+t.dataset.i); }
});
const syncSeg = () => document.querySelectorAll("#view button").forEach(b => b.setAttribute("aria-pressed", b.dataset.v === view));
document.addEventListener("keydown", e => {
  if (["SELECT", "INPUT"].includes(e.target.tagName)) return;
  const open = $("#viewer").classList.contains("open"), k = e.key.toLowerCase();
  if (k === "escape") { $("#exp").classList.remove("open"); return open && closeViewer(); }
  if (["arrowright", "j", "l"].includes(k)) { e.preventDefault(); const n = Math.min(visible.length - 1, sel + 1); open ? openViewer(n) : select(n); }
  else if (["arrowleft", "k", "h"].includes(k)) { e.preventDefault(); const n = Math.max(0, sel - 1); open ? openViewer(n) : select(n); }
  else if (k === "p" || k === "f") mark(sel, "pick");
  else if (k === "x") mark(sel, "reject");
  else if (k === "e" && open) { const c = $("#cmp"); const x = getComputedStyle(c).getPropertyValue("--x").trim(); c.style.setProperty("--x", x === "0%" ? "100%" : "0%"); }
  else if (k === " " || k === "enter") { e.preventDefault(); open ? closeViewer() : openViewer(sel); }
});
$("#sort").addEventListener("input", e => { sortBy = e.target.value; render(); });
$("#size").addEventListener("input", e => document.documentElement.style.setProperty("--tile", e.target.value + "px"));

function exportPicks() {
  const picks = photos.filter(p => marks[p.idx] === "pick");
  if (!picks.length) return toast("Pick some photos first");
  download("picks.txt", picks.map(p => p.file).join("\n") + "\n", "text/plain");
  toast(`Saved picks.txt (${picks.length})`);
}
$("#expCmd").textContent = `isr enhance --list picks.txt -o "${S.out_dir}${S.out_dir.includes("\\") ? "\\" : "/"}picks"`;
syncSeg(); render();
"""


def _body() -> str:
    seg = "".join(
        f'<button data-v="{v}" aria-pressed="false">{ic}{label} <span class="count">0</span></button>'
        for v, label, ic in (("top", "Suggested", icon("sparkles", 14)), ("best", "Best of each", icon("layers", 14)),
                             ("all", "All photos", ""), ("picked", "Picks", icon("heart", 14)), ("rejected", "Rejected", ""))
    )
    return f"""
<header class="topbar">
  <div class="brand">{LOGO}<div class="brand-name">ISR <span>/ Studio</span></div></div>
  <div class="crumb mono" id="src"></div>
  <div class="spacer"></div>
  <button class="btn primary" id="expBtn">{icon("download")}<span class="lbl">Export picks</span>
    <span class="num" style="opacity:.8" id="expCount">0</span></button>
  {theme_button()}
</header>
<div class="panel" id="exp" role="dialog" aria-label="Export picks">
  <div class="label">Export your picks</div>
  <p class="muted" style="font-size:13px;margin:8px 0">Download the list of picked files, then run this to make full-resolution
  edits of exactly those photos (originals are never changed):</p>
  <pre id="expCmd"></pre>
  <div style="display:flex;gap:8px"><button class="btn primary" id="expDl">{icon("download")} Download picks.txt</button>
  <button class="btn" id="expCopy">{icon("copy")} Copy command</button></div>
</div>
<div class="wrap">
  <section class="hero">
    <div><h1>Your best <em>aerial shots</em></h1><div class="sub" id="gen"></div></div>
  </section>
  <section class="stats" id="stats"></section>
  <div class="toolbar">
    <div class="seg" id="view" role="group" aria-label="View">{seg}</div>
    <label class="field">Sort <select class="input" id="sort">
      <option value="score">Overall score</option><option value="time">Time taken</option><option value="sharpness">Sharpness</option>
      <option value="colour">Colour</option><option value="composition">Composition</option></select></label>
    <div class="right">
      <label class="field" title="Thumbnail size">{icon("image")}<input type="range" id="size" min="180" max="520" value="300"></label>
      <span class="field kbdhint" style="font-size:12px"><kbd>P</kbd> pick <kbd>X</kbd> reject <kbd>←</kbd><kbd>→</kbd> <kbd>E</kbd> before/after</span>
    </div>
  </div>
  <section id="grid" aria-live="polite"></section>
</div>
<div id="viewer" role="dialog" aria-modal="true" aria-label="Photo viewer">
  <div class="stage">
    <div class="cmp" id="cmp" style="--x:50%"><img class="before" alt="Original"><img class="after" alt="Auto-edited">
      <div class="handle"></div><span class="lab b">Original</span><span class="lab a">Edited</span></div>
    <button class="btn icon vbtn l" aria-label="Previous">{icon("chevron-left", 18)}</button>
    <button class="btn icon vbtn r" aria-label="Next">{icon("chevron-right", 18)}</button>
    <button class="btn icon vbtn c" aria-label="Close">{icon("x", 18)}</button>
  </div>
  <aside class="side" id="side"></aside>
</div>
<script>document.addEventListener("DOMContentLoaded",()=>{{document.getElementById("src").textContent=DATA.summary.source;
document.getElementById("gen").textContent=`${{DATA.summary.photos}} photos · generated ${{DATA.summary.generated}}`;}});</script>
"""


def write_gallery(data: dict, out_dir: Path) -> Path:
    payload = dict(data)
    payload["icons"] = {k: icon(k, 14) for k in ("heart", "x", "layers", "sparkles", "check")}
    payload["icons"]["image"] = icon("image", 36)
    path = out_dir / "gallery.html"
    path.write_text(page("Studio", _body(), CSS, JS, payload, default_theme="dark"), encoding="utf-8")
    return path
