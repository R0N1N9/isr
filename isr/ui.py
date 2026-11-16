"""Shared design system for the offline HTML pages (triage report, studio gallery).

Pages are single self-contained files: no web fonts, CDNs or network access,
so they open from a USB stick on a laptop in the field. Light and dark themes
follow the system setting, with a toggle that remembers the choice.
"""

from __future__ import annotations

import json

# Lucide-style 24px stroke icons (MIT-licensed shapes, redrawn inline)
ICONS = {
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    "moon": '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>',
    "map": '<path d="M9 4 3 6v14l6-2 6 2 6-2V4l-6 2-6-2z"/><path d="M9 4v14M15 6v14"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "x": '<path d="M18 6 6 18M6 6l12 12"/>',
    "download": '<path d="M12 3v12M7 10l5 5 5-5"/><path d="M5 21h14"/>',
    "heart": '<path d="M19.5 12.6 12 20l-7.5-7.4A5 5 0 0 1 12 6a5 5 0 0 1 7.5 6.6z"/>',
    "sparkles": '<path d="M12 3l1.8 4.7L18.5 9.5l-4.7 1.8L12 16l-1.8-4.7L5.5 9.5l4.7-1.8z"/><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z"/>',
    "image": '<rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="9" cy="9" r="2"/><path d="m21 15-5-5L5 21"/>',
    "layers": '<path d="m12 2 10 5-10 5L2 7z"/><path d="m2 17 10 5 10-5M2 12l10 5 10-5"/>',
    "copy": '<rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/>',
    "chevron-left": '<path d="m15 18-6-6 6-6"/>',
    "chevron-right": '<path d="m9 18 6-6-6-6"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 16v-4M12 8h.01"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    "compare": '<path d="M12 3v18"/><rect x="3" y="5" width="18" height="14" rx="2"/>',
    "keyboard": '<rect x="2" y="6" width="20" height="12" rx="2"/><path d="M6 10h.01M10 10h.01M14 10h.01M18 10h.01M7 14h10"/>',
}


def icon(name: str, size: int = 16, cls: str = "") -> str:
    return (f'<svg class="ic {cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" '
            f'aria-hidden="true">{ICONS[name]}</svg>')


LOGO = ('<svg class="logo" width="26" height="26" viewBox="0 0 32 32" aria-hidden="true">'
        '<defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="var(--accent)"/>'
        '<stop offset="1" stop-color="var(--accent-2)"/></linearGradient></defs>'
        '<rect width="32" height="32" rx="9" fill="url(#lg)"/>'
        '<g fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round">'
        '<circle cx="9.5" cy="9.5" r="3"/><circle cx="22.5" cy="9.5" r="3"/>'
        '<circle cx="9.5" cy="22.5" r="3"/><circle cx="22.5" cy="22.5" r="3"/>'
        '<path d="M12 12l8 8M20 12l-8 8"/></g></svg>')

CSS = r"""
:root{
  color-scheme: light;
  --bg:#f6f7f9; --bg-2:#eef0f4; --surface:#ffffff; --surface-2:#f8f9fb; --raise:#ffffff;
  --ink:#0d1117; --ink-2:#3b4250; --muted:#6b7385; --faint:#9aa2b1;
  --line:rgba(13,17,23,.09); --line-2:rgba(13,17,23,.16);
  --accent:#5b5bf7; --accent-2:#22b8cf; --accent-ink:#ffffff; --accent-soft:rgba(91,91,247,.10);
  --good:#12a150; --warn:#e08a00; --bad:#e5484d; --hi:#e5484d; --mid:#f08c00; --lo:#d4a500;
  --shadow-sm:0 1px 2px rgba(16,24,40,.05);
  --shadow:0 1px 2px rgba(16,24,40,.04),0 8px 24px -8px rgba(16,24,40,.12);
  --shadow-lg:0 24px 64px -16px rgba(16,24,40,.28);
  --r-sm:8px; --r:12px; --r-lg:16px;
  --font: Inter, "SF Pro Text", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --mono: "JetBrains Mono", "SF Mono", ui-monospace, Menlo, Consolas, monospace;
  --glass: rgba(255,255,255,.72);
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --bg:#0a0b0e; --bg-2:#0f1115; --surface:#121419; --surface-2:#171a20; --raise:#1b1e25;
  --ink:#eef0f4; --ink-2:#c5cad4; --muted:#8b93a3; --faint:#5d6574;
  --line:rgba(255,255,255,.07); --line-2:rgba(255,255,255,.13);
  --accent:#7c7cff; --accent-2:#3dd6e8; --accent-soft:rgba(124,124,255,.14);
  --good:#3ccf7a; --warn:#f5a524; --bad:#ff6369; --hi:#ff5a63; --mid:#ff9f1a; --lo:#f2c94c;
  --shadow-sm:none; --shadow:0 1px 0 rgba(255,255,255,.03) inset,0 10px 30px -12px rgba(0,0,0,.6);
  --shadow-lg:0 30px 80px -20px rgba(0,0,0,.8);
  --glass: rgba(14,16,20,.72);
}
*{box-sizing:border-box}
html,body{margin:0}
body{background:var(--bg);color:var(--ink);font:14px/1.5 var(--font);-webkit-font-smoothing:antialiased;
  font-feature-settings:"cv11","ss01";min-height:100vh}
body::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:-1;
  background:radial-gradient(1200px 500px at 85% -10%,var(--accent-soft),transparent 60%),
             radial-gradient(900px 400px at -10% 0%,rgba(34,184,207,.06),transparent 60%)}
a{color:inherit}
.ic{flex:none;display:block}
.num{font-variant-numeric:tabular-nums}
.mono{font-family:var(--mono);font-size:12px}
.muted{color:var(--muted)}

/* top bar */
.topbar{position:sticky;top:0;z-index:30;display:flex;align-items:center;gap:14px;padding:12px 24px;
  background:var(--glass);backdrop-filter:saturate(1.6) blur(16px);-webkit-backdrop-filter:saturate(1.6) blur(16px);
  border-bottom:1px solid var(--line)}
.brand{display:flex;align-items:center;gap:10px;min-width:0}
.brand-name{font-weight:650;letter-spacing:-.01em;font-size:15px;white-space:nowrap}
.brand-name span{color:var(--muted);font-weight:500}
.crumb{color:var(--muted);font-size:12.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0}
.topbar .spacer{flex:1}
@media (max-width:700px){.topbar{padding:10px 16px;gap:8px}.crumb,.brand-name span,.topbar .lbl{display:none}
  .topbar .btn:not(.icon){padding:0 10px}}

/* buttons */
.btn{display:inline-flex;align-items:center;gap:7px;height:34px;padding:0 13px;border-radius:9px;
  border:1px solid var(--line-2);background:var(--surface);color:var(--ink);font:500 13px var(--font);
  cursor:pointer;white-space:nowrap;box-shadow:var(--shadow-sm);transition:background .15s,border-color .15s,transform .05s}
.btn:hover{border-color:var(--faint);background:var(--surface-2)}
.btn:active{transform:translateY(1px)}
.btn.primary{background:linear-gradient(180deg,color-mix(in srgb,var(--accent) 92%,#fff),var(--accent));
  border-color:transparent;color:var(--accent-ink);box-shadow:0 6px 18px -8px var(--accent)}
.btn.primary:hover{filter:brightness(1.07)}
.btn.ghost{background:transparent;border-color:transparent;box-shadow:none}
.btn.ghost:hover{background:var(--surface-2)}
.btn.icon{width:34px;padding:0;justify-content:center}
.btn.sm{height:28px;padding:0 10px;font-size:12px;border-radius:8px}
.btn.on-good{background:var(--good);border-color:var(--good);color:#fff}
.btn.on-bad{background:var(--surface-2);color:var(--muted);text-decoration:line-through}

/* segmented control */
.seg{display:inline-flex;max-width:100%;overflow-x:auto;scrollbar-width:none;padding:3px;gap:2px;background:var(--bg-2);border:1px solid var(--line);border-radius:10px}
.seg button{border:0;background:transparent;color:var(--muted);font:500 12.5px var(--font);height:28px;padding:0 11px;
  border-radius:7px;cursor:pointer;display:inline-flex;align-items:center;gap:6px;white-space:nowrap}
.seg button:hover{color:var(--ink)}
.seg button[aria-pressed="true"]{background:var(--raise);color:var(--ink);box-shadow:var(--shadow-sm),0 0 0 1px var(--line)}
.seg .count{font-size:11px;color:var(--faint);font-variant-numeric:tabular-nums}

/* form bits */
.field{display:inline-flex;align-items:center;gap:8px;color:var(--muted);font-size:12.5px;white-space:nowrap}
select.input{appearance:none;height:34px;padding:0 30px 0 11px;border-radius:9px;border:1px solid var(--line-2);
  background:var(--surface) url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%23888' stroke-width='2'%3E%3Cpath d='m6 9 6 6 6-6'/%3E%3C/svg%3E") no-repeat right 10px center;
  color:var(--ink);font:500 13px var(--font);cursor:pointer}
input[type=range]{appearance:none;-webkit-appearance:none;width:120px;height:4px;border-radius:4px;background:var(--line-2);outline:none}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;background:var(--surface);
  border:1px solid var(--line-2);box-shadow:0 1px 3px rgba(0,0,0,.25);cursor:pointer}
input[type=range]::-moz-range-thumb{width:16px;height:16px;border-radius:50%;background:var(--surface);border:1px solid var(--line-2);cursor:pointer}
.val{min-width:34px;color:var(--ink);font-variant-numeric:tabular-nums;font-weight:550}

/* cards / panels */
.panel{background:var(--surface);border:1px solid var(--line);border-radius:var(--r-lg);box-shadow:var(--shadow)}
.panel-h{display:flex;align-items:center;gap:8px;padding:12px 14px;border-bottom:1px solid var(--line);
  font-weight:600;font-size:13px}
.label{font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}

/* stats */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px}
.stat{padding:14px 16px;background:var(--surface);border:1px solid var(--line);border-radius:var(--r);box-shadow:var(--shadow-sm)}
.stat .v{font-size:22px;font-weight:650;letter-spacing:-.02em;font-variant-numeric:tabular-nums;line-height:1.2}
.stat .k{color:var(--muted);font-size:12px;margin-top:2px}
.stat.accent .v{background:linear-gradient(90deg,var(--accent),var(--accent-2));-webkit-background-clip:text;background-clip:text;color:transparent}

/* tags */
.tags{display:flex;flex-wrap:wrap;gap:5px}
.tag{display:inline-flex;align-items:center;gap:4px;height:22px;padding:0 8px;border-radius:999px;font-size:11.5px;font-weight:550;
  background:var(--surface-2);border:1px solid var(--line);color:var(--ink-2);white-space:nowrap}
.tag.good{color:var(--good);background:color-mix(in srgb,var(--good) 10%,transparent);border-color:color-mix(in srgb,var(--good) 25%,transparent)}
.tag.warn{color:var(--warn);background:color-mix(in srgb,var(--warn) 10%,transparent);border-color:color-mix(in srgb,var(--warn) 25%,transparent)}
.tag.accent{color:var(--accent);background:var(--accent-soft);border-color:color-mix(in srgb,var(--accent) 30%,transparent)}

kbd{font:600 10.5px var(--mono);padding:1px 5px;border-radius:5px;border:1px solid var(--line-2);border-bottom-width:2px;
  background:var(--surface);color:var(--ink-2)}
details.note{color:var(--muted);font-size:13px}
details.note summary{cursor:pointer;display:inline-flex;align-items:center;gap:6px;color:var(--ink-2);font-weight:550;list-style:none}
details.note summary::-webkit-details-marker{display:none}
details.note[open] summary{margin-bottom:6px}
details.note p{margin:6px 0;max-width:80ch}

/* toast */
#toast{position:fixed;left:50%;bottom:24px;transform:translate(-50%,12px);opacity:0;z-index:90;pointer-events:none;
  background:var(--ink);color:var(--bg);padding:9px 16px;border-radius:10px;font-weight:550;font-size:13px;
  box-shadow:var(--shadow-lg);transition:opacity .2s,transform .2s}
#toast.show{opacity:1;transform:translate(-50%,0)}

/* score ring */
.ring{--p:0;--c:var(--accent);width:var(--sz,40px);height:var(--sz,40px);border-radius:50%;display:grid;place-items:center;
  background:conic-gradient(var(--c) calc(var(--p)*1%),var(--line-2) 0);flex:none}
.ring>span{width:calc(100% - 6px);height:calc(100% - 6px);border-radius:50%;background:var(--surface);display:grid;
  place-items:center;font-weight:700;font-size:calc(var(--sz,40px)*.32);font-variant-numeric:tabular-nums}

.bar{height:6px;border-radius:6px;background:var(--line);overflow:hidden}
.bar>i{display:block;height:100%;border-radius:6px;background:linear-gradient(90deg,var(--accent),var(--accent-2))}

.empty{padding:60px 20px;text-align:center;color:var(--muted);grid-column:1/-1}
.empty .ic{margin:0 auto 10px;opacity:.5}
::selection{background:var(--accent-soft)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
"""

THEME_JS = r"""
(function(){
  const KEY="isr-theme", root=document.documentElement;
  let saved=null; try{saved=localStorage.getItem(KEY)}catch(e){}
  const sys=()=>matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light";
  const apply=t=>{root.dataset.theme=t;document.querySelectorAll("[data-theme-icon]").forEach(el=>{
    el.innerHTML=t==="dark"?el.dataset.sun:el.dataset.moon;});};
  apply(saved||root.dataset.defaultTheme||sys());
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change",()=>{if(!saved)apply(sys())});
  window.toggleTheme=()=>{const t=root.dataset.theme==="dark"?"light":"dark";saved=t;
    try{localStorage.setItem(KEY,t)}catch(e){}apply(t)};
  document.addEventListener("DOMContentLoaded",()=>apply(root.dataset.theme));
})();
window.toast=(m)=>{const t=document.getElementById("toast");if(!t)return;t.textContent=m;t.classList.add("show");
  clearTimeout(window.__tt);window.__tt=setTimeout(()=>t.classList.remove("show"),1600)};
window.esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
window.download=(name,text,type)=>{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([text],{type}));
  a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(a.href),1500)};
"""


def theme_button() -> str:
    sun, moon = icon("sun"), icon("moon")
    return (f'<button class="btn icon ghost" onclick="toggleTheme()" title="Toggle light / dark" aria-label="Toggle theme">'
            f'<span data-theme-icon data-sun=\'{sun}\' data-moon=\'{moon}\'>{moon}</span></button>')


def page(title: str, body: str, css: str = "", js: str = "", data: dict | None = None,
         default_theme: str | None = None) -> str:
    """Assemble a complete HTML document. ``data`` is embedded as ``window.DATA``."""
    blob = json.dumps(data, default=str).replace("</", "<\\/") if data is not None else "null"
    dt = f' data-default-theme="{default_theme}"' if default_theme else ""
    return f"""<!doctype html>
<html lang="en"{dt}>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script>{THEME_JS}</script>
<style>{CSS}{css}</style>
</head>
<body>
{body}
<div id="toast" role="status" aria-live="polite"></div>
<script>window.DATA = {blob};</script>
<script>{js}</script>
</body>
</html>
"""
