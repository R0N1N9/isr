"""Studio: cull a card of drone photos down to portfolio candidates.

1. score every photo (technical + artistic, see ``aesthetics``)
2. group near-duplicates (bursts, brackets, hovering re-takes) so each moment
   is represented by its best frame
3. suggest the top N moments and, optionally, auto-enhance them
4. write an offline gallery for picking favourites
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

from ..camera import read_photo_meta
from ..triage.pipeline import find_photos
from .aesthetics import dhash, hamming, load_work, score_image
from .enhance import enhance_file


@dataclass
class StudioConfig:
    top: int = 30
    enhance: bool = True
    strength: float = 0.6
    straighten: bool = True
    dup_bits: int = 30  # max differing bits (of 256) to call two frames "the same moment"
    dup_window_s: float = 180.0  # ...and taken within this many seconds of each other
    workers: int = 0
    thumb_px: int = 720
    preview_px: int = 2000


def _score_job(args):
    path, idx, out_dir, cfg = args
    rec = {"idx": idx, "file": str(path), "name": path.name}
    try:
        meta = read_photo_meta(path)
        rec.update(
            width=meta.width, height=meta.height, camera=meta.camera_model,
            taken_at=meta.taken_at.isoformat() if meta.taken_at else None,
            lat=meta.lat, lon=meta.lon, alt_m=meta.rel_alt_m, pitch=meta.gimbal_pitch, yaw=meta.gimbal_yaw,
            focal_35mm=meta.focal_35mm,
        )
    except Exception as e:
        rec["error"] = f"metadata: {e}"
        meta = None
    try:
        rgb = load_work(path)
        a = score_image(rgb, meta.gimbal_pitch if meta else None)
        rec.update(a.to_dict())
        rec["hash"] = format(dhash(rgb), "064x")
        thumb = Image.fromarray((rgb * 255).astype("uint8"))
        thumb.thumbnail((cfg.thumb_px, cfg.thumb_px), Image.LANCZOS)
        rel = f"thumbs/{idx:05d}_{path.stem}.jpg"
        thumb.save(out_dir / rel, quality=84, optimize=True)
        rec["thumb"] = rel
        rec["aspect"] = round(rgb.shape[1] / rgb.shape[0], 4)
    except Exception as e:
        rec["error"] = (rec.get("error", "") + f" score: {e}").strip()
        rec["score"] = 0.0
    return rec


def _enhance_job(args):
    rec, out_dir, cfg = args
    src = Path(rec["file"])
    stem = f"{rec['idx']:05d}_{src.stem}"
    dst = out_dir / "edited" / f"{src.stem}_edit.jpg"
    if dst.exists():  # two cards with the same file names: keep both
        dst = out_dir / "edited" / f"{stem}_edit.jpg"
    try:
        rep = enhance_file(src, dst, cfg.strength, tilt_deg=rec.get("tilt_deg"), auto_straighten=cfg.straighten,
                           gimbal_pitch=rec.get("pitch"))
        for f, name in ((src, f"previews/{stem}.jpg"), (dst, f"previews/{stem}_edit.jpg")):
            with Image.open(f) as im:
                im.draft("RGB", (cfg.preview_px, cfg.preview_px))
                im = im.convert("RGB")
                im.thumbnail((cfg.preview_px, cfg.preview_px), Image.LANCZOS)
                im.save(out_dir / name, quality=88, optimize=True)
        return {"edited": str(dst.relative_to(out_dir)), "edit_abs": str(dst.resolve()), "preview": f"previews/{stem}.jpg",
                "preview_edit": f"previews/{stem}_edit.jpg", "edit": {"applied": rep.applied, "notes": rep.notes,
                                                                     "summary": rep.summary()}}
    except Exception as e:
        return {"edit_error": str(e)}


def group_similar(recs: list[dict], max_bits: int, window_s: float) -> list[list[int]]:
    """Union-find over frames that look alike and were taken close together."""
    parent = list(range(len(recs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def ts(r):
        try:
            return datetime.fromisoformat(r["taken_at"]).timestamp() if r.get("taken_at") else None
        except ValueError:
            return None

    order = sorted(range(len(recs)), key=lambda i: (ts(recs[i]) is None, ts(recs[i]) or 0, recs[i]["name"]))
    times = [ts(recs[i]) for i in order]
    hashes = [int(recs[i]["hash"], 16) if recs[i].get("hash") else None for i in order]
    for a in range(len(order)):
        if hashes[a] is None:
            continue
        for b in range(a + 1, len(order)):
            if times[a] is not None and times[b] is not None and times[b] - times[a] > window_s:
                break
            if times[a] is None and b - a > 20:  # no timestamps: compare neighbours by name only
                break
            if hashes[b] is not None and hamming(hashes[a], hashes[b]) <= max_bits:
                ra, rb = find(order[a]), find(order[b])
                if ra != rb:
                    parent[rb] = ra
    groups: dict[int, list[int]] = {}
    for i in range(len(recs)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def run_studio(source: Path, out_dir: Path, cfg: StudioConfig, log=print) -> dict:
    t0 = time.time()
    source, out_dir = Path(source), Path(out_dir)
    photos = find_photos(source, exclude=out_dir)
    if not photos:
        raise SystemExit(f"No JPEG photos found under {source}")
    for d in ("thumbs", "previews", "edited"):
        (out_dir / d).mkdir(parents=True, exist_ok=True)
    workers = cfg.workers or max(1, (os.cpu_count() or 2) - 1)
    log(f"Scoring {len(photos)} photos from {source} with {workers} worker(s)...")

    ctx = mp.get_context("spawn")
    recs: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        jobs = [(p, i, out_dir, cfg) for i, p in enumerate(photos)]
        for n, rec in enumerate(pool.map(_score_job, jobs, chunksize=2), 1):
            rec["rel"] = str(Path(rec["file"]).relative_to(source))
            recs.append(rec)
            if n % 50 == 0 or n == len(photos):
                log(f"  {n}/{len(photos)} scored")

    groups = group_similar(recs, cfg.dup_bits, cfg.dup_window_s)
    leads = []
    for gi, members in enumerate(groups):
        members.sort(key=lambda i: -recs[i].get("score", 0))
        for rank, i in enumerate(members):
            recs[i]["group"] = gi
            recs[i]["group_size"] = len(members)
            recs[i]["group_rank"] = rank
        leads.append(members[0])
    leads.sort(key=lambda i: -recs[i].get("score", 0))
    top = [i for i in leads if not recs[i].get("error")][: cfg.top]
    for r, i in enumerate(top, 1):
        recs[i]["pick_rank"] = r
    log(f"{len(groups)} distinct moment(s); suggesting the top {len(top)}.")

    if cfg.enhance and top:
        ew = max(1, min(2, workers))  # full-resolution edits are memory heavy
        log(f"Auto-enhancing {len(top)} pick(s) at full resolution ({ew} at a time)...")
        with ProcessPoolExecutor(max_workers=ew, mp_context=ctx) as pool:
            jobs = [(recs[i], out_dir, cfg) for i in top]
            for n, (i, res) in enumerate(zip(top, pool.map(_enhance_job, jobs)), 1):
                recs[i].update(res)
                if n % 5 == 0 or n == len(top):
                    log(f"  {n}/{len(top)} edited")

    summary = {
        "source": str(source.resolve()),
        "out_dir": str(out_dir.resolve()),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "photos": len(recs),
        "groups": len(groups),
        "top": len(top),
        "edited": sum(1 for r in recs if r.get("edited")),
        "errors": sum(1 for r in recs if r.get("error")),
        "seconds": round(time.time() - t0, 1),
        "config": asdict(cfg),
    }
    data = {"summary": summary, "photos": recs}
    (out_dir / "studio.json").write_text(json.dumps(data, indent=1, default=str))
    return data
