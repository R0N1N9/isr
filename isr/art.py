"""Art as a by-product: stock-worthy frames found while doing the real work.

Every SAR, Survey and Stakeout job looks through its own photos after it
finishes. A frame is kept when it is the best of its moment, scores well, and
has no fault an edit cannot fix (blur, blown highlights). Keepers get the
light automatic touch-up and go into one gallery shared by all jobs, which can
be exported to a folder ready to upload to a stock library.
"""

from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path

MIN_SCORE = 66.0  # out of 100; below this a frame is a record shot, not a picture
MAX_PER_JOB = 12
MAX_SCORED = {"stakeout": 40}  # a fixed camera takes the same picture all week: sample it...
MAX_KEPT = {"stakeout": 1}  # ...and keep one frame of that view at most
SAME_BITS = 34  # two frames whose 256-bit fingerprints differ by this little are the same picture
UNFIXABLE = ("Soft", "Overexposed", "Underexposed", "Clipped", "Crushed", "Noisy")


def alike(a: str | None, b: str | None) -> bool:
    """Do two fingerprints (hex strings from the scorer) show the same picture?"""
    if not a or not b:
        return False
    return bin(int(a, 16) ^ int(b, 16)).count("1") <= SAME_BITS


def harvest(photos: list[Path], source: Path, art_dir: Path, section: str, job: str, name: str, log=print) -> list[dict]:
    """Score ``photos``, edit the keepers into ``art_dir`` and write art.json. Returns the keepers."""
    from .studio.gallery import write_gallery
    from .studio.pipeline import StudioConfig, _enhance_job, run_studio

    cap = MAX_SCORED.get(section)
    if cap and len(photos) > cap:
        photos = photos[:: max(1, len(photos) // cap)][:cap]
    if not photos:
        return []
    cfg = StudioConfig(top=MAX_PER_JOB, enhance=False)
    data = run_studio(source, art_dir, cfg, log=lambda *a: None, photos=photos)
    keep = []
    for r in sorted(data["photos"], key=lambda r: r.get("pick_rank") or 1e9):
        if not r.get("pick_rank") or r.get("score", 0) < MIN_SCORE:
            continue
        if any(w.startswith(UNFIXABLE) for w in r.get("warnings") or []):
            continue
        if len(keep) >= MAX_KEPT.get(section, MAX_PER_JOB):
            break
        if any(alike(r.get("hash"), k.get("hash")) for k in keep):
            continue  # the same view again, taken later: the better one is already in
        r.update(_enhance_job((r, art_dir, cfg)))
        if r.get("edited"):
            keep.append(r)
    data["summary"]["edited"] = len(keep)
    (art_dir / "studio.json").write_text(json.dumps(data, indent=1, default=str))
    write_gallery(data, art_dir)
    items = [{"id": f"{art_dir.name}/{r['name']}", "name": r["name"], "original": r["file"], "edited": r["edited"],
              "preview": r.get("preview_edit") or r.get("thumb"), "thumb": r.get("thumb"), "score": round(r["score"]),
              "tags": r.get("tags") or [], "warnings": r.get("warnings") or [], "taken": r.get("taken_at"),
              "lat": r.get("lat"), "lon": r.get("lon"), "hash": r.get("hash")}
             for r in sorted(keep, key=lambda r: -r["score"])]
    (art_dir / "art.json").write_text(json.dumps({"section": section, "job": job, "name": name, "scored": len(photos),
                                                  "found": time.strftime("%Y-%m-%d %H:%M"), "items": items}, indent=1))
    log(f"Art: {len(items)} stock-worthy frame{'s' if len(items) != 1 else ''} out of {len(photos)} looked at")
    return items


def _all(art_root: Path) -> list[dict]:
    out = []
    for d in sorted((p for p in art_root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            if (d / "art.json").exists():
                a = json.loads((d / "art.json").read_text())
                items, label, section = a["items"], a.get("name") or d.name, a.get("section") or ""
            elif (d / "studio.json").exists():  # a selection made with the old Art section or `isr art -o`
                s = json.loads((d / "studio.json").read_text())
                items = [{"id": f"{d.name}/{r['name']}", "name": r["name"], "original": r["file"], "edited": r["edited"],
                          "preview": r.get("preview_edit") or r.get("thumb"), "thumb": r.get("thumb"),
                          "score": round(r.get("score", 0)), "tags": r.get("tags") or [], "warnings": r.get("warnings") or [],
                          "hash": r.get("hash")} for r in s["photos"] if r.get("edited")]
                label, section = d.name, "art"
            else:
                continue
        except (OSError, ValueError, KeyError):
            continue
        out += [{**it, "dir": d.name, "from": label, "section": section} for it in items if (d / it["edited"]).exists()]
    return out


def collection(art_root: Path) -> list[dict]:
    """Everything in the gallery, newest job first, minus what was removed, one copy of each picture."""
    gone = _removed(art_root)
    ids, hashes = {g["id"] for g in gone}, [g.get("hash") for g in gone]
    out = [it for it in _all(art_root)
           if it["id"] not in ids and not any(alike(it.get("hash"), h) for h in hashes)]
    # the same picture reaching the gallery from two jobs is shown once: the better-scoring copy
    best = []
    for it in sorted(out, key=lambda it: -it["score"]):
        if not any(alike(it.get("hash"), k.get("hash")) for k in best):
            best.append(it)
    keep = {id(it) for it in best}
    return [it for it in out if id(it) in keep]


def _removed(art_root: Path) -> list[dict]:
    try:
        return [g if isinstance(g, dict) else {"id": g} for g in json.loads((art_root / "removed.json").read_text())]
    except (OSError, ValueError):
        return []


def remove(art_root: Path, item_id: str) -> None:
    """Take a picture out of the gallery for good: it will not come back from another job's copy either."""
    gone = _removed(art_root)
    if not any(g["id"] == item_id for g in gone):
        it = next((i for i in _all(art_root) if i["id"] == item_id), None)
        gone.append({"id": item_id, "hash": it.get("hash") if it else None})
    (art_root / "removed.json").write_text(json.dumps(gone))


def export(art_root: Path, dest_root: Path) -> tuple[Path, int]:
    """Copy every edited picture in the gallery into a fresh folder, with a list of what each one is."""
    items = collection(art_root)
    dest = dest_root / time.strftime("%Y-%m-%d_%H%M%S")
    dest.mkdir(parents=True, exist_ok=True)
    used, rows = set(), []
    for it in items:
        src = art_root / it["dir"] / it["edited"]
        stem, n = Path(it["name"]).stem, 1
        name = f"{stem}.jpg"
        while name.lower() in used:
            n += 1
            name = f"{stem}_{n}.jpg"
        used.add(name.lower())
        shutil.copy2(src, dest / name)
        rows.append([name, it["score"], "; ".join(it["tags"]), it["from"], it.get("taken") or "",
                     it.get("lat") if it.get("lat") is not None else "", it.get("lon") if it.get("lon") is not None else ""])
    with open(dest / "pictures.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "score", "qualities", "from job", "taken", "lat", "lon"])
        w.writerows(rows)
    return dest, len(rows)
