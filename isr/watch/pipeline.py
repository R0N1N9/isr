"""Read a stakeout: photos from one fixed camera over a day or a week.

The camera can be motion-triggered (a trail camera) or shooting at a fixed
interval. Each photo is compared with the same view at nearby times, which
separates real movement from an empty frame that the camera's own trigger
fired on, and shows *where* in the view things move. Movement close together
in time is one event. Events are then counted by hour and day of the week,
which is what staffing decisions need: when is this place busy, when is it
dead, and what moved when nothing should have.

Only timestamps and pixels are used. Nobody is identified or recognised.
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

EXTS = (".jpg", ".jpeg")
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
W, H = 160, 120  # working size for comparing frames
THRESH = 1.4  # how far, in units of the scene's typical contrast, a patch must differ to count as movement
EVENT_GAP_S = 60.0  # movement less than this far apart is the same event
MAX_PHOTOS = 20000


def taken_at(path: Path) -> datetime:
    """When the photo was taken: the camera's own timestamp, else the file's."""
    try:
        with Image.open(path) as im:
            ex = im.getexif()
            for v in (ex.get_ifd(0x8769).get(0x9003), ex.get(0x0132)):
                if v:
                    return datetime.strptime(str(v).strip("\x00 ")[:19], "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return datetime.fromtimestamp(path.stat().st_mtime)


def _load(path: Path):
    """Small greyscale view with lighting evened out, plus (is it a night / infra-red frame?)."""
    with Image.open(path) as im:
        im.draft("RGB", (W * 2, H * 2))
        rgb = np.asarray(im.convert("RGB").resize((W, H), Image.BILINEAR), np.float32) / 255
    g = rgb.mean(2)
    sat = float((rgb.max(2) - rgb.min(2)).mean())
    night = sat < 0.035 or float(g.mean()) < 0.12
    flat = g - ndimage.uniform_filter(g, 31)  # drop slow changes of light across the frame
    # scale by the typical contrast of the scene, not the spread: one bright figure must not rescale the frame
    return (flat / (1.4826 * float(np.median(np.abs(flat - np.median(flat)))) + 1e-3)).astype(np.float32), night


def _windows(mask: np.ndarray) -> list[tuple[int, int]]:
    """Runs of True in a 24-long ring of hours -> [(start_hour, end_hour_exclusive)], wrapping midnight."""
    if mask.all():
        return [(0, 24)]
    out, n = [], len(mask)
    start = next(i for i in range(n) if mask[i] and not mask[i - 1]) if mask.any() else None
    if start is None:
        return []
    i = start
    for _ in range(n):
        if mask[i % n] and not mask[(i - 1) % n]:
            s = i % n
        if mask[i % n] and not mask[(i + 1) % n]:
            out.append((s, (i % n) + 1))
        i += 1
    return out


def hours(a: int, b: int) -> str:
    return f"{a % 24:02d}:00–{b % 24:02d}:00"


def run_watch(src: Path, out: Path, name: str = "", watching: str = "", closed_from: int = 20, closed_to: int = 6,
              log=print) -> dict:
    """Analyse the folder and write activity.json, scene.jpg, heat.png and frames/ into ``out``."""
    t0 = time.time()
    src, out = Path(src), Path(out)
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in EXTS and not p.name.startswith(".")
                   and out.resolve() not in p.resolve().parents)
    if not files:
        raise SystemExit(f"No JPEG photos found under {src}")
    if len(files) > MAX_PHOTOS:
        log(f"  {len(files)} photos: using every {math.ceil(len(files) / MAX_PHOTOS)}th")
        files = files[:: math.ceil(len(files) / MAX_PHOTOS)]
    (out / "frames").mkdir(parents=True, exist_ok=True)
    log(f"Reading {len(files)} photos from {src}...")
    recs = []
    for n, p in enumerate(files, 1):
        try:
            g, night = _load(p)
            recs.append((taken_at(p), p, g, night))
        except Exception as e:
            log(f"  skipped {p.name}: {e}")
        if n % 250 == 0:
            log(f"  {n}/{len(files)}")
    if not recs:
        raise SystemExit("None of the photos could be read")
    recs.sort(key=lambda r: r[0])
    times = [r[0] for r in recs]
    G = np.stack([r[2] for r in recs])
    night = np.array([r[3] for r in recs])
    ts = np.array([t.timestamp() for t in times])
    n = len(recs)

    # -- movement in each frame: compare with the same view at the nearest other times --------------------
    log("Looking for movement...")
    heat = np.zeros((H, W), np.float32)
    moved = np.zeros(n, bool)
    frac = np.zeros(n, np.float32)
    boxes = [None] * n
    order = np.arange(n)
    for i in range(n):
        same = order[night == night[i]]
        same = same[same != i]
        if len(same) < 3:
            continue
        near = same[np.argsort(np.abs(ts[same] - ts[i]))[:9]]
        bg = np.median(G[near], axis=0)
        gain = float((G[i] * bg).sum() / ((bg * bg).sum() + 1e-6))  # same scene, slightly different contrast
        diff = np.abs(G[i] - gain * bg)
        m = ndimage.binary_opening(ndimage.uniform_filter(diff, 3) > THRESH, iterations=1)
        lab, k = ndimage.label(m)
        if not k:
            continue
        sizes = ndimage.sum(m, lab, index=np.arange(1, k + 1))
        keep = np.isin(lab, np.nonzero(sizes >= 0.004 * W * H)[0] + 1)
        f = float(keep.mean())
        if f == 0 or f > 0.45:  # nothing, or the whole view changed (light switched, camera knocked)
            continue
        moved[i], frac[i] = True, f
        heat += keep
        big = lab == int(np.argmax(sizes)) + 1
        jj, ii = np.nonzero(big)
        boxes[i] = (ii.min() / W, jj.min() / H, (ii.max() + 1) / W, (jj.max() + 1) / H)

    # -- events ---------------------------------------------------------------------------------------
    events, cur = [], None
    for i in np.nonzero(moved)[0]:
        if cur and ts[i] - cur["end_ts"] <= EVENT_GAP_S:
            cur["end_ts"] = ts[i]
            cur["frames"] += 1
            if frac[i] > cur["size"]:
                cur["size"], cur["best"] = float(frac[i]), int(i)
        else:
            cur = {"start_ts": ts[i], "end_ts": ts[i], "frames": 1, "size": float(frac[i]), "best": int(i)}
            events.append(cur)
    closed = np.zeros(24, bool)
    h = closed_from % 24
    while h != closed_to % 24:
        closed[h] = True
        h = (h + 1) % 24
    for e in events:
        t = times[e["best"]]
        e.update(time=t.isoformat(sep=" ", timespec="minutes"), weekday=t.weekday(), hour=t.hour,
                 night=bool(night[e["best"]]), out_of_hours=bool(closed[t.hour]),
                 seconds=round(e["end_ts"] - e["start_ts"]))

    # -- when was the camera actually watching? ----------------------------------------------------------
    gaps = np.diff(ts)
    regular = n >= 20 and float(np.percentile(gaps, 75)) <= 1.5 * float(np.median(gaps)) + 1 and float(np.median(gaps)) > 0
    mode = "interval" if regular else "triggered"
    cover = np.zeros((7, 24), np.float32)
    if regular:  # an hour counts if there was a frame in it
        for key in {(t.date(), t.hour) for t in times}:
            cover[key[0].weekday(), key[1]] += 1
    else:  # a triggered camera is assumed to have been armed from its first photo to its last
        t = times[0].replace(minute=0, second=0, microsecond=0)
        while t <= times[-1]:
            cover[t.weekday(), t.hour] += 1
            t += timedelta(hours=1)
    counts = np.zeros((7, 24), np.float32)
    for e in events:
        counts[e["weekday"], e["hour"]] += 1
    rate = np.divide(counts, cover, out=np.zeros_like(counts), where=cover > 0)  # events per hour watched
    by_hour = np.divide(counts.sum(0), cover.sum(0), out=np.zeros(24, np.float32), where=cover.sum(0) > 0)
    by_day = np.divide(counts.sum(1), cover.sum(1) / 24, out=np.zeros(7, np.float32), where=cover.sum(1) > 0)
    days = max(1e-9, float(cover.sum()) / 24)

    # -- pictures --------------------------------------------------------------------------------------
    log("Writing pictures...")
    day_idx = np.nonzero(~night & ~moved)[0]
    ref = recs[int(day_idx[len(day_idx) // 2])][1] if len(day_idx) else recs[n // 2][1]
    with Image.open(ref) as im:
        scene = im.convert("RGB")
        scene.thumbnail((1280, 1280))
        scene.save(out / "scene.jpg", quality=85)
    if heat.max() > 0:
        hm = ndimage.gaussian_filter(heat, 2.5)
        hm = np.clip(hm / hm.max(), 0, 1)
        rgba = np.zeros((H, W, 4), np.uint8)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = 255, (190 * (1 - hm)).astype(np.uint8), 40
        rgba[..., 3] = (np.clip(hm * 1.6, 0, 1) * 185).astype(np.uint8)
        Image.fromarray(rgba, "RGBA").resize(scene.size, Image.BICUBIC).save(out / "heat.png")
    shown = sorted([e for e in events if e["out_of_hours"]], key=lambda e: -e["size"])[:24]
    shown += sorted([e for e in events if not e["out_of_hours"]], key=lambda e: -e["size"])[:12]
    for k, e in enumerate(sorted(shown, key=lambda e: e["start_ts"]), 1):
        i = e["best"]
        try:
            with Image.open(recs[i][1]) as im:
                im = im.convert("RGB")
                im.thumbnail((900, 900))
                if boxes[i]:
                    x0, y0, x1, y1 = boxes[i]
                    ImageDraw.Draw(im).rectangle((x0 * im.width - 6, y0 * im.height - 6, x1 * im.width + 6, y1 * im.height + 6),
                                                 outline=(255, 60, 200), width=3)
                e["frame"] = f"frames/e{k:04d}.jpg"
                im.save(out / e["frame"], quality=82)
        except Exception:
            pass

    data = {
        "name": name or src.name, "watching": watching.strip(), "source": str(src), "mode": mode,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 1),
        "closed": [closed_from % 24, closed_to % 24],
        "summary": {"photos": n, "with_movement": int(moved.sum()), "events": len(events),
                    "first": times[0].isoformat(sep=" ", timespec="minutes"),
                    "last": times[-1].isoformat(sep=" ", timespec="minutes"), "days": round(days, 1),
                    "per_day": round(len(events) / days, 1),
                    "out_of_hours": sum(e["out_of_hours"] for e in events),
                    "night_frames": int(night.sum())},
        "counts": counts.astype(int).tolist(), "cover": cover.astype(int).tolist(),
        "rate": [[round(float(v), 2) for v in row] for row in rate],
        "by_hour": [round(float(v), 2) for v in by_hour], "by_day": [round(float(v), 1) for v in by_day],
        "events": [{k: e[k] for k in ("time", "weekday", "hour", "night", "out_of_hours", "frames", "seconds", "size",
                                      "frame") if k in e} for e in events],
        "has_heat": bool(heat.max() > 0),
    }
    data["advice"] = advise(data)
    (out / "activity.json").write_text(json.dumps(data, indent=1))
    log(f"{len(events)} events over {days:.1f} days ({data['summary']['out_of_hours']} out of hours)")
    return data


def advise(d: dict) -> list[dict]:
    """Plain staffing conclusions from the counts. Each is {kind, text}."""
    s = d["summary"]
    what = d["watching"] or "this view"
    rate, cover = np.array(d["rate"], float), np.array(d["cover"], float)
    by_hour = np.array(d["by_hour"], float)
    out = []
    add = lambda kind, text: out.append({"kind": kind, "text": text})  # noqa: E731
    if s["events"] == 0:
        add("info", f"No movement was found in {s['photos']} photos. Either {what} is very quiet or the camera did not "
                    "have a clear view; check a few frames by eye before relying on this.")
        return out
    c0, c1 = d["closed"]
    peak = by_hour.max()
    busy = _windows(by_hour >= max(0.6 * peak, 1e-9))
    wk = float(np.nansum(rate[:5]) / max(1, (cover[:5] > 0).sum()))
    we = float(np.nansum(rate[5:]) / max(1, (cover[5:] > 0).sum())) if (cover[5:] > 0).any() else None
    if busy and peak > 0:
        spans = " and ".join(hours(a, b) for a, b in busy[:3])
        add("peak", f"Busiest {spans}, at about {peak:.1f} movements an hour. If {what} is an entrance, that is when a "
                    f"guard posted there earns the most: arrivals are met rather than just recorded.")
    if we is not None and (cover[:5] > 0).any():
        if we < 0.4 * wk:
            add("pattern", f"Weekends are quiet: about {we / max(wk, 1e-9):.0%} of the weekday rate. Weekend cover can "
                           "be patrol-led, and any weekend movement deserves a second look.")
        elif we > 1.6 * wk:
            add("pattern", f"Weekends are busier than weekdays (about {we / max(wk, 1e-9):.1f} times). Staff the post "
                           "for the weekend, not the working week.")
    ooh = [e for e in d["events"] if e["out_of_hours"]]
    if ooh:
        hist = np.bincount([e["hour"] for e in ooh], minlength=24)
        top = int(np.argmax(hist))
        nights = len({e["time"][:10] for e in ooh})
        dead = [e for e in ooh if 0 <= e["hour"] < 5]
        add("out_of_hours", f"{len(ooh)} movement{'s' if len(ooh) != 1 else ''} between {c0:02d}:00 and {c1:02d}:00 on "
                            f"{nights} day{'s' if nights != 1 else ''}, most often around {top:02d}:00. Find out who it "
                            f"is (late staff, cleaners, deliveries) before treating the hours as closed.")
        if dead:
            add("out_of_hours", f"{len(dead)} of those {'were' if len(dead) != 1 else 'was'} between midnight and 05:00 "
                                f"({', '.join(e['time'][5:] for e in dead[:4])}{'…' if len(dead) > 4 else ''}). Time "
                                f"patrol passes of {what} for then, or put this view on an alarm.")
    else:
        add("out_of_hours", f"Nothing moved between {c0:02d}:00 and {c1:02d}:00 in the period watched.")
    quiet = [w for w in _windows((by_hour == 0) & (cover.sum(0) > 0)) if w[1] - w[0] >= 3 or (w[1] - w[0]) % 24 >= 3]
    if quiet:
        a, b = max(quiet, key=lambda w: (w[1] - w[0]) % 24 or 24)
        add("quiet", f"No movement at all {hours(a, b)} on any day. Those are also the hours when nobody is around to "
                     f"notice an intruder, so keep patrols going through them and vary their timing.")
    if s["days"] < 3:
        add("info", f"Only {s['days']:g} day{'s' if s['days'] != 1 else ''} watched: enough to see the daily rhythm, "
                    "not enough to compare days of the week. A full week is better.")
    if d["mode"] == "triggered":
        add("info", "The camera fired on movement, so the hours it stood armed are assumed to run from its first photo "
                    "to its last. If it was switched off in between, quiet hours are overstated.")
    return out


def compact(d: dict) -> dict:
    """The part of a stakeout a Survey keeps when one is attached to an access point."""
    return {k: d[k] for k in ("name", "watching", "mode", "closed", "summary", "counts", "cover", "rate", "by_hour",
                              "by_day", "advice")} | {
        "out_of_hours": [{"time": e["time"], "hour": e["hour"]} for e in d["events"] if e["out_of_hours"]][:60]}
