"""Scorecard: measure detectors (and verifiers) on labelled tiles before trusting them.

For a search the questions are:

* recall    - of the real people in the test tiles, how many does it flag?
* false alarms per photo - how much noise does a reviewer wade through,
  normalised to a 20 MP Mavic 3 photo so tiles of any size compare fairly?

Every system is run over the validation tiles of a dataset (your reviews,
imported public data, or both), predictions are matched to ground-truth
boxes, and the full trade-off curve is computed by sweeping the score
threshold. Matching is lenient on purpose: small boxes vary a lot, so a
prediction counts if IoU >= 0.25 or either box's centre lies in the other.
A second prediction on an already-found person is ignored rather than
counted as a false alarm (a reviewer loses nothing from it).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from ..triage.detectors import (ColorTargetDetector, Detection, RXAnomalyDetector, SizeFilter, WorkImage,
                                merge_detections)
from .dataset import Dataset

PHOTO_MP = 20.9  # Mavic 3 photo, megapixels
FA_LEVELS = (0.5, 1, 2, 5, 10)


@dataclass
class SystemSpec:
    name: str
    classic: bool = False
    colors: list[str] = field(default_factory=list)
    model: str | None = None
    verifier: str | None = None
    conf: float = 0.05  # low: the sweep chooses thresholds
    default_threshold: float = 0.35


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def _inside(p, box) -> bool:
    return box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]


def matches(pred, gt) -> bool:
    if _iou(pred, gt) >= 0.25:
        return True
    pc = ((pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2)
    gc = ((gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2)
    return _inside(pc, gt) or _inside(gc, pred)


class _Runner:
    def __init__(self, spec: SystemSpec):
        self.spec = spec
        self.model = None
        self.verifier = None
        if spec.model:
            from .onnx_detector import OnnxDetector
            self.model = OnnxDetector(spec.model, conf=spec.conf)
        if spec.verifier:
            from .verifier import Verifier
            self.verifier = Verifier(spec.verifier)

        self._photo_cache: dict[tuple, list[Detection]] = {}
        self.classic_on_tiles = 0

    def _classic(self, path: Path, gsd_m: float | None, work_px: int) -> list[Detection]:
        w = WorkImage.load(path, work_px, gsd_m)
        sizes = SizeFilter()
        dets = RXAnomalyDetector().detect(w, sizes)
        if self.spec.colors:
            dets += ColorTargetDetector(self.spec.colors).detect(w, sizes)
        return dets

    def _classic_from_photo(self, source: dict, gsd_m: float | None, T: int) -> list[Detection] | None:
        """Colour detectors run on the whole original photo (as in triage), mapped into the tile.

        Their thresholds come from the statistics of the whole frame, so running
        them on a small tile (where one person can be the rarest 0.5 % of pixels)
        would underrate them.
        """
        photo, scale = source.get("photo_abs"), float(source.get("scale") or 1.0)
        if not photo or not Path(photo).exists():
            return None
        key = (photo, scale)
        if key not in self._photo_cache:
            with Image.open(photo) as im:
                full = max(im.size)
            gsd_photo = gsd_m * scale if gsd_m else None  # tile gsd is after resampling by `scale`
            raw = self._classic(Path(photo), gsd_photo, min(2000, full))
            self._photo_cache[key] = [Detection(d.x0 * scale, d.y0 * scale, d.x1 * scale, d.y1 * scale, d.score,
                                                d.kinds, d.label, d.area_m2, d.extra) for d in raw]
        tx, ty = source.get("tile_xy", [0, 0])
        out = []
        for d in self._photo_cache[key]:
            cx, cy = (d.x0 + d.x1) / 2, (d.y0 + d.y1) / 2
            if tx <= cx < tx + T and ty <= cy < ty + T:
                out.append(Detection(d.x0 - tx, d.y0 - ty, d.x1 - tx, d.y1 - ty, d.score, list(d.kinds), d.label,
                                     d.area_m2, dict(d.extra)))
        return out

    def __call__(self, path: Path, gsd_m: float | None, source: dict | None = None) -> list[Detection]:
        dets: list[Detection] = []
        with Image.open(path) as im:
            im = im.convert("RGB")
            if self.spec.classic:
                got = self._classic_from_photo(source or {}, gsd_m, max(im.size))
                if got is None:
                    self.classic_on_tiles += 1
                    got = self._classic(path, gsd_m, max(im.size))
                dets += got
            if self.model is not None:
                self.model.tile = min(self.model.tile, max(im.size))
                for x0, y0, x1, y1, s, name in self.model.detect_image(im):
                    dets.append(Detection(x0, y0, x1, y1, s, [f"ai:{name}"], name))
            dets = merge_detections(dets, near_px=(1.0 / gsd_m) if gsd_m else 0.0)
            if self.verifier is not None and dets:
                from .verifier import apply_verification
                ps = self.verifier.score_boxes(im, [(d.x0, d.y0, d.x1, d.y1) for d in dets])
                for d, p in zip(dets, ps):
                    apply_verification(d, float(p), self.verifier.label)
        return dets


def evaluate(dataset_dir: Path, specs: list[SystemSpec], split: str = "val", classes: list[str] | None = None,
             limit: int | None = None, log=print) -> dict:
    ds = Dataset(dataset_dir)
    tiles = [s for s in ds.samples if s.split == split] or ([s for s in ds.samples] if split == "any" else [])
    if not tiles:
        raise ValueError(f"no '{split}' tiles in {dataset_dir} (use --split train or add more data)")
    if limit:
        tiles = tiles[:limit]
    want = set(classes) if classes else None
    n_gt = sum(1 for s in tiles for b in s.boxes if want is None or b["cls"] in want)
    mp = sum(s.width * s.height for s in tiles) / 1e6
    small = [b for s in tiles for b in s.boxes if (want is None or b["cls"] in want)]
    log(f"Evaluating on {len(tiles)} {split} tile(s): {n_gt} target(s), {mp:.1f} MP "
        f"(= {mp / PHOTO_MP:.2f} Mavic 3 photos)")
    results = []
    for spec in specs:
        run = _Runner(spec)
        t0 = time.time()
        scored: list[tuple[float, int]] = []  # (score, 1=TP / 0=FP)
        per_gt_best: list[float] = []  # best score that found each GT (0 if never)
        sizes_px: list[float] = []
        for s in tiles:
            gsd = s.source.get("gsd_cm")
            dets = run(ds.dir / s.file, gsd / 100 if gsd else None, s.source)
            gts = [b["box"] for b in s.boxes if want is None or b["cls"] in want]
            found = [0.0] * len(gts)
            for d in sorted(dets, key=lambda d: -d.score):
                box = (d.x0, d.y0, d.x1, d.y1)
                hit = next((k for k, g in enumerate(gts) if found[k] == 0 and matches(box, g)), None)
                if hit is not None:
                    found[hit] = d.score
                    scored.append((d.score, 1))
                elif any(matches(box, g) for g in gts):
                    continue  # duplicate on an already-found target
                else:
                    scored.append((d.score, 0))
            per_gt_best += found
            sizes_px += [((g[2] - g[0]) * (g[3] - g[1])) ** 0.5 for g in gts]
        secs = time.time() - t0
        if run.classic_on_tiles:
            log(f"  ! {spec.name}: {run.classic_on_tiles} tile(s) had no original photo, so the colour detectors ran "
                "on the tile alone (underrates them)")
        results.append(_summarise(spec, scored, per_gt_best, sizes_px, n_gt, mp, secs))
        r = results[-1]
        log(f"  {spec.name}: AP {r['ap']:.3f}, recall {r['at_default']['recall']:.0%} with "
            f"{r['at_default']['fa_per_photo']:.1f} false alarms/photo at {spec.default_threshold}")
    return {"dataset": str(dataset_dir), "split": split, "tiles": len(tiles), "targets": n_gt,
            "megapixels": round(mp, 2), "classes": classes, "systems": results,
            "generated": time.strftime("%Y-%m-%d %H:%M:%S")}


def _summarise(spec, scored, per_gt_best, sizes_px, n_gt, mp, secs) -> dict:
    photos = mp / PHOTO_MP
    sc = sorted(scored, key=lambda t: -t[0])
    tp = np.cumsum([t for _, t in sc]) if sc else np.zeros(0)
    fp = np.cumsum([1 - t for _, t in sc]) if sc else np.zeros(0)
    thr = np.array([s for s, _ in sc])
    recall = tp / max(n_gt, 1)
    precision = tp / np.maximum(tp + fp, 1)
    fa = fp / max(photos, 1e-9)
    # all-point interpolated average precision
    ap = 0.0
    if len(sc):
        r = np.concatenate([[0], recall, [recall[-1]]])
        p = np.concatenate([[1], precision, [0]])
        p = np.maximum.accumulate(p[::-1])[::-1]
        ap = float(np.sum((r[1:] - r[:-1]) * p[1:]))

    def at_threshold(t):
        m = thr >= t
        k = int(m.sum())
        if k == 0:
            return {"threshold": t, "recall": 0.0, "precision": 1.0, "fa_per_photo": 0.0}
        return {"threshold": t, "recall": round(float(recall[k - 1]), 4),
                "precision": round(float(precision[k - 1]), 4), "fa_per_photo": round(float(fa[k - 1]), 3)}

    def recall_at_fa(level):
        ok = fa <= level
        return round(float(recall[ok].max()), 4) if ok.any() else 0.0

    found = np.array(per_gt_best) >= spec.default_threshold
    buckets = {}
    for lo, hi, name in ((0, 16, "<16 px"), (16, 32, "16-32 px"), (32, 1e9, ">32 px")):
        idx = [i for i, z in enumerate(sizes_px) if lo <= z < hi]
        if idx:
            buckets[name] = {"targets": len(idx), "recall": round(float(found[idx].mean()), 4)}
    # curve, thinned to <= 300 points for the page
    step = max(1, len(sc) // 300)
    curve = [{"t": round(float(thr[i]), 4), "recall": round(float(recall[i]), 4),
              "precision": round(float(precision[i]), 4), "fa": round(float(fa[i]), 4)}
             for i in list(range(0, len(sc), step)) + ([len(sc) - 1] if len(sc) else [])]
    return {"name": spec.name, "spec": asdict(spec), "ap": round(ap, 4),
            "recall_at_fa": {str(level): recall_at_fa(level) for level in FA_LEVELS},
            "at_default": at_threshold(spec.default_threshold),
            "max_recall": round(float(recall[-1]), 4) if len(sc) else 0.0,
            "by_size": buckets, "seconds_per_photo": round(secs / max(photos, 1e-9), 2),
            "predictions": len(sc), "curve": curve}


def print_table(res: dict, log=print) -> None:
    hdr = f"{'system':28} {'AP':>6} {'recall@1FA':>10} {'recall@5FA':>10} {'recall':>7} {'FA/photo':>9} {'s/photo':>8}"
    log(hdr)
    log("-" * len(hdr))
    for r in res["systems"]:
        d = r["at_default"]
        log(f"{r['name'][:28]:28} {r['ap']:6.3f} {r['recall_at_fa']['1']:10.0%} {r['recall_at_fa']['5']:10.0%} "
            f"{d['recall']:7.0%} {d['fa_per_photo']:9.1f} {r['seconds_per_photo']:8.1f}")
    log("recall@NFA = best recall with at most N false alarms per 20 MP photo; recall / FA/photo at each "
        "system's default threshold")


def save(res: dict, out_dir: Path) -> Path:
    from .scorecard_report import write_scorecard
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scorecard.json").write_text(json.dumps(res, indent=1))
    return write_scorecard(res, out_dir)
