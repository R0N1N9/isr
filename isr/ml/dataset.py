"""Turn triage reviews into training data, flight after flight.

Every Confirm / Dismiss in the triage report is a label:

* confirmed candidate  -> positive example (a box of its class)
* dismissed candidate  -> hard negative: a tile with that false alarm in it and
  no box, which teaches a model what is *not* a person in your terrain (rocks,
  stumps, troughs, litter)

Tiles are cut from the full-resolution photo at the size the detector will
see in the field (default 640 px, i.e. no downscaling), so small people stay
as sharp as they really are. Tiles are assigned to train/val by photo, so the
validation set never contains another crop of a training photo.

The dataset folder keeps a ``manifest.jsonl`` (one line per tile) and the tile
images; ``export`` materialises it in YOLO (Ultralytics) or COCO (RF-DETR,
D-FINE, most other trainers) layout.

Box quality: candidate boxes come from the detectors, so they can be tight
(only the jacket) or loose. ``pad`` grows them a little; for best results,
open the exported dataset in a labelling tool (CVAT, Label Studio, Roboflow)
and adjust boxes before a serious training run.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image

MANIFEST = "manifest.jsonl"


@dataclass
class Sample:
    file: str  # relative to the dataset folder
    split: str  # train | val
    width: int
    height: int
    boxes: list[dict] = field(default_factory=list)  # {"cls": str, "box": [x0, y0, x1, y1]} in tile pixels
    kind: str = "positive"  # positive | negative
    key: str = ""
    source: dict = field(default_factory=dict)


class Dataset:
    def __init__(self, folder: Path):
        self.dir = Path(folder)
        self.samples: list[Sample] = []
        m = self.dir / MANIFEST
        if m.exists():
            for line in m.read_text().splitlines():
                if line.strip():
                    self.samples.append(Sample(**json.loads(line)))
        self.keys = {s.key for s in self.samples}

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / (MANIFEST + ".tmp")
        tmp.write_text("".join(json.dumps(asdict(s)) + "\n" for s in self.samples))
        os.replace(tmp, self.dir / MANIFEST)

    @property
    def classes(self) -> list[str]:
        seen: dict[str, None] = {}
        for s in self.samples:
            for b in s.boxes:
                seen.setdefault(b["cls"], None)
        return list(seen)

    def stats(self) -> dict:
        out = {"tiles": len(self.samples), "train": 0, "val": 0, "positive_tiles": 0, "negative_tiles": 0,
               "boxes": {}, "runs": len({s.source.get("run") for s in self.samples})}
        for s in self.samples:
            out[s.split] += 1
            out["positive_tiles" if s.boxes else "negative_tiles"] += 1
            for b in s.boxes:
                out["boxes"][b["cls"]] = out["boxes"].get(b["cls"], 0) + 1
        return out


def _h(*parts) -> int:
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def load_labels(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    if "labels" not in d:
        raise ValueError(f"{path} is not a labels export from the triage report")
    return d


def add_from_triage(triage_dir: Path, labels_path: Path, dataset_dir: Path, tile: int = 640, pad: float = 0.15,
                    val_share: float = 0.15, force: bool = False, log=print) -> dict:
    triage_dir = Path(triage_dir)
    results = json.loads((triage_dir / "results.json").read_text())
    labels = load_labels(labels_path)
    run = results["summary"]["generated"] + " " + results["summary"]["source"]
    if labels.get("generated") and labels["generated"] != results["summary"]["generated"] and not force:
        raise ValueError("these labels were exported from a different triage run "
                         f"({labels.get('generated')} vs {results['summary']['generated']}); pass --force to use anyway")
    lab = labels["labels"]
    ds = Dataset(dataset_dir)
    (ds.dir / "tiles").mkdir(parents=True, exist_ok=True)
    added = {"positive": 0, "negative": 0, "skipped_existing": 0, "missing_photos": 0}

    for photo in results["photos"]:
        conf = [(d, lab[d["id"]].get("class") or "person") for d in photo["detections"]
                if lab.get(d["id"], {}).get("status") == "confirmed"]
        dism = [d for d in photo["detections"] if lab.get(d["id"], {}).get("status") == "dismissed"]
        if not conf and not dism:
            continue
        src = Path(photo["file"])
        if not src.exists():
            added["missing_photos"] += 1
            log(f"  ! {src} not found (card not plugged in?); skipping its labels")
            continue
        split = "val" if _h(photo["rel"], src.stat().st_size) % 1000 < val_share * 1000 else "train"
        with Image.open(src) as im:
            im = im.convert("RGB")
            W, H = im.size
            T = min(tile, W, H)
            centres: list[tuple[float, float]] = []

            def padded(d):
                x0, y0, x1, y1 = d["box"]
                w, h = x1 - x0, y1 - y0
                return [x0 - w * pad, y0 - h * pad, x1 + w * pad, y1 + h * pad]

            for d, kind in [(d, "positive") for d, _ in conf] + [(d, "negative") for d in dism]:
                x0, y0, x1, y1 = d["box"]
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                if any(abs(cx - a) < T / 3 and abs(cy - b) < T / 3 for a, b in centres):
                    continue  # an earlier tile already covers this spot
                centres.append((cx, cy))
                j = _h(d["id"])
                jx, jy = ((j % 1000) / 1000 - 0.5) * T / 3, (((j // 1000) % 1000) / 1000 - 0.5) * T / 3
                tx = int(min(max(cx - T / 2 + jx, 0), W - T))
                ty = int(min(max(cy - T / 2 + jy, 0), H - T))
                key = f"{run}|{photo['rel']}|{tx},{ty}"
                if key in ds.keys:
                    added["skipped_existing"] += 1
                    continue
                boxes = []
                for dc, cls in conf:
                    bx0, by0, bx1, by1 = padded(dc)
                    cx0, cy0 = max(bx0, tx), max(by0, ty)
                    cx1, cy1 = min(bx1, tx + T), min(by1, ty + T)
                    full = (bx1 - bx0) * (by1 - by0)
                    if cx1 > cx0 and cy1 > cy0 and (cx1 - cx0) * (cy1 - cy0) >= 0.4 * full:
                        boxes.append({"cls": cls, "box": [round(cx0 - tx, 1), round(cy0 - ty, 1),
                                                          round(cx1 - tx, 1), round(cy1 - ty, 1)]})
                name = f"tiles/{_h(key):012x}.jpg"
                im.crop((tx, ty, tx + T, ty + T)).save(ds.dir / name, quality=93)
                s = Sample(file=name, split=split, width=T, height=T, boxes=boxes,
                           kind="positive" if boxes else "negative", key=key,
                           source={"run": run, "photo": photo["rel"], "candidate": d["id"], "label": kind,
                                   "tile_xy": [tx, ty], "lat": d.get("lat"), "lon": d.get("lon"),
                                   "cand_box": [round(x0 - tx, 1), round(y0 - ty, 1), round(x1 - tx, 1),
                                                round(y1 - ty, 1)],
                                   "gsd_cm": photo.get("gsd_cm"), "photo_abs": str(src.resolve())})
                ds.samples.append(s)
                ds.keys.add(key)
                added["positive" if boxes else "negative"] += 1
    ds.save()
    return {"added": added, "dataset": ds.stats()}


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def export(dataset_dir: Path, out: Path, fmt: str = "yolo", classes: list[str] | None = None) -> Path:
    """Write the dataset in YOLO or COCO layout. Returns the config file to give the trainer."""
    ds = Dataset(dataset_dir)
    if not ds.samples:
        raise ValueError(f"{dataset_dir} has no samples yet")
    names = classes or ds.classes or ["person"]
    idx = {n: i for i, n in enumerate(names)}
    out = Path(out)
    if fmt == "yolo":
        for s in ds.samples:
            stem = Path(s.file).stem
            _link(ds.dir / s.file, out / "images" / s.split / f"{stem}.jpg")
            lines = []
            for b in s.boxes:
                if b["cls"] not in idx:
                    continue
                x0, y0, x1, y1 = b["box"]
                lines.append(f"{idx[b['cls']]} {(x0 + x1) / 2 / s.width:.6f} {(y0 + y1) / 2 / s.height:.6f} "
                             f"{(x1 - x0) / s.width:.6f} {(y1 - y0) / s.height:.6f}")
            p = out / "labels" / s.split / f"{stem}.txt"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("\n".join(lines) + ("\n" if lines else ""))
        cfg = out / "data.yaml"
        cfg.write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
                       f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
        return cfg
    if fmt == "coco":
        cats = [{"id": i + 1, "name": n, "supercategory": "object"} for i, n in enumerate(names)]
        for split, folder in (("train", "train"), ("val", "valid"), ("val", "test")):
            images, anns = [], []
            for s in (x for x in ds.samples if x.split == split):
                iid = len(images) + 1
                fn = Path(s.file).name
                _link(ds.dir / s.file, out / folder / fn)
                images.append({"id": iid, "file_name": fn, "width": s.width, "height": s.height})
                for b in s.boxes:
                    if b["cls"] not in idx:
                        continue
                    x0, y0, x1, y1 = b["box"]
                    anns.append({"id": len(anns) + 1, "image_id": iid, "category_id": idx[b["cls"]] + 1,
                                 "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0), "iscrowd": 0})
            (out / folder).mkdir(parents=True, exist_ok=True)
            (out / folder / "_annotations.coco.json").write_text(
                json.dumps({"images": images, "annotations": anns, "categories": cats}))
        return out
    raise ValueError("format must be 'yolo' or 'coco'")
