"""Import public aerial datasets (HERIDAL, SARD, WiSARD, VisDrone, ...) into a dataset folder.

Annotation formats understood (auto-detected):

voc   one XML per image (<object><name>, <bndbox>) - HERIDAL and many SAR sets
yolo  images/ + labels/ with "class cx cy w h" lines, optional data.yaml names
coco  a JSON with images / annotations / categories (Roboflow exports, etc.)

Images are cut into full-resolution tiles the same way ``isr dataset add``
does for your own flights, so public data and your reviews train together:
tiles centred on objects (positives) plus background tiles with no objects
(negatives). ``scale`` resamples images first, to match the dataset's ground
resolution to yours (e.g. 0.5 if their photos were taken from half your
altitude). Check each dataset's licence before commercial use.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from .dataset import Dataset, Sample, _h

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _split_from_path(p: Path) -> str | None:
    parts = {x.lower() for x in p.parts}
    if parts & {"val", "valid", "validation", "test", "testing"}:
        return "val"
    if parts & {"train", "training"}:
        return "train"
    return None


def detect_format(src: Path) -> str:
    if any(src.rglob("*.xml")):
        return "voc"
    if any(src.rglob("*.json")) and any("annotations" in json.loads(f.read_text())
                                        for f in src.rglob("*.json") if f.stat().st_size < 500_000_000):
        return "coco"
    if any(src.rglob("*.txt")):
        return "yolo"
    raise ValueError(f"can't tell the annotation format of {src}; pass --format voc|yolo|coco")


def read_voc(src: Path):
    for xml in sorted(src.rglob("*.xml")):
        root = ET.parse(xml).getroot()
        fn = root.findtext("filename") or ""
        cands = [xml.with_name(fn), xml.with_suffix(".jpg"), xml.with_suffix(".JPG"), xml.with_suffix(".png")]
        cands += [p for p in xml.parent.parent.rglob(fn)] if fn else []
        img = next((c for c in cands if c.exists() and c.suffix.lower() in IMG_EXTS), None)
        if img is None:
            continue
        boxes = []
        for o in root.iter("object"):
            b = o.find("bndbox")
            if b is None:
                continue
            boxes.append((o.findtext("name", "object").strip(),
                          [float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]))
        yield img, boxes, _split_from_path(xml)


def read_yolo(src: Path):
    names: list[str] = []
    for y in list(src.rglob("data.yaml")) + list(src.rglob("*.names")) + list(src.rglob("classes.txt")):
        txt = y.read_text()
        if y.suffix == ".yaml":
            import re
            m = re.search(r"names:\s*\[(.*?)\]", txt, re.S)
            if m:
                names = [n.strip().strip("'\"") for n in m.group(1).split(",")]
            else:
                names = [ln.split(":", 1)[1].strip().strip("'\"") for ln in txt.splitlines()
                         if ln.strip() and ln.strip()[0].isdigit() and ":" in ln]
        else:
            names = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        if names:
            break
    for img in sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS):
        lab = img.with_suffix(".txt")
        parts = list(img.parts)
        if "images" in parts:  # images/<split>/x.jpg -> labels/<split>/x.txt
            parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
            lab = Path(*parts).with_suffix(".txt")
        if not lab.exists():
            lab = img.with_suffix(".txt")
        boxes = []
        if lab.exists():
            with Image.open(img) as im:
                W, H = im.size
            for ln in lab.read_text().splitlines():
                v = ln.split()
                if len(v) < 5:
                    continue
                c, cx, cy, w, h = int(float(v[0])), *map(float, v[1:5])
                name = names[c] if c < len(names) else f"class_{c}"
                boxes.append((name, [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]))
        yield img, boxes, _split_from_path(img)


def read_coco(src: Path):
    for jf in sorted(src.rglob("*.json")):
        d = json.loads(jf.read_text())
        if "annotations" not in d or "images" not in d:
            continue
        cats = {c["id"]: c["name"] for c in d.get("categories", [])}
        by_img: dict[int, list] = {}
        for a in d["annotations"]:
            x, y, w, h = a["bbox"]
            by_img.setdefault(a["image_id"], []).append((cats.get(a["category_id"], str(a["category_id"])),
                                                          [x, y, x + w, y + h]))
        for im in d["images"]:
            p = jf.parent / im["file_name"]
            if not p.exists():
                hits = list(src.rglob(Path(im["file_name"]).name))
                if not hits:
                    continue
                p = hits[0]
            yield p, by_img.get(im["id"], []), _split_from_path(jf)


READERS = {"voc": read_voc, "yolo": read_yolo, "coco": read_coco}


def import_dataset(src: Path, dataset_dir: Path, fmt: str | None = None, name: str | None = None,
                   class_map: dict[str, str] | None = None, keep_unmapped: bool = False, tile: int = 640,
                   scale: float = 1.0, negatives_per_image: int = 1, val_share: float = 0.15,
                   limit: int | None = None, split_override: str | None = None, gsd_cm: float | None = None,
                   log=print) -> dict:
    """Tile an annotated dataset into ``dataset_dir``.

    ``class_map`` renames source classes (e.g. {"human": "person"}); classes
    not in the map are dropped unless ``keep_unmapped``.
    """
    src = Path(src)
    fmt = fmt or detect_format(src)
    name = name or src.name
    ds = Dataset(dataset_dir)
    (ds.dir / "tiles").mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "positive": 0, "negative": 0, "boxes": 0, "skipped_existing": 0, "dropped_classes": {}}
    run = f"import:{name}"
    for n, (img, boxes, split) in enumerate(READERS[fmt](src)):
        if limit and n >= limit:
            break
        mapped = []
        for cls, b in boxes:
            new = (class_map or {}).get(cls, cls if (keep_unmapped or not class_map) else None)
            if new is None:
                stats["dropped_classes"][cls] = stats["dropped_classes"].get(cls, 0) + 1
                continue
            mapped.append((new, [v * scale for v in b]))
        split = split_override or split or ("val" if _h(name, img.name) % 1000 < val_share * 1000 else "train")
        with Image.open(img) as im:
            im = im.convert("RGB")
            if scale != 1.0:
                im = im.resize((max(1, round(im.size[0] * scale)), max(1, round(im.size[1] * scale))), Image.LANCZOS)
            W, H = im.size
            T = min(tile, W, H)
            stats["images"] += 1
            spots = []
            for _, (x0, y0, x1, y1) in mapped:
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                if any(abs(cx - a) < T / 3 and abs(cy - b) < T / 3 for a, b, _ in spots):
                    continue
                spots.append((cx, cy, "positive"))
            for k in range(negatives_per_image):  # background tiles away from every object
                for attempt in range(12):
                    j = _h(name, img.name, "neg", k, attempt)
                    tx, ty = j % max(1, W - T + 1), (j // 7919) % max(1, H - T + 1)
                    if not any(x1 > tx and x0 < tx + T and y1 > ty and y0 < ty + T for _, (x0, y0, x1, y1) in mapped):
                        spots.append((tx + T / 2, ty + T / 2, "negative"))
                        break
            for cx, cy, kind in spots:
                j = _h(name, img.name, round(cx), round(cy))
                jx, jy = (((j % 1000) / 1000 - 0.5) * T / 3, (((j // 1000) % 1000) / 1000 - 0.5) * T / 3) \
                    if kind == "positive" else (0, 0)
                tx = int(min(max(cx - T / 2 + jx, 0), W - T))
                ty = int(min(max(cy - T / 2 + jy, 0), H - T))
                key = f"{run}|{img.relative_to(src)}|{tx},{ty}|{scale}"
                if key in ds.keys:
                    stats["skipped_existing"] += 1
                    continue
                tb = []
                for cls, (x0, y0, x1, y1) in mapped:
                    c0, d0, c1, d1 = max(x0, tx), max(y0, ty), min(x1, tx + T), min(y1, ty + T)
                    if c1 > c0 and d1 > d0 and (c1 - c0) * (d1 - d0) >= 0.4 * (x1 - x0) * (y1 - y0):
                        tb.append({"cls": cls, "box": [round(c0 - tx, 1), round(d0 - ty, 1), round(c1 - tx, 1),
                                                       round(d1 - ty, 1)]})
                fn = f"tiles/{_h(key):012x}.jpg"
                im.crop((tx, ty, tx + T, ty + T)).save(ds.dir / fn, quality=93)
                ds.samples.append(Sample(file=fn, split=split, width=T, height=T, boxes=tb,
                                         kind="positive" if tb else "negative", key=key,
                                         source={"run": run, "photo": str(img.relative_to(src)), "tile_xy": [tx, ty],
                                                 "scale": scale,
                                                 "gsd_cm": round(gsd_cm / scale, 3) if gsd_cm else None,
                                                 "photo_abs": str(img.resolve())}))
                ds.keys.add(key)
                stats["positive" if tb else "negative"] += 1
                stats["boxes"] += len(tb)
        if stats["images"] % 100 == 0:
            log(f"  {stats['images']} images...")
    ds.save()
    stats["dataset"] = ds.stats()
    return stats
