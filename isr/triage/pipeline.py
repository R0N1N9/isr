"""Scan a folder / SD card of drone photos and produce a ranked list of hits."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw

from ..camera import PhotoMeta, photo_footprint, pixel_to_ground, read_photo_meta
from .detectors import (
    AIDetector,
    ColorTargetDetector,
    Detection,
    RXAnomalyDetector,
    SizeFilter,
    WorkImage,
    merge_detections,
)

IMAGE_EXTS = {".jpg", ".jpeg"}


@dataclass
class TriageConfig:
    colors: list[str] = field(default_factory=list)
    anomaly: bool = True
    sensitivity: float = 1.0
    ai: bool = False
    ai_model: str = "yolov8n.pt"
    ai_conf: float = 0.20
    ai_tile: int | None = None  # ONNX models: full-resolution tile size (default from the model card)
    verifier: str | None = None  # crop verifier .onnx (re-ranks candidates; never removes them)
    work_px: int = 2000
    min_area_m2: float = 0.04
    max_area_m2: float = 12.0
    ground_offset_m: float = 0.0
    max_per_photo: int = 8
    min_score: float = 0.35
    workers: int = 0  # 0 = auto
    preview_px: int = 1800
    crop_px: int = 420
    reference: dict | None = None  # ReferenceProfile as a dict (see reference.py)


def _ref_profile(cfg: TriageConfig):
    if not cfg.reference:
        return None
    from .reference import ReferenceProfile
    return ReferenceProfile(**cfg.reference)


def apply_reference(d: Detection, sim: float) -> None:
    """Raise candidates that look like the reference subject; lower clear mismatches."""
    d.extra["ref_sim"] = round(sim, 3)
    if sim >= 0.55:
        d.score = 1 - (1 - d.score) * (1 - 0.7 * sim)
        if "ref" not in d.kinds:
            d.kinds.append("ref")
        d.label += f" · {sim:.0%} like reference"
    elif sim < 0.25:
        d.score *= 0.85


@dataclass
class PhotoResult:
    file: str
    rel: str
    lat: float | None
    lon: float | None
    rel_alt_m: float | None
    gimbal_pitch: float | None
    gimbal_yaw: float | None
    taken_at: str | None
    gsd_cm: float | None
    footprint: list[tuple[float, float]] | None
    preview: str | None = None
    error: str | None = None
    detections: list[dict] = field(default_factory=list)

    @property
    def max_score(self) -> float:
        return max((d["score"] for d in self.detections), default=0.0)


def find_photos(root: Path, exclude: Path | None = None) -> list[Path]:
    """All JPEGs under ``root`` (e.g. an SD card's DCIM folder), sorted by name."""
    out = []
    excl = exclude.resolve() if exclude else None
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        if excl and (d.resolve() == excl or excl in d.resolve().parents):
            dirnames[:] = []
            continue
        dirnames[:] = [n for n in dirnames if not n.startswith(".")]
        for fn in filenames:
            if Path(fn).suffix.lower() in IMAGE_EXTS and not fn.startswith("._"):
                out.append(d / fn)
    return sorted(out)


def _classic_detect(path: Path, cfg: TriageConfig) -> tuple[PhotoMeta | None, list[Detection], str | None]:
    try:
        meta = read_photo_meta(path)
    except Exception as e:  # corrupt file, not an image, etc.
        return None, [], f"metadata: {e}"
    try:
        gsd = meta.gsd_m(cfg.ground_offset_m)
        w = WorkImage.load(path, cfg.work_px, gsd)
        sizes = SizeFilter(cfg.min_area_m2, cfg.max_area_m2)
        dets: list[Detection] = []
        if cfg.anomaly:
            dets += RXAnomalyDetector(sensitivity=cfg.sensitivity).detect(w, sizes)
        if cfg.colors:
            dets += ColorTargetDetector(cfg.colors).detect(w, sizes)
        prof = _ref_profile(cfg)
        if prof is not None:
            from .reference import similarity
            for d in dets:
                x0, y0 = int(d.x0 / w.scale), int(d.y0 / w.scale)
                x1, y1 = max(x0 + 1, int(d.x1 / w.scale)), max(y0 + 1, int(d.y1 / w.scale))
                d.extra["ref_sim"] = round(similarity(prof, w.rgb[y0:y1, x0:x1]), 3)
        return meta, dets, None
    except Exception as e:
        return meta, [], f"detect: {e}"


def make_ai_detector(cfg: TriageConfig, classes: set[str] | None = None):
    """ONNX file -> OnnxDetector (no PyTorch needed); anything else -> Ultralytics YOLO."""
    if str(cfg.ai_model).lower().endswith(".onnx"):
        from ..ml.onnx_detector import OnnxDetector
        return OnnxDetector(cfg.ai_model, conf=cfg.ai_conf, classes=classes, tile=cfg.ai_tile)
    return AIDetector(cfg.ai_model, conf=cfg.ai_conf, classes=classes)


def _score_reference(dets: list[Detection], prof, path: Path) -> None:
    """Apply appearance similarity; AI-only hits get it measured from the photo here."""
    from .reference import similarity
    todo = [d for d in dets if d.extra.get("ref_sim") is None]
    if todo:
        import numpy as np
        with Image.open(path) as im:
            for d in todo:
                patch = im.crop((int(d.x0), int(d.y0), int(d.x1) + 1, int(d.y1) + 1)).convert("RGB")
                patch.thumbnail((96, 96))
                d.extra["ref_sim"] = similarity(prof, np.asarray(patch, np.float32) / 255)
    for d in dets:
        apply_reference(d, float(d.extra["ref_sim"]))


def _render(path: Path, dets: list[dict], out_dir: Path, stem: str, cfg: TriageConfig) -> str | None:
    """Write a downscaled annotated preview and one crop per detection."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        W, H = im.size
        for i, d in enumerate(dets):
            x0, y0, x1, y1 = d["box"]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            half = max(cfg.crop_px / 2, (x1 - x0) * 1.6, (y1 - y0) * 1.6)
            bx0, by0 = max(0, int(cx - half)), max(0, int(cy - half))
            bx1, by1 = min(W, int(cx + half)), min(H, int(cy + half))
            crop = im.crop((bx0, by0, bx1, by1))
            dr = ImageDraw.Draw(crop)
            lw = max(2, int(crop.size[0] / 160))
            dr.rectangle((x0 - bx0 - lw * 2, y0 - by0 - lw * 2, x1 - bx0 + lw * 2, y1 - by0 + lw * 2),
                         outline=(255, 40, 200), width=lw)
            if crop.size[0] > 640:
                crop = crop.resize((640, round(crop.size[1] * 640 / crop.size[0])), Image.LANCZOS)
            rel = f"crops/{stem}_{i + 1}.jpg"
            crop.save(out_dir / rel, quality=88)
            d["crop"] = rel
        if not dets:
            return None
        ratio = cfg.preview_px / max(W, H)
        prev = im.resize((round(W * ratio), round(H * ratio)), Image.BILINEAR) if ratio < 1 else im
        ratio = min(ratio, 1.0)
        dr = ImageDraw.Draw(prev)
        for i, d in enumerate(dets):
            x0, y0, x1, y1 = (c * ratio for c in d["box"])
            pad = 10
            dr.rectangle((x0 - pad, y0 - pad, x1 + pad, y1 + pad), outline=(255, 40, 200), width=4)
            dr.text((x0 - pad, max(0, y0 - pad - 14)), str(i + 1), fill=(255, 40, 200))
        rel = f"previews/{stem}.jpg"
        prev.save(out_dir / rel, quality=85)
        return rel


def _render_job(args):
    path, dets, out_dir, stem, cfg = args
    try:
        return _render(path, dets, out_dir, stem, cfg), dets
    except Exception as e:
        print(f"  ! could not render {path.name}: {e}", file=sys.stderr)
        return None, dets


def run_triage(source: Path, out_dir: Path, cfg: TriageConfig, log=print) -> dict:
    t0 = time.time()
    source, out_dir = Path(source), Path(out_dir)
    photos = find_photos(source, exclude=out_dir)
    if not photos:
        raise SystemExit(f"No JPEG photos found under {source}")
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    (out_dir / "previews").mkdir(parents=True, exist_ok=True)
    workers = cfg.workers or max(1, (os.cpu_count() or 2) - 1)
    log(f"Scanning {len(photos)} photos from {source} with {workers} worker(s)...")

    prof = _ref_profile(cfg)
    ai_classes = None
    if prof is not None:
        from .reference import AI_CLASSES
        ai_classes = AI_CLASSES[prof.target]
    ai = make_ai_detector(cfg, ai_classes) if cfg.ai else None
    verifier = None
    if cfg.verifier:
        from ..ml.verifier import Verifier
        verifier = Verifier(cfg.verifier)
        log(f"Verifier {Path(cfg.verifier).name}: {verifier.card.type}, target '{verifier.label}'")
    if ai is not None and getattr(ai, "format", None):
        log(f"AI model {Path(cfg.ai_model).name}: {ai.format} format, {len(ai.class_names) or '?'} classes, "
            f"{ai.tile} px tiles")
    sizes = SizeFilter(cfg.min_area_m2, cfg.max_area_m2)

    results: list[PhotoResult] = []
    pending_render = []
    # 'spawn' behaves the same on Linux, macOS and Windows and is safe alongside torch
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futs = pool.map(_classic_detect, photos, [cfg] * len(photos), chunksize=2)
        for n, (path, (meta, dets, err)) in enumerate(zip(photos, futs), 1):
            if ai is not None and meta is not None:
                try:
                    dets = dets + ai.detect_path(path, meta.gsd_m(cfg.ground_offset_m), sizes)
                except Exception as e:
                    err = (err + "; " if err else "") + f"ai: {e}"
            # merge hits within ~1 m of each other (or 1 % of the frame if the scale is unknown)
            near = 0.0
            if meta is not None:
                gsd = meta.gsd_m(cfg.ground_offset_m)
                near = 1.0 / gsd if gsd else 0.01 * max(meta.width, meta.height)
            dets = merge_detections(dets, near_px=near)
            if prof is not None:
                _score_reference(dets, prof, path)
            dets = [d for d in dets if d.score >= cfg.min_score]
            dets.sort(key=lambda d: -d.score)
            dets = dets[: cfg.max_per_photo]
            if verifier is not None and dets:
                # after the cut-offs on purpose: the verifier re-ranks but never removes a candidate
                from ..ml.verifier import apply_verification
                with Image.open(path) as vim:
                    vim = vim.convert("RGB")
                    for d, p in zip(dets, verifier.score_boxes(vim, [(d.x0, d.y0, d.x1, d.y1) for d in dets])):
                        apply_verification(d, float(p), verifier.label)
                dets.sort(key=lambda d: -d.score)
            pr = _photo_result(path, source, meta, dets, cfg, err)
            results.append(pr)
            stem = f"{n:05d}_{path.stem}"
            if pr.detections:
                pending_render.append((path, pr.detections, out_dir, stem, cfg))
            if n % 25 == 0 or n == len(photos):
                hits = sum(len(r.detections) for r in results)
                log(f"  {n}/{len(photos)} photos, {hits} candidate(s) so far")

        if pending_render:
            log(f"Rendering crops for {len(pending_render)} photo(s) with candidates...")
            by_path = {r.file: r for r in results}
            for job, (prev, dets) in zip(pending_render, pool.map(_render_job, pending_render)):
                r = by_path[str(job[0])]
                r.preview, r.detections = prev, dets

    summary = _summary(results, source, cfg, time.time() - t0)
    data = {"summary": summary, "photos": [asdict(r) for r in results]}
    (out_dir / "results.json").write_text(json.dumps(data, indent=1, default=str))
    return data


def _photo_result(path: Path, root: Path, meta: PhotoMeta | None, dets: list[Detection], cfg, err) -> PhotoResult:
    if meta is None:
        return PhotoResult(str(path), str(path.relative_to(root)), None, None, None, None, None, None, None, None,
                           error=err)
    gsd = meta.gsd_m(cfg.ground_offset_m)
    pr = PhotoResult(
        file=str(path),
        rel=str(path.relative_to(root)),
        lat=meta.lat,
        lon=meta.lon,
        rel_alt_m=meta.rel_alt_m,
        gimbal_pitch=meta.gimbal_pitch,
        gimbal_yaw=meta.gimbal_yaw,
        taken_at=meta.taken_at.isoformat() if meta.taken_at else None,
        gsd_cm=round(gsd * 100, 2) if gsd else None,
        footprint=photo_footprint(meta, cfg.ground_offset_m),
        error=err,
    )
    for i, d in enumerate(dets):
        ll = pixel_to_ground(meta, d.cx, d.cy, cfg.ground_offset_m)
        pr.detections.append(
            {
                "id": f"{path.stem}-{i + 1}",
                "n": i + 1,
                "score": round(d.score, 3),
                "kinds": d.kinds,
                "label": d.label,
                "box": [round(d.x0), round(d.y0), round(d.x1), round(d.y1)],
                "area_m2": round(d.area_m2, 2) if d.area_m2 else None,
                "lat": ll[0] if ll else None,
                "lon": ll[1] if ll else None,
                "position_source": "projected" if ll else ("drone" if meta.has_position else None),
                "extra": d.extra,
            }
        )
        if not ll and meta.has_position:
            # fall back to the drone's own position so the hit is still mappable
            pr.detections[-1]["lat"], pr.detections[-1]["lon"] = meta.lat, meta.lon
    return pr


def _summary(results: list[PhotoResult], source: Path, cfg: TriageConfig, secs: float) -> dict:
    dets = [d for r in results for d in r.detections]
    return {
        "source": str(source),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "photos": len(results),
        "photos_with_gps": sum(1 for r in results if r.lat is not None),
        "photos_geolocatable": sum(1 for r in results if r.footprint),
        "photos_with_hits": sum(1 for r in results if r.detections),
        "detections": len(dets),
        "errors": sum(1 for r in results if r.error),
        "seconds": round(secs, 1),
        "config": asdict(cfg),
    }
