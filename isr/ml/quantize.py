"""Shrink a detector to 8-bit (INT8) for faster CPU inference.

Static quantisation needs calibration images so it can learn the range of
values inside the network; it uses tiles from your dataset (or photos) so the
ranges match your imagery. Always re-run the scorecard afterwards: small,
low-contrast targets are the first to suffer from lost precision.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image

from .onnx_detector import OnnxDetector


def _calibration_images(source: Path, n: int, tile: int, seed: int = 0) -> list[Image.Image]:
    source = Path(source)
    rng = random.Random(seed)
    if (source / "manifest.jsonl").exists():
        from .dataset import Dataset
        ds = Dataset(source)
        picks = rng.sample(ds.samples, min(n, len(ds.samples)))
        return [Image.open(ds.dir / s.file).convert("RGB") for s in picks]
    files = [p for p in source.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    out = []
    for p in rng.sample(files, min(len(files), n)):
        with Image.open(p) as im:
            im = im.convert("RGB")
            W, H = im.size
            T = min(tile, W, H)
            x, y = rng.randrange(0, W - T + 1), rng.randrange(0, H - T + 1)
            out.append(im.crop((x, y, x + T, y + T)))
    return out


def quantize_model(model: Path, out: Path, calibration: Path, samples: int = 64, log=print) -> Path:
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    det = OnnxDetector(model)
    imgs = _calibration_images(calibration, samples, det.tile)
    if not imgs:
        raise ValueError(f"no calibration images found in {calibration}")
    log(f"Calibrating on {len(imgs)} image(s)...")
    name = det.inputs[0].name

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(imgs)

        def get_next(self):
            im = next(self.it, None)
            return None if im is None else {name: det._prep([im])}

    out = Path(out)
    pre = out.with_suffix(".pre.onnx")
    try:
        quant_pre_process(str(model), str(pre))
        src = pre
    except Exception as e:  # pre-processing is an optimisation; quantise the original if it fails
        log(f"  (pre-processing skipped: {e.__class__.__name__})")
        src = Path(model)
    # Only the heavy layers (convolutions / matrix multiplies) go to 8-bit. Detector heads concatenate box
    # coordinates (0..640) with class scores (0..1) in one tensor; quantising that wipes out the scores.
    quantize_static(str(src), str(out), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    op_types_to_quantize=["Conv", "MatMul", "Gemm"])
    if pre.exists():
        pre.unlink()
    card = Path(model).with_suffix(".json")
    if card.exists():
        out.with_suffix(".json").write_text(card.read_text())
    return out


def compare(a: Path, b: Path, images: list[Path], conf: float = 0.25) -> dict:
    """Speed and agreement of two models on the same photos."""
    import time

    da, db = OnnxDetector(a, conf=conf), OnnxDetector(b, conf=conf)
    ta = tb = 0.0
    agree = total = 0
    for p in images:
        with Image.open(p) as im:
            im = im.convert("RGB")
            t = time.time()
            ra = da.detect_image(im)
            ta += time.time() - t
            t = time.time()
            rb = db.detect_image(im)
            tb += time.time() - t
        for x in ra:
            total += 1
            if any(x[5] == y[5] and _iou(x, y) > 0.5 for y in rb):
                agree += 1
    return {"seconds_a": round(ta / max(len(images), 1), 2), "seconds_b": round(tb / max(len(images), 1), 2),
            "speedup": round(ta / tb, 2) if tb else None,
            "agreement": round(agree / total, 3) if total else None, "detections_a": total}


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0
