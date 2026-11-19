"""Run any exported object-detection model (ONNX) on drone photos.

A fine-tuned model is just a file: drop ``model.onnx`` (and optionally a
``model.json`` model card next to it) into the app and pass it with
``--ai-model model.onnx``. Inference uses ONNX Runtime on the CPU (or a GPU
if one is available), so no PyTorch or Ultralytics install is needed in the
field, and Ultralytics' AGPL code stays out of the runtime.

Supported output layouts (auto-detected, or set ``"format"`` in the card):

``yolo``      one output (1, 4+C, N) or (1, N, 4+C): cx, cy, w, h + class
              scores (Ultralytics YOLOv8 / YOLO11 export, no NMS)
``yolo_e2e``  one output (1, N, 6): x1, y1, x2, y2, score, class
              (end-to-end / NMS-free YOLO exports)
``detr``      two outputs: boxes (1, Q, 4) normalised cx, cy, w, h and class
              logits (1, Q, C) (RF-DETR, DETR-style exports)
``dfine``     inputs (images, orig_target_sizes) and outputs labels, boxes
              (x1, y1, x2, y2 in pixels), scores (D-FINE / RT-DETR exports)

Model card (all fields optional)::

    {"format": "detr", "classes": ["person", "vehicle"], "input_size": 560,
     "normalize": "imagenet", "tile": 800, "conf": 0.3}

``tile`` is the size of the full-resolution square cut from the photo for
each inference; it is then resized to ``input_size``. Tiles close to the
input size keep small people sharp (slower); larger tiles are faster.
"""

from __future__ import annotations

import ast
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from ..triage.detectors import Detection, SizeFilter

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class ModelCard:
    format: str | None = None
    classes: list[str] = field(default_factory=list)
    input_size: int | None = None
    normalize: str | None = None  # "none" | "imagenet"
    tile: int | None = None
    conf: float | None = None
    notes: str = ""

    @classmethod
    def load(cls, model_path: Path) -> "ModelCard":
        p = Path(model_path).with_suffix(".json")
        if not p.exists():
            return cls()
        d = json.loads(p.read_text())
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, model_path: Path) -> Path:
        p = Path(model_path).with_suffix(".json")
        p.write_text(json.dumps({k: v for k, v in self.__dict__.items() if v not in (None, "", [])}, indent=2))
        return p


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float = 0.5) -> np.ndarray:
    """Greedy non-maximum suppression. boxes (N, 4) x1, y1, x2, y2. Returns kept indices."""
    if len(boxes) == 0:
        return np.zeros(0, int)
    x1, y1, x2, y2 = boxes.T
    area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        ov = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][ov <= iou]
    return np.array(keep, int)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


class OnnxDetector:
    """Tiled inference with an exported detector; same interface as the YOLO ``AIDetector``."""

    name = "ai"

    def __init__(self, path: str | Path, conf: float | None = None, classes: set[str] | None = None,
                 tile: int | None = None, overlap: float = 0.15, providers: list[str] | None = None):
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover - optional dependency
            raise RuntimeError("ONNX models need onnxruntime: pip install 'isr-tools[onnx]'") from e
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        avail = ort.get_available_providers()
        prefs = providers or [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider",
                                            "CPUExecutionProvider")
                              if p in avail]
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(self.path), so, providers=prefs)
        self.card = ModelCard.load(self.path)
        self.inputs = self.sess.get_inputs()
        self.outputs = self.sess.get_outputs()
        shp = self.inputs[0].shape
        fixed = [d for d in shp[2:] if isinstance(d, int)]
        self.input_size = int(self.card.input_size or (fixed[0] if fixed else 640))
        self.batch_fixed = isinstance(shp[0], int)
        self.format = self.card.format or self._detect_format()
        self.normalize = self.card.normalize or ("imagenet" if self.format == "detr" else "none")
        self.class_names = self.card.classes or self._metadata_names()
        self.conf = conf if conf is not None else (self.card.conf if self.card.conf is not None else 0.25)
        self.tile = int(tile or self.card.tile or round(self.input_size * 1.5))
        self.overlap = overlap
        names = set(self.class_names)
        wanted = set(classes or ())
        if wanted & names:
            self.keep_classes = wanted & names
        elif {"person", "traffic light", "frisbee"} <= names:
            # a stock COCO model: keep only classes that matter in a search (no frisbees or traffic lights)
            from ..triage.detectors import SAR_COCO_CLASSES
            self.keep_classes = SAR_COCO_CLASSES & names
        else:
            self.keep_classes = None  # custom (e.g. SAR-trained) model: every class is relevant

    # ---------------------------------------------------------------- introspection

    def _detect_format(self) -> str:
        outs = [o.shape for o in self.outputs]
        names = [o.name.lower() for o in self.outputs]
        if len(self.inputs) >= 2 and len(outs) >= 3:
            return "dfine"
        if len(outs) >= 2:
            return "detr"
        s = outs[0]
        if len(s) == 3 and s[-1] == 6 and isinstance(s[1], int) and s[1] <= 1000:
            return "yolo_e2e"
        if len(s) == 3:
            return "yolo"
        raise ValueError(f"can't tell this model's output layout ({names} {outs}); add a model card with \"format\"")

    def _metadata_names(self) -> list[str]:
        meta = self.sess.get_modelmeta().custom_metadata_map
        raw = meta.get("names")
        if raw:
            try:
                d = ast.literal_eval(raw)
                if isinstance(d, dict):
                    return [str(d[k]) for k in sorted(d)]
                if isinstance(d, (list, tuple)):
                    return [str(x) for x in d]
            except (ValueError, SyntaxError):
                pass
        return []

    def class_name(self, i: int) -> str:
        return self.class_names[i] if 0 <= i < len(self.class_names) else f"class_{i}"

    def describe(self) -> dict:
        return {"file": str(self.path), "format": self.format, "input_size": self.input_size,
                "classes": self.class_names or "(unnamed)", "normalize": self.normalize, "tile": self.tile,
                "conf": self.conf, "providers": self.sess.get_providers(),
                "inputs": [(i.name, i.shape) for i in self.inputs], "outputs": [(o.name, o.shape) for o in self.outputs]}

    # ---------------------------------------------------------------- inference

    def _prep(self, tiles: list[Image.Image]) -> np.ndarray:
        arr = np.stack([np.asarray(t.resize((self.input_size, self.input_size), Image.BILINEAR), np.float32) / 255
                        for t in tiles])
        if self.normalize == "imagenet":
            arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        return np.ascontiguousarray(arr.transpose(0, 3, 1, 2))

    def _run(self, batch: np.ndarray):
        feeds = {self.inputs[0].name: batch}
        if self.format == "dfine" and len(self.inputs) > 1:
            feeds[self.inputs[1].name] = np.tile(np.array([[self.input_size, self.input_size]], np.int64),
                                                 (len(batch), 1))
        return self.sess.run(None, feeds)

    def _decode(self, outs, b: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Boxes (x1, y1, x2, y2 in input pixels), scores, class ids for batch item b."""
        S = self.input_size
        if self.format == "yolo":
            o = outs[0][b]
            nc = len(self.class_names)
            if nc and o.shape[0] == 4 + nc and o.shape[1] != 4 + nc:
                o = o.T  # (4+C, N) -> (N, 4+C)
            elif not (nc and o.shape[1] == 4 + nc) and o.shape[0] < o.shape[1]:
                o = o.T  # unknown class count: anchors are the long axis
            cls_scores = o[:, 4:]
            if cls_scores.size and cls_scores.max() > 1.0:
                cls_scores = _sigmoid(cls_scores)
            ids = cls_scores.argmax(1)
            sc = cls_scores[np.arange(len(o)), ids]
            cx, cy, w, h = o[:, 0], o[:, 1], o[:, 2], o[:, 3]
            if o[:, :4].max() <= 1.5:  # normalised coordinates
                cx, cy, w, h = cx * S, cy * S, w * S, h * S
            boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        elif self.format == "yolo_e2e":
            o = outs[0][b]
            boxes, sc, ids = o[:, :4], o[:, 4], o[:, 5].astype(int)
            if boxes.max() <= 1.5:
                boxes = boxes * S
        elif self.format == "detr":
            a, c = outs[0][b], outs[1][b]
            if a.shape[-1] != 4:
                a, c = c, a
            prob = _sigmoid(c)
            ids = prob.argmax(1)
            sc = prob[np.arange(len(prob)), ids]
            cx, cy, w, h = (a[:, k] * S for k in range(4))
            boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        else:  # dfine: labels, boxes, scores (names vary; pick by shape)
            by_rank = sorted(outs, key=lambda t: t.ndim)
            two_d = [t for t in by_rank if t.ndim == 2]
            boxes = next(t for t in outs if t.ndim == 3)[b]
            labels, scores = (two_d[0], two_d[1]) if two_d[0].dtype.kind in "iu" else (two_d[1], two_d[0])
            ids, sc = labels[b].astype(int), scores[b].astype(np.float32)
        return boxes.astype(np.float32), sc.astype(np.float32), ids.astype(int)

    def _crop(self, im: Image.Image, x: int, y: int, T: int) -> Image.Image:
        """Square tile; past the image edge it's padded (grey for YOLO models, as in their training)."""
        W, H = im.size
        if x + T <= W and y + T <= H:
            return im.crop((x, y, x + T, y + T))
        fill = (114, 114, 114) if self.format.startswith("yolo") else (0, 0, 0)
        canvas = Image.new("RGB", (T, T), fill)
        canvas.paste(im.crop((x, y, min(W, x + T), min(H, y + T))), (0, 0))
        return canvas

    def detect_image(self, im: Image.Image) -> list[tuple[float, float, float, float, float, str]]:
        """Detections on a whole image as (x1, y1, x2, y2, score, class name) in image pixels."""
        im = im.convert("RGB")
        W, H = im.size
        T = min(self.tile, max(W, H))
        stride = max(1, int(T * (1 - self.overlap)))
        offs = []
        for y in range(0, max(1, H - int(T * self.overlap)), stride):
            for x in range(0, max(1, W - int(T * self.overlap)), stride):
                offs.append((min(x, max(0, W - T)), min(y, max(0, H - T))))
        offs = list(dict.fromkeys(offs))
        bs = 1 if self.batch_fixed else 8
        all_b, all_s, all_c = [], [], []
        for i in range(0, len(offs), bs):
            chunk = offs[i:i + bs]
            tiles = [self._crop(im, x, y, T) for x, y in chunk]
            outs = self._run(self._prep(tiles))
            k = T / self.input_size
            for b, (ox, oy) in enumerate(chunk):
                boxes, sc, ids = self._decode(outs, b)
                m = sc >= self.conf
                if self.keep_classes is not None:
                    m &= np.isin([self.class_name(j) for j in ids], list(self.keep_classes))
                if m.any():
                    all_b.append(boxes[m] * k + [ox, oy, ox, oy])
                    all_s.append(sc[m])
                    all_c.append(ids[m])
        if not all_b:
            return []
        boxes, sc, ids = np.concatenate(all_b), np.concatenate(all_s), np.concatenate(all_c)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
        out = []
        for c in np.unique(ids):  # per-class NMS across tiles
            m = np.where(ids == c)[0]
            for j in m[nms(boxes[m], sc[m], 0.5)]:
                out.append((*map(float, boxes[j]), float(sc[j]), self.class_name(int(c))))
        return sorted(out, key=lambda d: -d[4])

    def detect_path(self, path, gsd_full_m: float | None, sizes: SizeFilter) -> list[Detection]:
        with Image.open(path) as im:
            raw = self.detect_image(im)
        dets = []
        for x0, y0, x1, y1, s, name in raw:
            area = (x1 - x0) * (y1 - y0) * gsd_full_m**2 if gsd_full_m else None
            if area is not None and area > sizes.max_area_m2 * 4:
                continue
            dets.append(Detection(x0, y0, x1, y1, s, [f"ai:{name}"], f"AI: {name} ({s:.0%})", area))
        return dets

    def benchmark(self, path, repeats: int = 1) -> float:
        with Image.open(path) as im:
            im = im.convert("RGB")
            t = time.time()
            for _ in range(repeats):
                self.detect_image(im)
        return (time.time() - t) / repeats
