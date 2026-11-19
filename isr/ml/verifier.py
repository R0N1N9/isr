"""Second-opinion check on each candidate crop.

A verifier looks at a crop around every candidate (with some context) and
estimates the probability that it really is the target. Two kinds, both
plain ONNX files with a JSON card next to them:

``embedding``  a CLIP / SigLIP image encoder plus pre-computed text
               embeddings of prompts such as "an aerial photo of a person
               lying on the ground" vs "...a rock", "...a log". Zero-shot:
               no training needed. Build once with ``isr verifier build-clip``.
``classifier`` a small crop classifier (e.g. trained on your own reviews with
               ``isr verifier train``): softmax over classes, with the card
               naming which classes count as the target.

Verifiers re-rank; they never remove a candidate. In a search a miss costs
far more than a false alarm, so a doubtful crop moves down the list instead
of disappearing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
CLIP_NORM = ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])


def _providers(ort, avail=None):
    avail = avail or ort.get_available_providers()
    return [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider",
                        "CPUExecutionProvider") if p in avail]


def context_crop(im: Image.Image, box, context: float = 3.0, min_px: int = 48) -> Image.Image:
    """Square crop centred on ``box`` and ``context`` times its size (people need surroundings to judge)."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max((x1 - x0), (y1 - y0), min_px / context) * context / 2
    W, H = im.size
    l, t, r, b = cx - half, cy - half, cx + half, cy + half
    crop = im.crop((int(max(0, l)), int(max(0, t)), int(min(W, r)), int(min(H, b))))
    if (l < 0 or t < 0 or r > W or b > H):  # pad at photo edges so the target stays centred
        canvas = Image.new("RGB", (int(r - l), int(b - t)), (0, 0, 0))
        canvas.paste(crop, (int(max(0, -l)), int(max(0, -t))))
        crop = canvas
    return crop


@dataclass
class VerifierCard:
    type: str
    input_size: int = 224
    mean: list[float] | None = None
    std: list[float] | None = None
    classes: list[str] | None = None
    positive: list[str] | None = None
    prompts: dict | None = None  # {"positive": [...], "negative": [...]}
    logit_scale: float = 100.0
    context: float = 3.0
    target: str = "person"
    notes: str = ""

    @classmethod
    def load(cls, model_path: Path) -> "VerifierCard":
        d = json.loads(Path(model_path).with_suffix(".json").read_text())
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, model_path: Path) -> None:
        Path(model_path).with_suffix(".json").write_text(
            json.dumps({k: v for k, v in self.__dict__.items() if v is not None}, indent=2))


class Verifier:
    def __init__(self, path: str | Path):
        import onnxruntime as ort
        self.path = Path(path)
        self.card = VerifierCard.load(self.path)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(self.path), so, providers=_providers(ort))
        self.input = self.sess.get_inputs()[0]
        mean, std = (self.card.mean, self.card.std) if self.card.mean else (
            CLIP_NORM if self.card.type == "embedding" else IMAGENET)
        self.mean, self.std = np.array(mean, np.float32), np.array(std, np.float32)
        if self.card.type == "embedding":
            emb = np.load(self.path.with_suffix(".prompts.npz"))
            self.pos = emb["positive"].astype(np.float32)
            self.neg = emb["negative"].astype(np.float32)
        elif self.card.type == "classifier":
            pos = set(self.card.positive or [c for c in (self.card.classes or []) if c != "background"])
            self.pos_idx = [i for i, c in enumerate(self.card.classes or []) if c in pos]
        else:
            raise ValueError(f"unknown verifier type {self.card.type!r}")

    @property
    def label(self) -> str:
        return self.card.target

    def _batch(self, crops: list[Image.Image]) -> np.ndarray:
        S = self.card.input_size
        a = np.stack([np.asarray(c.convert("RGB").resize((S, S), Image.BICUBIC), np.float32) / 255 for c in crops])
        a = (a - self.mean) / self.std
        return np.ascontiguousarray(a.transpose(0, 3, 1, 2))

    def _run(self, x: np.ndarray) -> np.ndarray:
        fixed = isinstance(self.input.shape[0], int)
        if fixed and len(x) > 1:
            return np.concatenate([self.sess.run(None, {self.input.name: x[i:i + 1]})[0] for i in range(len(x))])
        return self.sess.run(None, {self.input.name: x})[0]

    def probability(self, crops: list[Image.Image]) -> np.ndarray:
        """P(target) for each crop."""
        if not crops:
            return np.zeros(0, np.float32)
        out = []
        for i in range(0, len(crops), 16):
            y = self._run(self._batch(crops[i:i + 16]))
            if self.card.type == "embedding":
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
                logits = self.card.logit_scale * np.concatenate([y @ self.pos.T, y @ self.neg.T], 1)
                logits -= logits.max(1, keepdims=True)
                e = np.exp(logits)
                p = e[:, : len(self.pos)].sum(1) / e.sum(1)
            else:
                z = y - y.max(1, keepdims=True)
                e = np.exp(z)
                p = e[:, self.pos_idx].sum(1) / e.sum(1)
            out.append(p.astype(np.float32))
        return np.concatenate(out)

    def score_boxes(self, im: Image.Image, boxes) -> np.ndarray:
        return self.probability([context_crop(im, b, self.card.context) for b in boxes])


def apply_verification(det, p: float, label: str) -> None:
    """Re-rank one detection by verifier probability ``p``; never removes it."""
    det.extra["verify_p"] = round(float(p), 3)
    if p >= 0.6:
        det.score = 1 - (1 - det.score) * (1 - 0.6 * p)
        if "verified" not in det.kinds:
            det.kinds.append("verified")
        det.label += f" · verifier {p:.0%} {label}"
    elif p < 0.3:
        det.score *= 0.5 + p  # 0.5x .. 0.8x
        det.label += f" · verifier doubts ({p:.0%})"
