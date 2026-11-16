"""Detectors that flag candidate people / objects in a drone photo.

Three independent detectors, each returning boxes in *full-resolution* pixel
coordinates with a 0..1 score:

``RXAnomalyDetector``
    Reed-Xiaoli colour anomaly detection, the classic SAR technique. It models
    the image's background colours as a multivariate Gaussian and flags pixels
    that are statistically far from it (a blue jacket in a forest, a tarp in a
    field). Needs no training data and works on any terrain.

``ColorTargetDetector``
    Matches specific colours, e.g. the missing person was last seen in a red
    jacket and blue jeans. Highest-value signal when the clothing is known.

``AIDetector`` (optional, needs ``pip install ultralytics``)
    Runs a YOLO model over overlapping full-resolution tiles so that people a
    few dozen pixels tall are still visible to the network. Stock COCO weights
    are trained on ground-level photos, so treat them as a supplement; a model
    fine-tuned on aerial SAR imagery (e.g. HERIDAL) can be passed in instead.

All detectors use the photo's ground sample distance (metres per pixel) when
available to reject blobs that are far too small or too large to be a person
or a person-sized object (roads, roofs, single noisy pixels).
"""

from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass
class Detection:
    x0: float
    y0: float
    x1: float
    y1: float
    score: float
    kinds: list[str]
    label: str
    area_m2: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    def iou(self, o: "Detection") -> float:
        ix = max(0.0, min(self.x1, o.x1) - max(self.x0, o.x0))
        iy = max(0.0, min(self.y1, o.y1) - max(self.y0, o.y0))
        inter = ix * iy
        a = (self.x1 - self.x0) * (self.y1 - self.y0)
        b = (o.x1 - o.x0) * (o.y1 - o.y0)
        return inter / (a + b - inter) if inter > 0 else 0.0


@dataclass
class WorkImage:
    """Downscaled RGB copy of a photo plus the scale back to full resolution."""

    rgb: np.ndarray  # float32 HxWx3 in 0..1
    scale: float  # full-res pixels per work pixel
    full_size: tuple[int, int]
    gsd_full_m: float | None  # metres per full-res pixel, None if unknown

    @property
    def gsd_work_m(self) -> float | None:
        return self.gsd_full_m * self.scale if self.gsd_full_m else None

    def local_mean(self, window_m: float = 8.0) -> np.ndarray:
        """Per-channel mean over a ~window_m square around each pixel (cached)."""
        cache = self.__dict__.setdefault("_local_cache", {})
        if window_m not in cache:
            if self.gsd_work_m:
                win = int(window_m / self.gsd_work_m)
            else:
                win = int(max(self.rgb.shape[:2]) * 0.03)
            win = max(7, win | 1)
            cache[window_m] = np.stack([ndimage.uniform_filter(self.rgb[..., c], win) for c in range(3)], -1)
        return cache[window_m]

    @classmethod
    def load(cls, path, work_px: int, gsd_full_m: float | None) -> "WorkImage":
        with Image.open(path) as im:
            full = im.size
            # JPEG DCT scaling makes this ~5x faster than a full decode + resize
            im.draft("RGB", (max(1, full[0] * work_px // max(full)), max(1, full[1] * work_px // max(full))))
            im = im.convert("RGB")
            if max(im.size) > work_px:
                ratio = work_px / max(im.size)
                im = im.resize((round(im.size[0] * ratio), round(im.size[1] * ratio)), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0
        return cls(rgb=arr, scale=full[0] / arr.shape[1], full_size=full, gsd_full_m=gsd_full_m)


@dataclass
class SizeFilter:
    """Physical size window for candidate blobs."""

    min_area_m2: float = 0.04  # ~20 x 20 cm: a head, a backpack
    max_area_m2: float = 12.0  # a two-person tent / small vehicle roof
    # fallbacks when the GSD is unknown (fractions of the work image area)
    min_frac: float = 2e-6
    max_frac: float = 2e-3

    def bounds_px(self, w: WorkImage) -> tuple[float, float]:
        if w.gsd_work_m:
            g2 = w.gsd_work_m**2
            return max(2.0, self.min_area_m2 / g2), self.max_area_m2 / g2
        n = w.rgb.shape[0] * w.rgb.shape[1]
        return max(2.0, self.min_frac * n), self.max_frac * n


def _blobs(mask: np.ndarray, value: np.ndarray, w: WorkImage, sizes: SizeFilter, max_blobs: int):
    """Connected components of ``mask`` within the size window.

    Yields (slice_y, slice_x, n_pixels, peak_value, mean_value, component_mask).
    """
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)), iterations=1)
    labels, n = ndimage.label(mask)
    if n == 0:
        return []
    lo, hi = sizes.bounds_px(w)
    idx = np.arange(1, n + 1)
    counts = ndimage.sum_labels(np.ones_like(value), labels, idx)
    peaks = ndimage.maximum(value, labels, idx)
    keep = [(i, c, p) for i, c, p in zip(idx, counts, peaks) if lo <= c <= hi]
    keep.sort(key=lambda t: -t[2])
    objs = ndimage.find_objects(labels)
    out = []
    for i, c, p in keep[:max_blobs]:
        sl = objs[i - 1]
        comp = labels[sl] == i
        h, wd = comp.shape
        # long thin features (fences, road lines, power lines) are not people
        if max(h, wd) > 12 and c / (h * wd) < 0.12:
            continue
        mean = float(value[sl][comp].mean())
        out.append((sl[0], sl[1], int(c), float(p), mean, comp))
    return out


def _to_detection(sy, sx, w: WorkImage, score, kinds, label, n_px, extra=None, pad=1) -> Detection:
    s = w.scale
    y0, y1 = max(0, sy.start - pad), sy.stop + pad
    x0, x1 = max(0, sx.start - pad), sx.stop + pad
    area = n_px * w.gsd_work_m**2 if w.gsd_work_m else None
    return Detection(
        x0=x0 * s,
        y0=y0 * s,
        x1=min(w.full_size[0], x1 * s),
        y1=min(w.full_size[1], y1 * s),
        score=float(score),
        kinds=list(kinds),
        label=label,
        area_m2=area,
        extra=extra or {},
    )


# --------------------------------------------------------------------------
# RX anomaly
# --------------------------------------------------------------------------


def _rx_distance(x: np.ndarray) -> np.ndarray:
    """Squared Mahalanobis distance of every pixel from a robust background fit."""
    flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
    sample = flat[:: max(1, flat.shape[0] // 200_000)]
    mu = sample.mean(0)
    inv = np.linalg.inv(np.cov(sample, rowvar=False) + np.eye(flat.shape[1]) * 1e-5)
    d = sample - mu
    d2s = np.einsum("ij,jk,ik->i", d, inv, d)
    # refit without the most extreme 2 % so targets don't inflate the background model
    bg = sample[d2s < np.percentile(d2s, 98)]
    mu = bg.mean(0)
    inv = np.linalg.inv(np.cov(bg, rowvar=False) + np.eye(flat.shape[1]) * 1e-5)
    d = flat - mu
    return np.einsum("ij,jk,ik->i", d, inv, d).reshape(x.shape[:-1]).astype(np.float32)


@dataclass
class RXAnomalyDetector:
    """Global + local RX.

    A pixel must be unusual for the photo as a whole *and* for its immediate
    surroundings (``local_window_m``). That suppresses large uniform features
    (roads, roofs, fields of a different crop) while keeping a person standing
    on a road. The threshold is a multiple of the 99.5th percentile, which a
    single large object (a tarp, a car) cannot shift, so one big anomaly in a
    frame doesn't hide a small one next to it.
    """

    sensitivity: float = 1.0  # >1 flags more, <1 flags less
    percentile: float = 99.5
    factor: float = 2.0
    min_d2: float = 30.0
    local_window_m: float = 8.0
    max_blobs: int = 12

    name = "anomaly"

    def detect(self, w: WorkImage, sizes: SizeFilter) -> list[Detection]:
        g = _rx_distance(w.rgb)
        d2 = np.minimum(g, _rx_distance(w.rgb - w.local_mean(self.local_window_m)))
        thr = max(self.min_d2, self.factor * float(np.percentile(d2, self.percentile))) / self.sensitivity

        # colourful anomalies matter more than bright/dark ones (glare, shadow)
        mx, mn = w.rgb.max(2), w.rgb.min(2)
        sat = np.where(mx > 1e-3, (mx - mn) / np.maximum(mx, 1e-3), 0)

        dets = []
        for sy, sx, n, peak, mean, comp in _blobs(d2 > thr, d2, w, sizes, self.max_blobs):
            s_mean = float(sat[sy, sx][comp].mean())
            rarity = 1.0 - math.exp(-(peak / thr - 1.0))
            score = 0.25 + 0.5 * rarity + 0.25 * min(1.0, s_mean / 0.6)
            rgb = w.rgb[sy, sx][comp].mean(0)
            dets.append(
                _to_detection(
                    sy,
                    sx,
                    w,
                    min(score, 0.95),
                    ["anomaly"],
                    f"colour anomaly ({describe_rgb(rgb)})",
                    n,
                    {"rx_peak": round(peak / thr, 2), "mean_rgb": [round(float(c) * 255) for c in rgb]},
                )
            )
        return dets


# --------------------------------------------------------------------------
# Target colours
# --------------------------------------------------------------------------

# name -> (hue_lo_deg, hue_hi_deg, min_sat, min_val, max_sat)
NAMED_COLORS: dict[str, tuple[float, float, float, float, float]] = {
    "red": (345, 12, 0.60, 0.30, 1.0),
    "orange": (12, 40, 0.70, 0.55, 1.0),
    "yellow": (42, 66, 0.65, 0.55, 1.0),
    "green": (75, 160, 0.55, 0.30, 1.0),  # warning: vegetation; works on snow/rock/sand
    "cyan": (165, 195, 0.45, 0.35, 1.0),
    "blue": (195, 250, 0.45, 0.25, 1.0),
    "purple": (250, 290, 0.35, 0.20, 1.0),
    "pink": (290, 345, 0.35, 0.40, 1.0),
    "magenta": (290, 330, 0.50, 0.35, 1.0),
    "white": (0, 360, 0.0, 0.88, 0.12),
}


def parse_color(spec: str) -> tuple[str, tuple[float, float, float, float, float]]:
    s = spec.strip().lower()
    if s in NAMED_COLORS:
        return s, NAMED_COLORS[s]
    if s.startswith("#") and len(s) in (4, 7):
        h = s[1:]
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        r, g, b = (int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))
        hh, ss, vv = colorsys.rgb_to_hsv(r, g, b)
        hd = hh * 360
        if ss < 0.15:
            raise ValueError(f"{spec}: greys/whites/blacks are too common to search by hex; use 'white'")
        return s, ((hd - 18) % 360, (hd + 18) % 360, max(0.30, ss * 0.6), max(0.15, vv * 0.5), 1.0)
    raise ValueError(f"unknown colour {spec!r}; use one of {', '.join(NAMED_COLORS)} or a #rrggbb hex")


@dataclass
class ColorTargetDetector:
    """Pixels of a target colour that also stand out from their surroundings.

    The contrast test (RGB distance from the ~8 m local mean) is what stops a
    field of dry grass from matching "orange" or red soil from matching "red":
    natural colour regions blend into their neighbourhood, clothing doesn't.
    """

    colors: list[str]
    max_blobs: int = 10
    min_contrast: float = 0.12
    local_window_m: float = 8.0

    name = "color"

    def __post_init__(self):
        self._specs = [parse_color(c) for c in self.colors]

    def detect(self, w: WorkImage, sizes: SizeFilter) -> list[Detection]:
        rgb = w.rgb
        mx, mn = rgb.max(2), rgb.min(2)
        v = mx
        s = np.where(mx > 1e-3, (mx - mn) / np.maximum(mx, 1e-3), 0)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        delta = np.maximum(mx - mn, 1e-6)
        h = np.where(
            mx == r,
            ((g - b) / delta) % 6,
            np.where(mx == g, (b - r) / delta + 2, (r - g) / delta + 4),
        ) * 60.0
        contrast = np.linalg.norm(rgb - w.local_mean(self.local_window_m), axis=2) > self.min_contrast
        dets = []
        for name, (hlo, hhi, smin, vmin, smax) in self._specs:
            if hlo <= hhi:
                hm = (h >= hlo) & (h <= hhi)
            else:  # wraps through 0 deg (red)
                hm = (h >= hlo) | (h <= hhi)
            mask = hm & (s >= smin) & (s <= smax) & (v >= vmin) & contrast
            strength = (s * v).astype(np.float32) if name != "white" else v.astype(np.float32)
            for sy, sx, n, peak, mean, comp in _blobs(mask, strength, w, sizes, self.max_blobs):
                # vivid and at least jacket-sized (~0.15 m² seen from above) scores highest
                size_term = min(1.0, n * w.gsd_work_m**2 / 0.15) if w.gsd_work_m else 0.6
                score = 0.45 + 0.25 * min(1.0, mean / 0.6) + 0.25 * size_term
                dets.append(
                    _to_detection(sy, sx, w, min(score, 0.95), [f"color:{name}"], f"{name} match", n)
                )
        return dets


# --------------------------------------------------------------------------
# AI (YOLO) — optional
# --------------------------------------------------------------------------

# COCO classes worth surfacing in a search
SAR_COCO_CLASSES = {
    "person",
    "backpack",
    "handbag",
    "suitcase",
    "umbrella",
    "bicycle",
    "motorcycle",
    "car",
    "truck",
    "boat",
    "dog",
    "horse",
}


class AIDetector:
    name = "ai"

    def __init__(self, model: str = "yolov8n.pt", conf: float = 0.20, tile: int = 1280, overlap: int = 160,
                 classes: set[str] | None = None, device: str | None = None):
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "AI detection needs the 'ultralytics' package: pip install 'isr-tools[ai]'"
            ) from e
        self.model = YOLO(model)
        self.conf = conf
        self.tile = tile
        self.overlap = overlap
        self.device = device
        names = set(self.model.names.values())
        # custom (e.g. SAR-trained) models: every class is relevant
        self.classes = classes or (SAR_COCO_CLASSES & names if SAR_COCO_CLASSES & names else names)

    def detect_path(self, path, gsd_full_m: float | None, sizes: SizeFilter) -> list[Detection]:
        with Image.open(path) as im:
            im = im.convert("RGB")
            W, H = im.size
            tiles, offs = [], []
            stride = self.tile - self.overlap
            for y in range(0, max(1, H - self.overlap), stride):
                for x in range(0, max(1, W - self.overlap), stride):
                    x0, y0 = min(x, max(0, W - self.tile)), min(y, max(0, H - self.tile))
                    tiles.append(im.crop((x0, y0, x0 + self.tile, y0 + self.tile)))
                    offs.append((x0, y0))
        dets: list[Detection] = []
        for i in range(0, len(tiles), 8):
            results = self.model.predict(
                tiles[i : i + 8], conf=self.conf, imgsz=self.tile, verbose=False, device=self.device
            )
            for res, (ox, oy) in zip(results, offs[i : i + 8]):
                for box, c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()):
                    name = self.model.names[int(k)]
                    if name not in self.classes:
                        continue
                    x0, y0, x1, y1 = box
                    area = (x1 - x0) * (y1 - y0) * gsd_full_m**2 if gsd_full_m else None
                    if area is not None and area > sizes.max_area_m2 * 4:
                        continue
                    dets.append(Detection(x0 + ox, y0 + oy, x1 + ox, y1 + oy, float(c), [f"ai:{name}"],
                                          f"AI: {name} ({c:.0%})", area))
        return merge_detections(dets, iou=0.3)


# --------------------------------------------------------------------------
# Merging & helpers
# --------------------------------------------------------------------------


def merge_detections(dets: list[Detection], iou: float = 0.05, near_px: float = 0.0) -> list[Detection]:
    """Greedily merge overlapping detections; scores combine as noisy-OR."""
    dets = sorted(dets, key=lambda d: -d.score)
    merged: list[Detection] = []
    for d in dets:
        for m in merged:
            close = near_px and math.hypot(d.cx - m.cx, d.cy - m.cy) < near_px
            if m.iou(d) > iou or close:
                m.x0, m.y0 = min(m.x0, d.x0), min(m.y0, d.y0)
                m.x1, m.y1 = max(m.x1, d.x1), max(m.y1, d.y1)
                new = [k for k in d.kinds if k not in m.kinds]
                if new:
                    # independent detectors agreeing is strong evidence
                    m.score = 1 - (1 - m.score) * (1 - d.score * 0.8)
                    m.kinds += new
                    m.label = f"{m.label} + {d.label}"
                m.extra.update({k: v for k, v in d.extra.items() if k not in m.extra})
                if m.area_m2 is None:
                    m.area_m2 = d.area_m2
                break
        else:
            merged.append(Detection(d.x0, d.y0, d.x1, d.y1, d.score, list(d.kinds), d.label, d.area_m2,
                                    dict(d.extra)))
    return merged


def describe_rgb(rgb) -> str:
    r, g, b = (float(c) for c in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if v < 0.15:
        return "very dark"
    if s < 0.15:
        return "white/bright" if v > 0.8 else "grey"
    hd = h * 360
    for name in ("red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink"):
        lo, hi = NAMED_COLORS[name][:2]
        if (lo <= hd <= hi) if lo <= hi else (hd >= lo or hd <= hi):
            return name
    return "mixed"
