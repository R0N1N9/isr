"""Light, automatic touch-ups for aerial photos.

The goal is the first five minutes of a professional edit, not a look:

1. white balance    partial grey-world correction of colour casts (aerial
                    haze is usually blue), clamped to +/-8 % per channel
2. dehaze / levels  remove the haze veil (lifted black point) and set the
                    white point, never clipping more than a sliver
3. tone             gentle lift of dark frames / pull-down of bright frames
4. contrast         adaptive S-curve: flat frames get more, punchy frames none
5. clarity          local contrast on mid-tones (large-radius unsharp mask)
6. vibrance         boosts muted colours more than already-saturated ones
7. straighten       levels a tilted horizon (oblique shots, when confident)
8. sharpen          fine output sharpening sized to the image

Every step is measured from the photo itself and scaled by ``strength``
(0..1, default 0.6), so a well-exposed frame gets almost nothing. Originals
are never modified: results are written as new files with the original
EXIF / GPS / DJI XMP metadata and colour profile carried over.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage

from .aesthetics import horizon_tilt, luma


@dataclass
class EditReport:
    applied: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        names = {
            "wb": "white balance",
            "dehaze": "dehaze",
            "white_point": "white point",
            "tone": "tone",
            "contrast": "contrast",
            "clarity": "clarity",
            "vibrance": "vibrance",
            "straighten": "straighten",
            "sharpen": "sharpen",
        }
        return ", ".join(names.get(k, k) for k in self.applied) or "no changes needed"


def _soft_clip(x: np.ndarray, knee: float = 0.92) -> np.ndarray:
    """Roll highlights off smoothly instead of hard-clipping at 1.0."""
    over = x > knee
    if over.any():
        span = 1.0 - knee
        x = x.copy()
        x[over] = knee + span * np.tanh((x[over] - knee) / span)
    return x


def _small(arr: np.ndarray, px: int = 800) -> np.ndarray:
    h, w = arr.shape[:2]
    step = max(1, int(max(h, w) / px))
    return arr[::step, ::step]


def straighten(im: Image.Image, deg: float) -> Image.Image:
    """Level a horizon tilted ``deg`` clockwise, cropping to the largest same-aspect rectangle."""
    W, H = im.size
    t = math.radians(abs(deg))
    c, s = math.cos(t), math.sin(t)
    k = min(W / (W * c + H * s), H / (W * s + H * c))
    rot = im.rotate(deg, resample=Image.BICUBIC, expand=False)  # PIL rotates counter-clockwise
    cw, ch = int(W * k), int(H * k)
    left, top = (W - cw) // 2, (H - ch) // 2
    return rot.crop((left, top, left + cw, top + ch))


@dataclass
class _Params:
    gains: np.ndarray | None = None
    bp: np.ndarray | None = None
    wp_gain: float = 1.0
    gamma: float = 1.0
    curve_k: float = 0.0
    clarity: float = 0.0
    vibrance: float = 0.0


def _levels(x: np.ndarray, p: _Params) -> np.ndarray:
    if p.gains is not None:
        x = x * p.gains
    if p.bp is not None:
        x = (x - p.bp) / (1 - p.bp)
    if p.wp_gain != 1.0:
        x = x * p.wp_gain
    return x


def _tone(L: np.ndarray, p: _Params) -> np.ndarray:
    Ln = L
    if p.gamma != 1.0:
        Ln = np.clip(Ln, 0, 1) ** p.gamma
    if p.curve_k:
        x = np.clip(Ln, 0, 1)
        Ln = x + p.curve_k * (x * x * (3 - 2 * x) - x)
    return Ln


def _render(x: np.ndarray, p: _Params, blur: np.ndarray | None) -> np.ndarray:
    """All per-pixel steps for one strip of the image (float32 0..1 in, out)."""
    x = _levels(x, p)
    L = luma(x)
    Ln = _tone(L, p)
    if blur is not None and p.clarity:
        c = np.clip(Ln, 0, 1)
        Ln = Ln + p.clarity * np.clip(Ln - blur, -0.15, 0.15) * (4 * c * (1 - c))
    x *= (np.clip(Ln, 0, 1.2) / np.maximum(L, 1e-4))[..., None]
    if p.vibrance:
        L = luma(x)[..., None]
        mx, mn = x.max(2, keepdims=True), x.min(2, keepdims=True)
        sat = np.clip((mx - mn) / np.maximum(mx, 1e-3), 0, 1)
        x = L + (x - L) * (1 + p.vibrance * (1 - sat) ** 2)
    return _soft_clip(np.clip(x, 0, None))


def enhance_image(
    im: Image.Image,
    strength: float = 0.6,
    tilt_deg: float | None = None,
    auto_straighten: bool = True,
    sharpen: bool = True,
    gimbal_pitch: float | None = None,
) -> tuple[Image.Image, EditReport]:
    """Return an enhanced copy of ``im`` and a report of what was changed.

    Adjustments are measured on a small copy, then applied to the full image
    in strips, so a 20 MP frame needs a few hundred MB rather than gigabytes.
    """
    rep = EditReport()
    p = _Params()
    s = max(0.0, min(1.0, strength))
    im = ImageOps.exif_transpose(im).convert("RGB")
    arr = np.asarray(im)  # uint8, shared with the PIL image
    H, W = arr.shape[:2]
    small = _small(arr).astype(np.float32) / 255.0

    # 1. white balance from genuinely neutral surfaces (roads, roofs, rock, cloud)
    sl = luma(small)
    mx, mn = small.max(2), small.min(2)
    sat = (mx - mn) / np.maximum(mx, 1e-3)
    neutral = (sl > 0.2) & (sl < 0.9) & (sat < 0.12)
    if neutral.mean() > 0.02:
        means = small[neutral].mean(0)
        gains = np.clip(means.mean() / np.maximum(means, 1e-3), 0.94, 1.06)
        gains = (1 + (gains - 1) * 0.6 * s).astype(np.float32)
        if np.abs(gains - 1).max() > 0.004:
            p.gains = gains
            rep.applied["wb"] = round(float(np.abs(gains - 1).max()), 3)
    else:
        rep.notes.append("white balance left alone (no neutral surfaces to measure)")
    small = _levels(small, p)

    # 2. dehaze (black point) and white point
    bp = np.clip(np.percentile(small.reshape(-1, 3), 0.3, axis=0) * 0.9 * s, 0, 0.12).astype(np.float32)
    if bp.max() > 0.006:
        p.bp = bp
        small = (small - bp) / (1 - bp)
        rep.applied["dehaze"] = round(float(bp.max()), 3)
    wp = float(np.percentile(luma(small), 99.7))
    if wp < 0.95:
        p.wp_gain = float(1 + (min(1.15, 0.96 / max(wp, 0.2)) - 1) * s)
        small = small * p.wp_gain
        rep.applied["white_point"] = round(p.wp_gain - 1, 3)

    # 3. tone and 4. adaptive S-curve
    sL = luma(small)
    med = float(np.median(sL))
    target = med + (0.44 - med) * 0.5 * s
    if abs(target - med) > 0.01 and 0.03 < med < 0.97:
        p.gamma = float(np.clip(math.log(target) / math.log(med), 0.75, 1.3))
        rep.applied["tone"] = round(1 - p.gamma, 3)
    p2, p98 = np.percentile(sL, (2, 98))
    k = 0.35 * s * float(np.clip((0.9 - (p98 - p2)) / 0.4, 0.15, 1.0))
    if k > 0.01:
        p.curve_k = k
        rep.applied["contrast"] = round(k, 3)

    # 5. clarity: blur of the toned luma, computed at 1/4 scale
    f = 4
    lo = _levels(arr[::f, ::f].astype(np.float32) / 255.0, p)
    lo_t = ndimage.gaussian_filter(_tone(luma(lo), p), max(H, W) * 0.008 / f)
    blur_full = np.asarray(Image.fromarray(lo_t.astype(np.float32), "F").resize((W, H), Image.BILINEAR))
    del lo, lo_t
    p.clarity = 0.45 * s
    rep.applied["clarity"] = round(p.clarity, 3)
    # 6. vibrance
    p.vibrance = 0.35 * s
    rep.applied["vibrance"] = round(p.vibrance, 3)

    out_arr = np.empty_like(arr)
    step = 256
    for r0 in range(0, H, step):
        r1 = min(H, r0 + step)
        x = arr[r0:r1].astype(np.float32) / 255.0
        y = _render(x, p, blur_full[r0:r1])
        out_arr[r0:r1] = (np.clip(y, 0, 1) * 255 + 0.5).astype(np.uint8)
    del blur_full
    out = Image.fromarray(out_arr)

    # 7. straighten (only when a horizon can be in frame)
    if gimbal_pitch is not None and gimbal_pitch <= -60:
        auto_straighten = False
    if auto_straighten:
        if tilt_deg is None:
            prev = out.resize((1024, max(1, round(1024 * H / W))), Image.BILINEAR)
            t, conf = horizon_tilt(luma(np.asarray(prev, np.float32) / 255))
            tilt_deg = t if (t is not None and conf > 0.5) else None
        if tilt_deg is not None and 0.4 <= abs(tilt_deg) <= 6:
            out = straighten(out, tilt_deg)
            rep.applied["straighten"] = round(tilt_deg, 2)
            rep.notes.append(f"rotated {tilt_deg:+.1f}° and cropped to {out.size[0]}x{out.size[1]}")

    # 8. output sharpening
    if sharpen:
        r = max(0.6, 1.1 * max(out.size) / 5280)
        out = out.filter(ImageFilter.UnsharpMask(radius=r, percent=int(40 + 60 * s), threshold=2))
        rep.applied["sharpen"] = round(r, 2)
    return out, rep


def enhance_file(src: Path, dst: Path, strength: float = 0.6, tilt_deg: float | None = None,
                 auto_straighten: bool = True, quality: int = 95, gimbal_pitch: float | None = None) -> EditReport:
    """Enhance ``src`` into a new file ``dst`` (never overwrites the source)."""
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        raise ValueError("refusing to overwrite the original photo")
    with Image.open(src) as im:
        exif = im.info.get("exif")
        icc = im.info.get("icc_profile")
        xmp = im.info.get("xmp")
        if xmp is None:  # older Pillow: keep the raw XMP packet if present
            raw = src.read_bytes()[:262144]
            a, b = raw.find(b"<x:xmpmeta"), raw.find(b"</x:xmpmeta>")
            xmp = raw[a:b + 12] if a >= 0 and b > a else None
        if exif:
            # pixels are now upright; reset the orientation tag
            e = im.getexif()
            e[0x0112] = 1
            exif = e.tobytes()
        if gimbal_pitch is None:
            try:
                from ..camera import read_photo_meta
                gimbal_pitch = read_photo_meta(src).gimbal_pitch
            except Exception:
                gimbal_pitch = None
        out, rep = enhance_image(im, strength, tilt_deg, auto_straighten, gimbal_pitch=gimbal_pitch)
    dst.parent.mkdir(parents=True, exist_ok=True)
    kw = {"quality": quality, "subsampling": 0, "optimize": True}
    if exif:
        kw["exif"] = exif
    if icc:
        kw["icc_profile"] = icc
    if xmp:
        kw["xmp"] = xmp if isinstance(xmp, bytes) else str(xmp).encode()
    out.save(dst, "JPEG", **kw)
    return rep
