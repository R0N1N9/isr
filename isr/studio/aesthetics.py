"""Score drone photos for portfolio / print potential.

"Visually appealing" is subjective, so this module doesn't pretend to know
taste. It measures the things that reliably separate keepers from rejects in
aerial work, and explains each score with plain tags:

Technical (can the photo be sold or printed at all?)
    sharpness     motion blur / missed focus, measured in the sharpest region
    exposure      mid-tone placement and clipped highlights / crushed shadows
    contrast      tonal range; flat, hazy frames score low
    noise         sensor noise (low light, high ISO)

Artistic (does it have the qualities aerial art is known for?)
    colour        colourfulness (Hasler & Suesstrunk), penalising garish extremes
    composition   a clear subject placed on a third or dead centre, or strong
                  mirror symmetry (a top-down signature)
    simplicity    large calm areas: water, sand, snow, fields (minimalism sells)
    pattern       strong repeating structure: crop rows, solar farms, waves

All measurements run on a ~1024 px copy, so a 20 MP Mavic 3 frame scores in
well under a second.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

WORK_PX = 1024


@dataclass
class Aesthetics:
    score: float  # 0..100 overall
    technical: float  # 0..1
    artistic: float  # 0..1
    parts: dict[str, float]  # each 0..1
    tags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)  # raw numbers for debugging / tuning
    tilt_deg: float | None = None  # horizon tilt estimate (oblique shots only)

    def to_dict(self) -> dict:
        return asdict(self)


def load_work(path, px: int = WORK_PX) -> np.ndarray:
    """Downscaled RGB float32 (0..1), orientation corrected."""
    with Image.open(path) as im:
        im.draft("RGB", (px, px))
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((px, px), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0


def luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _sat(x: float, k: float) -> float:
    """Smooth 0..1 saturation: 0 -> 0, k -> ~0.63, large -> 1."""
    return 1.0 - math.exp(-max(0.0, x) / k)


# --------------------------------------------------------------------------
# Technical
# --------------------------------------------------------------------------


def sharpness(lum: np.ndarray, noise: float = 0.0) -> tuple[float, float]:
    """Laplacian energy in the sharpest tiles (aerial frames are focused at infinity,
    so the best region tells us about motion blur and focus).

    White noise of std ``noise`` adds 20 * noise**2 to the variance of the
    5-point Laplacian; that is subtracted so grain isn't mistaken for detail.
    """
    lap = ndimage.laplace(lum)
    h, w = lum.shape
    ty, tx = 4, 6
    vals = []
    for i in range(ty):
        for j in range(tx):
            t = lap[i * h // ty:(i + 1) * h // ty, j * w // tx:(j + 1) * w // tx]
            vals.append(float(t.var()))
    v = max(0.0, float(np.percentile(vals, 80)) - 20.0 * noise**2)
    # normalised by contrast so a flat-but-sharp scene (water) isn't called blurry
    c = float(lum.std()) + 0.02
    raw = math.sqrt(v) / c
    return _sat(raw, 0.30), raw


def exposure(lum: np.ndarray) -> tuple[float, float, float, float]:
    m = float(lum.mean())
    hi = float((lum > 0.985).mean())
    lo = float((lum < 0.015).mean())
    s = math.exp(-((m - 0.46) ** 2) / (2 * 0.16**2))
    s -= min(0.6, hi * 6) + min(0.4, lo * 3)
    return max(0.0, min(1.0, s)), m, hi, lo


def contrast(lum: np.ndarray) -> tuple[float, float]:
    p2, p98 = np.percentile(lum, (2, 98))
    rng = float(p98 - p2)
    rms = float(lum.std())
    s = 0.5 * min(1.0, rng / 0.75) + 0.5 * min(1.0, rms / 0.2)
    return s, rng


def noise_sigma(lum: np.ndarray) -> float:
    """Immerkaer's fast noise estimate (on 0..1 luma)."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    h, w = lum.shape
    conv = np.abs(ndimage.convolve(lum, k))
    return float(math.sqrt(math.pi / 2) * conv.sum() / (6 * (w - 2) * (h - 2)))


# --------------------------------------------------------------------------
# Artistic
# --------------------------------------------------------------------------


def colourfulness(rgb: np.ndarray) -> tuple[float, float]:
    # light smoothing first so chroma noise doesn't read as colour
    rgb = ndimage.uniform_filter(rgb, size=(3, 3, 1))
    r, g, b = (rgb[..., i] * 255 for i in range(3))
    rg = r - g
    yb = 0.5 * (r + g) - b
    c = math.hypot(float(rg.std()), float(yb.std())) + 0.3 * math.hypot(float(rg.mean()), float(yb.mean()))
    # ~15 dull, ~35 average, ~55+ vivid; past ~110 tends to look garish
    s = min(1.0, c / 55.0)
    if c > 110:
        s -= min(0.3, (c - 110) / 150)
    return max(0.0, s), c


def saliency(lum: np.ndarray) -> np.ndarray:
    """Spectral-residual saliency (Hou & Zhang 2007) at 64 px, returned 0..1."""
    small = np.asarray(Image.fromarray((lum * 255).astype(np.uint8)).resize((64, 64), Image.BILINEAR), np.float32)
    f = np.fft.fft2(small)
    log_amp = np.log(np.abs(f) + 1e-6)
    phase = np.angle(f)
    resid = log_amp - ndimage.uniform_filter(log_amp, 3)
    sal = np.abs(np.fft.ifft2(np.exp(resid + 1j * phase))) ** 2
    sal = ndimage.gaussian_filter(sal, 2.5)
    sal -= sal.min()
    return sal / (sal.max() + 1e-9)


def composition(lum: np.ndarray) -> dict[str, float]:
    sal = saliency(lum)
    h, w = sal.shape
    flat = np.sort(sal.ravel())[::-1]
    top = flat[: max(1, len(flat) // 20)].sum() / (flat.sum() + 1e-9)  # mass in top 5 %
    focus = _sat(top - 0.08, 0.12)  # one clear subject vs scattered interest
    ys, xs = np.mgrid[0:h, 0:w]
    m = sal > np.percentile(sal, 95)
    cy, cx = float(ys[m].mean() / h), float(xs[m].mean() / w)
    thirds = min(math.hypot(cx - a, cy - b) for a in (1 / 3, 2 / 3) for b in (1 / 3, 2 / 3))
    centre = math.hypot(cx - 0.5, cy - 0.5)
    placement = max(1 - thirds / 0.18, 1 - centre / 0.12, 0.0)

    small = np.asarray(Image.fromarray((lum * 255).astype(np.uint8)).resize((128, 96), Image.BILINEAR), np.float32)
    z = small - small.mean()
    den = float((z * z).sum()) + 1e-6
    sym_lr = float((z * z[:, ::-1]).sum()) / den
    sym_tb = float((z * z[::-1, :]).sum()) / den
    symmetry = max(0.0, max(sym_lr, sym_tb))
    return {"focus": focus, "placement": placement, "symmetry": symmetry, "subject_x": cx, "subject_y": cy}


def simplicity(lum: np.ndarray) -> tuple[float, float]:
    gx = ndimage.sobel(lum, 1)
    gy = ndimage.sobel(lum, 0)
    mag = np.hypot(gx, gy)
    busy = ndimage.uniform_filter((mag > 0.25).astype(np.float32), 25)
    calm = float((busy < 0.03).mean())  # fraction of the frame that is calm
    return _sat(calm - 0.15, 0.35), calm


def pattern(lum: np.ndarray) -> tuple[float, float]:
    """Peakiness of the mid-frequency power spectrum: repetition makes sharp peaks."""
    small = np.asarray(Image.fromarray((lum * 255).astype(np.uint8)).resize((256, 192), Image.BILINEAR), np.float32)
    z = (small - small.mean()) * np.outer(np.hanning(192), np.hanning(256))
    p = np.abs(np.fft.fftshift(np.fft.fft2(z))) ** 2
    cy, cx = 96, 128
    yy, xx = np.mgrid[0:192, 0:256]
    r = np.hypot((yy - cy) / 96, (xx - cx) / 128)
    band = p[(r > 0.06) & (r < 0.6)]
    if band.size == 0 or band.mean() <= 0:
        return 0.0, 0.0
    peak = float(np.percentile(band, 99.95) / (np.median(band) + 1e-9))
    return _sat(math.log10(max(peak, 1.0)) - 2.3, 0.6), peak


def horizon_tilt(lum: np.ndarray) -> tuple[float | None, float]:
    """Find a sky/ground horizon and return (tilt_deg, confidence 0..1).

    Tilt is positive when the horizon is rotated clockwise (falls to the
    right). For each column the strongest vertical transition is located, a
    Theil-Sen line is fitted through those points, and the fit is accepted only
    if most columns agree and the region above the line looks like sky
    (smoother and brighter than the ground below).
    """
    h, w = lum.shape
    tex = ndimage.gaussian_filter(np.hypot(ndimage.sobel(lum, 0), ndimage.sobel(lum, 1)), 3.0)
    feats = [ndimage.gaussian_filter(lum, 3.0), tex / (tex.mean() + 1e-6)]
    n = h
    r = np.arange(1, n)
    lo, hi = int(0.05 * n), int(0.9 * n)
    xs, ys = [], []
    for c in np.linspace(0.04 * w, 0.96 * w, 48).astype(int):
        total = np.zeros(n - 1)
        for f in feats:
            col = f[:, max(0, c - 3):c + 4].mean(1)
            cs = np.cumsum(col)[:-1]
            above = cs / r
            below = (col.sum() - cs) / (n - r)
            # between-class separation of a two-region split at each row (Otsu-style)
            sep = (above - below) ** 2 * r * (n - r) / n**2
            total += sep / (sep.max() + 1e-12)
        k = lo + int(total[lo:hi].argmax())
        xs.append(c)
        ys.append(k + 1)
    if len(xs) < 20:
        return None, 0.0
    xs_a, ys_a = np.array(xs, float), np.array(ys, float)
    i, j = np.triu_indices(len(xs_a), 1)
    dx = xs_a[j] - xs_a[i]
    ok = dx > w * 0.1
    slope = float(np.median((ys_a[j] - ys_a[i])[ok] / dx[ok]))
    icpt = float(np.median(ys_a - slope * xs_a))
    inl = np.abs(ys_a - (slope * xs_a + icpt)) < 0.012 * h
    frac = float(inl.mean())
    # sky check: above the line should be calmer and brighter than below
    yy, xx = np.mgrid[0:h, 0:w]
    line = slope * xx + icpt
    above, below = yy < line - 0.02 * h, yy > line + 0.02 * h
    if above.mean() < 0.05 or below.mean() < 0.05:
        return None, 0.0
    calm_sky = float(tex[below].mean()) > 1.5 * float(tex[above].mean())
    bright_sky = float(lum[above].mean()) > float(lum[below].mean())
    if not (calm_sky and bright_sky):
        return None, 0.0
    tilt = math.degrees(math.atan(slope))
    return tilt, max(0.0, min(1.0, (frac - 0.5) / 0.4))


# --------------------------------------------------------------------------
# Overall
# --------------------------------------------------------------------------

WEIGHTS_TECH = {"sharpness": 0.40, "exposure": 0.30, "contrast": 0.20, "noise": 0.10}
WEIGHTS_ART = {"colour": 0.30, "composition": 0.30, "simplicity": 0.20, "pattern": 0.20}


def score_image(rgb: np.ndarray, gimbal_pitch: float | None = None) -> Aesthetics:
    lum = luma(rgb)
    sigma = noise_sigma(lum)
    sharp, sharp_raw = sharpness(lum, sigma)
    expo, mean_l, clip_hi, clip_lo = exposure(lum)
    cont, rng = contrast(lum)
    noise = 1.0 - _sat(sigma - 0.012, 0.02)
    col, col_raw = colourfulness(rgb)
    comp = composition(lum)
    simp, calm = simplicity(lum)
    patt, peak = pattern(lum)

    compo = max(0.55 * comp["placement"] + 0.45 * comp["focus"], comp["symmetry"] * 0.95)
    parts = {
        "sharpness": sharp,
        "exposure": expo,
        "contrast": cont,
        "noise": noise,
        "colour": col,
        "composition": compo,
        "simplicity": simp,
        "pattern": patt,
    }
    tech = sum(parts[k] * w for k, w in WEIGHTS_TECH.items())
    # the strongest artistic quality carries a photo: an abstract pattern shot
    # needn't be colourful, a minimalist one needn't have a pattern
    art_avg = sum(parts[k] * w for k, w in WEIGHTS_ART.items())
    art_best = max(parts[k] for k in WEIGHTS_ART)
    art = 0.6 * art_avg + 0.4 * art_best
    score = 0.45 * tech + 0.55 * art
    # a blurry or badly exposed frame should never outrank a clean one
    gate = min(1.0, 0.35 + sharp) * min(1.0, 0.5 + expo) * min(1.0, 0.6 + noise)
    score *= gate

    tags, warns = [], []
    if sharp > 0.8:
        tags.append("Tack sharp")
    if col > 0.8:
        tags.append("Vivid colour")
    elif col < 0.25:
        tags.append("Muted / monochrome")
    if comp["symmetry"] > 0.6:
        tags.append("Symmetry")
    if simp > 0.6:
        tags.append("Minimal")
    if patt > 0.5:
        tags.append("Pattern")
    if comp["placement"] > 0.7 and comp["focus"] > 0.5:
        tags.append("Strong subject")
    if gimbal_pitch is not None and gimbal_pitch <= -80:
        tags.append("Top-down")
    if sharp < 0.35:
        warns.append("Soft / motion blur")
    if clip_hi > 0.02:
        warns.append("Clipped highlights")
    if clip_lo > 0.04:
        warns.append("Crushed shadows")
    if mean_l < 0.22:
        warns.append("Underexposed")
    elif mean_l > 0.72:
        warns.append("Overexposed")
    if rng < 0.45:
        warns.append("Flat / hazy")
    if noise < 0.5:
        warns.append("Noisy")

    tilt = None
    if gimbal_pitch is None or gimbal_pitch > -60:  # horizon may be in frame
        t, conf = horizon_tilt(lum)
        if t is not None and conf > 0.5 and 0.4 <= abs(t) <= 6:
            tilt = round(t, 2)
            warns.append(f"Horizon tilted {abs(t):.1f}°")

    stats = {
        "sharp_raw": round(sharp_raw, 3),
        "mean_luma": round(mean_l, 3),
        "clip_hi": round(clip_hi, 4),
        "clip_lo": round(clip_lo, 4),
        "range": round(rng, 3),
        "noise_sigma": round(sigma, 4),
        "colourfulness": round(col_raw, 1),
        "calm": round(calm, 3),
        "pattern_peak": round(peak, 1),
        "symmetry": round(comp["symmetry"], 3),
        "subject_x": round(comp["subject_x"], 3),
        "subject_y": round(comp["subject_y"], 3),
    }
    return Aesthetics(
        score=round(100 * score, 1),
        technical=round(tech, 3),
        artistic=round(art, 3),
        parts={k: round(v, 3) for k, v in parts.items()},
        tags=tags,
        warnings=warns,
        stats=stats,
        tilt_deg=tilt,
    )


def dhash(rgb: np.ndarray, size: int = 16) -> int:
    """Difference hash (size*size bits, 256 by default) for spotting near-duplicate frames.

    256 bits rather than the usual 64: low-detail aerial scenes (fields, water)
    all look alike to a 64-bit hash.
    """
    g = Image.fromarray((luma(rgb) * 255).astype(np.uint8)).resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(g, np.int16)
    bits = (a[:, 1:] > a[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()
