"""Reference-photo target profiles for search and rescue.

Given a photo of the missing person (or their vehicle or dog), extract what a
drone can actually see from altitude: the dominant colours of the subject and
its physical size class. Faces, licence plates and fine features are not
resolvable from tens of metres up, and this module makes no attempt at
biometric identification.

The profile is used in three ways during triage:
* its distinctive colours become colour targets;
* the size window is set for the target type (person / vehicle / animal);
* each candidate gets an appearance-similarity score (colour histogram match
  with the reference subject) that raises or lowers its rank.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

TARGET_SIZES = {"person": (0.04, 2.5), "vehicle": (2.5, 30.0), "animal": (0.05, 4.0)}
AI_CLASSES = {
    "person": {"person", "backpack", "handbag", "suitcase", "umbrella"},
    "vehicle": {"car", "truck", "bus", "motorcycle", "bicycle", "boat"},
    "animal": {"dog", "horse", "cow", "sheep", "bear", "cat", "bird"},
}
HUE_BINS, SAT_BINS = 18, 3


@dataclass
class ReferenceProfile:
    target: str
    colors: list[dict] = field(default_factory=list)  # {"hex", "name", "share"}
    hist: list[float] = field(default_factory=list)  # hue x saturation histogram of the subject
    notes: list[str] = field(default_factory=list)

    def color_specs(self) -> list[str]:
        return [c["hex"] if c["hex"] != "white" else "white" for c in self.colors]


def _hsv(rgb: np.ndarray):
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = np.maximum(mx - mn, 1e-6)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60.0
    s = np.where(mx > 1e-3, (mx - mn) / np.maximum(mx, 1e-3), 0)
    return h, s, mx


def colour_hist(rgb: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Normalised hue x saturation histogram of chromatic pixels (+1 bin for bright neutrals)."""
    h, s, v = _hsv(rgb.reshape(-1, 3))
    w = np.ones_like(h) if weights is None else weights.reshape(-1)
    chrom = (s >= 0.25) & (v >= 0.15)
    hb = np.minimum((h / 360 * HUE_BINS).astype(int), HUE_BINS - 1)
    sb = np.minimum(((s - 0.25) / 0.75 * SAT_BINS).astype(int), SAT_BINS - 1)
    hist = np.bincount((hb * SAT_BINS + sb)[chrom], weights=w[chrom], minlength=HUE_BINS * SAT_BINS + 1)
    white = (s < 0.12) & (v > 0.85)
    hist[-1] += w[white].sum()
    tot = hist.sum()
    return hist / tot if tot > 0 else hist


def similarity(profile: ReferenceProfile, rgb_patch: np.ndarray) -> float:
    """Bhattacharyya coefficient between the reference subject and a candidate patch (0..1)."""
    if not profile.hist or rgb_patch.size == 0:
        return 0.0
    q = colour_hist(rgb_patch)
    if q.sum() == 0:
        return 0.0
    return float(np.sqrt(np.asarray(profile.hist) * q).sum())


def _name(rgb) -> str:
    from .detectors import describe_rgb
    return describe_rgb(rgb)


def build_profile(path: str | Path, target: str = "person", box: tuple[int, int, int, int] | None = None,
                  save_crop: Path | None = None) -> ReferenceProfile:
    """Extract a colour profile of the subject in a reference photo.

    ``box`` (x0, y0, x1, y1 in the photo's pixels) isolates the subject; without
    it the subject is assumed to fill the middle of the frame and colours that
    dominate the photo's border (background) are down-weighted.
    """
    if target not in TARGET_SIZES:
        raise ValueError(f"target must be one of {', '.join(TARGET_SIZES)}")
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        if box:
            im = im.crop(box)
        im.thumbnail((480, 480), Image.LANCZOS)
        if save_crop:
            im.save(save_crop, quality=88)
        rgb = np.asarray(im, np.float32) / 255.0
    h_, w_ = rgb.shape[:2]
    prof = ReferenceProfile(target=target)

    yy, xx = np.mgrid[0:h_, 0:w_]
    if box:
        weight = np.ones((h_, w_), np.float32)
    else:
        weight = np.exp(-(((xx / w_ - 0.5) / 0.28) ** 2 + ((yy / h_ - 0.5) / 0.32) ** 2) / 2).astype(np.float32)
        # colours that dominate the border are background
        b = max(2, int(min(h_, w_) * 0.07))
        border = np.zeros((h_, w_), bool)
        border[:b], border[-b:], border[:, :b], border[:, -b:] = True, True, True, True
        q = _quant(rgb)
        all_f = np.bincount(q.ravel(), minlength=512) / q.size
        bor_f = np.bincount(q[border], minlength=512) / border.sum()
        ratio = bor_f / np.maximum(all_f, 1e-6)
        weight *= (1.0 / (1.0 + ratio[q] ** 2)).astype(np.float32)
    h, s, v = _hsv(rgb)
    if target == "person":  # skin is not clothing
        skin = (h > 5) & (h < 35) & (s > 0.18) & (s < 0.6) & (v > 0.35)
        weight[skin] *= 0.15

    prof.hist = colour_hist(rgb, weight).round(5).tolist()
    chrom = (s >= 0.3) & (v >= 0.18)
    tot_w = float(weight.sum()) + 1e-9
    if chrom.any():
        feats = np.stack([np.cos(np.radians(h[chrom])) * s[chrom], np.sin(np.radians(h[chrom])) * s[chrom],
                          v[chrom] * 0.5], -1)
        cols = rgb[chrom]
        wts = weight[chrom]
        cent, lab = _kmeans(feats, wts, k=min(4, max(1, int(chrom.sum() // 50))))
        for c in range(len(cent)):
            m = lab == c
            share = float(wts[m].sum()) / tot_w
            if share < 0.06:
                continue
            mean = (cols[m] * wts[m, None]).sum(0) / wts[m].sum()
            hh, ss, vv = colorsys.rgb_to_hsv(*mean)
            if ss < 0.2:
                continue
            hexc = "#%02x%02x%02x" % tuple(int(round(x * 255)) for x in mean)
            prof.colors.append({"hex": hexc, "name": _name(mean), "share": round(share, 3)})
    white = (s < 0.12) & (v > 0.85)
    wshare = float(weight[white].sum()) / tot_w
    if wshare > 0.15:
        prof.colors.append({"hex": "white", "name": "white", "share": round(wshare, 3)})
    dark = v < 0.2
    if float(weight[dark].sum()) / tot_w > 0.3:
        prof.notes.append("subject is largely black/dark: dark colours can't be searched by colour (shadows look the "
                          "same); rely on the anomaly detector and size")
    prof.colors.sort(key=lambda c: -c["share"])
    prof.colors = prof.colors[:3]
    for c in prof.colors:
        if c["name"] in ("green", "yellow") and target == "person":
            prof.notes.append(f"{c['name']} clothing blends with vegetation; expect weaker colour matches")
    if not prof.colors:
        prof.notes.append("no distinctive colours found in the reference; matching will rely on size and anomalies")
    return prof


def _quant(rgb: np.ndarray) -> np.ndarray:
    q = np.minimum((rgb * 8).astype(int), 7)
    return q[..., 0] * 64 + q[..., 1] * 8 + q[..., 2]


def _kmeans(x: np.ndarray, w: np.ndarray, k: int, iters: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    if len(x) > 20000:
        idx = rng.choice(len(x), 20000, replace=False, p=w / w.sum())
        xs, ws = x[idx], np.ones(len(idx))
    else:
        xs, ws = x, w
    # k-means++ style init weighted by distance
    cent = [xs[rng.choice(len(xs), p=ws / ws.sum())]]
    for _ in range(1, k):
        d = np.min([((xs - c) ** 2).sum(1) for c in cent], axis=0) * ws
        if d.sum() <= 0:
            break
        cent.append(xs[rng.choice(len(xs), p=d / d.sum())])
    cent = np.array(cent)
    for _ in range(iters):
        lab = np.argmin(((xs[:, None] - cent[None]) ** 2).sum(-1), 1)
        new = np.array([(xs[lab == c] * ws[lab == c, None]).sum(0) / max(ws[lab == c].sum(), 1e-9)
                        if (lab == c).any() else cent[c] for c in range(len(cent))])
        if np.allclose(new, cent, atol=1e-4):
            break
        cent = new
    lab_full = np.argmin(((x[:, None] - cent[None]) ** 2).sum(-1), 1)
    return cent, lab_full
