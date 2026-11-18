"""Made-up photo sets for trying the SAR and Art sections without a drone.

Everything is painted, not photographed, but carries real GPS and gimbal
metadata, so the tools treat it exactly like a card from an aircraft.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .synth import draw_person, mavic3_gsd, save_dji_jpeg, terrain

LAT, LON = 39.7000, -105.2000  # an arbitrary spot; the scenes do not exist
W, H = 2640, 1978


def write_sar_demo(folder: Path) -> list[Path]:
    """Six straight-down frames along a search leg at 50 m. Three of them hold someone.

    A walker in a red jacket standing in the open, a casualty in orange lying
    beside a track, and a third person in blue close to tree cover.
    """
    folder = Path(folder)
    g = mavic3_gsd(50) * 2  # metres per pixel at this image size
    people = {1: [(1750, 620, dict(jacket=(205, 28, 34)))],
              3: [(840, 1310, dict(jacket=(240, 140, 20), pants=(40, 44, 60), lying=True))],
              4: [(1980, 1480, dict(jacket=(36, 84, 200), pants=(60, 60, 64)))]}
    out = []
    for i in range(6):
        im = terrain(W, H, seed=20 + i)
        for x, y, kw in people.get(i, []):
            draw_person(im, x, y, g, **kw)
        out.append(save_dji_jpeg(im, folder / f"DJI_{i + 1:04d}.JPG", LAT + 0.00004 * i, LON + 0.00075 * i, 50,
                                 yaw=90, when=f"2026:11:16 10:0{i}:00"))
    return out


def _grain(a: np.ndarray, rng, amount=0.012) -> np.ndarray:
    return a + rng.normal(0, amount, a.shape[:2] + (1,)).astype(np.float32)


def _img(a: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def _coast(w, h, rng, shift=0):
    """A beach seen from above: pale sand, a foam line, water deepening offshore."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    shore = w * 0.36 + (yy - h / 2) * 0.22 + 26 * np.sin(yy / 95.0) + 12 * np.sin(yy / 31.0 + 1.3) + shift
    d = (xx - shore) / w  # < 0 sand, > 0 sea
    sand = np.array([0.90, 0.82, 0.66], np.float32) - 0.05 * np.clip(-d * 3, 0, 1)[..., None]
    t = np.clip(d * 2.4, 0, 1)[..., None]
    sea = np.array([0.20, 0.74, 0.72], np.float32) * (1 - t) + np.array([0.03, 0.22, 0.42], np.float32) * t
    a = np.where((d > 0)[..., None], sea, sand)
    foam = np.exp(-(d * w / 9.0) ** 2) + 0.5 * np.exp(-((d * w - 46 - 10 * np.sin(yy / 40.0)) / 6.0) ** 2)
    a = a + foam[..., None] * 0.55
    sea_m, sand_m = (d > 0.006)[..., None], (d < -0.006)[..., None]
    a += (0.035 * np.sin(xx / 13.0 + yy / 47.0) + 0.03 * np.sin((xx * 0.8 + yy * 0.6) / 2.6))[..., None] * sea_m  # swell, ripples
    a += (0.03 * np.sin((xx * 0.3 + yy) / 2.2) + 0.02 * np.sin(yy / 9.0 + xx / 70.0))[..., None] * sand_m  # wind ripples
    im = _img(_grain(a, rng))
    dr = ImageDraw.Draw(im)
    spots = np.random.default_rng(5)  # the same beach in both frames of the burst
    for _ in range(26):  # parasols and towels
        y = float(spots.uniform(40, h - 40))
        x = float(w * 0.36 + (y - h / 2) * 0.22 + shift - spots.uniform(40, 330))
        c = [(214, 44, 38), (245, 190, 40), (40, 110, 200), (245, 245, 240)][int(spots.integers(0, 4))]
        r = float(spots.uniform(5, 9))
        dr.ellipse((x - r + 5, y - r + 5, x + r + 5, y + r + 5), fill=(170, 150, 118))
        dr.ellipse((x - r, y - r, x + r, y + r), fill=c)
    return np.asarray(im.filter(ImageFilter.UnsharpMask(radius=1.4, percent=220, threshold=0)), np.float32) / 255


def _fields(w, h, rng):
    """Crop rows in alternating strips, with a farm track cutting across."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = xx * 0.92 + yy * 0.39
    strip = (u // 150).astype(int) % 3
    base = np.array([[0.36, 0.52, 0.20], [0.62, 0.50, 0.26], [0.22, 0.42, 0.18]], np.float32)[strip]
    rows = 0.07 * np.sin(u / 4.2)[..., None]
    a = base + rows
    track = np.abs(yy - (h * 0.62 - xx * 0.18)) < 9
    a[track] = (0.70, 0.62, 0.48)
    return _grain(a, rng, 0.02)


def _boat(w, h, rng):
    """Calm water, one small red boat on a third, a wake behind it."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = np.array([0.05, 0.36, 0.44], np.float32) + (0.05 * (yy / h))[..., None] + 0.012 * np.sin(xx / 60 + yy / 23)[..., None]
    a = a + (0.03 * np.sin((xx + yy * 0.4) / 2.4) * (0.6 + 0.4 * np.sin(xx / 90 + yy / 140)))[..., None]  # fine ripples
    im = _img(_grain(a, rng, 0.006))
    d = ImageDraw.Draw(im)
    bx, by = w * 0.66, h * 0.36
    d.polygon([(bx - 150, by + 190), (bx - 6, by + 8), (bx + 6, by + 14), (bx - 96, by + 215)], fill=(150, 205, 205))
    d.polygon([(bx - 20, by + 22), (bx + 16, by - 26), (bx + 30, by - 16), (bx - 4, by + 32)], fill=(214, 44, 38))
    d.polygon([(bx - 6, by + 6), (bx + 8, by - 12), (bx + 16, by - 6), (bx + 2, by + 12)], fill=(245, 240, 232))
    return np.asarray(im, np.float32) / 255


def _lake(w, h, seed):
    a = np.asarray(terrain(w, h, seed=seed), np.float32) / 255
    yy, xx = np.mgrid[0:h, 0:w]
    lake = ((xx - w * 0.62) / (w * 0.22)) ** 2 + ((yy - h * 0.45) / (h * 0.25)) ** 2 < 1
    a[lake] = a[lake] * 0.25 + np.array([0.05, 0.25, 0.42], np.float32) * 0.75
    a[int(h * .2):int(h * .3), int(w * .2):int(w * .3)] = (0.75, 0.15, 0.1)
    return a


def _horizon(w, h, tilt):
    g = np.asarray(terrain(w + 480, h + 480, seed=3), np.float32) / 255
    hh = g.shape[0]
    y = np.linspace(0, 1, hh)[:, None, None]
    sky = np.array([0.45, 0.62, 0.9], np.float32) * (1 - y * 0.4)
    g[: int(hh * 0.42)] = sky[: int(hh * 0.42)]
    im = _img(g).rotate(tilt, resample=Image.BICUBIC).crop((240, 240, w + 240, h + 240))
    return np.asarray(im, np.float32) / 255


def write_art_demo(folder: Path) -> list[Path]:
    """Ten frames from one imaginary flight: a few keepers, a burst, and the usual rejects.

    Keepers: a coastline, crop rows, a lone boat, a lake. Then a second frame
    of the coastline taken seconds later (so the two are grouped), and the same
    lake blurred, blown out and hazy, plus a tilted horizon to straighten.
    """
    folder = Path(folder)
    rng = np.random.default_rng(7)
    w, h = 1600, 1200
    lake = _lake(w, h, 0)
    frames = [
        ("coast", _coast(w, h, rng), "10:00:00", -90),
        ("coast_again", _coast(w, h, rng, shift=7), "10:00:04", -90),
        ("crop_rows", _fields(w, h, rng), "10:04:00", -90),
        ("lone_boat", _boat(w, h, rng), "10:09:00", -90),
        ("lake", lake, "10:14:00", -90),
        ("lake_blurred", np.asarray(_img(lake).filter(ImageFilter.GaussianBlur(4)), np.float32) / 255, "10:19:00", -90),
        ("lake_overexposed", lake * 2.4, "10:24:00", -90),
        ("lake_hazy", lake * 0.45 + 0.45, "10:29:00", -90),
        ("forest_track", np.asarray(terrain(w, h, seed=11), np.float32) / 255, "10:34:00", -90),
        ("tilted_horizon", _horizon(w, h, 3.0), "10:39:00", -12),
    ]
    out = []
    for i, (name, a, when, pitch) in enumerate(frames):
        out.append(save_dji_jpeg(_img(a), folder / f"DJI_{i + 1:04d}_{name}.JPG", LAT + 0.001 * i, LON, 80,
                                 pitch=pitch, when=f"2026:11:16 {when}"))
    return out
