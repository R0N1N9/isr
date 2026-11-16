import json

import numpy as np
import pytest
from PIL import Image, ImageFilter

from isr.camera import read_photo_meta
from isr.cli import main
from isr.studio.aesthetics import dhash, hamming, horizon_tilt, luma, score_image
from isr.studio.enhance import enhance_file, enhance_image
from isr.studio.gallery import write_gallery
from isr.studio.pipeline import StudioConfig, group_similar, run_studio

from synth import save_dji_jpeg, terrain


def _scene(seed=0, w=1600, h=1200):
    """A reasonably 'photographic' aerial scene: terrain, a lake, a red roof, contrast."""
    im = terrain(w, h, seed=seed)
    a = np.asarray(im, np.float32) / 255
    yy, xx = np.mgrid[0:h, 0:w]
    lake = ((xx - w * 0.62) / (w * 0.22)) ** 2 + ((yy - h * 0.45) / (h * 0.25)) ** 2 < 1
    a[lake] = a[lake] * 0.25 + np.array([0.05, 0.25, 0.42]) * 0.75
    a[int(h * .2):int(h * .3), int(w * .2):int(w * .3)] = (0.75, 0.15, 0.1)
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def _arr(im):
    return np.asarray(im.convert("RGB"), np.float32) / 255


def _small(im, w=1024):
    """Proportional downscale (non-uniform scaling would skew measured angles)."""
    return im.resize((w, round(w * im.size[1] / im.size[0])))


def _horizon(tilt_ccw=0.0, w=2400, h=1800, seed=3):
    g = np.asarray(terrain(w, h, seed=seed), np.float32) / 255
    y = np.linspace(0, 1, h)[:, None, None]
    sky = np.array([0.45, 0.62, 0.9]) * (1 - y * 0.4)
    g[: int(h * 0.42)] = sky[: int(h * 0.42)]
    im = Image.fromarray((g * 255).astype(np.uint8))
    return im.rotate(tilt_ccw, resample=Image.BICUBIC).crop((240, 240, w - 240, h - 240))


# ---------------------------------------------------------------- aesthetics

@pytest.mark.parametrize("kind", ["blur", "over", "under", "haze", "noise"])
def test_degradations_score_lower(kind):
    base = _scene()
    a = _arr(base)
    if kind == "blur":
        d = _arr(base.filter(ImageFilter.GaussianBlur(3)))
    elif kind == "over":
        d = np.clip(a * 2.3, 0, 1)
    elif kind == "under":
        d = a * 0.22
    elif kind == "haze":
        d = a * 0.5 + 0.42
    else:
        d = np.clip(a + np.random.default_rng(1).normal(0, 0.06, a.shape), 0, 1).astype(np.float32)
    assert score_image(d).score < score_image(a).score - 5


def test_warnings_explain_problems():
    a = _arr(_scene())
    assert "Overexposed" in score_image(np.clip(a * 2.6, 0, 1)).warnings or \
        "Clipped highlights" in score_image(np.clip(a * 2.6, 0, 1)).warnings
    assert "Flat / hazy" in score_image(a * 0.4 + 0.45).warnings


def test_dhash_separates_duplicates_from_different_scenes():
    a, b = _scene(0), _scene(5)
    shifted = a.crop((8, 6, a.size[0], a.size[1])).resize(a.size)
    assert hamming(dhash(_arr(a)), dhash(_arr(shifted))) <= 30
    assert hamming(dhash(_arr(a)), dhash(_arr(b))) > 30


@pytest.mark.parametrize("ccw", [3.0, -2.0, 0.8])
def test_horizon_tilt(ccw):
    t, conf = horizon_tilt(luma(_arr(_small(_horizon(ccw)))))
    assert conf > 0.5
    assert t == pytest.approx(-ccw, abs=0.25)


def test_no_horizon_in_top_down_scene():
    _, conf = horizon_tilt(luma(_arr(_small(_scene()))))
    assert conf < 0.5


def test_group_similar_respects_time_window():
    h = "0" * 64
    recs = [{"name": "a", "hash": h, "taken_at": "2026-01-01T10:00:00"},
            {"name": "b", "hash": h, "taken_at": "2026-01-01T10:00:20"},
            {"name": "c", "hash": h, "taken_at": "2026-01-01T11:00:00"},
            {"name": "d", "hash": "f" * 64, "taken_at": "2026-01-01T10:00:05"}]
    groups = sorted(sorted(recs[i]["name"] for i in g) for g in group_similar(recs, 30, 180))
    assert groups == [["a", "b"], ["c"], ["d"]]


# ---------------------------------------------------------------- enhance

def test_enhance_dehazes_without_clipping():
    hazy = Image.fromarray((np.clip(_arr(_scene()) * 0.55 + 0.35, 0, 1) * 255).astype(np.uint8))
    out, rep = enhance_image(hazy, 0.6)
    a, b = luma(_arr(hazy)), luma(_arr(out))
    assert "dehaze" in rep.applied
    assert np.percentile(b, 98) - np.percentile(b, 2) > (np.percentile(a, 98) - np.percentile(a, 2)) * 1.2
    assert (b > 0.995).mean() < 0.005  # no blown highlights introduced
    assert out.size == hazy.size


def test_enhance_is_gentle_at_zero_strength():
    im = _scene()
    out, _ = enhance_image(im, 0.0, sharpen=False)
    assert np.abs(_arr(out) - _arr(im)).mean() < 0.01


def test_enhance_straightens_oblique_shot():
    out, rep = enhance_image(_horizon(2.5), 0.6, gimbal_pitch=-15)
    assert rep.applied.get("straighten") == pytest.approx(-2.5, abs=0.3)
    t, _ = horizon_tilt(luma(_arr(_small(out))))
    assert abs(t) < 0.3
    _, rep2 = enhance_image(_horizon(2.5), 0.6, gimbal_pitch=-90)  # top-down: never rotate
    assert "straighten" not in rep2.applied


def test_enhance_file_keeps_metadata_and_never_overwrites(tmp_path):
    src = save_dji_jpeg(_scene(), tmp_path / "DJI_0001.JPG", 39.7, -105.2, 80, yaw=10, pitch=-90)
    before = src.read_bytes()
    with pytest.raises(ValueError):
        enhance_file(src, src)
    dst = tmp_path / "out" / "DJI_0001_edit.jpg"
    enhance_file(src, dst, 0.6)
    assert src.read_bytes() == before
    m = read_photo_meta(dst)
    assert m.lat == pytest.approx(39.7) and m.lon == pytest.approx(-105.2)
    assert m.rel_alt_m == 80 and m.gimbal_pitch == -90 and m.focal_35mm == 24


# ---------------------------------------------------------------- pipeline / CLI

@pytest.fixture(scope="module")
def card(tmp_path_factory):
    root = tmp_path_factory.mktemp("scard")
    d = root / "DCIM" / "100MEDIA"
    for i in range(4):
        save_dji_jpeg(_scene(i * 7), d / f"DJI_{i:04d}.JPG", 39.7, -105.2, 80, when=f"2026:11:16 10:{i * 10:02d}:00")
    base = _scene(0)
    # a burst of the first scene a few seconds later, plus a blurred bad frame
    save_dji_jpeg(base.crop((6, 4, base.size[0], base.size[1])).resize(base.size), d / "DJI_0100.JPG", 39.7, -105.2, 80,
                  when="2026:11:16 10:00:03")
    save_dji_jpeg(base.filter(ImageFilter.GaussianBlur(4)), d / "DJI_0101.JPG", 39.7, -105.2, 80,
                  when="2026:11:16 10:00:06")
    return root


def test_studio_end_to_end(card, tmp_path):
    out = tmp_path / "studio"
    before = sorted((p.name, p.stat().st_size) for p in card.rglob("*") if p.is_file())
    data = run_studio(card, out, StudioConfig(top=3, workers=2), log=lambda *a: None)
    gallery = write_gallery(data, out)

    s = data["summary"]
    assert s["photos"] == 6 and s["groups"] == 4 and s["top"] == 3 and s["edited"] == 3
    by = {p["name"]: p for p in data["photos"]}
    burst = {by[n]["group"] for n in ("DJI_0000.JPG", "DJI_0100.JPG", "DJI_0101.JPG")}
    assert len(burst) == 1 and by["DJI_0101.JPG"]["group_rank"] == 2  # blurred frame ranks last in its group
    for p in data["photos"]:
        assert (out / p["thumb"]).exists()
        if p.get("pick_rank"):
            assert (out / p["edited"]).exists() and (out / p["preview_edit"]).exists()
    html = gallery.read_text()
    assert "window.DATA = " in html and "Export picks" in html
    assert json.loads((out / "studio.json").read_text())["summary"]["photos"] == 6
    assert sorted((p.name, p.stat().st_size) for p in card.rglob("*") if p.is_file()) == before


def test_cli_enhance_list(card, tmp_path):
    lst = tmp_path / "picks.txt"
    files = sorted(card.rglob("*.JPG"))[:2]
    lst.write_text("\n".join(str(f) for f in files) + "\n")
    assert main(["enhance", "--list", str(lst), "-o", str(tmp_path / "e")]) == 0
    assert sorted(p.name for p in (tmp_path / "e").iterdir()) == sorted(f"{f.stem}_edit.jpg" for f in files)
