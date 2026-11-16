import math

import pytest
from PIL import Image

from isr.camera import photo_footprint, pixel_to_ground, read_photo_meta
from isr.geo import haversine_m, to_local

from synth import mavic3_gsd, save_dji_jpeg

LAT, LON = 39.70, -105.20


def _photo(tmp_path, yaw=0.0, pitch=-90.0, alt=60.0, size=(1320, 989)):
    # small image, same aspect ratio as a Mavic 3 photo; 24 mm eq focal length
    return read_photo_meta(save_dji_jpeg(Image.new("RGB", size, (90, 110, 60)), tmp_path / "p.jpg", LAT, LON, alt,
                                         yaw=yaw, pitch=pitch))


def test_reads_dji_metadata(tmp_path):
    m = _photo(tmp_path, yaw=-35.5, pitch=-60)
    assert m.lat == pytest.approx(LAT, abs=1e-7)
    assert m.lon == pytest.approx(LON, abs=1e-7)
    assert m.rel_alt_m == 60.0
    assert m.gimbal_yaw == -35.5
    assert m.gimbal_pitch == -60.0
    assert m.focal_35mm == 24
    assert m.can_geolocate


def test_gsd_matches_mavic3_spec(tmp_path):
    m = _photo(tmp_path, size=(5280, 3956))
    # spec sheet maths: 17.3 mm * 100 m / (12.29 mm * 5280 px) = 2.67 cm/px at 100 m
    assert m.gsd_m() * 100 / 60 == pytest.approx(2.67 / 100, rel=0.04)
    assert mavic3_gsd(60) == pytest.approx(m.gsd_m(), rel=1e-6)


def test_nadir_centre_is_drone_position(tmp_path):
    m = _photo(tmp_path)
    lat, lon = pixel_to_ground(m, m.width / 2, m.height / 2)
    assert haversine_m(lat, lon, LAT, LON) < 0.01


@pytest.mark.parametrize("yaw, expect", [
    (0, (1, 0)),     # facing north: image right = east
    (90, (0, -1)),   # facing east: image right = south
    (180, (-1, 0)),  # facing south: image right = west
])
def test_nadir_right_offset_direction(tmp_path, yaw, expect):
    m = _photo(tmp_path, yaw=yaw)
    px = 300
    lat, lon = pixel_to_ground(m, m.width / 2 + px, m.height / 2)
    e, n = to_local(LAT, LON, lat, lon)
    dist = px * m.gsd_m()
    assert e == pytest.approx(expect[0] * dist, abs=0.05)
    assert n == pytest.approx(expect[1] * dist, abs=0.05)


def test_image_top_is_forward(tmp_path):
    m = _photo(tmp_path, yaw=0)
    lat, lon = pixel_to_ground(m, m.width / 2, 0)
    e, n = to_local(LAT, LON, lat, lon)
    assert n > 0 and abs(e) < 0.01


def test_oblique_centre_range(tmp_path):
    m = _photo(tmp_path, yaw=90, pitch=-45, alt=50)
    lat, lon = pixel_to_ground(m, m.width / 2, m.height / 2)
    e, n = to_local(LAT, LON, lat, lon)
    # 45 deg down from 50 m -> 50 m ahead, to the east
    assert e == pytest.approx(50, abs=0.1)
    assert abs(n) < 0.1


def test_ground_offset_shrinks_range(tmp_path):
    m = _photo(tmp_path, yaw=0, pitch=-45, alt=50)
    lat, _ = pixel_to_ground(m, m.width / 2, m.height / 2, ground_offset_m=10)  # ground 10 m higher
    _, n = to_local(LAT, LON, lat, LON)
    assert n == pytest.approx(40, abs=0.1)


def test_horizon_pixels_do_not_project(tmp_path):
    m = _photo(tmp_path, pitch=-10)
    assert pixel_to_ground(m, m.width / 2, 0) is None


def test_footprint_size(tmp_path):
    m = _photo(tmp_path, size=(5280, 3956), alt=100)
    fp = photo_footprint(m)
    w = haversine_m(*fp[0], *fp[1])
    h = haversine_m(*fp[1], *fp[2])
    assert w == pytest.approx(141, rel=0.04)
    assert h == pytest.approx(106, rel=0.04)


def test_non_dji_photo(tmp_path):
    p = tmp_path / "plain.jpg"
    Image.new("RGB", (100, 80)).save(p)
    m = read_photo_meta(p)
    assert not m.can_geolocate
    assert m.gsd_m() is None
    assert pixel_to_ground(m, 1, 1) is None
    assert math.isfinite(m.focal_px())
