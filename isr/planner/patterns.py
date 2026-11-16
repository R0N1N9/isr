"""Flight patterns for mapping and search, computed in local east/north metres.

All functions return lists of (east, north) points relative to an origin; the
mission module turns them into lat/lon waypoints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

EN = tuple[float, float]


@dataclass(frozen=True)
class CameraSpec:
    name: str
    sensor_w_mm: float
    sensor_h_mm: float
    focal_mm: float
    image_w_px: int
    image_h_px: int

    def footprint_m(self, alt_m: float) -> tuple[float, float]:
        """Ground footprint (cross-track width, along-track height) for a nadir photo.

        The image's long side is across the flight line because the aircraft
        faces along the line with the gimbal pointing straight down.
        """
        return self.sensor_w_mm / self.focal_mm * alt_m, self.sensor_h_mm / self.focal_mm * alt_m

    def gsd_cm(self, alt_m: float) -> float:
        return self.sensor_w_mm * alt_m / (self.focal_mm * self.image_w_px) * 100

    def alt_for_gsd(self, gsd_cm: float) -> float:
        return gsd_cm / 100 * self.focal_mm * self.image_w_px / self.sensor_w_mm


CAMERAS = {
    # Hasselblad L2D-20c, 4/3" CMOS, 24 mm equivalent. Mavic 3 / Classic / Pro / Cine main camera.
    "mavic3": CameraSpec("Mavic 3 Hasselblad (main)", 17.3, 13.0, 12.29, 5280, 3956),
    # Mavic 3 Pro 70 mm medium tele, 1/1.3" 48 MP (12 MP binned output shown)
    "mavic3pro-70mm": CameraSpec("Mavic 3 Pro 70 mm tele", 9.6, 7.2, 18.7, 4032, 3024),
    # Mavic 3 Enterprise (mechanical shutter) for reference
    "m3e": CameraSpec("Mavic 3 Enterprise", 17.3, 13.0, 12.29, 5280, 3956),
    # Mavic 3 Thermal wide camera (1/2" 48 MP)
    "m3t-wide": CameraSpec("Mavic 3T wide", 6.4, 4.8, 4.4, 8000, 6000),
}


def _rot(p: EN, ang: float) -> EN:
    c, s = math.cos(ang), math.sin(ang)
    return p[0] * c - p[1] * s, p[0] * s + p[1] * c


def lawnmower(polygon: list[EN], spacing: float, angle_deg: float = 0.0, overshoot: float = 0.0,
              swath: float = 0.0) -> list[list[EN]]:
    """Back-and-forth lines covering ``polygon``.

    ``angle_deg`` is the flight-line direction, clockwise from north (0 = lines
    run north/south). ``swath`` is the ground width each line sees; when given,
    the outermost lines are inset so their swath just reaches the boundary
    (fewer lines for a search). With 0, lines are spread so the outer ones sit
    about half a line spacing inside the boundary. Returns a list of line
    segments in serpentine order;
    concave polygons may produce several segments per scan line.
    """
    if spacing <= 0:
        raise ValueError("line spacing must be positive")
    # rotate the world so flight lines are along +y (north)
    ang = math.radians(angle_deg)
    poly = [_rot(p, ang) for p in polygon]
    xs = [p[0] for p in poly]
    xmin, xmax = min(xs), max(xs)
    width = xmax - xmin
    if swath > 0:
        n = max(1, math.ceil(max(0.0, width - swath) / spacing - 1e-9) + 1)
    else:
        n = max(1, math.ceil(width / spacing - 1e-9))
    # centre the set of lines in the polygon
    start = xmin + (width - (n - 1) * spacing) / 2 if n > 1 else (xmin + xmax) / 2
    eps = 1e-6 * max(1.0, width)
    lines: list[list[EN]] = []
    for i in range(n):
        x = min(max(start + i * spacing, xmin + eps), xmax - eps)  # never exactly on a vertex
        ys = []
        for j in range(len(poly)):
            (x1, y1), (x2, y2) = poly[j], poly[(j + 1) % len(poly)]
            if (x1 <= x < x2) or (x2 <= x < x1):
                t = (x - x1) / (x2 - x1)
                ys.append(y1 + t * (y2 - y1))
        ys.sort()
        segs = [(ys[k], ys[k + 1]) for k in range(0, len(ys) - 1, 2)]
        if i % 2:
            segs = [(b, a) for a, b in reversed(segs)]
        for a, b in segs:
            d = overshoot if b > a else -overshoot
            lines.append([(x, a - d), (x, b + d)])
    return [[_rot(p, -ang) for p in seg] for seg in lines]


def expanding_square(spacing: float, max_radius: float, first_heading_deg: float = 0.0) -> list[EN]:
    """IAMSAR expanding square from the datum (0, 0): legs S, S, 2S, 2S, 3S ..."""
    pts = [(0.0, 0.0)]
    h = math.radians(first_heading_deg)
    x = y = 0.0
    k = 1
    while True:
        for _ in range(2):
            x += math.sin(h) * k * spacing
            y += math.cos(h) * k * spacing
            pts.append((x, y))
            h += math.pi / 2
        if k * spacing / 2 >= max_radius:
            break
        k += 1
    return pts


def sector_search(radius: float, first_heading_deg: float = 0.0, second_pass: bool = True) -> list[EN]:
    """IAMSAR sector search: three 120 deg triangles through the datum.

    With ``second_pass`` the pattern is repeated rotated by 30 deg, giving 6
    radial legs 30 deg apart (good for a small, high-probability area).
    """
    pts = [(0.0, 0.0)]
    passes = [0.0, 30.0] if second_pass else [0.0]
    for off in passes:
        h = first_heading_deg + off
        for _ in range(3):
            a = math.radians(h)
            p1 = (math.sin(a) * radius, math.cos(a) * radius)
            a2 = math.radians(h + 120)
            # equilateral triangle: out, across, back to the datum
            p2 = (p1[0] + math.sin(a2) * radius, p1[1] + math.cos(a2) * radius)
            pts += [p1, p2, (0.0, 0.0)]
            h += 120  # next triangle starts 120 deg on
    return pts


def densify(path: list[EN], step: float) -> list[EN]:
    """Points every ``step`` metres along a polyline (always keeps vertices)."""
    out: list[EN] = [path[0]]
    for (x1, y1), (x2, y2) in zip(path, path[1:]):
        d = math.hypot(x2 - x1, y2 - y1)
        n = max(1, math.ceil(d / step - 1e-9))
        for i in range(1, n + 1):
            t = i / n
            out.append((x1 + (x2 - x1) * t, y1 + (y2 - y1) * t))
    return out


def path_length(path: list[EN]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))
