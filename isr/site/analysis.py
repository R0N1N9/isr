"""Sightlines, blind spots, camera placement and access-point checks for a site model.

The site is rasterised to a grid (0.5 m by default) holding, per cell, the
height of whatever blocks the view there. Visibility is then a 2.5-D line of
sight: an observer at height ``h_obs`` sees a point at height ``h_tgt`` unless
something between them rises above the straight line joining the two. That is
why a camera at 2.7 m sees over a 1.2 m garden wall but a passer-by's view is
cut by a 1.8 m hedge.

Three kinds of observer are modelled:

* **cameras** (existing and planned): a sector with a bearing, field of view and range;
* **the public way**: points every few metres along each drawn street, at eye height;
* **windows**: each ground-floor window is also somewhere an occupant looks out from.

A *blind spot* is ground inside the boundary that none of them can see.
"""

from __future__ import annotations

import base64
import io
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from ..geo import from_local
from . import model as M

NEG = -1e9


MEASURED = {"id": None, "kind": "measured", "name": "something standing there that is not drawn on the plan"}
MIN_MEASURED_M = 0.6  # lower than this in the elevation model is kerbs, grass and noise
_grid_cache: dict = {}


def measured_grid(project, terrain: dict, xmin, ymax, cell, nx, ny):
    """Measured heights (metres above ground) on the analysis grid: the tallest thing in each cell."""
    from pathlib import Path
    from .terrain import load_heights

    p = Path(project) / terrain["file"]
    if not p.exists():
        return None
    key = (str(p), p.stat().st_mtime_ns, round(xmin, 3), round(ymax, 3), round(cell, 4), nx, ny)
    if key not in _grid_cache:
        h, _ = load_heights(project, terrain)
        bx0, _, _, by1 = terrain["bounds"]
        ppm = terrain["px_per_m"]
        jj, ii = np.nonzero(h >= MIN_MEASURED_M)
        gi = np.floor((bx0 + (ii + 0.5) / ppm - xmin) / cell).astype(np.int64)
        gj = np.floor((ymax - (by1 - (jj + 0.5) / ppm)) / cell).astype(np.int64)
        ok = (gi >= 0) & (gi < nx) & (gj >= 0) & (gj < ny)
        out = np.zeros((ny, nx), np.float32)
        np.maximum.at(out, (gj[ok], gi[ok]), h[jj[ok], ii[ok]])
        _grid_cache.clear()
        _grid_cache[key] = out
    return _grid_cache[key]


class Scene:
    """Rasterised site."""

    def __init__(self, model: dict, project=None, use_measured: bool = True):
        self.model = model
        self.s = model["settings"]
        F = model["features"]
        self.by_kind = {k: [f for f in F if f["kind"] == k] for k in M.GEOMETRY}
        self.buildings = self.by_kind["building"]
        self.boundaries = self.by_kind["boundary"]

        # extent: the property plus a margin wide enough to take in the street outside it
        pts = [p for f in (self.boundaries or [f for f in F if f["kind"] != "street"]) for p in f["pts"]]
        if pts:
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            m = 15.0 if self.boundaries else 10.0
            ext = [min(xs) - m, min(ys) - m, max(xs) + m, max(ys) + m]
        else:
            ext = list(model["bounds"])
        self.xmin, self.ymin, self.xmax, self.ymax = ext
        w, h = self.xmax - self.xmin, self.ymax - self.ymin
        self.cell = max(self.s["cell_m"], max(w, h) / 700.0)
        self.nx, self.ny = max(2, int(math.ceil(w / self.cell))), max(2, int(math.ceil(h / self.cell)))
        self.xmax, self.ymin = self.xmin + self.nx * self.cell, self.ymax - self.ny * self.cell
        self.cx = self.xmin + (np.arange(self.nx) + 0.5) * self.cell  # cell-centre coordinates
        self.cy = self.ymax - (np.arange(self.ny) + 0.5) * self.cell

        shape = (self.ny, self.nx)
        self.solid = np.zeros(shape, bool)  # cannot stand here: buildings and barriers
        self.building_mask = np.zeros(shape, bool)
        self.opaque = []  # (feature, mask, height) of everything that blocks sight
        for f in self.buildings:
            mk = self._poly(f["pts"])
            self.solid |= mk
            self.building_mask |= mk
            self.opaque.append((f, mk, f["height"]))
        for f in self.by_kind["barrier"]:
            mk = self._line(f["pts"])
            self.solid |= mk
            if M.BARRIER_TYPES[f["subtype"]]["opaque"]:
                self.opaque.append((f, mk, f["height"]))
        for f in self.by_kind["vegetation"]:
            self.opaque.append((f, self._poly(f["pts"]), f["height"]))
        t = model.get("terrain")
        self.Hm = measured_grid(project, t, self.xmin, self.ymax, self.cell, self.nx, self.ny) \
            if t and project and use_measured and model["settings"].get("use_measured", 1) else None
        self.H, self.fid = self.heights()

        if self.boundaries:
            self.inside = np.zeros(shape, bool)
            for f in self.boundaries:
                self.inside |= self._poly(f["pts"])
        else:
            self.inside = np.ones(shape, bool)
        self.domain = self.inside & ~self.solid
        self.area_m2 = float(self.domain.sum()) * self.cell ** 2

    # -- rasterising ---------------------------------------------------------

    def _px(self, pts):
        return [((x - self.xmin) / self.cell - 0.5, (self.ymax - y) / self.cell - 0.5) for x, y in pts]

    def _poly(self, pts) -> np.ndarray:
        im = Image.new("L", (self.nx, self.ny), 0)
        ImageDraw.Draw(im).polygon(self._px(pts), fill=1, outline=1)
        return np.asarray(im, bool)

    def _line(self, pts) -> np.ndarray:
        # two cells wide so a diagonal fence has no gaps a sightline could slip through
        im = Image.new("L", (self.nx, self.ny), 0)
        ImageDraw.Draw(im).line(self._px(pts), fill=1, width=2)
        return np.asarray(im, bool)

    def heights(self, without: dict | None = None):
        """Blocking-height raster and the index (into ``self.opaque``) of the feature responsible."""
        H = np.zeros((self.ny, self.nx), np.float32)
        fid = np.full((self.ny, self.nx), -1, np.int16)
        for i, (f, mk, h) in enumerate(self.opaque):
            if f is without:
                continue
            upd = mk & (h > H)
            H[upd] = h
            fid[upd] = i
        if self.Hm is not None:  # what the elevation model shows standing there, drawn or not
            hm = self.Hm
            if without is not None:  # asking "what if this were gone" removes its measured bulk too
                gone = next((ndimage.binary_dilation(mk, iterations=2) for f, mk, _ in self.opaque if f is without), None)
                hm = np.where(gone, 0, hm) if gone is not None else hm
            upd = hm > H
            fid[upd & (fid < 0)] = -2
            H = np.maximum(H, hm)
        return H, fid

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        return int((self.ymax - y) // self.cell), int((x - self.xmin) // self.cell)

    def in_grid(self, x: float, y: float) -> bool:
        return self.xmin <= x < self.xmax and self.ymin < y <= self.ymax

    def latlon(self, x: float, y: float) -> tuple[float, float]:
        o = self.model["origin"]
        lat, lon = from_local(o["lat"], o["lon"], x, y)
        return round(lat, 7), round(lon, 7)

    # -- visibility ----------------------------------------------------------

    def viewshed(self, x, y, h_obs, max_r, h_tgt=None, bearing=None, hfov=360.0, H=None, raw=False):
        """Cells visible from (x, y).

        Rays are marched outward in polar coordinates; along each ray the
        steepest slope to any obstruction so far is tracked, and a target is
        visible if the slope up (or down) to it is at least that steep.
        Returns a full-grid boolean mask, or with ``raw=True`` the window
        ``(rows, cols, mask, bearing_deg)`` so callers can cut their own sectors.
        """
        H = self.H if H is None else H
        h_tgt = self.s["target_height_m"] if h_tgt is None else h_tgt
        cell = self.cell
        step = cell * 0.5
        nr = max(2, int(max_r / step))
        r = (np.arange(nr) + 1) * step
        na = max(120, int(math.ceil(2 * math.pi * max_r / cell * 1.4)))
        th = np.arange(na) * (2 * math.pi / na)
        ci = np.floor((x + np.outer(np.sin(th), r) - self.xmin) / cell).astype(np.int32)
        rj = np.floor((self.ymax - (y + np.outer(np.cos(th), r))) / cell).astype(np.int32)
        ok = (ci >= 0) & (ci < self.nx) & (rj >= 0) & (rj < self.ny)
        hs = np.zeros((na, nr), np.float32)
        hs[ok] = H[rj[ok], ci[ok]]
        # ignore what the observer is standing on or fixed to; measured roofs overhang the walls drawn under them
        near = cell * 1.0 if self.Hm is None else max(cell, 1.5)
        slope = (hs - h_obs) / r
        slope[:, r < near] = NEG
        cm = np.maximum.accumulate(slope, axis=1)
        before = np.concatenate([np.full((na, 1), NEG, np.float32), cm[:, :-1]], axis=1)
        vis = before <= (h_tgt - h_obs) / r + 1e-6

        j0, i0 = self.cell_of(x - max_r, y + max_r)
        j1, i1 = self.cell_of(x + max_r, y - max_r)
        i0, i1 = max(0, i0), min(self.nx, i1 + 1)
        j0, j1 = max(0, j0), min(self.ny, j1 + 1)
        rows, cols = slice(j0, max(j0, j1)), slice(i0, max(i0, i1))
        dx = self.cx[cols][None, :] - x
        dy = self.cy[rows][:, None] - y
        rr = np.hypot(dx, dy)
        ang = np.arctan2(dx, dy) % (2 * math.pi)
        k = np.clip(np.rint(rr / step).astype(np.int32) - 1, 0, nr - 1)
        a = np.rint(ang / (2 * math.pi / na)).astype(np.int32) % na
        m = vis[a, k] & (rr <= max_r)
        m[rr < near] = True
        deg = np.degrees(ang)
        if bearing is not None and hfov < 360:
            m &= np.abs((deg - bearing + 180) % 360 - 180) <= hfov / 2
        if raw:
            return rows, cols, m, deg
        out = np.zeros((self.ny, self.nx), bool)
        out[rows, cols] = m
        return out

    def first_blocker(self, x0, y0, h0, x1, y1, h1):
        """The feature that first cuts the line of sight from point 0 to point 1, or None."""
        D = math.hypot(x1 - x0, y1 - y0)
        n = max(2, int(D / (self.cell * 0.5)))
        for i in range(1, n):
            t = i / n
            if t * D < self.cell or (1 - t) * D < self.cell:
                continue
            x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            if not self.in_grid(x, y):
                continue
            j, c = self.cell_of(x, y)
            if 0 <= j < self.ny and 0 <= c < self.nx and self.H[j, c] > h0 + (h1 - h0) * t:
                if self.fid[j, c] >= 0:
                    return self.opaque[self.fid[j, c]][0]
                # measured bulk: if it is the edge or overhang of something drawn, name that
                near = self.fid[max(0, j - 4): j + 5, max(0, c - 4): c + 5]
                near = near[near >= 0]
                return self.opaque[int(np.bincount(near).argmax())][0] if near.size else MEASURED
        return None

    # -- placing things relative to walls -------------------------------------

    def off_wall(self, x, y, dist, snap=1.5):
        """If (x, y) is on or near a building wall, the point ``dist`` metres out from that wall.

        Doors, windows and wall-mounted cameras are drawn *on* the wall line; the
        analysis needs the spot just outside it. Returns (x, y, building or None).
        """
        best = None
        for b in self.buildings:
            d, px, py, i = M.closest_on_path(x, y, b["pts"], True)
            inside = M.point_in_poly(x, y, b["pts"])
            if (inside or d <= snap) and (best is None or d < best[0]):
                best = (d, px, py, i, b)
        if best is None:
            return x, y, None
        d, px, py, i, b = best
        pts = b["pts"]
        n = len(pts)
        inside = M.point_in_poly(x, y, pts)
        corner = next((k for k in (i, (i + 1) % n) if math.hypot(px - pts[k][0], py - pts[k][1]) < 0.15), None)
        if not inside and d > 0.1:  # already drawn clear of the wall: keep its direction
            nx, ny = (x - px) / d, (y - py) / d
        elif corner is not None:  # on a corner: straight out along the diagonal
            a, c = _outward(pts, (corner - 1) % n), _outward(pts, corner)
            L = math.hypot(a[0] + c[0], a[1] + c[1])
            nx, ny = ((a[0] + c[0]) / L, (a[1] + c[1]) / L) if L > 0.3 else c
        else:
            nx, ny = _outward(pts, i)
        return px + nx * dist, py + ny * dist, b

    def disc(self, x, y, radius) -> np.ndarray:
        """Standable cells within ``radius`` of a point (falls back to a wider disc if empty)."""
        d = np.hypot(self.cx[None, :] - x, self.cy[:, None] - y)
        for ok in (~self.solid & (self.H < 0.3), ~self.solid):  # prefer open ground to the inside of a shrub
            for r in (radius, radius * 1.8, radius * 3):
                m = (d <= r) & ok
                if m.any():
                    return m
        return m


def _outward(pts, i):
    """Unit normal of ring segment ``i`` pointing out of the polygon."""
    ax, ay = pts[i]
    bx, by = pts[(i + 1) % len(pts)]
    dx, dy = bx - ax, by - ay
    L = math.hypot(dx, dy) or 1.0
    nx, ny = dy / L, -dx / L
    mx, my = (ax + bx) / 2, (ay + by) / 2
    if M.point_in_poly(mx + nx * 0.05, my + ny * 0.05, pts):
        nx, ny = -nx, -ny
    return nx, ny


def _samples(pts, spacing):
    out = []
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        L = math.hypot(bx - ax, by - ay)
        n = max(1, int(math.ceil(L / spacing)))
        out += [(ax + (bx - ax) * i / n, ay + (by - ay) * i / n) for i in range(n)]
    out.append(tuple(pts[-1]))
    return out


def _dist_to_segments(P: np.ndarray, segs: np.ndarray) -> np.ndarray:
    """Distance from each point (n,2) to the nearest of the segments (m,4)."""
    if not len(segs):
        return np.full(len(P), np.inf)
    a, b = segs[:, :2], segs[:, 2:]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-12)
    t = np.clip(((P[:, None, :] - a[None]) * ab[None]).sum(2) / L2[None], 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    return np.sqrt(((P[:, None, :] - proj) ** 2).sum(2)).min(1)


# ---------------------------------------------------------------------------
# the assessment
# ---------------------------------------------------------------------------

STAND = 1.0  # how far out from a wall someone at a door or window stands, metres
COVERED = 0.35  # share of the ground at an opening that must be seen for it to count as watched


class Analysis:
    """Runs every check. ``result`` is plain JSON; ``layers`` holds the rasters behind the map overlays."""

    def __init__(self, model: dict, project=None, use_measured: bool = True):
        M.normalise(model)
        self.model = model
        self.sc = sc = Scene(model, project, use_measured)
        self.s = s = model["settings"]
        self.cams = sc.by_kind["camera"]
        self.access = sc.by_kind["access"]

        # observers
        self.cam_masks = {c["id"]: self._camera_mask(c) for c in self.cams}
        zero = np.zeros((sc.ny, sc.nx), bool)
        self.cam_now = _union([self.cam_masks[c["id"]] for c in self.cams if c["status"] == "existing"], zero)
        self.cam_plan = _union(list(self.cam_masks.values()), zero)
        self.street_pts = [p for f in sc.by_kind["street"] for p in _samples(f["pts"], 3.0) if sc.in_grid(*p)]
        self.street = self._street_mask(sc.H)
        self.windows = {}
        for a in self.access:
            if a["subtype"] == "window":
                x, y, b = sc.off_wall(*a["pts"][0], dist=sc.cell * 1.5)
                if b is not None:
                    self.windows[a["id"]] = sc.viewshed(x, y, s["eye_height_m"], s["window_range_m"])
        self.window_all = _union(list(self.windows.values()), zero)
        self.natural = self.street | self.window_all
        dom = sc.domain
        self.blind = dom & ~self.cam_now & ~self.natural

        self.proposals, self.prop_masks = [], []
        self.urgency = {a["id"]: a["score"] for a in self._access()}  # scored before any recommendation
        self.proposals, self.prop_masks = self._recommend()
        self.cam_final = _union(self.prop_masks, self.cam_plan)
        self.vantage_masks = {v["id"]: sc.viewshed(v["pts"][0][0], v["pts"][0][1], v["height"], v["range"], h_tgt=1.2)
                              for v in sc.by_kind["vantage"]}
        self.layers = {
            "vantage": _union(list(self.vantage_masks.values()), zero) & dom,
            "cameras": self.cam_plan & dom,
            "cameras_now": self.cam_now & dom,
            "natural": self.natural & dom,
            "hidden": dom & ~self.natural,
            "blind": self.blind,
            "proposed": self.cam_final & dom & ~self.cam_plan,
            "blind_plan": dom & ~self.cam_plan & ~self.natural,
            "blind_after": dom & ~self.cam_final & ~self.natural,
        }

        access = self._access()
        blind = self._blind_spots(access)
        perimeter = self._perimeter()
        blockers = self._blockers(access)
        A = max(sc.area_m2, 1e-9)
        c2 = sc.cell ** 2
        pct = lambda m: round(100.0 * float((m & dom).sum()) * c2 / A, 1)  # noqa: E731
        stats = {
            "area_m2": round(sc.area_m2, 0),
            "camera_pct_now": pct(self.cam_now), "camera_pct_planned": pct(self.cam_plan),
            "camera_pct_recommended": pct(self.cam_final),
            "street_pct": pct(self.street), "natural_pct": pct(self.natural),
            "blind_pct": pct(self.blind), "blind_m2": round(float(self.blind.sum()) * c2, 0),
            "blind_pct_recommended": pct(dom & ~self.cam_final & ~self.natural),
            "cameras_existing": sum(c["status"] == "existing" for c in self.cams),
            "cameras_planned": sum(c["status"] == "planned" for c in self.cams),
            "cameras_recommended": len(self.proposals),
            "access_total": len(access),
            "access_high": sum(a["priority"] == "high" for a in access),
            "access_medium": sum(a["priority"] == "medium" for a in access),
            "access_unwatched": sum(not a["camera_now"] for a in access),
            "has_boundary": bool(sc.boundaries), "has_street": bool(self.street_pts),
            "has_buildings": bool(sc.buildings), "cell_m": round(sc.cell, 2),
            "measured": sc.Hm is not None, "has_terrain": bool(model.get("terrain")),
        }
        cameras = []
        for c in self.cams:
            mk = self.cam_masks[c["id"]]
            lat, lon = sc.latlon(*c["pts"][0])
            cameras.append({"id": c["id"], "code": f"C{len(cameras) + 1}", "name": c["name"], "status": c["status"], "x": c["pts"][0][0],
                            "y": c["pts"][0][1], "lat": lat, "lon": lon, "height": c["height"],
                            "bearing": round(c["bearing"]), "hfov": round(c["hfov"]), "range": c["range"],
                            "covers_m2": round(float((mk & dom).sum()) * c2, 0),
                            "covers": [a["name"] for a in access if c["id"] in a["camera_ids"]],
                            "notes": c["notes"]})
        self.result = {
            "stats": stats, "access": access, "blind_spots": blind, "cameras": cameras,
            "recommended": self.proposals, "perimeter": perimeter, "blockers": blockers,
            "grid": {"bounds": [sc.xmin, sc.ymin, sc.xmax, sc.ymax], "cell_m": sc.cell},
        }
        self.result["detections"] = self._detections(access)
        vant = []
        for v in sc.by_kind["vantage"]:
            mk = self.vantage_masks[v["id"]] & dom
            x, y = v["pts"][0]
            lat, lon = sc.latlon(x, y)
            sees = [a["name"] for a in access if _frac(mk, self._stand_disc_id(a["id"])) >= COVERED]
            vant.append({"id": v["id"], "code": f"V{len(vant) + 1}", "name": v["name"], "x": x, "y": y, "lat": lat,
                         "lon": lon, "height": v["height"], "range": v["range"], "inside": bool(sc.in_grid(x, y) and
                         sc.inside[sc.cell_of(x, y)]), "sees_pct": round(100 * float(mk.sum()) * c2 / A),
                         "sees_blind_m2": round(float((mk & self.blind).sum()) * c2),
                         "sees": sees, "misses": [a["name"] for a in access if a["name"] not in sees],
                         "notes": v["notes"]})
        self.result["vantages"] = vant
        self.result["activity"] = self._activity(access)
        self.result["measured_heights"] = {}
        if sc.Hm is not None:  # what the elevation model says each drawn thing stands at
            for f, mk, _ in sc.opaque:
                v = sc.Hm[mk & (sc.Hm >= MIN_MEASURED_M)]
                if v.size >= 3:
                    self.result["measured_heights"][f["id"]] = round(float(np.percentile(v, 75)), 1)
        self.result["findings"] = self._findings()
        from .guards import plan as guard_plan

        self.result["guards"] = guard_plan(self)
        if self.result["guards"]:
            posts = self.result["guards"]["posts"]
            timing = []
            for w in self.result["activity"]:
                post = next((p["id"] for p in posts if w["access_name"] in p["sees"]), None)
                seen = f" Post {post} sees it." if post else " No recommended post sees it, so this falls to the patrol."
                if w["busy"]:
                    timing.append(f"{w['label']}: busiest {w['busy']}.{seen} That is when a guard there does most good.")
                if w["out_of_hours"]:
                    timing.append(f"{w['label']}: {w['out_of_hours']} movement{'s' if w['out_of_hours'] != 1 else ''} "
                                  f"while closed, most often around {w['ooh_hour']:02d}:00. Time a patrol pass for then.")
                if w["quiet"]:
                    timing.append(f"{w['label']}: nothing moves {w['quiet']}. Keep patrols going through it but expect "
                                  "any movement then to be worth a challenge.")
            self.result["guards"]["timing"] = timing

    # -- observers -----------------------------------------------------------

    def _camera_point(self, c):
        x, y = c["pts"][0]
        px, py, b = self.sc.off_wall(x, y, dist=self.sc.cell * 1.5, snap=1.0)
        return px, py, b

    def _camera_mask(self, c):
        x, y, _ = self._camera_point(c)
        return self.sc.viewshed(x, y, c["height"], c["range"], bearing=c["bearing"], hfov=c["hfov"])

    def _street_mask(self, H, every=1):
        s, sc = self.s, self.sc
        out = np.zeros((sc.ny, sc.nx), bool)
        for x, y in self.street_pts[::every]:
            out |= sc.viewshed(x, y, s["eye_height_m"], s["street_range_m"], h_tgt=1.2, H=H)
        return out

    # -- camera placement ----------------------------------------------------

    def _candidates(self):
        """Places a camera could reasonably be fixed: building corners and along walls.

        With no buildings drawn, boundary corners stand in (a post or fence mount).
        """
        sc = self.sc
        off = sc.cell * 1.5
        out = []
        for b in sc.buildings:
            pts = b["pts"]
            n = len(pts)
            cx, cy = M.centroid(pts)
            for i in range(n):
                x, y, _ = sc.off_wall(*pts[i], dist=off)
                side = M.compass(M.bearing_deg(pts[i][0] - cx, pts[i][1] - cy))
                out.append((x, y, f"{b['name']}, {side} corner", pts[i]))
                n1 = _outward(pts, i)
                ax, ay = pts[i]
                ex, ey = pts[(i + 1) % n]
                k = int(math.hypot(ex - ax, ey - ay) // 5.0)
                for j in range(1, k + 1):
                    t = j / (k + 1)
                    wx, wy = ax + (ex - ax) * t, ay + (ey - ay) * t
                    side = M.compass(M.bearing_deg(n1[0], n1[1]))
                    out.append((wx + n1[0] * off, wy + n1[1] * off, f"{b['name']}, {side} wall", (wx, wy)))
        if not sc.buildings:
            for f in sc.boundaries:
                cx, cy = M.centroid(f["pts"])
                for p in f["pts"]:
                    d = math.hypot(cx - p[0], cy - p[1]) or 1.0
                    x, y = p[0] + (cx - p[0]) / d * 1.0, p[1] + (cy - p[1]) / d * 1.0
                    side = M.compass(M.bearing_deg(p[0] - cx, p[1] - cy))
                    out.append((x, y, f"Post at the {side} boundary corner", (x, y)))
        good = []
        for x, y, where, shown in out:
            if not sc.in_grid(x, y):
                continue
            j, i = sc.cell_of(x, y)
            if sc.inside[j, i] and not sc.building_mask[j, i]:
                good.append((x, y, where, shown))
        return good

    def _weights(self):
        """How much it is worth to put each cell on camera."""
        sc = self.sc
        c2 = sc.cell ** 2
        W = np.where(sc.domain, c2, 0.0).astype(np.float32)  # in m2: plain ground counts its own area
        W[sc.domain & ~self.natural] += c2  # nobody can see it at all: counts double
        if sc.boundaries:
            edge = sc.inside & ~ndimage.binary_erosion(sc.inside, iterations=max(1, int(round(2.0 / sc.cell))))
            W[edge & sc.domain] += 0.5 * c2
        for a in self.access:  # watching a badly exposed door is worth as much as several hundred m2 of lawn
            d = self._stand_disc(a, 2.0) & sc.domain
            n = int(d.sum())
            if n:
                W[d] += 60.0 * (1 + self.urgency.get(a["id"], 0)) / n
        return W

    def _stand_disc(self, a, radius=1.5):
        x, y, _ = self.sc.off_wall(*a["pts"][0], dist=STAND)
        return self.sc.disc(x, y, radius)

    def _recommend(self):
        sc, s = self.sc, self.s
        n_max = s["max_new_cameras"]
        if n_max <= 0 or not sc.domain.any():
            return [], []
        cands = []
        for x, y, where, shown in self._candidates():
            rows, cols, m, deg = sc.viewshed(x, y, s["camera_height_m"], s["camera_range_m"], raw=True)
            m = m & sc.domain[rows, cols]
            if m.any():
                cands.append((x, y, where, shown, rows, cols, m, np.rint(deg).astype(np.int32) % 360))
        if not cands:
            return [], []
        W = self._weights()
        total = float(W.sum()) or 1.0
        covered = self.cam_plan.copy()
        half = int(round(min(s["camera_hfov"], 360) / 2))
        kernel = np.ones(min(360, 2 * half + 1), np.float32)
        access_discs = [(a, self._stand_disc(a)) for a in self.access]
        props, masks = [], []
        for _ in range(n_max):
            best = None
            for ci, (x, y, where, shown, rows, cols, m, deg) in enumerate(cands):
                w = np.where(m & ~covered[rows, cols], W[rows, cols], 0.0)
                bins = np.bincount(deg.ravel(), weights=w.ravel(), minlength=360)[:360]
                if not bins.any():
                    continue
                ext = np.concatenate([bins[-half:], bins, bins[:half]]) if half else bins
                gains = np.convolve(ext, kernel, mode="valid")[:360] if half < 180 else np.full(360, bins.sum())
                az = int(np.argmax(gains))
                if best is None or gains[az] > best[0]:
                    best = (float(gains[az]), ci, az)
            if best is None or best[0] < 0.02 * total:
                break
            gain, ci, az = best
            x, y, where, shown, rows, cols, m, deg = cands[ci]
            sector = m & (np.abs((deg - az + 180) % 360 - 180) <= s["camera_hfov"] / 2)
            full = np.zeros((sc.ny, sc.nx), bool)
            full[rows, cols] = sector
            new = full & sc.domain & ~covered
            covered |= full
            masks.append(full)
            lat, lon = sc.latlon(*shown)
            props.append({
                "id": f"R{len(props) + 1}", "x": round(shown[0], 2), "y": round(shown[1], 2), "lat": lat, "lon": lon,
                "mount": where, "height": s["camera_height_m"], "bearing": az, "aim": M.compass(az),
                "hfov": round(s["camera_hfov"]), "range": s["camera_range_m"],
                "adds_m2": round(float(new.sum()) * sc.cell ** 2, 0),
                "adds_pct": round(100.0 * float(new.sum()) * sc.cell ** 2 / max(sc.area_m2, 1e-9), 1),
                "covers": [a["name"] for a, d in access_discs if _frac(full, d) >= COVERED
                           and _frac(self.cam_plan, d) < COVERED],
            })
            cands.pop(ci)
        return props, masks

    # -- access points -------------------------------------------------------

    def _access(self):
        sc, s = self.sc, self.s
        out = []
        for a in self.access:
            x0, y0 = a["pts"][0]
            x, y, b = sc.off_wall(x0, y0, dist=STAND)
            d = sc.disc(x, y, 1.5)
            cam_ids = [c["id"] for c in self.cams if _frac(self.cam_masks[c["id"]], d) >= COVERED]
            cam_now = any(c["status"] == "existing" and c["id"] in cam_ids for c in self.cams)
            cam_planned = bool(cam_ids)
            rec = [p["id"] for p, m in zip(self.proposals, self.prop_masks) if _frac(m, d) >= COVERED]
            street = _frac(self.street, d) >= COVERED
            others = _union([m for k, m in self.windows.items() if k != a["id"]], np.zeros_like(sc.solid))
            window = _frac(others, d) >= COVERED
            near = sc.disc(x, y, 8.0) & sc.domain
            conceal = float((self.blind & near).sum()) / max(1, int(near.sum()))
            blocker, removable = None, False
            if self.street_pts and not street:
                sx, sy = min(self.street_pts, key=lambda p: math.hypot(p[0] - x, p[1] - y))
                f = sc.first_blocker(sx, sy, s["eye_height_m"], x, y, 1.2)
                blocker = f["name"] if f else None
                removable = bool(f) and f["kind"] != "building"

            weight = M.ACCESS_TYPES[a["subtype"]][1]
            score, why = 0, []
            if not cam_now:
                score += weight - 1
                why.append("no camera covers it")
            if not street and not window:
                score += 2
                why.append("cannot be seen from the street or from a window" if self.street_pts or self.windows
                           else "no natural surveillance recorded")
            if conceal >= 0.5:
                score += 2
                why.append(f"{conceal:.0%} of the ground within 8 m is unobserved")
            elif conceal >= 0.25:
                score += 1
                why.append(f"{conceal:.0%} of the ground within 8 m is unobserved")
            priority = "high" if score >= 5 else "medium" if score >= 3 else "low"

            if cam_now:
                action = "Covered by an existing camera."
            elif cam_planned:
                action = "Will be covered once the planned camera is installed."
            elif rec:
                action = f"Install recommended camera {rec[0]} to cover it."
            else:
                action = ("None of the recommended cameras covers it: raise the camera limit, or add a camera, "
                          "sensor or light here.")
            if removable and not cam_now:
                action += f" {blocker} blocks the view from the street; lowering or thinning it would restore it."
            lat, lon = sc.latlon(x0, y0)
            out.append({
                "id": a["id"], "name": a["name"], "type": M.ACCESS_TYPES[a["subtype"]][0], "subtype": a["subtype"],
                "x": x0, "y": y0, "lat": lat, "lon": lon, "on": b["name"] if b else None,
                "camera_now": cam_now, "camera_planned": cam_planned, "camera_ids": cam_ids,
                "cameras": [c["name"] for c in self.cams if c["id"] in cam_ids], "recommended": rec,
                "street": street, "window": window, "street_blocker": blocker,
                "concealment": round(conceal, 2), "score": score, "priority": priority, "why": why,
                "action": action, "notes": a["notes"],
            })
        order = {"high": 0, "medium": 1, "low": 2}
        out.sort(key=lambda r: (order[r["priority"]], -r["score"], r["name"]))
        for i, r in enumerate(out):
            r["code"] = f"A{i + 1}"
            sx, sy, _ = sc.off_wall(r["x"], r["y"], dist=STAND)
            r["photos"] = self.photos_of(sx, sy)
        return out

    def photos_of(self, x, y, limit=3):
        """Survey photos that look at a point: inside the view cone, in range, and not behind something."""
        sc, hits = self.sc, []
        for p in self.model["photos"]:
            if p.get("x") is None or p["kind"] == "nadir":
                continue
            d = math.hypot(x - p["x"], y - p["y"])
            ground = p["kind"] == "ground"
            if d > (40 if ground else 200):
                continue
            if p.get("heading") is None:
                if d > 12:
                    continue
            elif d > 1.5 and abs((M.bearing_deg(x - p["x"], y - p["y"]) - p["heading"] + 180) % 360 - 180) \
                    > (p.get("hfov") or 65) / 2:
                continue
            if ground and d > 2 and sc.first_blocker(p["x"], p["y"], 1.5, x, y, 1.0) is not None:
                continue
            hits.append((d, p["id"]))
        return [pid for _, pid in sorted(hits)[:limit]]

    # -- blind spots ---------------------------------------------------------

    def _blind_spots(self, access):
        sc, s = self.sc, self.s
        lab, n = ndimage.label(self.blind, structure=np.ones((3, 3)))
        if not n:
            return []
        c2 = sc.cell ** 2
        sizes = ndimage.sum(self.blind, lab, index=np.arange(1, n + 1)) * c2
        sx, sy = (sc.xmin + sc.xmax) / 2, (sc.ymin + sc.ymax) / 2
        if sc.boundaries:
            sx, sy = M.centroid([p for f in sc.boundaries for p in f["pts"]])
        span = max(sc.xmax - sc.xmin, sc.ymax - sc.ymin) - (30 if sc.boundaries else 20)
        grow = max(1, int(round(1.0 / sc.cell)))
        out = []
        for k in np.argsort(-sizes)[:40]:
            area = float(sizes[k])
            if area < s["min_blind_m2"]:
                break
            comp = lab == k + 1
            jj, ii = np.nonzero(comp)
            mx, my = sc.cx[ii].mean(), sc.cy[jj].mean()
            q = int(np.argmin((sc.cx[ii] - mx) ** 2 + (sc.cy[jj] - my) ** 2))  # a point that is inside the shape
            px, py = float(sc.cx[ii[q]]), float(sc.cy[jj[q]])
            near_acc = []
            for a in access:
                d = np.hypot(sc.cx[ii] - a["x"], sc.cy[jj] - a["y"]).min()
                if d <= 4.0:
                    near_acc.append(a["name"])
            fat = ndimage.binary_dilation(comp, iterations=grow + 1)
            ids = sc.fid[fat & (sc.fid >= 0)]
            hidden_by = []
            if ids.size:
                cnt = np.bincount(ids)
                hidden_by = [sc.opaque[i][0]["name"] for i in np.argsort(-cnt)[:3] if cnt[i] > 0]
            at_building = bool((fat & sc.building_mask).any())
            whole = area > 0.8 * sc.area_m2
            sev = "high" if near_acc else "medium" if at_building or area >= 40 else "low"
            dist = math.hypot(px - sx, py - sy)
            where = "centre of the site" if dist < 0.15 * max(span, 1) else \
                f"{M.compass(M.bearing_deg(px - sx, py - sy))} side"
            fixed = float((comp & self.cam_final).sum()) / max(1, int(comp.sum()))
            planned = float((comp & self.cam_plan).sum()) / max(1, int(comp.sum()))
            lat, lon = sc.latlon(px, py)
            out.append({
                "x": round(px, 2), "y": round(py, 2), "lat": lat, "lon": lon, "area_m2": round(area, 0),
                "where": "most of the grounds" if whole else where, "severity": sev, "near_access": near_acc,
                "hidden_by": hidden_by, "at_building": at_building, "fixed_by_plan": round(fixed, 2),
                "covered_by_planned": round(planned, 2),
                "extent_m": round(float(max(np.ptp(sc.cx[ii]), np.ptp(sc.cy[jj])) + sc.cell), 1),
            })
        order = {"high": 0, "medium": 1, "low": 2}
        out.sort(key=lambda b: (order[b["severity"]], -b["area_m2"]))
        for i, b in enumerate(out):
            b["id"] = f"B{i + 1}"
            b["photos"] = self.photos_of(b["x"], b["y"])
        return out

    # -- perimeter -----------------------------------------------------------

    def _perimeter(self):
        sc = self.sc
        if not sc.boundaries:
            return None
        segs = []
        for f in sc.by_kind["barrier"]:
            segs += [(*a, *b) for a, b in zip(f["pts"], f["pts"][1:])]
        for f in sc.buildings:
            p = f["pts"]
            segs += [(*p[i], *p[(i + 1) % len(p)]) for i in range(len(p))]
        segs = np.array(segs, float).reshape(-1, 4)
        total, closed_len, gaps = 0.0, 0.0, []
        openings = [a for a in self.access if a["subtype"] in ("gate", "vehicle_gate", "other", "door", "garage")]
        for f in sc.boundaries:
            ring = f["pts"] + [f["pts"][0]]
            P = np.array(_samples(ring, 0.5)[:-1], float)
            n = len(P)
            L = M.path_length(f["pts"], True)
            total += L
            step = L / n
            is_open = _dist_to_segments(P, segs) > 0.5
            closed_len += float((~is_open).sum()) * step
            if is_open.all():
                gaps.append(self._gap(P, 0, n, step, openings, whole=True))
                continue
            start = int(np.argmin(is_open))  # begin on a closed sample so runs do not wrap
            idx = (np.arange(n) + start) % n
            run = []
            for i in list(idx) + [idx[0]]:
                if is_open[i] and len(run) < n:
                    run.append(i)
                    continue
                if len(run) * step >= 1.0:
                    gaps.append(self._gap(P, run[0], len(run), step, openings))
                run = []
        gaps.sort(key=lambda g: (g["declared"] is not None, -g["length_m"]))
        for i, g in enumerate(gaps):
            g["code"] = f"G{i + 1}"
        barriers = []
        for f in sc.by_kind["barrier"]:
            t = M.BARRIER_TYPES[f["subtype"]]
            barriers.append({"name": f["name"], "type": t["label"], "height": f["height"],
                             "length_m": round(M.path_length(f["pts"]), 1), "see_through": not t["opaque"],
                             "low": f["height"] < 1.2, "notes": f["notes"]})
        return {"length_m": round(total, 1), "closed_m": round(closed_len, 1),
                "closed_pct": round(100 * closed_len / max(total, 1e-9)), "gaps": gaps, "barriers": barriers}

    def _gap(self, P, first, count, step, openings, whole=False):
        n = len(P)
        pts = P[(first + np.arange(count)) % n]
        mid = pts[count // 2]
        declared = None
        for a in openings:
            if np.hypot(pts[:, 0] - a["pts"][0][0], pts[:, 1] - a["pts"][0][1]).min() <= 2.5:
                declared = a["name"]
                break
        lat, lon = self.sc.latlon(float(mid[0]), float(mid[1]))
        cx, cy = M.centroid([p for f in self.sc.boundaries for p in f["pts"]])
        return {"length_m": round(count * step, 1), "x": round(float(mid[0]), 2), "y": round(float(mid[1]), 2),
                "lat": lat, "lon": lon, "declared": declared, "whole": whole,
                "side": M.compass(M.bearing_deg(mid[0] - cx, mid[1] - cy)),
                "line": [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in pts[:: max(1, count // 40)]] +
                        [[round(float(pts[-1][0]), 2), round(float(pts[-1][1]), 2)]]}

    # -- what a stakeout camera saw over time ---------------------------------------

    def _activity(self, access):
        """Attached stakeouts, each tied to the access point it watched, boiled down for the guard plan."""
        from ..stakeout.pipeline import _windows, hours

        by_id = {a["id"]: a for a in access}
        out = []
        for w in self.model.get("stakeouts") or []:
            d = w["data"]
            a = by_id.get(w["access"])
            bh = np.array(d["by_hour"], float)
            cover = np.array(d["cover"], float)
            busy = " and ".join(hours(x, y) for x, y in _windows(bh >= max(0.6 * bh.max(), 1e-9))[:3]) if bh.max() > 0 else ""
            quiet = [q for q in _windows((bh == 0) & (cover.sum(0) > 0)) if ((q[1] - q[0]) % 24 or 24) >= 3]
            ooh = d.get("out_of_hours") or []
            hist = np.bincount([e["hour"] for e in ooh], minlength=24) if ooh else None
            out.append({
                "id": w["id"], "name": w["name"], "access": w["access"], "access_name": a["name"] if a else None,
                "label": a["name"] if a else (d.get("watching") or w["name"]),
                "summary": d["summary"], "rate": d["rate"], "cover": d["cover"], "closed": d["closed"],
                "advice": d.get("advice") or [], "busy": busy, "peak_rate": round(float(bh.max()), 1),
                "quiet": hours(*max(quiet, key=lambda q: (q[1] - q[0]) % 24 or 24)) if quiet else "",
                "out_of_hours": len(ooh), "ooh_hour": int(np.argmax(hist)) if ooh else None,
            })
        return out

    # -- things the detector found in the drone photos ---------------------------

    def _detections(self, access):
        """Site context for each detection: inside or outside, who can see it, what it is near."""
        sc = self.sc
        segs = np.array([(*a, *b) for f in sc.by_kind["barrier"] for a, b in zip(f["pts"], f["pts"][1:])],
                        float).reshape(-1, 4)
        out = []
        for d in self.model.get("detections") or []:
            if d["status"] == "dismissed" or not sc.in_grid(d["x"], d["y"]):
                continue
            j, i = sc.cell_of(d["x"], d["y"])
            spot = np.hypot(sc.cx[None, :] - d["x"], sc.cy[:, None] - d["y"]) <= max(1.0, sc.cell * 1.5)
            inside = bool(sc.inside[j, i])
            on_camera = bool((self.cam_now & spot).any())
            overlooked = bool((self.natural & spot).any())
            near = min(access, key=lambda a: math.hypot(a["x"] - d["x"], a["y"] - d["y"]), default=None)
            near_d = math.hypot(near["x"] - d["x"], near["y"] - d["y"]) if near else None
            fence_d = float(_dist_to_segments(np.array([[d["x"], d["y"]]]), segs)[0]) if len(segs) else None
            parts = [("Inside the boundary" if inside else "Outside the boundary") if sc.boundaries else "On the site"]
            parts.append("on camera" if on_camera else "seen from the street or a window" if overlooked
                         else "where nobody can see")
            if near and near_d <= 10:
                parts.append(f"{near_d:.0f} m from {near['name']}")
            if fence_d is not None and fence_d <= 1.5:
                parts.append("against a fence, wall or hedge")
            out.append({**{k: d.get(k) for k in ("code", "x", "y", "lat", "lon", "score", "label", "area_m2", "photo",
                                                 "photo_id", "crop", "seen", "status", "type", "note")},
                        "inside": inside, "on_camera": on_camera, "overlooked": overlooked,
                        "unseen": inside and not on_camera and not overlooked,
                        "near_access": near["name"] if near and near_d <= 10 else None,
                        "near_access_m": round(near_d, 1) if near and near_d <= 10 else None,
                        "at_barrier": fence_d is not None and fence_d <= 1.5, "context": ", ".join(parts) + "."})
        return out

    # -- what hides the grounds from the street -------------------------------

    def _blockers(self, access):
        """For each hedge, fence, wall or planting: what the street could see if it were not there."""
        sc = self.sc
        if not self.street_pts:
            return []
        feats = [f for f, _, _ in sc.opaque if f["kind"] != "building"]
        feats.sort(key=lambda f: -M.path_length(f["pts"]))
        discs = {a["id"]: self._stand_disc_id(a["id"]) for a in access if not a["street"]}
        out = []
        for f in feats[:14]:  # a coarser walk along the street is enough for a what-if
            H, _ = sc.heights(without=f)
            opened = self._street_mask(H, every=2) & sc.domain & ~self.street
            area = float(opened.sum()) * sc.cell ** 2
            reveals = [a["name"] for a in access if a["id"] in discs
                       and _frac(opened | self.street, discs[a["id"]]) >= COVERED]
            unwatched = float((opened & self.blind).sum()) * sc.cell ** 2
            if area >= 10 or reveals:
                out.append({"id": f["id"], "name": f["name"], "type": M.feature_label(f), "height": f["height"],
                            "hides_m2": round(area, 0), "hides_blind_m2": round(unwatched, 0),
                            "hides_access": reveals})
        out.sort(key=lambda b: (-len(b["hides_access"]), -b["hides_blind_m2"], -b["hides_m2"]))
        return out

    def _stand_disc_id(self, fid):
        return self._stand_disc(next(a for a in self.access if a["id"] == fid))

    # -- prioritised findings ------------------------------------------------

    def _findings(self):
        r = self.result
        st = r["stats"]
        out = []
        add = lambda sev, cat, title, detail, action, ref=None: out.append(  # noqa: E731
            {"severity": sev, "category": cat, "title": title, "detail": detail, "action": action, "ref": ref})
        for a in r["access"]:
            if a["priority"] == "low":
                continue
            add(a["priority"], "Access point", f"{a['name']} is poorly watched",
                _sentence(a["why"]), a["action"], a["id"])
        for b in r["blind_spots"]:
            if b["severity"] == "low":
                continue
            by = f" It is screened by {_join(b['hidden_by'])}." if b["hidden_by"] else ""
            near = f" It reaches {_join(b['near_access'])}." if b["near_access"] else \
                " It runs up to a building wall." if b["at_building"] else ""
            fix = (f"The recommended cameras cover {b['fixed_by_plan']:.0%} of it."
                   if b["fixed_by_plan"] >= 0.2 else
                   "No recommended camera reaches most of it: consider lighting, cutting back planting, "
                   "or a camera on a post.")
            add(b["severity"], "Blind spot", f"{b['id']}: {b['area_m2']:.0f} m² unobserved on the {b['where']}"
                if "side" in b["where"] else f"{b['id']}: {b['area_m2']:.0f} m² unobserved, {b['where']}",
                f"Nobody on the street, at a window or on camera can see this area.{by}{near}", fix, b["id"])
        for w in r["activity"]:
            if w["out_of_hours"]:
                c0, c1 = w["closed"]
                add("medium", "Activity", f"{w['label']}: {w['out_of_hours']} movement"
                    f"{'s' if w['out_of_hours'] != 1 else ''} while the site is closed",
                    f"The stakeout ({w['summary']['days']:g} days) recorded movement between {c0:02d}:00 and {c1:02d}:00, "
                    f"most often around {w['ooh_hour']:02d}:00.",
                    "Find out who it is. If it is not expected, time patrols for that hour or alarm that approach.",
                    w["access"] or None)
        for d in r["detections"]:
            if d["status"] != "confirmed":
                continue
            what = M.DETECTION_TYPES.get(d["type"], "Item")
            sev = "medium" if d["inside"] and (d["unseen"] or (d["type"] == "object" and d["at_barrier"])) else "low"
            if d["type"] == "object" and d["at_barrier"]:
                act = "Anything left against a barrier can be used to climb it: move it away from the boundary."
            elif d["unseen"]:
                act = "It was in ground no camera or passer-by can see: this is where cover is most needed."
            else:
                act = "Recorded for reference."
            add(sev, "Seen in photos", f"{d['code']}: {what.lower()} seen during the survey",
                d["context"] + (f" {d['note']}" if d["note"] else ""), act, d["code"])
        per = r["perimeter"]
        if per:
            for g in per["gaps"]:
                if g["declared"]:
                    continue
                if g["whole"]:
                    add("medium", "Perimeter", "The boundary has no fence, wall or hedge drawn",
                        f"All {g['length_m']:.0f} m of the property line is open.",
                        "If barriers exist, draw them; otherwise anyone can walk in from any side.")
                else:
                    add("medium" if g["length_m"] >= 3 else "low", "Perimeter",
                        f"{g['code']}: {g['length_m']:.0f} m gap in the perimeter on the {g['side']} side",
                        "This stretch of the boundary has no barrier and no gate or driveway marked on it.",
                        "Close it, or mark it as an access point so it is assessed as one.")
            for b in per["barriers"]:
                if b["low"]:
                    add("low", "Perimeter", f"{b['name']} is only {b['height']:.1f} m high",
                        "A barrier under about 1.2 m marks the line but is stepped over easily.",
                        "Treat it as a marker, not as protection.")
        for b in r["blockers"]:
            if not b["hides_access"]:
                continue
            add("medium", "Sightline", f"{_join(b['hides_access'])} cannot be seen from the street because of "
                f"{b['name']}",
                f"Without it, a passer-by could see {b['hides_m2']:.0f} m² more of the grounds "
                f"({b['hides_blind_m2']:.0f} m² of that is currently seen by nobody).",
                "Lowering it below eye level (about 1 m), thinning it, or replacing it with a see-through "
                "fence opens the view while keeping the boundary.", b["id"])
        if not st["has_boundary"]:
            add("info", "Model", "No boundary drawn", "Without a property line the whole surveyed area is analysed.",
                "Draw the boundary so the figures refer to the property only.")
        if not st["has_street"]:
            add("info", "Model", "No public way drawn",
                "Sightlines from the street cannot be assessed until the street or footpath is drawn.",
                "Draw a line along each street, lane or footpath that passes the property.")
        if st["access_total"] == 0:
            add("info", "Model", "No access points marked", "Doors, windows, gates and driveways are not assessed.",
                "Mark each one on the plan.")
        order = {"high": 0, "medium": 1, "low": 2, "info": 3}
        out.sort(key=lambda f: order[f["severity"]])
        return out

    # -- overlays ------------------------------------------------------------

    def overlay_png(self, name: str) -> bytes:
        rgba = {"cameras": (59, 130, 246, 88), "cameras_now": (59, 130, 246, 88), "natural": (250, 204, 21, 96), "hidden": (10, 10, 16, 165), "blind": (229, 72, 77, 120),
                "proposed": (16, 163, 90, 92), "blind_after": (229, 72, 77, 120), "blind_plan": (229, 72, 77, 120),
                "vantage": (20, 184, 166, 105)}[name]
        mk = self.layers[name]
        arr = np.zeros((*mk.shape, 4), np.uint8)
        arr[mk] = rgba
        buf = io.BytesIO()
        Image.fromarray(arr, "RGBA").save(buf, "PNG", optimize=True)
        return buf.getvalue()

    def height_png(self) -> bytes:
        """The blocking-height grid for the 3-D view: centimetres packed into red (high byte) and green."""
        cm = np.clip(np.round(self.sc.H * 100), 0, 65535).astype(np.uint16)
        buf = io.BytesIO()
        Image.fromarray(np.dstack([cm >> 8, cm & 255, np.zeros_like(cm)]).astype(np.uint8), "RGB").save(buf, "PNG")
        return buf.getvalue()

    def overlays_b64(self) -> dict:
        out = {k: "data:image/png;base64," + base64.b64encode(self.overlay_png(k)).decode() for k in self.layers}
        out["height"] = "data:image/png;base64," + base64.b64encode(self.height_png()).decode()
        return out


def _union(masks, zero):
    out = zero.copy()
    for m in masks:
        out |= m
    return out


def _frac(mask, disc) -> float:
    n = int(disc.sum())
    return float((mask & disc).sum()) / n if n else 0.0


def _join(names) -> str:
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1] if names else ""


def _sentence(parts) -> str:
    s = "; ".join(parts)
    return (s[0].upper() + s[1:] + ".") if s else ""


def analyse(model: dict, project=None, use_measured: bool = True) -> Analysis:
    return Analysis(model, project, use_measured)
