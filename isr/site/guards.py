"""Guard posts, patrol routes and staffing scenarios for a site.

Given how many officers might be on site, this works out:

* **posts**: where a standing officer sees the most that matters (access
  points first, then ground no camera covers, then everything else), chosen
  one at a time so each new post adds what the others cannot see;
* **a patrol route**: the shortest walking loop, around buildings and through
  gates, that visits every door, gate, blind spot and gap in the perimeter
  which the posts do not already watch;
* **a scenario per headcount**: for one officer, two, three..., a recommended
  split between posts and patrol and an alternative, each with what it gains,
  what it leaves exposed, and what to do about that.

It is geometry, not doctrine: line of sight at eye height on flat ground and
walking distance on the plan. It knows nothing about lighting, shelter, radio
cover, welfare, or what the client's threat actually is.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from . import model as M

STOP_S = 20.0  # time spent at each checkpoint (try the door, look round)


def plan(an) -> dict | None:
    """``an`` is a finished Analysis. Returns the guard plan, or None when it is switched off."""
    s, sc = an.s, an.sc
    n_max = int(s.get("guards") or 0)
    if n_max <= 0 or not sc.domain.any():
        return None
    r = an.result
    access = r["access"]

    # -- where an officer can walk ---------------------------------------------
    walk = sc.inside & ~sc.solid
    for a in access:  # a gate or driveway is a way through the barrier it sits on
        if a["subtype"] in ("gate", "vehicle_gate", "other"):
            walk |= (np.hypot(sc.cx[None, :] - a["x"], sc.cy[:, None] - a["y"]) <= 1.6) & ~sc.building_mask
    idx = np.full(walk.shape, -1, np.int64)
    idx[walk] = np.arange(int(walk.sum()))
    slow = np.where(sc.H > 0.3, 4.0, 1.0).astype(np.float32)  # pushing through planting is slow
    rows, cols, wts = [], [], []
    for dj, di in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a0 = idx[: sc.ny - dj, max(0, -di): sc.nx - max(0, di)]
        b0 = idx[dj:, max(0, di): sc.nx - max(0, -di)]
        ok = (a0 >= 0) & (b0 >= 0)
        c = (slow[: sc.ny - dj, max(0, -di): sc.nx - max(0, di)] + slow[dj:, max(0, di): sc.nx - max(0, -di)]) / 2
        rows.append(a0[ok])
        cols.append(b0[ok])
        wts.append(c[ok] * sc.cell * math.hypot(dj, di))
    n_nodes = int(walk.sum())
    graph = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(n_nodes, n_nodes)).tocsr()
    wj, wi = np.nonzero(walk)
    wx, wy = sc.cx[wi], sc.cy[wj]

    def node_at(x, y, reach=3.0):
        d = np.hypot(wx - x, wy - y)
        k = int(np.argmin(d)) if len(d) else -1
        return k if k >= 0 and d[k] <= reach else None

    # -- what a patrol should look at -----------------------------------------
    feats = {a["id"]: a for a in an.access}
    cps = []
    for a in access:
        x, y, _ = sc.off_wall(a["x"], a["y"], dist=1.0)
        cps.append({"name": a["name"], "kind": "access", "x": x, "y": y, "priority": a["priority"],
                    "disc": an._stand_disc(feats[a["id"]]), "entry": a["subtype"] in ("gate", "vehicle_gate")})
    for b in r["blind_spots"]:
        cps.append({"name": f"blind spot {b['id']}", "kind": "blind", "x": b["x"], "y": b["y"], "priority": b["severity"]})
    for g in (r["perimeter"] or {}).get("gaps", []):
        if not g["declared"] and not g["whole"]:
            cps.append({"name": f"perimeter gap {g['code']}", "kind": "gap", "x": g["x"], "y": g["y"], "priority": "medium"})
    kept, unreachable = [], []
    for c in cps:
        if any(math.hypot(c["x"] - k["x"], c["y"] - k["y"]) < 3.0 for k in kept if k["kind"] != "access" or c["kind"] != "access"):
            continue
        c["cell"] = sc.cell_of(c["x"], c["y"])
        c["node"] = node_at(c["x"], c["y"])
        (kept if c["node"] is not None else unreachable).append(c)
    cps = kept
    if cps:
        dist, pred = dijkstra(graph, directed=False, indices=[c["node"] for c in cps], return_predecessors=True)
        D = dist[:, [c["node"] for c in cps]]
        start = next((i for i, c in enumerate(cps) if c.get("entry")), 0)
        far = [c for i, c in enumerate(cps) if not np.isfinite(D[start, i])]
        if far:  # cut off from the entrance by something drawn across the way
            unreachable += far
            keep = [i for i, c in enumerate(cps) if np.isfinite(D[start, i])]
            cps, D, pred = [cps[i] for i in keep], D[np.ix_(keep, keep)], pred[keep]
            start = next((i for i, c in enumerate(cps) if c.get("entry")), 0)
    else:
        D, pred, start = np.zeros((0, 0)), None, 0

    def tour(ids):
        """Shortest closed loop through checkpoints ``ids``: nearest neighbour, then 2-opt."""
        if len(ids) < 2:
            return list(ids)
        first = start if start in ids else ids[0]
        order, left = [first], set(ids) - {first}
        while left:
            nxt = min(left, key=lambda j: D[order[-1], j])
            order.append(nxt)
            left.discard(nxt)
        L = lambda o: sum(D[o[k], o[(k + 1) % len(o)]] for k in range(len(o)))  # noqa: E731
        improved = True
        while improved and len(order) > 3:
            improved = False
            for i in range(1, len(order) - 1):
                for j in range(i + 1, len(order)):
                    cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                    if L(cand) < L(order) - 1e-6:
                        order, improved = cand, True
        return order

    def leg(i, j):
        """Cell-by-cell path from checkpoint i to j, thinned to a drawable line."""
        pts, n = [], cps[j]["node"]
        while n >= 0 and n != cps[i]["node"] and len(pts) < 20000:
            pts.append((float(wx[n]), float(wy[n])))
            n = int(pred[i, n])
        pts.append((cps[i]["x"], cps[i]["y"]))
        pts = pts[::-1]
        return pts[::3] + [pts[-1]]

    def route(ids, rovers):
        order = tour(ids)
        if len(order) < 2:
            return None
        line, length = [], 0.0
        for k in range(len(order)):
            a, b = order[k], order[(k + 1) % len(order)]
            length += float(D[a, b])
            line += leg(a, b)
        lap = length / max(0.3, s["patrol_speed_ms"]) + STOP_S * len(order)
        return {"length_m": round(length), "lap_min": round(lap / 60, 1), "revisit_min": round(lap / 60 / rovers, 1),
                "stops": [{"n": k + 1, "name": cps[i]["name"], "x": round(cps[i]["x"], 2), "y": round(cps[i]["y"], 2)}
                          for k, i in enumerate(order)],
                "line": [[round(x, 2), round(y, 2)] for x, y in line]}

    # -- best places to stand ----------------------------------------------------
    c2 = sc.cell ** 2
    W = np.where(sc.domain, c2, 0.0).astype(np.float32)
    W[sc.domain & ~an.cam_now] += c2  # ground no camera records: worth more to a guard
    for c in cps:
        if c["kind"] == "access":
            n = int((c["disc"] & sc.domain).sum())
            if n:
                W[c["disc"] & sc.domain] += (120.0 if c["priority"] != "low" or c.get("entry") else 60.0) / n
    total_w = float(W.sum()) or 1.0
    open_ground = walk & sc.inside & (sc.H < 0.3)
    step = max(1, int(round(3.0 / sc.cell)))
    while int(open_ground[::step, ::step].sum()) > 140:
        step += 1
    cand = [(float(sc.cx[i]), float(sc.cy[j])) for j in range(step // 2, sc.ny, step) for i in range(step // 2, sc.nx, step)
            if open_ground[j, i]]
    cand += [(c["x"], c["y"]) for c in cps if c.get("entry") and node_at(c["x"], c["y"], 1.5) is not None]
    rng = min(float(s["guard_range_m"]), math.hypot(sc.xmax - sc.xmin, sc.ymax - sc.ymin))
    views = [sc.viewshed(x, y, s["eye_height_m"], rng, h_tgt=1.2) & sc.domain for x, y in cand]
    covered = np.zeros_like(sc.domain)
    has_camera = {a["name"]: a["camera_planned"] for a in access}
    posts, masks = [], []
    cxm, cym = M.centroid([p for f in sc.boundaries for p in f["pts"]]) if sc.boundaries else \
        ((sc.xmin + sc.xmax) / 2, (sc.ymin + sc.ymax) / 2)
    for _ in range(min(n_max, len(cand))):
        gains = [float(W[v & ~covered].sum()) for v in views]
        k = int(np.argmax(gains)) if gains else -1
        if k < 0 or gains[k] < 0.04 * total_w:
            break
        x, y = cand[k]
        v = views[k]
        sees = [c["name"] for c in cps if c["kind"] == "access" and _seen(v, c)]
        misses = [c["name"] for c in cps if c["kind"] == "access" and not _seen(v, c)]
        near = min((c for c in cps if c["kind"] == "access"), key=lambda c: math.hypot(c["x"] - x, c["y"] - y), default=None)
        nd = math.hypot(near["x"] - x, near["y"] - y) if near else 99
        where = f"by {near['name']}" if nd <= 6 else \
            f"{M.compass(M.bearing_deg(x - cxm, y - cym))} side of the grounds" if math.hypot(x - cxm, y - cym) > 6 else \
            "middle of the grounds"
        j, i = sc.cell_of(x, y)
        lat, lon = sc.latlon(x, y)
        no_cam = float((v & ~an.cam_now).sum()) * c2
        p = {"id": f"P{len(posts) + 1}", "x": round(x, 2), "y": round(y, 2), "lat": lat, "lon": lon, "where": where,
             "sees_pct": round(100 * float(v.sum()) * c2 / max(sc.area_m2, 1e-9)), "sees": sees, "misses": misses,
             "adds_pct": round(100 * float((v & ~covered).sum()) * c2 / max(sc.area_m2, 1e-9)),
             "controls": near["name"] if near and near.get("entry") and nd <= 6 else None,
             "seen_from_street": bool(an.street[j, i])}
        p["pros"] = [f"Sees {p['sees_pct']}% of the grounds" + (f" and {len(sees)} of {len(sees) + len(misses)} access points "
                     f"({M_join(sees)})" if sees else "")]
        if p["controls"]:
            p["pros"].append(f"Close enough to {p['controls']} to challenge anyone coming in")
        if no_cam >= 20:
            p["pros"].append(f"Watches {no_cam:.0f} m² that no existing camera records")
        if p["seen_from_street"]:
            p["pros"].append("In view of the street, so the guard is a visible deterrent")
        p["cons"] = []
        if misses:
            p["cons"].append(f"Cannot see {M_join(misses[:5])}" + (" and others" if len(misses) > 5 else ""))
        if p["seen_from_street"]:
            p["cons"].append("Anyone outside can see where the guard is and when they leave")
        if p["sees_pct"] < 40:
            p["cons"].append("Sees under half the grounds: buildings and planting cut the view")
        on_cam = [m for m in misses if has_camera.get(m)]
        rest = [m for m in misses if m not in on_cam]
        p["suggestion"] = " ".join(x for x in (
            f"Give this post a screen for the camera on {M_join(on_cam[:3])}." if on_cam else "",
            f"Cover {M_join(rest[:3])}{' and the rest' if len(rest) > 3 else ''} from the patrol or with a camera." if rest else "",
            "" if misses else "Nothing at an access point is hidden from here; rotate officers to keep attention up.") if x)
        posts.append(p)
        masks.append(v)
        covered |= v
    worthwhile = len(posts)
    strong = max(1, sum(p["adds_pct"] >= 15 for p in posts)) if posts else 0  # posts that earn a full-time officer

    # -- scenarios --------------------------------------------------------------
    cam = an.cam_now & sc.domain
    area = max(sc.area_m2, 1e-9)
    all_ids = list(range(len(cps)))

    def option(statics, rovers):
        seen = np.zeros_like(sc.domain)
        for k in range(statics):
            seen |= masks[k]
        watched = [i for i in all_ids if _seen(seen, cps[i])]
        unwatched = [i for i in all_ids if i not in watched]
        rt = None
        if rovers:
            ids = unwatched + [i for i in watched if cps[i]["kind"] == "access" and cps[i]["priority"] == "high"]
            rt = route(sorted(ids) if len(ids) >= 2 else all_ids, rovers)
        g_pct = round(100 * float(seen.sum()) * c2 / area)
        any_pct = round(100 * float((seen | cam).sum()) * c2 / area)
        acc_w = [cps[i]["name"] for i in watched if cps[i]["kind"] == "access"]
        acc_u = [cps[i]["name"] for i in unwatched if cps[i]["kind"] == "access"]
        title = " + ".join(x for x in (
            f"{statics} static post{'s' if statics != 1 else ''}" if statics else "",
            f"{rovers} roving patrol{'s' if rovers != 1 else ''}" if rovers else "") if x)
        pros, cons, tips = [], [], []
        if statics:
            pros.append(f"{g_pct}% of the grounds is watched all the time by a guard" +
                        (f" ({any_pct}% counting cameras)" if any_pct > g_pct else ""))
            if acc_w:
                pros.append(f"{M_join(acc_w[:6])} {'are' if len(acc_w) > 1 else 'is'} always in view")
            if posts[0]["controls"]:
                pros.append(f"{posts[0]['controls']} is controlled: arrivals are met, not just seen")
            cons.append("A fixed guard can be watched and timed from outside")
        if rt:
            pros.append(f"Every checkpoint on the route is visited every {rt['revisit_min']:g} min "
                        f"({rt['length_m']} m loop, {len(rt['stops'])} stops)")
            pros.append("Doors and gates are tried by hand, not only looked at")
            if statics:
                cons.append(f"{M_join(acc_u[:5]) if acc_u else 'Ground the posts cannot see'} "
                            f"{'are' if len(acc_u) != 1 else 'is'} seen only when the patrol passes, every "
                            f"{rt['revisit_min']:g} min")
            else:
                cons.append(f"Nothing is watched continuously: an intruder has up to {rt['revisit_min']:g} min between passes")
                if any(c.get("entry") for c in cps):
                    cons.append("The entrance is unattended whenever the patrol is on the far side")
            tips.append("Vary the direction and the start time of each lap so the pattern cannot be learned.")
            if rovers > 1:
                tips.append(f"Keep the {rovers} officers on the same loop about {rt['revisit_min']:g} min apart, "
                            "or walking it in opposite directions.")
        elif acc_u:
            cons.append(f"{M_join(acc_u[:6])} {'are' if len(acc_u) > 1 else 'is'} never seen by a guard")
        if not rt and statics:
            cons.append("Nobody checks doors and gates by hand")
        blind_left = float((an.blind & ~seen).sum()) * c2
        if blind_left >= 20 and not rt:
            cons.append(f"{blind_left:.0f} m² stays unseen by anyone")
        if acc_u and r["recommended"]:
            cams = [p["id"] for p in r["recommended"] if set(p["covers"]) & set(acc_u)]
            if cams:
                tips.append(f"Recommended camera{'s' if len(cams) > 1 else ''} {M_join(cams)} would put "
                            f"{'some of ' if len(acc_u) > 1 else ''}what the guards miss on a screen.")
        if statics and not r["cameras"] and not rt:
            tips.append("With no patrol, cameras are the only way to see behind the buildings.")
        return {"title": title, "statics": [p["id"] for p in posts[:statics]], "rovers": rovers, "route": rt,
                "guard_pct": g_pct, "any_pct": any_pct, "watched": acc_w, "unwatched": acc_u,
                "pros": pros, "cons": cons, "suggestions": tips}

    scenarios = []
    can_patrol = len(cps) >= 2
    for n in range(1, n_max + 1):
        opts = []
        if n == 1:
            if can_patrol:
                opts.append(option(0, 1))
            if worthwhile:
                opts.append(option(1, 0))
            # one officer who can already see nearly everything from one spot should stay there
            if len(opts) == 2 and opts[1]["guard_pct"] >= 85 and not opts[1]["unwatched"]:
                opts.reverse()
        else:
            st = min(strong, n - 1) if can_patrol else min(worthwhile, n)
            if st:
                opts.append(option(st, n - st if can_patrol else 0))
            if can_patrol:  # the alternative: one post fewer and one more officer walking
                opts.append(option(st - 1, n - st + 1) if st else option(0, n))
        if not opts:
            continue
        for k, o in enumerate(opts):
            o["recommended"] = k == 0
        best = opts[0]
        why = ("One officer cannot both hold a post and check the doors; walking covers more of what matters here."
               if n == 1 and best["rovers"] else
               "From this one spot an officer sees nearly everything, so staying put beats walking away from it."
               if n == 1 else
               f"Holding {len(best['statics'])} post{'s' if len(best['statics']) != 1 else ''} keeps the main view and "
               f"entrance covered while {best['rovers']} officer{'s' if best['rovers'] != 1 else ''} "
               f"check{'s' if best['rovers'] == 1 else ''} what cannot be seen from there." if best["rovers"] else
               "There is too little to patrol; fixed posts see what there is to see.")
        scenarios.append({"guards": n, "why": why, "options": opts})
    return {"posts": posts, "scenarios": scenarios,
            "checkpoints": [{"name": c["name"], "kind": c["kind"], "x": round(c["x"], 2), "y": round(c["y"], 2)} for c in cps],
            "unreachable": [c["name"] for c in unreachable],
            "assumptions": {"eye_height_m": s["eye_height_m"], "range_m": round(rng), "speed_ms": s["patrol_speed_ms"],
                            "stop_s": STOP_S}}


def _seen(mask: np.ndarray, c: dict) -> bool:
    """Is this checkpoint in view: most of the ground at a door, or the spot itself for anything else."""
    if c["kind"] == "access":
        n = int(c["disc"].sum())
        return n > 0 and float((mask & c["disc"]).sum()) / n >= 0.35
    j, i = c["cell"]
    return bool(mask[max(0, j - 2): j + 3, max(0, i - 2): i + 3].any())


def M_join(names) -> str:
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1] if names else ""
