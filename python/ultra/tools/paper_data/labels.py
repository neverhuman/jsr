"""Collision-free label placement for a scatter plot whose plot box is known in points.

Each label picks one of a fixed set of candidate boxes: eight that touch its marker, and rings of
positions reached by a leader line. A box nearer another point than its own (by less than NEAREST) counts
as a collision, so a name is never read as another point's, whatever its leader; a number that replaces a
name in a dense cluster pays a large cost for it instead, so a crowded figure still gets its key. A deterministic search (coordinate descent, then seeded annealing)
looks for a choice without collisions; `place` reports which labels still collide, so the caller can
replace them by short numbers and a key.
"""

from __future__ import annotations

import math
import random

LINE_PT, PAD, GAP = 7.4, 0.8, 1.4  # label height, clearance between labels, clearance from the marker
HARD = 1000.0                       # cost of one collision
DIAGONAL_MAX = 3 * LINE_PT          # pt: only labels this short may sit at a marker's corner
NEAREST = 1.0                       # pt: a label is this much nearer its own point than any other, or it collides
_NEAR = [(1, 0, 0.0), (-1, 0, 0.3), (0, 1, 0.6), (0, -1, 0.6), (1, 1, 0.9), (1, -1, 0.9), (-1, 1, 1.1), (-1, -1, 1.1)]
_REACH = (9.0, 18.0, 30.0, 46.0, 64.0)  # leader lengths in pt
_ANGLES = 12


def overlap(a, b, pad: float = 0.0) -> bool:
    return a[0] - pad < b[2] and b[0] - pad < a[2] and a[1] - pad < b[3] and b[1] - pad < a[3]


def seg_hits_box(seg, box) -> bool:
    """Liang-Barsky: does the segment pass through the open box?"""
    (x0, y0), (x1, y1) = seg
    t0, t1 = 0.0, 1.0
    for p, q in ((x0 - x1, x0 - box[0]), (x1 - x0, box[2] - x0), (y0 - y1, y0 - box[1]), (y1 - y0, box[3] - y0)):
        if abs(p) < 1e-12:
            if q <= 0:
                return False
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
    return t0 < t1


def segs_cross(a, b) -> bool:
    def side(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    return side(a[0], a[1], b[0]) * side(a[0], a[1], b[1]) < 0 and side(b[0], b[1], a[0]) * side(b[0], b[1], a[1]) < 0


def candidates(p: dict) -> list[dict]:
    hw, hh, r = p["w"] / 2, p.get("h", LINE_PT) / 2, p["clear"]
    out = []
    for sx, sy, pen in _NEAR:
        if sx and sy and p["w"] > DIAGONAL_MAX and not p["bold"]:
            pen = 400.0  # a long label at a corner reads as the label of whatever it runs under: beside or a leader first
        reach = r + GAP if 0 in (sx, sy) else 0.72 * r + GAP
        cx, cy = p["px"] + sx * (hw + reach), p["py"] + sy * (hh + reach)
        out.append({"box": (cx - hw, cy - hh, cx + hw, cy + hh), "leader": None, "pen": pen})
    for ring, length in enumerate(_REACH):
        for k in range(_ANGLES):
            ang = 2 * math.pi * (k + 0.5) / _ANGLES
            ux, uy = math.cos(ang), math.sin(ang)
            tip = (p["px"] + ux * (r + length), p["py"] + uy * (r + length))
            if abs(ux) > 0.4:  # the label sits beside the tip
                cx, cy = tip[0] + math.copysign(hw + 0.8, ux), tip[1] + uy * hh
            else:              # the label sits above or below the tip
                cx, cy = tip[0], tip[1] + math.copysign(hh + 0.8, uy)
            out.append({"box": (cx - hw, cy - hh, cx + hw, cy + hh), "pen": 3.0 + 0.3 * length + 0.01 * k + ring,
                        "leader": ((p["px"] + ux * (r + 0.6), p["py"] + uy * (r + 0.6)), tip)})
    for c in out:  # extent: the box and its leader, with the clearance
        xs = [c["box"][0], c["box"][2]] + ([c["leader"][0][0], c["leader"][1][0]] if c["leader"] else [])
        ys = [c["box"][1], c["box"][3]] + ([c["leader"][0][1], c["leader"][1][1]] if c["leader"] else [])
        c["ext"] = (min(xs) - PAD, min(ys) - PAD, max(xs) + PAD, max(ys) + PAD)
    return out


def gap(box, x: float, y: float) -> float:
    """Distance from a point to the nearest edge of a box (0 inside it)."""
    return math.hypot(max(box[0] - x, 0.0, x - box[2]), max(box[1] - y, 0.0, y - box[3]))


def _static(i: int, cand: dict, pts: list[dict], fixed: list, lines: list, size: tuple) -> float:
    box, cost = cand["box"], cand["pen"]
    if box[0] < 1 or box[1] < 1 or box[2] > size[0] - 1 or box[3] > size[1] - 1:
        return math.inf
    own = gap(box, pts[i]["px"], pts[i]["py"])  # a label nearer another point reads as that point's label
    corners = {v for seg in lines for v in seg}  # nor may another label sit right under or over a frontier system
    for q in ([] if pts[i]["bold"] else pts):
        on = any(math.hypot(q["px"] - x, q["py"] - y) < 0.5 for x, y in corners)
        if on and q is not pts[i] and box[0] <= q["px"] <= box[2] and not box[1] <= q["py"] <= box[3]:
            cost += HARD * (min(abs(q["py"] - box[1]), abs(q["py"] - box[3])) < 2 * LINE_PT)
    wrong = HARD if pts[i].get("number") is None else 500.0  # a number in a dense cluster may have no clear spot
    cost += wrong * sum(gap(box, q["px"], q["py"]) < own + NEAREST for q in pts
                        if math.hypot(q["px"] - pts[i]["px"], q["py"] - pts[i]["py"]) >= 2.5)
    cost += HARD * sum(overlap(box, f) or bool(cand["leader"] and seg_hits_box(cand["leader"], f)) for f in fixed)
    cost += HARD * sum(seg_hits_box(seg, box) for seg in lines)  # a label never sits on the frontier line
    for j, q in enumerate(pts):
        near = math.hypot(q["px"] - pts[i]["px"], q["py"] - pts[i]["py"]) < 2.5  # the point itself or a twin on it
        mark = (q["px"] - q["r"], q["py"] - q["r"], q["px"] + q["r"], q["py"] + q["r"])
        if overlap(box, mark, 0.3 if near else 0.6) or (cand["leader"] and not near and seg_hits_box(cand["leader"], mark)):
            cost += HARD
    return cost


def _clash(a: dict, b: dict) -> float:
    if not overlap(a["ext"], b["ext"]):
        return 0.0
    if overlap(a["box"], b["box"], PAD):
        return HARD
    if (a["leader"] and seg_hits_box(a["leader"], b["box"])) or (b["leader"] and seg_hits_box(b["leader"], a["box"])):
        return HARD
    return 400.0 if a["leader"] and b["leader"] and segs_cross(a["leader"], b["leader"]) else 0.0


class _Search:
    def __init__(self, cands: list[list[dict]]):
        self.cands, self.memo = cands, {}

    def clash(self, i: int, k: int, j: int, m: int) -> float:
        key = (i, k, j, m) if i < j else (j, m, i, k)
        hit = self.memo.get(key)
        if hit is None:
            hit = self.memo[key] = _clash(self.cands[i][k], self.cands[j][m])
        return hit

    def cost(self, choice: list, i: int, k: int) -> float:
        return self.cands[i][k]["static"] + sum(self.clash(i, k, j, choice[j]) for j in range(len(choice))
                                                if j != i and choice[j] is not None)

    def descend(self, choice: list[int]) -> float:
        """Move one label at a time to its cheapest candidate until nothing moves. Returns the total cost."""
        for _ in range(40):
            moved = False
            for i in range(len(choice)):
                best = min(range(len(self.cands[i])), key=lambda k: (self.cost(choice, i, k), k))
                if best != choice[i]:
                    choice[i], moved = best, True
            if not moved:
                break
        return sum(self.cost(choice, i, choice[i]) for i in range(len(choice)))


def place(pts: list[dict], fixed: list, lines: list, size: tuple[float, float]) -> list[dict]:
    """Choose a box for every label. Each returned dict has `box`, `leader` and `ok` (False = still collides)."""
    for p in pts:  # two systems on one spot: every label clears the larger of the stacked markers
        p["clear"] = max(q["r"] for q in pts if math.hypot(q["px"] - p["px"], q["py"] - p["py"]) < 2.5)
    cands = []
    for i, p in enumerate(pts):
        usable = []
        for c in candidates(p):
            c["static"] = _static(i, c, pts, fixed, lines, size)
            if c["static"] < math.inf:
                usable.append(c)
        cands.append(sorted(usable, key=lambda c: c["static"]) or [dict(candidates(p)[0], static=HARD)])
    search = _Search(cands)
    choice: list = [None] * len(pts)
    crowd = [sum(math.hypot(q["px"] - p["px"], q["py"] - p["py"]) < 45 for q in pts) for p in pts]
    for i in sorted(range(len(pts)), key=lambda i: (-crowd[i], i)):  # the most crowded labels choose first
        choice[i] = min(range(len(cands[i])), key=lambda k: (search.cost(choice, i, k), k))
    best = search.descend(choice)
    if len(pts) > 1:  # annealing from the greedy state: removes collisions, then shortens leaders; seeded
        rng = random.Random(20261122)
        state, energy = list(choice), best
        steps, hot = 1500 * len(pts), 400.0 if best >= 150 else 12.0
        for step in range(steps):
            temp = hot * (1.0 - step / steps) ** 2 + 0.05
            i = rng.randrange(len(pts))
            k = rng.randrange(len(cands[i]))
            if k == state[i]:
                continue
            delta = search.cost(state, i, k) - search.cost(state, i, state[i])
            if delta <= 0 or rng.random() < math.exp(-delta / temp):
                state[i], energy = k, energy + delta
                if energy < best - 1e-9:
                    best, choice = energy, list(state)
        best = search.descend(choice)
    return [{"box": cands[i][k]["box"], "leader": cands[i][k]["leader"], "ok": search.cost(choice, i, k) < HARD}
            for i, k in enumerate(choice)]
