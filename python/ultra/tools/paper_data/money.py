"""Figure 1 geometry: axis limits, the streaming frontier and collision-free label placement.

The plot box is fixed in points (pgfplots `scale only axis`), so every marker and label box is
known here. Placement is deterministic; if two boxes would still overlap the build fails.
Outputs: money_layout.tex (axis macros, labels, leaders) and money_frontier.csv.
"""

from __future__ import annotations

import math

from .common import PaperDataError, csv_text
from .evidence import HEADLINE, derived

AXIS_W, AXIS_H = 197.0, 140.0  # pt; with tick and axis labels the figure stays under 243 pt
FONT_PT, LINE_PT, PAD = 7.0, 7.4, 0.8  # \scriptsize labels
MARK_R = {"ours": 3.3, "comparator": 2.3}  # half-extent of the drawn marker
GAP = 1.4
# Width of a 7 pt Latin Modern glyph in em (upper bounds, checked against TeX by the tests).
_WIDTHS = {**dict.fromkeys("il.,:;'", 0.32), **dict.fromkeys("fj", 0.36), **dict.fromkeys("rst()", 0.45),
           **dict.fromkeys("cez", 0.51), **dict.fromkeys("ago0123456789/", 0.57), **dict.fromkeys("kqvxy", 0.60),
           **dict.fromkeys("bdhnpu", 0.64), "w": 0.82, "m": 0.95, " ": 0.39, "-": 0.39, "I": 0.43, "J": 0.59,
           **dict.fromkeys("SZL", 0.71), **dict.fromkeys("FEPB", 0.80), **dict.fromkeys("CTR", 0.83),
           **dict.fromkeys("AHNUVXYDKGOQ", 0.89), "M": 1.04, "W": 1.16}
# Candidate label positions: (dx, dy) of the label box centre from the marker centre in units of
# (half width + reach, half height + reach), a leader flag and a preference penalty.
_NEAR = [(1, 0, 0.0), (-1, 0, 0.3), (0, 1, 0.6), (0, -1, 0.6), (1, 1, 0.9), (1, -1, 0.9), (-1, 1, 1.1), (-1, -1, 1.1)]
_FAR = [(9.0, 4.0), (15.0, 6.0), (22.0, 8.0)]  # leader lengths in pt and their penalties


def text_width(text: str, bold: bool = False) -> float:
    return sum(_WIDTHS.get(c, 0.89) for c in text) * FONT_PT * (1.16 if bold else 1.0)


def axis_limits(points: list[dict]) -> dict:
    lo = min(p["speed"] for p in points)
    hi = max(p["speed"] for p in points)
    xmin = 2.0 ** math.floor(math.log2(lo * 0.75))
    xmax = 2.0 ** math.ceil(math.log2(hi * 1.6))
    top = max(p["wer"] for p in points) + 1.5
    step = 2 if top <= 20 else 5
    return {"xmin": xmin, "xmax": xmax, "ymax": step * math.ceil(top / step), "ystep": step}


def frontier(points: list[dict]) -> list[tuple[float, float]]:
    """Staircase through the streaming systems no other streaming system beats on both axes."""
    best: list[dict] = []
    for p in sorted((q for q in points if q["class"] == "streaming" and q["frontier"]),
                    key=lambda q: (-q["speed"], q["wer"])):
        if not best or p["wer"] < best[-1]["wer"]:
            best.append(p)
    best.reverse()  # slowest and most accurate first
    stairs: list[tuple[float, float]] = []
    for a, b in zip(best, best[1:]):
        stairs += [(a["speed"], a["wer"]), (a["speed"], b["wer"])]
    if best:
        stairs.append((best[-1]["speed"], best[-1]["wer"]))
    return stairs


def _overlap(a, b, pad: float = 0.0) -> bool:
    return a[0] - pad < b[2] and b[0] - pad < a[2] and a[1] - pad < b[3] and b[1] - pad < a[3]


def _hits(seg, box) -> bool:
    """Does the segment ((x0, y0), (x1, y1)) pass through the box? Sampled every 1.5 pt."""
    (x0, y0), (x1, y1) = seg
    n = max(1, int(math.hypot(x1 - x0, y1 - y0) / 1.5))
    return any(box[0] < x0 + (x1 - x0) * i / n < box[2] and box[1] < y0 + (y1 - y0) * i / n < box[3]
               for i in range(n + 1))


def _candidates(p: dict) -> list[dict]:
    hw, hh, r = p["w"] / 2, LINE_PT / 2, p["clear"]
    out = []
    for sx, sy, pen in _NEAR:
        reach = r + GAP if 0 in (sx, sy) else 0.72 * r + GAP
        cx, cy = p["px"] + sx * (hw + reach), p["py"] + sy * (hh + reach)
        out.append({"box": (cx - hw, cy - hh, cx + hw, cy + hh), "leader": None, "pen": pen})
    for length, pen in _FAR:
        for k in range(12):
            ang = math.radians(15 + 30 * k)
            ux, uy = math.cos(ang), math.sin(ang)
            tip = (p["px"] + ux * (r + length), p["py"] + uy * (r + length))
            cx = tip[0] + (hw + 0.8 if ux > 0 else -hw - 0.8)
            cy = tip[1] + uy * hh
            out.append({"box": (cx - hw, cy - hh, cx + hw, cy + hh), "pen": pen + 0.01 * k,
                        "leader": ((p["px"] + ux * (r + 0.6), p["py"] + uy * (r + 0.6)), tip)})
    return out


def _cost(i: int, cand: dict, pts: list[dict], chosen: list, fixed: list, lines: list) -> float:
    box = cand["box"]
    cost = cand["pen"]
    if box[0] < 1 or box[1] < 1 or box[2] > AXIS_W - 1 or box[3] > AXIS_H - 1:
        cost += 1000
    cost += 1000 * sum(_overlap(box, f) for f in fixed)
    cost += 300 * sum(_hits(seg, box) for seg in lines)
    for j, q in enumerate(pts):
        mark = (q["px"] - q["r"], q["py"] - q["r"], q["px"] + q["r"], q["py"] + q["r"])
        near = math.hypot(q["px"] - pts[i]["px"], q["py"] - pts[i]["py"]) < 2.5  # the point itself or a twin on it
        if _overlap(box, mark, 0.3 if near else 0.6) or (cand["leader"] and not near and _hits(cand["leader"], mark)):
            cost += 1000
        if j != i and chosen[j] is not None:
            other = chosen[j]
            if _overlap(box, other["box"], PAD) or (cand["leader"] and _hits(cand["leader"], other["box"])) or (
                    other["leader"] and _hits(other["leader"], box)):
                cost += 1000
    return cost


def place(pts: list[dict], fixed: list, lines: list) -> list[dict]:
    """Greedy start, then sweeps that move one label at a time to its cheapest position."""
    order = sorted(range(len(pts)), key=lambda i: (-sum(math.hypot(pts[i]["px"] - q["px"], pts[i]["py"] - q["py"]) < 45
                                                         for q in pts), pts[i]["id"]))
    for p in pts:  # two systems on one spot: every label clears the larger of the stacked markers
        p["clear"] = max(q["r"] for q in pts if math.hypot(q["px"] - p["px"], q["py"] - p["py"]) < 2.5)
    cands = [_candidates(p) for p in pts]
    chosen: list = [None] * len(pts)
    for _ in range(60):
        changed = False
        for i in order:
            best = min(cands[i], key=lambda c: _cost(i, c, pts, chosen, fixed, lines))
            if best is not chosen[i]:
                chosen[i], changed = best, True
        if not changed:
            break
    bad = [pts[i]["id"] for i in range(len(pts)) if _cost(i, chosen[i], pts, chosen, fixed, lines) >= 300]
    if bad:
        raise PaperDataError(f"Figure 1 labels cannot be placed without collisions: {bad}; shorten their `short` labels")
    return chosen


def add_money(rows: list[dict], out: dict[str, str]) -> None:
    pts = []
    for r in rows:
        if (r["machine"], r["threads"]) != HEADLINE or not r["plot"]:
            continue
        d = derived(r)
        label = r["short"]
        pts.append({"id": r["id"], "speed": d["speed"], "wer": d["wer"], "class": r["class"], "role": r["role"],
                    "frontier": r["frontier"], "label": label,
                    "w": text_width(label, r["role"] == "ours") + 2 * 0.6, "r": MARK_R[r["role"]]})
    if not pts:
        out["money_layout.tex"] = "% Generated by python/ultra/tools/paper_data. No measured rows yet.\n"
        out["money_frontier.csv"] = csv_text(["speed", "wer"], [])
        return
    lim = axis_limits(pts)
    span = math.log10(lim["xmax"] / lim["xmin"])

    def to_pt(speed: float, wer: float) -> tuple[float, float]:
        return AXIS_W * math.log10(speed / lim["xmin"]) / span, AXIS_H * wer / lim["ymax"]

    for p in pts:
        p["px"], p["py"] = to_pt(p["speed"], p["wer"])
    stairs = frontier(pts)
    stair_pt = [to_pt(*v) for v in stairs]
    lines = list(zip(stair_pt, stair_pt[1:]))
    better = (AXIS_W - 47.0, 3.0, AXIS_W - 3.0, 14.0)  # the "better" arrow, bottom right
    chosen = place(pts, [better], lines)
    ticks = []
    t = lim["xmin"]
    while t <= lim["xmax"] * 1.0001:
        ticks.append(f"{t:g}")
        t *= 2
    body = ["% Generated by python/ultra/tools/paper_data (money.py). Do not edit by hand.",
            f"\\def\\MoneyAxisW{{{AXIS_W:g}pt}}\\def\\MoneyAxisH{{{AXIS_H:g}pt}}%",
            f"\\def\\MoneyXmin{{{lim['xmin']:g}}}\\def\\MoneyXmax{{{lim['xmax']:g}}}\\def\\MoneyYmax{{{lim['ymax']:g}}}%",
            f"\\def\\MoneyXticks{{{','.join(ticks)}}}\\def\\MoneyYstep{{{lim['ystep']:g}}}%",
            "\\def\\MoneyAnnotations{%"]
    for p, c in sorted(zip(pts, chosen), key=lambda t: t[0]["id"]):
        cx, cy = (c["box"][0] + c["box"][2]) / 2, (c["box"][1] + c["box"][3]) / 2
        text = p["label"].replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")
        body.append(f"  \\moneylabel{{{cx:.2f}}}{{{cy:.2f}}}{{{p['role']}}}{{{text}}}% {p['id']} w={p['w']:.2f}")
        if c["leader"]:
            (ax, ay), (bx, by) = c["leader"]
            body.append(f"  \\moneyleader{{{ax:.2f}}}{{{ay:.2f}}}{{{bx:.2f}}}{{{by:.2f}}}%")
    body.append("}%")
    out["money_layout.tex"] = "\n".join(body) + "\n"
    out["money_frontier.csv"] = csv_text(["speed", "wer"], [[f"{x:.4f}", f"{y:.3f}"] for x, y in stairs])
