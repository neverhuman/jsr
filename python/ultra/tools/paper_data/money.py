"""Figure 1 geometry: axis limits, the streaming frontier, labels, and the watts axis.

x is the speed on one pinned core, y the WER on the dev-clean half of the panel (400 utterances).

The plot box is fixed in points (pgfplots `scale only axis`), so every marker and label box is known
here. Labels are placed without collisions (labels.py); a label that cannot be placed becomes a number
with a key under the plot. Outputs: money_layout.tex (axis macros, labels, leaders, key, watts ticks)
and money_frontier.csv.
"""

from __future__ import annotations

from functools import lru_cache
import math
import statistics

from .cold import ENERGY_WORDS, speed_text
from .common import PaperDataError, csv_text, fmt
from .evidence import HEADLINE, derived
from .labels import LINE_PT, place
from .records import Records
from .words import name

AXIS_W, AXIS_H = 197.0, 205.0  # pt; with tick and axis labels the figure stays under 243 pt wide
FONT_PT = 7.0  # \scriptsize labels
MARK_R = {"ours": 3.3, "comparator": 2.3}  # half-extent of the drawn marker
BETTER = (AXIS_W - 47.0, 3.0, AXIS_W - 3.0, 14.0)  # the "better" arrow, bottom right
BAND = (0.0, AXIS_H - 62.0, 9.5, AXIS_H)           # the rotated "slower than real time" note, top left
WATTS_SPREAD = 0.15  # a watts axis is drawn only if every system is within this of the common power per core
# Width of a 7 pt Latin Modern glyph in em (upper bounds, checked against TeX by the tests).
_WIDTHS = {**dict.fromkeys("il.,:;'", 0.32), **dict.fromkeys("fj", 0.36), **dict.fromkeys("rst()", 0.45),
           **dict.fromkeys("cez", 0.51), **dict.fromkeys("ago0123456789/", 0.57), **dict.fromkeys("kqvxy", 0.60),
           **dict.fromkeys("bdhnpu", 0.64), "w": 0.82, "m": 0.95, " ": 0.39, "-": 0.39, "I": 0.43, "J": 0.59,
           **dict.fromkeys("SZL", 0.71), **dict.fromkeys("FEPB", 0.80), **dict.fromkeys("CTR", 0.83),
           **dict.fromkeys("AHNUVXYDKGOQ", 0.89), "M": 1.04, "W": 1.16, "\u00b7": 0.32}


def text_width(text: str, bold: bool = False) -> float:
    return sum(_WIDTHS.get(c, 0.89) for c in text) * FONT_PT * (1.16 if bold else 1.0)


def axis_limits(points: list[dict]) -> dict:
    lo = min(p["speed"] for p in points)
    hi = max(p["speed"] for p in points)
    top = max(p["wer"] for p in points) + 0.5  # the top point's label goes beside it, not above
    step = 2 if top <= 20 else 5
    return {"xmin": 2.0 ** math.floor(math.log2(lo * 0.75)), "xmax": 2.0 ** math.ceil(math.log2(hi * 1.6)),
            "ymax": step * math.ceil(top / step), "ystep": step}


def frontier(points: list[dict]) -> list[dict]:
    """The streaming systems no other streaming system beats on both axes, slowest first."""
    best: list[dict] = []
    for p in sorted((q for q in points if q["class"] == "streaming" and q["frontier"]),
                    key=lambda q: (-q["speed"], q["wer"])):
        if not best or p["wer"] < best[-1]["wer"]:
            best.append(p)
    return best[::-1]


def stairs(best: list[dict]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in zip(best, best[1:]):
        out += [(a["speed"], a["wer"]), (a["speed"], b["wer"])]
    return out + ([(best[-1]["speed"], best[-1]["wer"])] if best else [])


def watts_axis(points: list[dict], lim: dict) -> dict | None:
    """Ticks of a top axis in watts per live stream, or None when the rows do not support one.

    watts = power per busy core x CPU-seconds per audio-second, so one common power per core turns the
    speed axis into a watts axis. It is drawn only if every drawn system has a measured value and lies
    within WATTS_SPREAD of the median power per core.
    """
    if not points or any(p["watts"] is None for p in points):
        return None
    per_core = [p["watts"] * p["speed"] for p in points]  # watts while one core is busy
    k = statistics.median(per_core)
    spread = max(abs(x / k - 1.0) for x in per_core)
    if spread > WATTS_SPREAD:
        return None
    ticks, decade = [], 10.0 ** math.floor(math.log10(k / lim["xmax"]))
    while decade <= k / lim["xmin"]:
        ticks += [w for w in (decade, 2 * decade, 5 * decade) if k / lim["xmax"] <= w <= k / lim["xmin"]]
        decade *= 10
    return {"k": k, "spread": spread, "ticks": [(k / w, f"{w:g}") for w in ticks]}


@lru_cache(maxsize=32)
def _layout(spec: tuple, lines: tuple) -> tuple:
    """Place the labels; a label that cannot be placed becomes a number. Cached: builds repeat it."""
    pts = [{"id": i, "label": label, "bold": bold, "px": px, "py": py, "r": r, "number": None,
            "w": max(text_width(line, bold) for line in label.split("\n")) + 1.2,  # label: plain text, callout included
            "h": LINE_PT * (1 + 1.15 * label.count("\n"))} for i, label, bold, px, py, r in spec]
    numbered: list[dict] = []
    for _ in range(8):
        res = place(pts, [BETTER, BAND], list(lines), (AXIS_W, AXIS_H))
        bad = [p for p, c in zip(pts, res) if not c["ok"]]
        if not bad:
            return tuple((p["id"], p["number"], c["box"], c["leader"]) for p, c in zip(pts, res))
        fresh = [p for p in bad if p["number"] is None]
        if not fresh:
            break
        fresh = [p for p in fresh if not p["bold"]] or fresh  # our point keeps its name while any other can give way
        fresh.sort(key=lambda p: (-sum(math.hypot(q["px"] - p["px"], q["py"] - p["py"]) < 25 for q in pts), -p["w"], p["id"]))
        for p in fresh[:max(1, (len(fresh) + 1) // 2)]:  # the most crowded half first
            numbered.append(p)
            p["number"] = len(numbered)
            p["w"] = text_width(str(p["number"]), True) + 1.2
    raise PaperDataError(f"Figure 1 cannot be labelled without collisions: {[p['id'] for p in bad]}")


def _tex(text: str) -> str:
    return (text.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")
            .replace(" \u00b7 ", "\\,$\\cdot$\\,"))


def _note(rec: Records, rows: list[dict], pts: list[dict], axis: dict | None) -> str:
    """Record the caption's watts sentence (TeX) and return the same statement as plain text."""
    have = sum(p["watts"] is not None for p in pts)
    if axis:
        text = (f"Top axis: watts one live stream costs, at the median measured power per busy core "
                f"({axis['k']:.2f}\\,W; every system is within {100 * axis['spread']:.0f}\\% of it).")
    elif have:
        method = ENERGY_WORDS[next(p["row"]["energy_method"] for p in pts if p["watts"] is not None)]
        text = f"Power is reported in watts per always-on stream ({method}, same run)."
    else:
        text = "No watts drawn yet: no same-run energy measurement has been admitted."
    first = rows[0] if rows else None
    rec.add("MoneyWattsCaption", text, unit="sentence", label="reported",
            configuration="generated from the energy rows of the systems drawn in Figure 1",
            workload=first["workload"] if first else "pending (no evidence rows yet)",
            workload_digest=first["workload_digest"] if first else "pending", source="docs/paper/data/evidence/*.jsonl")
    return text.replace("\\,", " ").replace("\\%", "%")


def add_money(rows: list[dict], out: dict[str, str], rec: Records, comparison: tuple[dict, ...]) -> str:
    """Write the Figure 1 files, mark the rows on the frontier, record the caption's watts sentence.
    Returns that sentence as plain text (for the README block)."""
    pts = []
    for r in rows:
        r["frontier_point"] = False
        if (r["machine"], r["threads"]) != HEADLINE or not r["plot"]:
            continue
        d = derived(r)
        # y is the WER on the dev-clean half: the number model cards and our dashboard lead with
        pts.append({"id": r["id"], "speed": d["speed"], "wer": d["wer_clean"], "class": r["class"], "role": r["role"],
                    "frontier": r["frontier"], "label": r["short"], "watts": d["watts"], "row": r})
    if not pts:
        out["money_layout.tex"] = "% Generated by python/ultra/tools/paper_data. No measured rows yet.\n"
        out["money_frontier.csv"] = csv_text(["speed", "wer_clean"], [])
        first = rows[0] if rows else None
        rec.add("MoneyMarksCaption", "\\relax", unit="sentence", label="reported",
                configuration="generated: the marks drawn in Figure 1 (none yet)",
                workload=first["workload"] if first else "pending (no evidence rows yet)",
                workload_digest=first["workload_digest"] if first else "pending", source="docs/paper/data/evidence/*.jsonl")
        return _note(rec, rows, pts, None)
    lim = axis_limits(pts)
    span = math.log10(lim["xmax"] / lim["xmin"])

    def to_pt(speed: float, wer: float) -> tuple[float, float]:
        return round(AXIS_W * math.log10(speed / lim["xmin"]) / span, 3), round(AXIS_H * wer / lim["ymax"], 3)

    best = frontier(pts)
    for p in best:
        p["row"]["frontier_point"] = True
    line = stairs(best)
    line_pt = [to_pt(*v) for v in line]
    axis = watts_axis(pts, lim)
    for p in pts:  # call out power beside each point when neither the axis nor the comparison table shows it
        callout = None if axis or p["watts"] is None or p["row"] in comparison else fmt(p["watts"], 3)
        p["plain"] = p["label"] + (f" - {callout} W" if callout else "")
        p["tex"] = _tex(p["label"]) + (f"\\,$\\cdot$\\,{callout}\\,W" if callout else "")
    spec = tuple((p["id"], p["plain"], p["role"] == "ours", *to_pt(p["speed"], p["wer"]), MARK_R[p["role"]]) for p in pts)
    placed = {i: (number, box, leader) for i, number, box, leader in _layout(spec, tuple(zip(line_pt, line_pt[1:])))}
    ticks, t = [], lim["xmin"]
    while t <= lim["xmax"] * 1.0001:
        ticks.append(f"{t:g}")
        t *= 2
    has = {"frontier": len(best) > 1,
           "stream": any(p["class"] == "streaming" and p["role"] != "ours" and not p["row"]["faded"] for p in pts),
           "whole": any(p["class"] == "whole_utterance" and not p["row"]["faded"] for p in pts),
           "faded": any(p["row"]["faded"] for p in pts)}
    marks = ["Diamond: ours."]
    if has["whole"] or any(p["class"] == "whole_utterance" for p in pts):
        marks.append("Hollow: whole-utterance systems, which see the whole clip before they emit a word.")
    if has["frontier"]:
        marks.append("Dashed: no other streaming system is better on both axes.")
    first = rows[0]
    rec.add("MoneyMarksCaption", " ".join(marks), unit="sentence", label="reported",
            configuration="generated: the marks drawn in Figure 1", workload=first["workload"],
            workload_digest=first["workload_digest"], source="docs/paper/data/evidence/*.jsonl")
    comparison_rows = []
    for row in comparison:
        d = derived(row)
        comparison_rows.append(" & ".join((_tex(name(row)), f"{speed_text(d['speed'])}$\\times$",
                                            fmt(d["watts"], 3), f"{fmt(d['wer_clean'], 2)}\\%",
                                            f"{fmt(d['wer'], 2)}\\%")) + r"\\")
    body = ["% Generated by python/ultra/tools/paper_data (money.py). Do not edit by hand.",
            f"\\def\\MoneyAxisW{{{AXIS_W:g}pt}}\\def\\MoneyAxisH{{{AXIS_H:g}pt}}%",
            f"\\def\\MoneyXmin{{{lim['xmin']:g}}}\\def\\MoneyXmax{{{lim['xmax']:g}}}\\def\\MoneyYmax{{{lim['ymax']:g}}}%",
            f"\\def\\MoneyXticks{{{','.join(ticks)}}}\\def\\MoneyYstep{{{lim['ystep']:g}}}%",
            f"\\def\\MoneyWattsOn{{{1 if axis else 0}}}%",
            "\\def\\MoneyHasFrontier{%d}\\def\\MoneyHasStream{%d}\\def\\MoneyHasWhole{%d}\\def\\MoneyHasFaded{%d}%%" % (
                has["frontier"], has["stream"], has["whole"], has["faded"]),
            f"\\def\\MoneyWattTicks{{{','.join(f'{x:.5g}' for x, _ in axis['ticks']) if axis else ''}}}%",
            f"\\def\\MoneyWattLabels{{{','.join(w for _, w in axis['ticks']) if axis else ''}}}%",
            "\\def\\MoneyComparisonRows{" + " ".join(comparison_rows) + "}%",
            "\\def\\MoneyAnnotations{%"]
    key = []
    for p in sorted(pts, key=lambda p: p["id"]):
        number, box, leader = placed[p["id"]]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        text = p["tex"] if number is None else str(number)
        style = "ours" if p["role"] == "ours" else "comparator" if number is None else "number"
        body.append(f"  \\moneylabel{{{cx:.2f}}}{{{cy:.2f}}}{{{style}}}{{{text}}}% {p['id']} w={box[2] - box[0]:.2f}")
        if leader:
            (ax, ay), (bx, by) = leader
            body.append(f"  \\moneyleader{{{ax:.2f}}}{{{ay:.2f}}}{{{bx:.2f}}}{{{by:.2f}}}%")
        if number is not None:
            key.append((number, p["tex"]))
    body.append("}%")
    body.append("\\def\\MoneyKey{" + "\\quad ".join(f"\\mbox{{\\textbf{{{n}}}~{name}}}" for n, name in sorted(key)) + "}%")
    out["money_layout.tex"] = "\n".join(body) + "\n"
    out["money_frontier.csv"] = csv_text(["speed", "wer_clean"], [[f"{x:.4f}", f"{y:.3f}"] for x, y in line])
    return _note(rec, rows, pts, axis)
