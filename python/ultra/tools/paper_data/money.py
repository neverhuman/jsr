"""Figure 1 geometry: axis limits, the streaming frontier, labels, and the watts axis.

x is the speed on one pinned core on a linear scale, y the pooled WER of the 730-utterance panel.

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
from .words import name, signed as _sg, tex as _words_tex

AXIS_W, AXIS_H = 197.0, 168.0  # pt; with tick, 4-row header table, legend and key the figure fits page 1 cleanly
FONT_PT = 7.0  # \scriptsize labels
MARK_R = {"ours": 3.3, "comparator": 2.3}  # half-extent of the drawn marker
BETTER = (AXIS_W - 47.0, 3.0, AXIS_W - 3.0, 14.0)  # the "better" arrow, bottom right
WATTS_SPREAD = 0.15  # a watts axis is drawn only if every system is within this of the common power per core
# Width of a 7 pt Latin Modern glyph in em (upper bounds, checked against TeX by the tests).
_WIDTHS = {**dict.fromkeys("il.,:;'", 0.32), **dict.fromkeys("fj", 0.36), **dict.fromkeys("rst()", 0.45),
           **dict.fromkeys("cez", 0.51), **dict.fromkeys("ago0123456789/", 0.57), **dict.fromkeys("kqvxy", 0.60),
           **dict.fromkeys("bdhnpu", 0.64), "w": 0.82, "m": 0.95, " ": 0.39, "-": 0.39, "I": 0.43, "J": 0.59,
           **dict.fromkeys("SZL", 0.71), **dict.fromkeys("FEPB", 0.80), **dict.fromkeys("CTR", 0.83),
           **dict.fromkeys("AHNUVXYDKGOQ", 0.89), "M": 1.04, "W": 1.16, "\u00b7": 0.32, "\u00a7": 0.57}


def text_width(text: str, bold: bool = False) -> float:
    return sum(_WIDTHS.get(c, 0.89) for c in text) * FONT_PT * (1.16 if bold else 1.0)


def axis_limits(points: list[dict]) -> dict:
    hi = max(p["speed"] for p in points)
    xstep = 5 * math.ceil(hi / 35)
    top = max(p["wer"] for p in points) + 0.5  # the top point's label goes beside it, not above
    step = 2 if top <= 20 else 5
    return {"xmin": 0, "xmax": xstep * math.ceil(hi * 1.1 / xstep), "xstep": xstep,
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
    low = k / lim["xmax"]
    high = k / min(p["speed"] for p in points)  # the reciprocal scale stops at the slowest measured system
    ticks, decade = [], 10.0 ** math.floor(math.log10(low))
    while decade <= high:
        ticks += [w for w in (decade, 2 * decade, 5 * decade) if low <= w <= high]
        decade *= 10
    return {"k": k, "spread": spread, "ticks": [(k / w, f"{w:g}") for w in ticks]}


@lru_cache(maxsize=32)
def _layout(spec: tuple, lines: tuple, reserved: tuple = ()) -> tuple:
    """Place the labels; a label that cannot be placed becomes a number. Cached: builds repeat it."""
    pts = [{"id": i, "label": label, "bold": bold, "px": px, "py": py, "r": r, "number": None,
            "w": max(text_width(line, bold) for line in label.split("\n")) + 1.2,  # label: plain text, callout included
            "h": LINE_PT * (1 + 1.15 * label.count("\n"))} for i, label, bold, px, py, r in spec]
    numbered: list[dict] = []
    for _ in range(8):
        res = place(pts, [BETTER, *reserved], list(lines), (AXIS_W, AXIS_H))
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
        text = f"Active core W per stream ({method}, same run), not device power."
    else:
        text = "No watts drawn yet: no same-run energy measurement has been admitted."
    first = rows[0] if rows else None
    rec.add("MoneyWattsCaption", text, unit="sentence", label="reported",
            configuration="generated from the energy rows of the systems drawn in Figure 1",
            workload=first["workload"] if first else "pending (no evidence rows yet)",
            workload_digest=first["workload_digest"] if first else "pending", source="docs/paper/data/evidence/*.jsonl")
    return text.replace("\\,", " ").replace("\\%", "%")


HERO_VARIANTS = {
    # Our 4 operating points on one 60 MB INT8 artifact (quiet H1, same interleaved blocks)
    "LiveLmWide": "rc2-live-lm320",
    "LiveLm": "rc2-live-lm",
    "LiveGreedy": "rc2-live-greedy",
    "Final": "rc2-final",
    # All 12 timed Zen 2 public comparators (quiet H1 primary, plus 3 H0 interim rows)
    "Zip": "r0-int8-r6",
    "MoonTiny": "moonshine-tiny",
    "MoonSmall": "moonshine-small",
    "VoskSmall": "vosk-small",
    "VoskLarge": "vosk-large",
    "ParakeetOnnx": "parakeet-v3-onnxasr-int8",
    "ParakeetSherpa": "parakeet-tdt-v3-int8",
    "NemoFiveSixty": "nemotron-en-560",
    "NemoOneSixty": "nemotron-en-160",
    "NemoThreeFive": "nemotron-35",
    "WhisperTiny": "whispercpp-stream-tiny",
    "WhisperBase": "whispercpp-stream-base",
}
_HERO_QUIET_FULL = {"rc2-live-lm320", "rc2-live-lm", "rc2-live-greedy", "rc2-final", "r0-int8-r6", "parakeet-v3-onnxasr-int8"}


def hero_rows(rows: list[dict]) -> dict[str, dict]:
    """Select all 4 Isosyll modes and all 12 timed Zen 2 public comparators on one 730-utt DEV workload."""
    matched: dict[str, dict] = {}
    for row in rows:
        if (row["machine"], row["threads"]) != HEADLINE or row.get("anchor"):
            continue
        v = row["variant"]
        if v not in HERO_VARIANTS.values():
            continue
        if v in _HERO_QUIET_FULL:
            if row["host"] != "xbabe0" or row["interim"] or row["timing_panel"] != "full" or len(row.get("cpu_reps", [])) != 3:
                continue
        elif not row["plot"]:
            continue
        matched[v] = row
    if not set(HERO_VARIANTS.values()) <= matched.keys():
        return {}
    chosen = {label: matched[variant] for label, variant in HERO_VARIANTS.items()}
    return chosen if len({row["workload_digest"] for row in chosen.values()}) == 1 else {}


def hero_sentences(hero: dict[str, dict], rec: Records) -> dict[str, str]:
    """Figure 1 headline and README summary. The figure draws live +LM at 160 ms and final only."""
    d = {label: derived(row) for label, row in hero.items()}
    live, fin, zip_ = d["LiveLm"], d["Final"], d["Zip"]
    gap_c = rec.rows["CandLiveLmCDelta"]["value"].replace("$-$", "\u2212")
    headline = (f"Isosyll vs. all 12 timed public CPU speech recognizers on one pinned Zen 2 core: "
                f"live +LM runs at {speed_text(live['speed'])}\u00d7 ({fmt(live['watts'], 3)} J/audio-s, {fmt(live['wer'], 2)}% WER at 160 ms; "
                f"{gap_c} pp on unselected DEV) and the utterance-end commit pass reaches {fmt(fin['wer'], 2)}% at {speed_text(fin['speed'])}\u00d7 "
                f"vs. Zipformer INT8 at {speed_text(zip_['speed'])}\u00d7 ({fmt(zip_['watts'], 3)} J/audio-s, {fmt(zip_['wer'], 2)}%).")
    subtitle = (f"Live +LM: {speed_text(live['speed'])}\u00d7, {fmt(live['watts'], 3)} J/audio-s, {fmt(live['wer'], 2)}% "
                f"\u00b7 Final: {speed_text(fin['speed'])}\u00d7, {fmt(fin['wer'], 2)}%")
    return {"headline": headline, "subtitle": subtitle}


def nemotron_timing_note(rows: list[dict]) -> str:
    row = next((row for row in rows if row["variant"] == "nemotron-en-560"
                and row["host"] == "xbabe0" and row["timing_panel"] == "p73"
                and not row["interim"] and len(row.get("cpu_reps", [])) == 3), None)
    if row is None:
        return ""
    data = derived(row)
    return (f"Nemotron 560 ms scores {fmt(data['wer'], 2)}% WER on the same 730 utterances "
            f"and runs at {speed_text(data['speed'])}× in the separate quiet 73-clip timing test.")


# Energy ramp: green (lowest J/audio-s) -> amber -> red (highest). OKLab lightness falls monotonically with energy,
# so the order survives grayscale and colour-blind viewing; hue carries the green/amber/red reading.
RAMP_ANCHORS = ("#10B981", "#84CC16", "#F59E0B", "#EA580C", "#DC2626")  # emerald, lime, amber, orange, red (saturated twins of the table cell tints); hue and chroma only # hue and chroma come from these; lightness is set linearly below
RAMP_L = (0.73, 0.43)  # OKLab lightness at the lowest and the highest energy
RAMP_STEPS = 48
NOT_METERED_HEX = "9AA1AD"
HERO_SHOW_FINAL = False  # one switch: True also plots Isosyll final (its spot has a free label slot in money_focused.tex)
HERO_PLOTTED = (*(("Final",) if HERO_SHOW_FINAL else ()), "LiveLm", "Zip", "MoonTiny", "MoonSmall", "VoskSmall", "VoskLarge", "ParakeetOnnx", "ParakeetSherpa",
                "NemoFiveSixty", "NemoOneSixty", "NemoThreeFive", "WhisperTiny", "WhisperBase")


def _oklab(hex_: str) -> tuple[float, float, float]:
    lin = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(int(hex_.lstrip("#")[i:i + 2], 16) / 255) for i in (0, 2, 4))
    l, m, s_ = (math.cbrt(v) for v in (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b,
                                       0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b,
                                       0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b))
    return (0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s_,
            1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s_,
            0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s_)


def _from_oklab(lab: tuple[float, float, float]) -> str:
    L, a, b = lab
    l, m, s_ = ((L + 0.3963377774 * a + 0.2158037573 * b) ** 3, (L - 0.1055613458 * a - 0.0638541728 * b) ** 3,
                (L - 0.0894841775 * a - 1.291485548 * b) ** 3)
    rgb = (4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s_, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s_,
           -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s_)
    enc = lambda c: 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
    return "".join(f"{round(255 * min(1.0, max(0.0, enc(max(c, 0.0))))):02X}" for c in rgb)


def ramp_hex(t: float) -> str:
    """Colour at position t in [0, 1] along the energy ramp (0 = lowest energy = green)."""
    labs = [_oklab(h) for h in RAMP_ANCHORS]
    t = min(1.0, max(0.0, t)) * (len(labs) - 1)
    i = min(int(t), len(labs) - 2)
    f = t - i
    _, a, b = (x + (y - x) * f for x, y in zip(labs[i], labs[i + 1]))
    return _from_oklab((RAMP_L[0] + (RAMP_L[1] - RAMP_L[0]) * t / (len(labs) - 1), a, b))


def _decade_ticks(lo: float, hi: float) -> list[float]:
    return [m * 10.0 ** k for k in range(-3, 4) for m in (1, 3) if lo * 0.999 <= m * 10.0 ** k <= hi * 1.001]


def energy_range(values: list[float]) -> tuple[float, float]:
    """Colour-bar range on the 1, 2, 3, 5 series: the series value at or below the lowest energy and at or above the highest."""
    series = sorted(round(m * 10.0 ** k, 6) for k in range(-3, 4) for m in (1, 2, 3, 5))
    return max(v for v in series if v <= min(values) * 1.0001), min(v for v in series if v >= max(values) * 0.9999)


def hero_floor_included(chosen: dict[str, dict]) -> list[str]:
    """Plotted systems whose energy value still includes the idle floor (release-run receipts without the R7 value)."""
    return [k for k in HERO_PLOTTED if k in hero_energy(chosen) and chosen[k].get("energy_idle_subtracted") is None
            and chosen[k].get("energy_includes_idle")]


def hero_energy(chosen: dict[str, dict]) -> dict[str, float]:
    """J per audio-second, idle-subtracted, per plotted system: the matched-campaign (R7) value when its row has one,
    else the release-run core energy. A system in neither is absent (drawn grey, named in the caption)."""
    energy = {}
    for label in HERO_PLOTTED:
        row = chosen[label]
        value = row.get("energy_idle_subtracted")
        if value is None and row.get("energy_method") == "core_msr_active":
            value = row.get("watts")
        if value is not None:
            energy[label] = float(value)
    return energy


def _hero_layout(chosen: dict[str, dict], rec: Records) -> list[str]:
    points = {label: derived(row) for label, row in chosen.items()}
    energy = hero_energy(chosen)
    live_ratio = chosen["LiveLm"]["paired_vs_anchor"]["speed"]
    macros = {"Xmax": 35, "Xticks": "0,5,10,15,20,25,30,35", "Ymax": 20, "Ratio": fmt(live_ratio, 1)}
    for label in HERO_PLOTTED:
        point = points[label]
        macros.update({label + "X": f"{point['speed']:.6f}", label + "Y": f"{point['wer']:.6f}"})
    lo, hi = energy_range(list(energy.values())) if energy else (0.3, 10.0)
    span = math.log(hi / lo)
    pos = lambda v: math.log(v / lo) / span
    for label in HERO_PLOTTED:
        macros[label + "Color"] = ramp_hex(pos(energy[label])) if label in energy else NOT_METERED_HEX
        macros[label + "Metered"] = int(label in energy)
        macros[label + "Mark"] = "\\,\\ddag" if chosen[label]["faded"] else ""  # shared-host interim timing
        if label in energy:
            macros[label + "Energy"] = fmt(energy[label], 3)
    macros["RampHex"] = ",".join(ramp_hex(i / (RAMP_STEPS - 1)) for i in range(RAMP_STEPS))
    macros["RampTicks"] = ",".join(f"{pos(v):.5f}/{fmt(v, 0) if v >= 1 else fmt(v, 1)}" for v in sorted({round(v, 6) for v in (lo, hi, *_decade_ticks(lo, hi))}))
    macros["RampLo"], macros["RampHi"] = f"{lo:g}", f"{hi:g}"
    macros["NotMetered"] = ", ".join(_tex(chosen[k]["short"]) for k in HERO_PLOTTED if k not in energy) or "none"
    macros["ShowFinal"] = int(HERO_SHOW_FINAL)
    # the colorbar may only say "idle-subtracted" when every colored value is idle-subtracted
    macros["BarTitle"] = ("Energy per audio-second (J, idle-subtracted, log)" if not hero_floor_included(chosen)
                          else "Energy per audio-second (J, log)")
    floor = [_tex(chosen[k]["short"]) for k in hero_floor_included(chosen)]
    macros["EnergyNote"] = ("" if not floor else
                            "Energy: idle floor subtracted, except " + ", ".join(floor[:-1]) + (" and " if len(floor) > 1 else "")
                            + floor[-1] + " (floor included; not a paired ratio). ")
    macros["AnyNotMetered"] = int(len(energy) < len(HERO_PLOTTED))
    macros["AnyFaded"] = int(any(chosen[k]["faded"] for k in HERO_PLOTTED))
    for key, tag, mode in (("Panel", "Panel", "LiveLm"), ("C", "C", "LiveLm"), ("CWide", "C", "LiveLmEight")):
        macros["Gap" + key] = "%s [%s, %s]" % tuple(rec.rows[f"Cand{mode}{tag}{x}"]["value"] for x in ("Delta", "Lo", "Hi"))
    macros["GapCDelta"] = rec.rows["CandLiveLmCDelta"]["value"]
    macros["GapCWideDelta"] = rec.rows["CandLiveLmEightCDelta"]["value"]
    return [f"\\def\\Hero{key}{{{value}}}%" for key, value in macros.items()]


def add_money(rows: list[dict], out: dict[str, str], rec: Records, comparison: tuple[dict, ...], hero: dict[str, dict], nemotron_note: str) -> str:
    """Write the Figure 1 files, mark the rows on the frontier, record the caption's watts sentence.
    Returns that sentence as plain text (for the README block)."""
    pts = []
    for r in rows:
        r["frontier_point"] = False
        if (r["machine"], r["threads"]) != HEADLINE or not r["plot"]:
            continue
        d = derived(r)
        # y is pooled WER across all 730 utterances (400 dev-clean + 330 dev-other)
        lbl = r["system"] if r["role"] == "ours" else r["short"]
        pts.append({"id": r["id"], "speed": d["speed"], "wer": d["wer"], "class": r["class"], "role": r["role"],
                    "frontier": r["frontier"], "label": lbl, "watts": d["watts"], "row": r})
    if not pts:
        out["money_layout.tex"] = "% Generated by python/ultra/tools/paper_data. No measured rows yet.\n"
        out["money_layout.tex"] += "\\def\\MoneyHeroOn{0}%\n"
        out["money_frontier.csv"] = csv_text(["speed", "wer"], [])
        first = rows[0] if rows else None
        rec.add("MoneyMarksCaption", "\\relax", unit="sentence", label="reported",
                configuration="generated: the marks drawn in Figure 1 (none yet)",
                workload=first["workload"] if first else "pending (no evidence rows yet)",
                workload_digest=first["workload_digest"] if first else "pending", source="docs/paper/data/evidence/*.jsonl")
        return _note(rec, rows, pts, None)
    lim = axis_limits(pts)
    span = lim["xmax"] - lim["xmin"]

    def to_pt(speed: float, wer: float) -> tuple[float, float]:
        return round(AXIS_W * (speed - lim["xmin"]) / span, 3), round(AXIS_H * wer / lim["ymax"], 3)

    best = frontier(pts)
    for p in best:
        p["row"]["frontier_point"] = True
    line = stairs(best)
    line_pt = [to_pt(*v) for v in line]
    frontier_len = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(line_pt, line_pt[1:]))
    visible_frontier = len(best) > 1 and frontier_len >= 12.0
    axis = watts_axis(pts, lim) if not hero else None
    for p in pts:  # call out power only when the figure has neither a watts axis nor an inset table
        callout = None if axis or comparison or p["watts"] is None else fmt(p["watts"], 3)
        sub_p = " \u00a7" if p["row"]["timing_panel"] != "full" else ""
        sub_t = "~\\S" if p["row"]["timing_panel"] != "full" else ""
        if p["role"] == "ours" and not callout and " (" in p["label"]:
            head, tail = p["label"].split(" (", 1)
            p["plain"] = f"{head}\n({tail}{sub_p}"
            p["tex"] = f"{_tex(head)}\\\\({_tex(tail)}{sub_t}"
        else:
            p["plain"] = p["label"] + sub_p + (f" - {callout} W" if callout else "")
            p["tex"] = _tex(p["label"]) + sub_t + (f"\\,$\\cdot$\\,{callout}\\,W" if callout else "")
    spec = tuple((p["id"], p["plain"], p["role"] == "ours", *to_pt(p["speed"], p["wer"]), MARK_R[p["role"]]) for p in pts)
    quiet_full = {p["id"]: p for p in pts if p["frontier"] and p["row"]["timing_panel"] == "full"}
    zipformer = quiet_full.get("r0-int8-r6-zen2-t1")
    live = quiet_full.get("rc2-live-lm-zen2-t1")
    speed_gap, reserved = "", ()
    if zipformer and live and live["speed"] > zipformer["speed"]:
        left = to_pt(zipformer["speed"], 0)[0]
        right = to_pt(live["speed"], 0)[0]
        y = min(AXIS_H - 14.0, to_pt(0, max(live["wer"], zipformer["wer"]))[1] + 9.0)
        ratio = live["speed"] / zipformer["speed"]
        text = f"{ratio:.1f}× as fast as Zipformer INT8"
        half_width = max((right - left) / 2, text_width(text, bold=True) / 2)
        center = (left + right) / 2
        reserved = ((center - half_width - 2, y - 2, center + half_width + 2, y + 13),)
        speed_gap = (f"\\draw[{{Stealth[length=3.5pt]}}-{{Stealth[length=3.5pt]}}, okBlue, line width=0.6pt] "
                     f"([xshift={left:.2f}pt, yshift={y:g}pt] rel axis cs:0,0) -- "
                     f"([xshift={right:.2f}pt, yshift={y:g}pt] rel axis cs:0,0) "
                     f"node[midway, above=2pt, font=\\scriptsize\\bfseries] {{{ratio:.1f}$\\times$ as fast as Zipformer INT8}};")
    placed = {i: (number, box, leader) for i, number, box, leader in _layout(spec, tuple(zip(line_pt, line_pt[1:])), reserved)}
    ticks = range(lim["xmin"], lim["xmax"] + 1, lim["xstep"])
    has = {"frontier": visible_frontier,
           "stream": any(p["class"] == "streaming" and p["role"] != "ours" and not p["row"]["faded"] for p in pts),
           "whole": any(p["class"] == "whole_utterance" and not p["row"]["faded"] for p in pts),
           "faded": any(p["row"]["faded"] for p in pts)}
    ours_pts = sorted((p for p in pts if p["role"] == "ours"), key=lambda p: p["speed"])
    ours_conn = ""
    if len(ours_pts) == 2:
        a, b = to_pt(ours_pts[0]["speed"], ours_pts[0]["wer"]), to_pt(ours_pts[1]["speed"], ours_pts[1]["wer"])
        ours_conn = (f"\\draw[okBlue!70, line width=0.8pt, densely dotted] "
                     f"([xshift={a[0]:.2f}pt, yshift={a[1]:.2f}pt] rel axis cs:0,0) -- "
                     f"([xshift={b[0]:.2f}pt, yshift={b[1]:.2f}pt] rel axis cs:0,0);")
        names = [_tex(p["row"]["system"]) for p in reversed(ours_pts)]
        ours_legend = r"\shortstack[l]{" + r"\\".join(names) + "}"
        marks = ["Orange diamonds: ours (same INT8 weights)."]
    else:
        ours_legend = "Isosyll v3.0.0 (final, seg.)"
        marks = ["Orange diamond: ours (final commit pass)."]
    if has["whole"] or any(p["class"] == "whole_utterance" for p in pts):
        marks.append("Hollow: offline.")
    n_faded = sum(1 for p in pts if p["row"]["faded"])
    if n_faded:
        marks.append(f"Lighter: H0 interim ({n_faded}).")
    n_sub = sum(1 for p in pts if p["row"]["timing_panel"] != "full")
    if n_sub:
        marks.append(f"\\S: 73-clip timing ({n_sub}).")
    first = rows[0]
    rec.add("MoneyFullMarksCaption", " ".join(marks), unit="sentence", label="reported",
            configuration="generated: broader comparison figure", workload=first["workload"],
            workload_digest=first["workload_digest"], source="docs/paper/data/evidence/*.jsonl")
    if hero:
        unmetered = [_tex(hero[k]["short"]) for k in HERO_PLOTTED if k not in hero_energy(hero)]
        floor = [_tex(hero[k]["short"]) for k in hero_floor_included(hero)]
        floor_txt = f"idle floor subtracted except for {', '.join(floor)}, which include it" if floor else "idle floor subtracted"
        grey = f"; grey = not metered: {', '.join(unmetered)}" if unmetered else ""
        marks = ["Single-core throughput against pooled WER on the 730-utterance DEV panel; names and marker fills show core energy "
                 f"per audio-second on a log scale, green lowest (AMD core-energy registers, {floor_txt}; Isosyll live and Zipformer INT8 in the same blocks, others in separate sessions; not wall power{grey}).",
                 "Isosyll live is the 160\\,ms chunk of one \\ColdHeadModel\\,MB INT8 artifact, timed with Zipformer INT8 in the same quiet blocks; "
                 "circles stream, squares are offline whole-clip, \\S\\ marks timing on a 73-clip subset only"
                 + (", and \\ddag\\ marks interim timing on a shared host" if any(hero[k]["faded"] for k in HERO_PLOTTED) else "")
                 + "."]
    rec.add("MoneyMarksCaption", " ".join(marks), unit="sentence", label="reported",
            configuration="generated: the marks drawn in Figure 1", workload=first["workload"],
            workload_digest=first["workload_digest"], source="docs/paper/data/evidence/*.jsonl")
    comparison_rows = []
    for row in comparison:
        d = derived(row)
        lbl = f"\\textbf{{{_tex(row['system'])}}}" if row["role"] == "ours" else _tex(row["short"])
        w_cell = "--" if d["watts"] is None else fmt(d["watts"], 3)
        comparison_rows.append(" & ".join((lbl, f"{speed_text(d['speed'])}$\\times$",
                                            w_cell, f"{fmt(d['wer_clean'], 2)}\\%",
                                            f"{fmt(d['wer'], 2)}\\%")) + r"\\")
    body = ["% Generated by python/ultra/tools/paper_data (money.py). Do not edit by hand.",
            f"\\def\\MoneyAxisW{{{AXIS_W:g}pt}}\\def\\MoneyAxisH{{{AXIS_H:g}pt}}%",
            f"\\def\\MoneyXmin{{{lim['xmin']:g}}}\\def\\MoneyXmax{{{lim['xmax']:g}}}\\def\\MoneyYmax{{{lim['ymax']:g}}}%",
            f"\\def\\MoneyXticks{{{','.join(map(str, ticks))}}}\\def\\MoneyYstep{{{lim['ystep']:g}}}%",
            f"\\def\\MoneyWattsOn{{{1 if axis else 0}}}%",
            "\\def\\MoneyHasFrontier{%d}\\def\\MoneyHasStream{%d}\\def\\MoneyHasWhole{%d}\\def\\MoneyHasFaded{%d}%%" % (
                has["frontier"], has["stream"], has["whole"], has["faded"]),
            f"\\def\\MoneyWattTicks{{{','.join(f'{x:.5g}' for x, _ in axis['ticks']) if axis else ''}}}%",
            f"\\def\\MoneyWattLabels{{{','.join(w for _, w in axis['ticks']) if axis else ''}}}%",
            "\\def\\MoneyComparisonRows{" + " ".join(comparison_rows) + "}%",
            "\\def\\MoneyOursLine{" + ours_conn + "}%",
            "\\def\\MoneyOursLegend{" + ours_legend + "}%",
            "\\def\\MoneySpeedGap{" + speed_gap + "}%",
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
    key_items = [f"\\mbox{{\\textbf{{{n}}}~{name}}}" for n, name in sorted(key)]
    if any(p["row"]["timing_panel"] != "full" for p in pts):
        key_items.append("\\mbox{\\S~73-clip subset timing}")
    body.append("\\def\\MoneyKey{" + "\\quad ".join(key_items) + "}%")
    body.append(f"\\def\\MoneyHeroOn{{{int(bool(hero))}}}%")
    if hero:
        body.extend(_hero_layout(hero, rec))
    out["money_layout.tex"] = "\n".join(body) + "\n"
    out["money_frontier.csv"] = csv_text(["speed", "wer"], [[f"{x:.4f}", f"{y:.3f}"] for x, y in line])
    return _note(rec, rows, pts, axis)
