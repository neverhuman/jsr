"""Reports over energy_bench receipts: ladder fits, package against core registers, tables, evidence lines.

    energy_report.py fit RECEIPT...                  (watts against busy workers, least squares)
    energy_report.py uplift RECEIPT...               (package watts per core-register watt)
    energy_report.py table --ours ARM RECEIPT...     (one Markdown table)
    energy_report.py sidecar --row ID RECEIPT        (energy lines for the paper evidence file)
    energy_report.py sidecar --power-bench SYSTEM=ID --host H SUMMARY   (the same lines from power_bench.py rows)
    energy_report.py resummarise RECEIPT             (re-apply the distrust rules to stored repetitions)
    energy_report.py join --log LOG --run NAME=CPUS:START:END:AUDIO_S   (watts for a run someone else timed)
    energy_report.py stair SERIES --plateau LABEL:WORKERS:START:END ...  (levels inside one staircase run)
    energy_report.py idle RECEIPT...                 (idle against idle: the noise floor of run minus idle)
    energy_report.py paced-lag RECEIPT... --timestamps T --utterances N   (word lag of paced streams)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

try:
    from . import energy_bench as bench
except ImportError:  # run as a script next to energy_bench.py
    import energy_bench as bench

SCHEMA = bench.SCHEMA
CORE_DIRTY = bench.CORE_DIRTY
DEFAULT_THRESHOLDS = bench.DEFAULT_THRESHOLDS
flags_for = bench.flags_for
headline = bench.headline
one_line = bench.one_line
rounded = bench.rounded
sha256_file = bench.sha256_file


def linear_fit(xs, ys):
    """Least-squares line with R^2 and the largest residual relative to the fitted value."""
    if len(xs) != len(ys) or len(xs) < 2 or len(set(xs)) < 2:
        raise ValueError("a line needs at least two distinct x values")
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    intercept = mean_y - slope * mean_x
    fitted = [intercept + slope * x for x in xs]
    ss_res = sum((y - f) ** 2 for y, f in zip(ys, fitted))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    return {
        "slope": slope,
        "intercept": intercept,
        "r2": 1.0 - ss_res / ss_tot if ss_tot else 1.0,
        "max_rel_residual": max(abs(y - f) / abs(f) if f else math.inf for y, f in zip(ys, fitted)),
    }


def least_squares(rows, ys):
    """Ordinary least squares with an intercept: ys ~ a + sum(b_i * rows[.][i]).

    Returns {"intercept", "coefficients", "r2", "residual_stdev", "residuals", "n"}.
    """
    if not rows or len(rows) != len(ys) or len(rows) <= len(rows[0]) + 1:
        raise ValueError("not enough points for the fit")
    design = [[1.0, *row] for row in rows]
    size = len(design[0])
    normal = [[sum(r[i] * r[j] for r in design) for j in range(size)] + [sum(r[i] * y for r, y in zip(design, ys))]
              for i in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(normal[r][column]))
        if abs(normal[pivot][column]) < 1e-12:
            raise ValueError("the regressors do not vary independently")
        normal[column], normal[pivot] = normal[pivot], normal[column]
        for other in range(size):
            if other != column:
                factor = normal[other][column] / normal[column][column]
                normal[other] = [a - factor * b for a, b in zip(normal[other], normal[column])]
    beta = [normal[i][size] / normal[i][i] for i in range(size)]
    residuals = [y - sum(b * x for b, x in zip(beta, r)) for r, y in zip(design, ys)]
    mean_y = statistics.fmean(ys)
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum(e * e for e in residuals)
    return {
        "intercept": beta[0],
        "coefficients": beta[1:],
        "r2": 1.0 - ss_res / ss_tot if ss_tot else 1.0,
        "residual_stdev": math.sqrt(ss_res / (len(ys) - size)),
        "residuals": residuals,
        "n": len(ys),
    }


def phase_points(receipt):
    """(label, pinned-core watts, other-core watts, package watts) for every phase of a receipt.

    The idle window between two repetitions is counted once.
    """
    points = []
    reps = receipt["reps"]
    for index, rep in enumerate(reps):
        core, other, package = rep.get("core"), rep.get("other_cores"), rep.get("package")
        if not (core and other and package):
            continue
        points.append((f"{receipt['arm']}:run{rep['rep']}", core["run_watts"], other["run_watts"], package["run_watts"]))
        if "idle_before_watts" not in other:  # older receipts kept only the two idle windows together
            points.append((f"{receipt['arm']}:idle{rep['rep']}", core["idle_watts"], other["idle_watts"],
                           package["idle_watts"]))
            continue
        points.append((f"{receipt['arm']}:idle{rep['rep']}", core["idle_before_watts"],
                       other["idle_before_watts"], package["idle_before_watts"]))
        if index == len(reps) - 1:
            points.append((f"{receipt['arm']}:idle{rep['rep'] + 1}", core["idle_after_watts"],
                           other["idle_after_watts"], package["idle_after_watts"]))
    return points


def load_receipts(paths):
    receipts = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    for receipt in receipts:
        if receipt.get("schema") != SCHEMA:
            raise SystemExit(f"energy_bench: {receipt.get('arm')} is not a {SCHEMA} receipt")
    return receipts


def fit_main(args):
    """Net watts against the number of busy workers, per view, over several receipts."""
    receipts = sorted(load_receipts(args.receipts), key=lambda r: r["workers"])
    xs = [r["workers"] for r in receipts]
    result = {"workers": xs, "arms": [r["arm"] for r in receipts]}
    def worker_core_watts(receipt):
        """Median net watts of the cores that host workers (the receipt may have measured more)."""
        reps = receipt.get("reps") or []
        clean = [rep for rep in reps if not CORE_DIRTY & set(rep["flags"])] or reps
        if not clean or not all("net_watts_by_core" in rep for rep in clean):
            return receipt["summary"]["net_watts"]["median"]
        hosts = [str(c) for c in receipt["cpus"][:receipt["workers"]]]
        return statistics.median(sum(rep["net_watts_by_core"][c] for c in hosts) for rep in clean)

    def corrected_package_watts(receipt):
        """Median package net watts with the other cores' change taken out at `--foreign-slope`.

        Uses every repetition whose core view is clean: a foreign step is corrected, not discarded.
        Cores that were measured but carried no worker count as foreign.
        """
        hosts = {str(c) for c in receipt["cpus"][:receipt["workers"]]}
        values = []
        for rep in receipt["reps"]:
            other = rep.get("other_cores")
            if CORE_DIRTY & set(rep["flags"]) or not other:
                continue
            foreign = other["run_watts"] - other["idle_watts"]
            foreign += sum(w for c, w in rep.get("net_watts_by_core", {}).items() if c not in hosts)
            values.append(rep["package"]["net_watts"] - args.foreign_slope * foreign)
        return statistics.median(values) if values else None

    core_watts = [worker_core_watts(r) for r in receipts]
    package_watts = [r["summary"]["net_watts_package"]["median"] for r in receipts]
    result["core"] = dict(linear_fit(xs, core_watts), net_watts=core_watts)
    result["package"] = dict(linear_fit(xs, package_watts), net_watts=package_watts)
    result["clean_repetitions"] = [r.get("clean_repetitions") for r in receipts]
    if args.foreign_slope is not None:
        corrected = [corrected_package_watts(r) for r in receipts]
        if None not in corrected:
            result["package_foreign_corrected"] = dict(
                linear_fit(xs, corrected), net_watts=corrected, foreign_slope=args.foreign_slope)
            result["package_over_core_slope_corrected"] = (
                result["package_foreign_corrected"]["slope"] / result["core"]["slope"])
    per_audio = [r["joules_per_audio_s"] for r in receipts]
    if None not in per_audio:
        low, high = min(per_audio), max(per_audio)
        result["joules_per_audio_s"] = per_audio
        result["joules_per_audio_s_range_pct"] = 100.0 * (high - low) / statistics.median(per_audio)
    if result["core"]["slope"]:
        result["package_over_core_slope"] = result["package"]["slope"] / result["core"]["slope"]
    text = json.dumps(rounded(result), indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def uplift_main(args):
    """Package watts against core-register watts over every phase of the given receipts.

    package = floor + k_pinned * (pinned cores) + k_other * (all other cores). k is the number
    of package watts that appear per watt the core registers report: the part a per-core figure
    leaves out is (k - 1) per core-register watt.
    """
    receipts = load_receipts(args.receipts)
    points = [point for receipt in receipts for point in phase_points(receipt)]
    if args.idle_only:
        points = [point for point in points if ":idle" in point[0]]
    both = least_squares([[mine, other] for _, mine, other, _ in points], [pkg for *_, pkg in points])
    one = least_squares([[mine + other] for _, mine, other, _ in points], [pkg for *_, pkg in points])
    result = {
        "arms": [r["arm"] for r in receipts],
        "phases": len(points),
        "all_cores": {"floor_watts": one["intercept"], "package_watts_per_core_watt": one["coefficients"][0],
                      "r2": one["r2"], "residual_stdev_watts": one["residual_stdev"]},
        "split": {"floor_watts": both["intercept"], "pinned_cores": both["coefficients"][0],
                  "other_cores": both["coefficients"][1], "r2": both["r2"],
                  "residual_stdev_watts": both["residual_stdev"]},
        "core_watts_range": [min(m + o for _, m, o, _ in points), max(m + o for _, m, o, _ in points)],
    }
    if args.points:
        result["points"] = [
            {"phase": label, "pinned": mine, "other": other, "package": pkg, "residual": error}
            for (label, mine, other, pkg), error in zip(points, one["residuals"])
        ]
    text = json.dumps(rounded(result), indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def resummarise_main(args):
    """Re-evaluate the distrust flags and the headline of a receipt with the current rules."""
    path = Path(args.receipt)
    receipt = load_receipts([path])[0]
    thresholds = dict(DEFAULT_THRESHOLDS)
    busy_workers = receipt.get("busy_workers")
    if busy_workers is None:
        paced_or_threaded = receipt.get("audio_rate_per_worker") or receipt["command"][:1] == ["sleep"]
        busy_workers = 0 if paced_or_threaded else receipt["workers"]
    for rep in receipt["reps"]:
        rep["flags"] = flags_for(rep, thresholds, len(receipt["measured_cores"]), busy_workers)
    previous_tool = receipt.get("resummarised", {}).get("measured_with_tool_sha256", receipt["tool_sha256"])
    receipt.update(headline(receipt["reps"], thresholds))
    receipt["thresholds"] = thresholds
    receipt["busy_workers"] = busy_workers
    receipt["resummarised"] = {
        "measured_with_tool_sha256": previous_tool,
        "rules_from_tool_sha256": sha256_file(Path(__file__).resolve()),
        "note": "flags and headline recomputed from the stored repetitions; no measured value changed",
    }
    Path(args.out or path).write_text(json.dumps(rounded(receipt), indent=1) + "\n", encoding="utf-8")
    print(one_line(rounded(receipt)))
    return 0


def table_main(args):
    receipts = load_receipts(args.receipts)
    ours = next((r for r in receipts if r["arm"] == args.ours), None)
    if ours is None:
        raise SystemExit(f"energy_bench table: no receipt for --ours {args.ours}")

    def cell(value, digits=3):
        return "n/a" if value is None else f"{value:.{digits}g}"

    lines = [
        "| System | CPU-s per audio-s | J per audio-s = W per live stream (core) | same, package view "
        "| W per busy core (core / package) | Ratio to ours (core) | Reps (spread) | Flags |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in sorted(receipts, key=lambda r: r["joules_per_audio_s"] or math.inf):
        ratio = r["joules_per_audio_s"] / ours["joules_per_audio_s"] if r["joules_per_audio_s"] else None
        spread = r["summary"]["joules_per_audio_s"]
        lines.append(
            f"| {r['arm']} | {cell(r['cpu_s_per_audio_s'])} | {cell(r['joules_per_audio_s'])} "
            f"| {cell(r['joules_per_audio_s_package'])} "
            f"| {cell(r['watts_per_busy_core'])} / {cell(r['watts_per_busy_core_package'])} "
            f"| {cell(ratio)}x | {r['repetitions']} ({cell(spread['spread_pct'] if spread else None, 2)}%) "
            f"| {', '.join(r['flags']) or 'none'} |"
        )
    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def sidecar_lines(receipt, row, receipt_sha256, session=None):
    """Energy lines for the paper evidence file: one per (row, method), joined to a cost row by `row`.

    core_msr_active   core register while running, idle floor included (runtime-lane's CorWatt basis)
    core_msr_dynamic  core register above idle
    pkg_rapl_delta    package counter above idle (adds the shared uncore power a busy core causes)
    """
    methods = (
        ("core_msr_active", "core", "gross_joules_per_audio_s", "watts_per_busy_core_active"),
        ("core_msr_dynamic", "core", "joules_per_audio_s", "watts_per_busy_core"),
        ("pkg_rapl_delta", "package", "joules_per_audio_s", "watts_per_busy_core_package"),
    )
    lines = []
    for method, view, key, _ in methods:
        reps = [rep[view][key] for rep in receipt["reps"] if view in rep]
        if not reps or None in reps:
            continue
        per_core = [
            (rep[view]["run_watts"] if method == "core_msr_active" else rep[view]["net_watts"])
            * rep[view]["run_window_seconds"] / rep["cpu_seconds"]
            for rep in receipt["reps"] if rep["cpu_seconds"]
        ]
        value = statistics.median(reps)
        lines.append({
            "kind": "energy",
            "row": row,
            "host": receipt["machine"],
            "energy_method": method,
            "joules_per_audio_s": value,
            "watts_per_stream": value,
            "energy_reps": reps,
            "energy_idle_w": statistics.median(rep[view]["idle_watts"] for rep in receipt["reps"]),
            "energy_active_w": statistics.median(rep[view]["run_watts"] for rep in receipt["reps"]),
            "watts_per_busy_core": statistics.median(per_core) if per_core else None,
            "energy_cpu_s_per_audio_s": receipt["cpu_s_per_audio_s"],
            "energy_workload_digest": receipt["audio_manifest_sha256"],
            "energy_receipt_sha256": receipt_sha256,
            "energy_host": receipt["machine"],
            "energy_tool": SCHEMA,
            "energy_flags": receipt["flags"],
            "session": session,
        })
    return lines


def idle_pairs(receipts):
    """Idle against idle from every repetition: the idle window after a run minus the one before it.

    Core view per pinned core, package view as measured. The absolute differences are the noise floor
    of a run-minus-idle figure on this host.
    """
    core, package = [], []
    for receipt in receipts:
        pinned = len(receipt["measured_cores"])
        for rep in receipt["reps"]:
            if "core" in rep:
                core.append((rep["core"]["idle_after_watts"] - rep["core"]["idle_before_watts"]) / pinned)
            if "package" in rep:
                package.append(rep["package"]["idle_after_watts"] - rep["package"]["idle_before_watts"])

    def spread(values):
        ordered = sorted(abs(v) for v in values)
        if not ordered:
            return None
        return {"pairs": len(ordered), "median_abs_watts": statistics.median(ordered),
                "p90_abs_watts": ordered[int(0.9 * (len(ordered) - 1))], "max_abs_watts": ordered[-1],
                "mean_watts": statistics.fmean(values)}

    return {"core_per_pinned_core": spread(core), "package": spread(package)}


def idle_main(args):
    result = dict(arms=[Path(p).stem for p in args.receipts], **idle_pairs(load_receipts(args.receipts)))
    text = json.dumps(rounded(result), indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def stair_levels(series, plateaus, repetition=1, edge=3.0):
    """Median-second levels of each plateau of a staircase run, and the line through them.

    `series` is a `--series-out` file; `plateaus` are (label, busy workers, start, end) in seconds from
    the start of the run. The first and last `edge` seconds of each plateau are left out. Returns the
    levels (pinned cores' register watts and package watts), the idle levels around the run and the
    least-squares line package = floor + slope x core-register watts.
    """
    times, labels = series["t"], series["labels"]
    first, last = labels[f"run{repetition}_start"], labels[f"run{repetition}_end"]
    origin = times[first]

    def seconds(key, a, b):
        values = series[key]
        return [((times[i] + times[i + 1]) / 2 - origin, (values[i + 1] - values[i]) / (times[i + 1] - times[i]))
                for i in range(a, b) if times[i + 1] - times[i] >= 0.3]

    package, cores = seconds("package_joules", first, last), seconds("core_joules", first, last)
    idle_a = labels[f"idle{repetition}_start"]
    after = f"idle{repetition + 1}_start"
    after_end = f"idle{repetition + 1}_end" if f"idle{repetition + 1}_end" in labels else f"run{repetition + 1}_start"
    idle_package = ([w for _, w in seconds("package_joules", idle_a, first)]
                    + [w for _, w in seconds("package_joules", labels[after], labels[after_end])])
    idle_cores = ([w for _, w in seconds("core_joules", idle_a, first)]
                  + [w for _, w in seconds("core_joules", labels[after], labels[after_end])])
    levels = []
    for label, workers, start, end in plateaus:
        inside_package = [w for t, w in package if start + edge <= t < end - edge]
        inside_cores = [w for t, w in cores if start + edge <= t < end - edge]
        if len(inside_package) < 5:
            continue
        levels.append({
            "plateau": label, "workers": workers, "seconds": len(inside_package),
            "core_register_watts": statistics.median(inside_cores),
            "package_watts": statistics.median(inside_package),
            "package_watts_p25": statistics.quantiles(inside_package, n=4)[0],
        })
    line = least_squares([[level["core_register_watts"]] for level in levels],
                         [level["package_watts"] for level in levels]) if len(levels) >= 3 else None
    return {
        "run_seconds": times[last] - origin,
        "idle_package_watts": statistics.median(idle_package),
        "idle_core_register_watts": statistics.median(idle_cores),
        "levels": levels,
        "package_floor_watts_when_busy": line["intercept"] if line else None,
        "package_watts_per_core_register_watt": line["coefficients"][0] if line else None,
        "line_r2": line["r2"] if line else None,
        "line_residual_stdev_watts": line["residual_stdev"] if line else None,
    }


def stair_main(args):
    series = json.loads(Path(args.series).read_text(encoding="utf-8"))
    plateaus = []
    for spec in args.plateau:
        label, workers, start, end = spec.split(":")
        plateaus.append((label, int(workers), float(start), float(end)))
    result = dict(series=Path(args.series).name, **stair_levels(series, plateaus, args.repetition))
    text = json.dumps(rounded(result), indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


WORD_RE = __import__("re").compile(r"[a-z0-9']+")


def words(text):
    return WORD_RE.findall((text or "").lower())


def align_matches(ref, hyp):
    """(ref index, hyp index) pairs of exact matches along one optimal Levenshtein alignment."""
    n, m = len(ref), len(hyp)
    dp = [list(range(m + 1))] + [[i] + [0] * m for i in range(1, n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    out, i, j = [], n, m
    while i > 0 and j > 0:
        if ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            i, j = i - 1, j - 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return out[::-1]


def confirmed_counts(snapshots, final_words):
    """For each (time, snapshot) the number of final words already emitted and unchanged.

    A snapshot is the tail of the hypothesis at that moment (the event payload is a bounded window),
    so its first word may be cut. The window is placed in the final hypothesis where its words, after
    the first, match longest; everything before the matched run counts as emitted.
    """
    counts, floor = [], 0
    for _, text in snapshots:
        window = words(text)
        best = (floor, 0)
        if window:
            for start in range(max(0, floor - len(window)), len(final_words) + 1):
                run = 0
                while (run + 1 < len(window) and start + run + 1 <= len(final_words)
                       and window[run + 1] == final_words[start + run]):
                    run += 1
                head = 1 if start >= 1 and window[0] == final_words[start - 1] else 0
                if run and (start + run > sum(best) or (run == len(window) - 1 and start + run == sum(best))):
                    best = (start - head, run + head)
            if best[1] == 0 and window == final_words[:len(window)]:
                best = (0, len(window))
        floor = max(floor, best[0])
        counts.append(sum(best))
    return counts


def stitch(previous, window, slack=16):
    """The segment's text after a new event, given its text before and the event's bounded tail window.

    The window is the tail of the new text: it starts near len(new) - len(window). The overlap with the
    old text that matches longest decides where it is placed; whatever followed in the old text (a
    revised tail) is replaced.
    """
    if not previous:
        return window
    best_k, best_m = None, 0
    for k in range(max(0, len(previous) - len(window) - slack), len(previous) + 1):
        tail = previous[k:]
        m = 0
        while m < len(tail) and m < len(window) and tail[m] == window[m]:
            m += 1
        if m > best_m:
            best_k, best_m = k, m
    if best_k is None or best_m < min(4, len(window)):
        return window if len(window) >= len(previous) else previous[:max(0, len(previous) - len(window))] + window
    return previous[:best_k] + window


def stitched_texts(events):
    """(time, whole hypothesis so far) for every transcript event, segments joined in order."""
    segments, out = {}, []
    for event in events:
        body = event["event"]
        if body.get("event_type") not in ("Preview", "Commit", "Final"):
            continue
        seg = body.get("segment_id", 0)
        segments[seg] = stitch(segments.get(seg, ""), body.get("snapshot_utf8", ""))
        out.append((event["t_s"], " ".join(segments[k] for k in sorted(segments))))
    return out


def prefix_counts(texts, final_words):
    """For each (time, whole text): how many leading final words it already holds unchanged."""
    counts = []
    for _, text in texts:
        current = words(text)
        n = 0
        while n < len(current) and n < len(final_words) and current[n] == final_words[n]:
            n += 1
        counts.append(n)
    return counts


def stable_lags_from_events(events, ref_words, ref_ends):
    """Stable-word lag (s) per matched reference word from a stream's own events (no external text)."""
    texts = stitched_texts(events)
    if not texts:
        return [], len(ref_words), ""
    final_words = words(texts[-1][1])
    counts = prefix_counts(texts, final_words)
    emitted = [texts[-1][0]] * len(final_words)
    suffix_min = len(final_words)
    for (moment, _), count in zip(reversed(texts), reversed(counts)):
        suffix_min = min(suffix_min, count)
        for j in range(suffix_min):
            emitted[j] = moment
    lags = [emitted[j] - ref_ends[i] for i, j in align_matches(ref_words, final_words)]
    return lags, len(ref_words), texts[-1][1]


def stable_word_lags(snapshots, final_text, ref_words, ref_ends):
    """Stable-word lag (s) of every reference word that the final hypothesis matches.

    A word counts as emitted at the first snapshot after which it is never withdrawn; its lag is that
    time minus the word's end in the audio (input paced at real time, both from the start of the stream).
    """
    final_words = words(final_text)
    counts = confirmed_counts(snapshots, final_words)
    emitted = [None] * len(final_words)
    suffix_min = len(final_words)
    for (moment, _), count in zip(reversed(snapshots), reversed(counts)):
        suffix_min = min(suffix_min, count)
        for j in range(suffix_min):
            emitted[j] = moment
    if snapshots:
        for j in range(len(final_words)):
            if emitted[j] is None:
                emitted[j] = snapshots[-1][0]
    lags = [emitted[j] - ref_ends[i] for i, j in align_matches(ref_words, final_words) if emitted[j] is not None]
    return lags, len(ref_words)


def paced_lag_main(args):
    """Word lag of every paced stream of energy_bench runs, next to that run's watts per stream."""
    timing = json.loads(Path(args.timestamps).read_text(encoding="utf-8"))[:args.utterances]
    ref_words, ref_ends, offset = [], [], 0.0
    for utterance in timing:
        ref_words += utterance["ref_words"]
        ref_ends += [offset + end for end in utterance["ref_word_end_s"]]
        offset += float(utterance["duration_s"])
    rows = []
    for path in args.receipts:
        receipt = load_receipts([path])[0]
        logs = Path(args.log_root) / receipt["arm"] if args.log_root else Path(path).parent / "out" / receipt["arm"]
        lags, matched, total, finals = [], 0, 0, []
        for events_path in sorted(logs.glob("rep*-w*.events.jsonl")):
            events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            stream_lags, refs, final_text = stable_lags_from_events(events, ref_words, ref_ends)
            lags += stream_lags
            matched += len(stream_lags)
            total += refs
            finals.append(events[-1]["t_s"] - offset if events else None)
        ordered = sorted(lags)
        pick = (lambda q: ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))] * 1000.0) if ordered else None
        rows.append({
            "arm": receipt["arm"], "streams": receipt["workers"],
            "watts_per_stream_core": receipt["joules_per_audio_s"],
            "gross_watts_per_stream_core": receipt["summary"]["gross_joules_per_audio_s"]["median"]
            if receipt["summary"].get("gross_joules_per_audio_s") else None,
            "cpu_s_per_audio_s_per_stream": receipt["cpu_s_per_audio_s"],
            "core_busy_fraction": receipt["reps"][0]["busy"]["pinned_busy_fraction_run"],
            "lag_p50_ms": pick(0.50) if ordered else None, "lag_p95_ms": pick(0.95) if ordered else None,
            "lag_max_ms": ordered[-1] * 1000.0 if ordered else None,
            "words_matched": matched, "words_reference": total,
            "final_after_audio_end_s_max": max((f for f in finals if f is not None), default=None),
            "flags": receipt["flags"],
        })
        print(json.dumps(rounded(rows[-1])))
    if args.out:
        Path(args.out).write_text("".join(json.dumps(rounded(r)) + "\n" for r in rows), encoding="utf-8")
    return 0


class PowerLog:
    """The continuous 1 Hz log (`energy_bench.py sample` lines with `wall`): cumulative joules by UTC time.

    A gap longer than `max_gap` seconds between two samples (a restart) is remembered; an interval that
    spans one cannot be trusted, because a counter may have wrapped unseen.
    """

    def __init__(self, paths, max_gap=30.0):
        unit, previous = None, None
        self.wall, self.package, self.gaps = [], [], []
        self.cores = {}
        self.listed = []
        totals, package_total = {}, 0
        for path in sorted(str(p) for p in paths):
            with open(path, encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get("type") == "header":
                        unit = record["energy_unit_j"]
                        self.listed = [int(c) for c in record.get("cpus", [])]
                        continue
                    if "wall" not in record or unit is None:
                        continue
                    if previous is not None:
                        if record["wall"] <= previous["wall"]:
                            continue
                        if record["wall"] - previous["wall"] > max_gap:
                            self.gaps.append((previous["wall"], record["wall"]))
                        package_total += bench.wrap_delta(previous["pkg"], record["pkg"], bench.COUNTER_MODULUS)
                        for cpu, value in record["core"].items():
                            totals[cpu] = totals.get(cpu, 0) + bench.wrap_delta(
                                previous["core"].get(cpu, value), value, bench.COUNTER_MODULUS)
                    self.wall.append(record["wall"])
                    self.package.append(package_total * unit)
                    for cpu in record["core"]:
                        self.cores.setdefault(cpu, []).append(totals.get(cpu, 0) * unit)
                    previous = record
        if len(self.wall) < 2:
            raise SystemExit("energy_report: the power log holds fewer than two samples")

    def core_of(self, cpu):
        """The logged core whose register a hardware thread shares.

        The log lists one thread per physical core. When it lists 0..N-1, thread N+k is the SMT sibling
        of core k (the numbering on these hosts); any other thread must be listed itself.
        """
        if str(cpu) in self.cores:
            return str(cpu)
        count = len(self.listed)
        if self.listed == list(range(count)) and count <= int(cpu) < 2 * count:
            return str(int(cpu) - count)
        raise SystemExit(f"energy_report: CPU {cpu} is not in the power log")

    def covers(self, start, end):
        return (self.wall[0] <= start and end <= self.wall[-1]
                and not any(a < end and start < b for a, b in self.gaps))

    def joules(self, series, start, end):
        """Energy between two UTC times, interpolated inside the one-second samples at both ends."""
        def at(moment):
            low, high = 0, len(self.wall) - 1
            while high - low > 1:
                middle = (low + high) // 2
                if self.wall[middle] <= moment:
                    low = middle
                else:
                    high = middle
            share = (moment - self.wall[low]) / (self.wall[high] - self.wall[low])
            return series[low] + share * (series[high] - series[low])
        return at(end) - at(start)

    def per_second(self, series, start, end):
        return [(series[i + 1] - series[i]) / (self.wall[i + 1] - self.wall[i])
                for i in range(len(self.wall) - 1) if start <= self.wall[i] and self.wall[i + 1] <= end]


def join_run(log, cpus, start, end, audio_seconds, margin=1.5, busy_watts=2.0):
    """Energy of one repetition measured by somebody else, from the continuous log.

    `cpus` are the service CPUs (hardware threads; an SMT sibling shares its core's register), `start`
    and `end` UTC epoch seconds. The window is widened by `margin` seconds on both sides because other
    receipts time to the second; the extra idle seconds add a fraction of a joule.
    """
    cores = sorted({log.core_of(c) for c in cpus}, key=int)
    if not log.covers(start, end):
        return {"covered": False}
    duration = end - start
    # The widened window may reach past the first or last sample of the log by less than the margin.
    low, high = max(start - margin, log.wall[0]), min(end + margin, log.wall[-1])
    joules = sum(log.joules(log.cores[c], low, high) for c in cores)
    idle = []
    for c in cores:
        around = (log.per_second(log.cores[c], start - 60 - margin, start - margin - 2)
                  + log.per_second(log.cores[c], end + margin + 5, end + margin + 65))
        idle.append(statistics.median(around) if around else 0.0)
    idle_watts = sum(idle)
    busy = [c for c, series in log.cores.items()
            if log.joules(series, start, end) / duration >= busy_watts and c not in cores]
    package = log.per_second(log.package, start, end)
    return {
        "covered": True,
        "cores": [int(c) for c in cores],
        "seconds": duration,
        "core_watts": joules / duration,
        "idle_core_watts": idle_watts,
        "idle_is_trustworthy": idle_watts < 1.0 * len(cores),
        "joules_per_audio_s_active": joules / audio_seconds,
        "joules_per_audio_s_dynamic": (joules - idle_watts * (high - low)) / audio_seconds,
        "package_watts_mean": statistics.fmean(package) if package else None,
        "package_watts_median": statistics.median(package) if package else None,
        "other_busy_cores": len(busy),
    }


def join_lines(joined, row, receipt_sha256, host, workload_digest=None, session=None):
    """Evidence lines for a cost row from its joined repetitions (the same runs that were timed).

    `core_msr_active` always; `core_msr_dynamic` only when every repetition had a believable idle level
    (a core that ran another arm right before or after has none).
    """
    usable = [r for r in joined if r.get("covered")]
    if not usable:
        return []
    methods = [("core_msr_active", "joules_per_audio_s_active")]
    if all(r["idle_is_trustworthy"] for r in usable):
        methods.append(("core_msr_dynamic", "joules_per_audio_s_dynamic"))
    lines = []
    for method, key in methods:
        reps = [r[key] for r in usable]
        value = statistics.median(reps)
        lines.append({
            "kind": "energy", "row": row, "host": host, "energy_method": method,
            "joules_per_audio_s": value, "watts_per_stream": value, "energy_reps": reps,
            "energy_active_w": statistics.median(r["core_watts"] for r in usable),
            "energy_idle_w": statistics.median(r["idle_core_watts"] for r in usable),
            "energy_package_w": statistics.median(r["package_watts_median"] for r in usable),
            "energy_other_busy_cores": [r["other_busy_cores"] for r in usable],
            "energy_workload_digest": workload_digest, "energy_receipt_sha256": receipt_sha256,
            "energy_host": host, "energy_tool": "jsr.energy-bench/v1 power-log join",
            "energy_flags": [] if len(usable) == len(joined) else ["repetition_outside_power_log"],
            "session": session,
        })
    return lines


def join_main(args):
    """Join the continuous power log with repetitions: NAME=CPUS:START:END:AUDIO_SECONDS (UTC epoch seconds)."""
    log = PowerLog(args.log)
    out = []
    for spec in args.run:
        name, _, rest = spec.partition("=")
        cpus, start, end, audio = rest.split(":")
        row = join_run(log, bench.parse_cpu_list(cpus), float(start), float(end), float(audio), margin=args.margin)
        out.append(dict(name=name, **row))
        print(json.dumps(rounded(out[-1])))
    text = "".join(json.dumps(rounded(row)) + "\n" for row in out)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    if args.row:
        digest = hashlib.sha256(text.encode()).hexdigest()
        for line in join_lines(out, args.row, digest, args.host, args.workload_digest, args.session):
            print(json.dumps(rounded(line), separators=(", ", ": ")))
    return 0


POWER_BENCH_SCHEMA = "jsr.power-bench/v1"


def power_bench_lines(summary, rows, receipt_sha256, host, session=None):
    """The same evidence lines from a power_bench.py summary (turbostat CorWatt, one run per arm).

    `rows` maps a power_bench system name to the id of its cost row. power_bench multiplies core watts
    by wall seconds per audio-second of the same run, which is the `core_msr_active` definition here;
    its `cpu_s_per_audio_s` is that wall figure and is passed on under its true name.
    """
    if summary.get("schema") != POWER_BENCH_SCHEMA:
        raise SystemExit(f"energy_report: not a {POWER_BENCH_SCHEMA} summary")
    lines = []
    for row in summary["core_power"]:
        if row["system"] not in rows:
            continue
        for method, joules, watts in (
            ("core_msr_active", row["mw_per_live_stream"] / 1000.0, row["active_core_w"]),
            ("core_msr_dynamic", row["dynamic_mw_per_live_stream"] / 1000.0, row["dynamic_core_w"]),
        ):
            lines.append({
                "kind": "energy",
                "row": rows[row["system"]],
                "host": host,
                "energy_method": method,
                "joules_per_audio_s": joules,
                "watts_per_stream": joules,
                "energy_reps": [joules],
                "energy_idle_w": row["idle_core_w"],
                "energy_active_w": row["active_core_w"],
                "watts_per_busy_core": watts,
                "energy_wall_s_per_audio_s": row["cpu_s_per_audio_s"],
                "energy_workload_digest": summary.get("timestamps_sha256"),
                "energy_receipt_sha256": receipt_sha256,
                "energy_host": host,
                "energy_tool": POWER_BENCH_SCHEMA,
                "energy_flags": ["single_run"],
                "session": session,
            })
    return lines


def sidecar_main(args):
    path = Path(args.receipt)
    if args.power_bench:
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows = dict(item.split("=", 1) for item in args.power_bench)
        lines = power_bench_lines(summary, rows, sha256_file(path), args.host, args.session)
    else:
        if not args.row:
            raise SystemExit("energy_report sidecar: --row is required for an energy_bench receipt")
        lines = sidecar_lines(load_receipts([path])[0], args.row, sha256_file(path), args.session)
    for line in lines:
        print(json.dumps(rounded(line), separators=(", ", ": ")))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    fit = sub.add_parser("fit", help="watts against busy workers over several receipts")
    fit.add_argument("receipts", nargs="+")
    fit.add_argument("--foreign-slope", type=float, default=None,
                     help="package watts per core-register watt of the other cores (from `uplift --idle-only`)")
    fit.add_argument("--out", default=None)
    fit.set_defaults(handler=fit_main)

    uplift = sub.add_parser("uplift", help="package watts per core-register watt over all phases")
    uplift.add_argument("receipts", nargs="+")
    uplift.add_argument("--points", action="store_true", help="include every phase and its residual")
    uplift.add_argument("--idle-only", action="store_true",
                        help="idle windows only: the line then describes the other cores' load alone")
    uplift.add_argument("--out", default=None)
    uplift.set_defaults(handler=uplift_main)

    again = sub.add_parser("resummarise", help="recompute flags and headline from stored repetitions")
    again.add_argument("receipt")
    again.add_argument("--out", default=None)
    again.set_defaults(handler=resummarise_main)

    sidecar = sub.add_parser("sidecar", help="energy lines for docs/paper/data/evidence, one per method")
    sidecar.add_argument("receipt")
    sidecar.add_argument("--row", default=None, help="id of the cost row these lines belong to")
    sidecar.add_argument("--power-bench", action="append", metavar="SYSTEM=ROW_ID",
                         help="the file is a power_bench.py summary; map each of its systems to a row id")
    sidecar.add_argument("--host", default=None, help="host of a power_bench summary (it does not record one)")
    sidecar.add_argument("--session", default=None)
    sidecar.set_defaults(handler=sidecar_main)

    idle = sub.add_parser("idle", help="idle against idle over every repetition of the given receipts")
    idle.add_argument("receipts", nargs="+")
    idle.add_argument("--out", default=None)
    idle.set_defaults(handler=idle_main)

    lag = sub.add_parser("paced-lag", help="stable-word lag of paced streams beside their watts per stream")
    lag.add_argument("receipts", nargs="+", help="energy_bench receipts of paced runs (--events written)")
    lag.add_argument("--timestamps", required=True, help="word end times: [{duration_s, ref_words, ref_word_end_s}]")
    lag.add_argument("--utterances", type=int, required=True, help="how many leading utterances the paced audio holds")
    lag.add_argument("--log-root", default=None, help="directory holding <arm>/rep*-w*.events.jsonl")
    lag.add_argument("--out", default=None)
    lag.set_defaults(handler=paced_lag_main)

    stair = sub.add_parser("stair", help="plateau levels of a staircase run and the package-against-core line")
    stair.add_argument("series", help="the --series-out file of the staircase run")
    stair.add_argument("--plateau", action="append", required=True, metavar="LABEL:WORKERS:START:END",
                       help="seconds from the start of the run (repeatable)")
    stair.add_argument("--repetition", type=int, default=1)
    stair.add_argument("--out", default=None)
    stair.set_defaults(handler=stair_main)

    join = sub.add_parser("join", help="energy of other agents' repetitions from the continuous power log")
    join.add_argument("--log", action="append", required=True, help="power-YYYYMMDD.jsonl (repeatable)")
    join.add_argument("--run", action="append", required=True, metavar="NAME=CPUS:START:END:AUDIO_S",
                      help="service CPUs, UTC epoch start and end, audio seconds decoded")
    join.add_argument("--margin", type=float, default=1.5,
                      help="seconds added on both sides for receipts timed to the second (0 for exact times)")
    join.add_argument("--out", default=None)
    join.add_argument("--row", default=None, help="also print the evidence lines for this cost row id")
    join.add_argument("--host", default=None)
    join.add_argument("--workload-digest", default=None)
    join.add_argument("--session", default=None)
    join.set_defaults(handler=join_main)

    table = sub.add_parser("table", help="one Markdown table over several receipts")
    table.add_argument("receipts", nargs="+")
    table.add_argument("--ours", required=True, help="arm name the ratio column divides by")
    table.add_argument("--out", default=None)
    table.set_defaults(handler=table_main)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
