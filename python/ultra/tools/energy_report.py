"""Reports over energy_bench receipts: ladder fits, package against core registers, tables, evidence lines.

    energy_report.py fit RECEIPT...                  (watts against busy workers, least squares)
    energy_report.py uplift RECEIPT...               (package watts per core-register watt)
    energy_report.py table --ours ARM RECEIPT...     (one Markdown table)
    energy_report.py sidecar --row ID RECEIPT        (energy lines for the paper evidence file)
    energy_report.py resummarise RECEIPT             (re-apply the distrust rules to stored repetitions)
"""

from __future__ import annotations

import argparse
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


def sidecar_main(args):
    path = Path(args.receipt)
    receipt = load_receipts([path])[0]
    for line in sidecar_lines(receipt, args.row, sha256_file(path), args.session):
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
    sidecar.add_argument("--row", required=True, help="id of the cost row these lines belong to")
    sidecar.add_argument("--session", default=None)
    sidecar.set_defaults(handler=sidecar_main)

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
