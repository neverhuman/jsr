"""The evidence row format (schema v2): constants and the checks every row and sidecar line must pass.

One row per (system, runtime, machine, threads). Format and field meanings: docs/paper/data/README.md.
"""

from __future__ import annotations

import statistics

from .common import EVIDENCE_REQUIRED, MACHINES, NAME_RE, PaperDataError

ROLES = ("ours", "comparator", "ablation")  # ablation: our weights in another runtime; tables only
CLASSES = ("streaming", "whole_utterance")
# How a row uses its cores. intra_op: one recogniser with N threads, one clip at a time. workers: one process
# with N single-thread workers that share the weights (the serving path). processes: N separate
# single-thread processes, each with its own copy of the weights.
PARALLEL = ("intra_op", "workers", "processes")
EVIDENCE_LABELS = ("exploratory", "qualified")
# core_msr_contended: core registers in a separate run on a shared core. Kept as a cross-check only: it is
# never the drawn or tabulated value, and a group's headline method is chosen among the others.
ENERGY_METHODS = ("core_msr_active", "core_msr_dynamic", "pkg_rapl_delta", "wall_meter", "core_msr_contended")
CROSS_CHECK_METHODS = ("core_msr_contended",)
# Word lag = end to end, input paced at real time, at the row's own thread budget. `lag_design_*` is the
# lag with unbounded compute and may be given alone. A whole-utterance row has no word lag; it may give
# `final_latency_*` (wait for the end of the clip, then infer). All of them share the three proof fields.
LAG_PROOF = ("lag_method", "lag_workload_digest", "lag_receipt_sha256")
LAG_LIVE = ("lag_p50_ms", "lag_p95_ms")
LAG_FIELDS = LAG_LIVE + LAG_PROOF
LAG_DESIGN = ("lag_design_p50_ms", "lag_design_p95_ms")
FINAL_LATENCY = ("final_latency_p50_ms", "final_latency_p95_ms")
LAG_KEYS = LAG_FIELDS + LAG_DESIGN + FINAL_LATENCY + ("lag_host",)  # lag_host: where the lag was measured
# A paced lag measured on a shared host is backlog more than lag for a system that is close to real time
# there, so it is admitted from a shared host only for rows at least this many times faster than real time.
LAG_SHARED_MIN_SPEED = 4.0
ENERGY_FIELDS = ("joules_per_audio_s", "energy_method", "energy_receipt_sha256")
# A slow system is timed on a 73-utterance subset; its WER then comes from a full-panel pass elsewhere.
TIMING_PANELS = ("full", "p73")
# What data/systems.json may set per variant: names and order for the paper, never a number.
PRESENTATION = ("system", "short", "macro", "cite", "order", "plot", "runtime_short", "text", "detail")
MIN_REPS = 3
SHORT_MAX = 22
MAX_OTHER_BUSY = 2  # a primary row has at most this many other busy cores on the host in every repetition
_DIGITS = "Zero One Two Three Four Five Six Seven Eight Nine".split()


def spell(text: str) -> str:
    """'d2e-16457' -> 'DTwoeOneSixFourFiveSeven': a letters-only macro stem."""
    return "".join(_DIGITS[int(c)] if c.isdigit() else c for part in text.split("-") for c in part.capitalize())


def _median_ok(value, reps) -> bool:
    med = statistics.median(float(x) for x in reps)
    return abs(med - float(value)) <= 1e-9 * max(1.0, med)


def _check_lag(r: dict, where: str) -> None:
    have = [k for k in LAG_KEYS if r.get(k) not in (None, "") and k != "lag_host"]
    if not have:
        return
    r["lag_host"] = r.get("lag_host") or r["host"]
    quiet = MACHINES[r["machine"]]["primary_host"]
    if r.get("lag_p50_ms") is not None and quiet not in (None, r["lag_host"]) \
            and 1.0 / float(r["cpu_s_per_audio_s"]) < LAG_SHARED_MIN_SPEED:
        raise PaperDataError(f"{where}: a paced lag measured on the shared host {r['lag_host']} is not admitted for a "
                             f"system slower than {LAG_SHARED_MIN_SPEED:g}x real time; give lag_design_* only until "
                             f"it is measured on {quiet}")
    live, design = [k for k in have if k in LAG_LIVE], [k for k in have if k in LAG_DESIGN]
    final = [k for k in have if k in FINAL_LATENCY]
    if r["class"] == "whole_utterance":
        if live or design:
            raise PaperDataError(f"{where}: a whole_utterance row reports throughput only; it cannot carry a word "
                                 f"lag (use final_latency_p50_ms / final_latency_p95_ms)")
        need = FINAL_LATENCY + LAG_PROOF
    else:
        if final:
            raise PaperDataError(f"{where}: final_latency_* is for whole_utterance rows; a streaming row has a word lag")
        need = (LAG_LIVE if live else ()) + (LAG_DESIGN if design or not live else ()) + LAG_PROOF
    if set(need) - set(have):
        raise PaperDataError(f"{where}: lag fields are all-or-none; missing {sorted(set(need) - set(have))}")


def check_row(r: dict, where: str) -> None:
    miss = [k for k in EVIDENCE_REQUIRED if r.get(k) in (None, "", [])]
    if miss:
        raise PaperDataError(f"{where}: evidence row missing {miss}")
    if r["label"] == "reported":
        raise PaperDataError(f"{where}: a reported (published) number cannot be a measured row; use data/reported.json")
    r.setdefault("timing_panel", "full")
    for key, allowed in (("label", EVIDENCE_LABELS), ("role", ROLES), ("class", CLASSES), ("parallel", PARALLEL),
                         ("machine", tuple(MACHINES)), ("timing_panel", TIMING_PANELS)):
        if r[key] not in allowed:
            raise PaperDataError(f"{where}: bad {key} {r[key]!r}; allowed: {', '.join(allowed)}")
    if not isinstance(r["threads"], int) or r["threads"] < 1:
        raise PaperDataError(f"{where}: threads must be a positive integer")
    base = f"{r['variant']}-{r['machine']}-t{r['threads']}"
    if r["id"] != base and not r["id"].startswith(base + "-"):
        raise PaperDataError(f"{where}: id must be <variant>-<machine>-t<threads>, optionally followed by "
                             f"-<host> or -anchor-<host>, got {r['id']!r}")
    for value, reps in (("cpu_s_per_audio_s", "cpu_reps"), ("wall_s_per_audio_s", "wall_reps")):
        if len(r[reps]) < MIN_REPS:
            raise PaperDataError(f"{where}: {reps} needs at least {MIN_REPS} repetitions")
        if not _median_ok(r[value], r[reps]):
            raise PaperDataError(f"{where}: {value} is not the median of {reps}")
    if r["timing_panel"] != "full" and any(r.get(k) in (None, "") for k in ("timing_workload_digest", "wer_source")):
        raise PaperDataError(f"{where}: a row timed on {r['timing_panel']} names its timing_workload_digest and its "
                             f"wer_source (the full-panel pass its WER comes from)")
    _check_lag(r, where)
    r["short"] = r.get("short") or r["system"]
    if len(r["short"]) > SHORT_MAX or set(r["short"]) & set(',"{}\\$'):
        raise PaperDataError(f"{where}: `short` (the plot label) must be at most {SHORT_MAX} plain characters, "
                             f"no comma: {r['short']!r}")
    if "," in r["session"] or "," in r["host"]:
        raise PaperDataError(f"{where}: session and host must not contain a comma")
    r["macro"] = r.get("macro") or spell(r["variant"])
    if not NAME_RE.match(r["macro"]):
        raise PaperDataError(f"{where}: give the row a letters-only `macro` (derived {r['macro']!r} is not one)")
    busy = r.get("n_busy_cores_reps") or []
    r["loaded"] = bool(busy) and max(busy) - 1 > MAX_OTHER_BUSY  # the arm's own core is one of the busy ones


def attach_energy(target: dict, src: dict, where: str) -> None:
    """Attach one energy measurement (inline fields or a sidecar line) to its row, keyed by method."""
    have = [k for k in ENERGY_FIELDS if src.get(k) not in (None, "")]
    if not have:
        return
    if len(have) != len(ENERGY_FIELDS):
        raise PaperDataError(f"{where}: energy fields are all-or-none; missing {sorted(set(ENERGY_FIELDS) - set(have))}")
    if src is not target and src.get("host") != target.get("host"):
        raise PaperDataError(f"{where}: an energy line names its host, and it must be the row's own host "
                             f"({target.get('host')}); watts and seconds come from one run")
    method = src["energy_method"]
    if method not in ENERGY_METHODS:
        raise PaperDataError(f"{where}: bad energy_method {method!r}; allowed: {', '.join(ENERGY_METHODS)}")
    watts = src.get("watts_per_stream")
    if watts is not None and abs(float(watts) - float(src["joules_per_audio_s"])) > 1e-9:
        raise PaperDataError(f"{where}: watts_per_stream must equal joules_per_audio_s (the same quantity)")
    if method in target.setdefault("energy", {}):
        raise PaperDataError(f"{where}: energy by {method} is given twice for row {target['id']}")
    target["energy"][method] = float(src["joules_per_audio_s"])
