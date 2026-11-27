"""Outputs of the measured comparison rows: cold_dev.csv (read by the figures) and one record per number."""

from __future__ import annotations

from pathlib import Path

from .common import HOST_NAMES, MACHINES, csv_text, fmt
from .evidence import HEADLINE, derived, r7_campaign, read_rows, spell
from .records import Records

# cold_dev.csv is read by pgfplots, which splits on every comma: no free-text column except `short`.
# Columns are append-only (figures depend on the names). `plot` = 1 when the lead's rule lets the row be
# drawn: a primary, non-ablation row whose session also holds a row of ours.
CSV_HEADER = ["id", "variant", "short", "role", "class", "machine", "threads", "series", "cpu", "speed", "rtfx",
              "wer", "wer_clean", "wer_other", "rss_mb", "model_mb", "lag_p50_ms", "lag_p95_ms", "watts", "label",
              "host", "plot", "lag_design_p50_ms", "lag_design_p95_ms", "final_latency_p50_ms", "final_latency_p95_ms",
              "faded", "timing_panel", "energy_r7"]


PARALLEL_TAG = {"intra_op": "", "workers": "Workers", "processes": "Procs"}
ENERGY_WORDS = {"core_msr_active": "core energy registers", "core_msr_dynamic": "core energy registers above idle",
                "pkg_rapl_delta": "package energy above idle", "wall_meter": "wall power meter"}


def macro_suffix(r: dict) -> str:
    """Xeon / Mthree for the machine, Eight for eight cores, and how they are used (Workers, Procs)."""
    cores = "" if r["threads"] == 1 else spell(str(r["threads"])) + PARALLEL_TAG[r["parallel"]]
    return MACHINES[r["machine"]]["macro"] + cores


def speed_text(x: float) -> str:
    return fmt(x, 0 if x >= 100 else 1 if x >= 10 else 2)


def _csv_row(r: dict) -> list:
    d = derived(r)

    def num(key: str, nd: int) -> str:
        return "" if d[key] is None else f"{float(d[key]):.{nd}f}"

    series = f"{r['machine']}-t{r['threads']}-{r['role']}-{r['class'].split('_')[0]}"
    return [r["id"], r["variant"], r["short"], r["role"], r["class"], r["machine"], r["threads"], series,
            num("cpu", 5), num("speed", 3), num("rtfx", 3), num("wer", 2), num("wer_clean", 2), num("wer_other", 2),
            num("rss_mb", 1), num("model_mb", 1), num("lag_p50_ms", 0), num("lag_p95_ms", 0), num("watts", 3),
            r["label"], r["host"], int(r["plot"]), num("lag_design_p50_ms", 0), num("lag_design_p95_ms", 0),
            num("final_latency_p50_ms", 0), num("final_latency_p95_ms", 0), int(r["faded"]), r["timing_panel"],
            num("energy_r7", 6)]


def _add_row_records(rec: Records, r: dict) -> None:
    d = derived(r)
    suffix = macro_suffix(r)
    conf = (f"{r['system']}; {r['runtime']}; {r['configuration']}; class {r['class']}; {r['threads']} thread(s) "
            f"({r['parallel']}) on {r['hardware']} (host {r['host']}, session {r['session']}); scope {r['scope']}; "
            f"median of {len(r['cpu_reps'])} repetitions")
    common = dict(label=r["label"], workload=r["workload"], workload_digest=r["workload_digest"],
                  source=f"docs/paper/data/evidence/{r['_where']} id={r['id']}; receipt {r['source']} "
                         f"sha256 {str(r['receipt_sha256'])[:12]}")
    fields = [
        ("Speed", speed_text(d["speed"]), d["speed"], "audio-s per CPU-s (x real time on the pinned cores)", ""),
        ("Rtfx", speed_text(d["rtfx"]), d["rtfx"], "audio-s per wall-s", ""),
        ("Cpu", fmt(d["cpu"], 3), d["cpu"], "CPU-s/audio-s", ""),
        ("Wer", fmt(d["wer"], 2), d["wer"], "% WER", "; pooled over the 730 utterances"),
        ("WerClean", fmt(d["wer_clean"], 2), d["wer_clean"], "% WER", "; dev-clean half (400 utterances)"),
        ("WerOther", fmt(d["wer_other"], 2), d["wer_other"], "% WER", "; dev-other half (330 utterances)"),
        ("Rss", fmt(d["rss_mb"], 0), d["rss_mb"], "MB peak resident memory of the process tree (10^6 bytes)", ""),
        ("Model", fmt(d["model_mb"], 0), d["model_mb"], "MB of model files (10^6 bytes)", ""),
    ]
    if d["installed_mb"] is not None:
        fields.append(("Installed", fmt(d["installed_mb"], 0), d["installed_mb"], "MB installed: every file of the bundle "
                       "the engine loads, FP32 weights included (10^6 bytes)", "; raw/bundle listing"))
    for key, name, unit in (
        ("lag_p50_ms", "LagPfifty", "ms word lag, input paced at real time, p50"),
        ("lag_p95_ms", "LagPninetyFive", "ms word lag, input paced at real time, p95"),
        ("lag_design_p50_ms", "LagDesignPfifty", "ms word lag with unbounded compute, p50"),
        ("lag_design_p95_ms", "LagDesignPninetyFive", "ms word lag with unbounded compute, p95"),
        ("final_latency_p50_ms", "FinalLatencyPfifty", "ms from the end of the clip to the transcript, p50"),
        ("final_latency_p95_ms", "FinalLatencyPninetyFive", "ms from the end of the clip to the transcript, p95"),
    ):
        if d[key] is not None:
            fields.append((name, fmt(d[key], 0), d[key], unit,
                           f"; {r['lag_method']}; lag workload {str(r['lag_workload_digest'])[:12]}"))
    fields.append(("CorePct", fmt(100 * d["cpu"], 1), 100 * d["cpu"], "% of one core for one always-on stream", ""))
    if d["watts"] is not None:
        fields.append(("Watts", fmt(d["watts"], 3), d["watts"], "W per live stream (= J per audio-s)",
                       f"; energy method {r['energy_method']}"))
    for name, value, raw, unit, note in fields:
        rec.add(f"Cold{r['macro']}{name}{suffix}", value, raw=raw, unit=unit, configuration=conf + note, **common)


R7_WORDS = "R7 matched campaign on the timing subset (one pinned Zen 2 core, counterbalanced blocks)"
R8_WORDS = ("R8 follow-up on the full panel (R7 protocol: same pinned Zen 2 core, idle windows and counterbalanced "
            "blocks; v3.1.0 live and Zipformer INT8 anchors, block-1 anchors within 5% of R6)")
R7_PAIR = ("rc2-live-lm", "r0-int8-r6")  # the paired headline systems: Isosyll live and Zipformer INT8


def add_r7_records(root: Path, rows: list[dict], rec: Records) -> None:
    """Campaign-level R7 numbers for the protocol text, captions and briefs, read from the verdict line and rows."""
    camp = r7_campaign(root)
    for r in rows:  # one record per R7 row, with the campaign's workload and receipt, not the timing row's
        if r.get("anchor") or r.get("energy_idle_subtracted") is None:
            continue
        value = float(r["energy_idle_subtracted"])
        r8 = r.get("energy_protocol") == "R8"
        rec.add(f"Cold{r['macro']}EnergyRseven{macro_suffix(r)}", fmt(value, 3), raw=value,
                unit="J per audio-s (idle-subtracted core)", label=r["label"],
                configuration=f"{r['system']}; {r['runtime']}; {R8_WORDS if r8 else R7_WORDS}, median of "
                              f"{r['energy_idle_subtracted_n_clean']} core-clean blocks",
                workload=("R8: the full 730-utterance LibriSpeech DEV panel" if r8 else "R7 timing subset of the LibriSpeech DEV panel"),
                workload_digest=camp["workload_digest"] if camp else "pending (no R7 verdict line)",
                source=r["energy_r7_source"])
    have = [r for r in rows if (r["machine"], r["threads"]) == HEADLINE and not r.get("anchor")
            and r.get("energy_idle_subtracted") is not None and r.get("energy_protocol", "R7") == "R7"]
    if camp is None or not have:
        return
    common = dict(label="exploratory", workload=f"R7: {camp['clips']}-clip subset of the 730-utterance DEV panel",
                  workload_digest=camp["workload_digest"], source="docs/paper/data/evidence/xbabe0-r7-energy.jsonl")
    n_clean = [int(r["energy_idle_subtracted_n_clean"]) for r in have]

    def put(rid: str, value: str, raw, unit: str, what: str) -> None:
        rec.add(rid, value, raw=raw, unit=unit, configuration=f"R7 matched energy campaign: {what}", **common)

    put("RsevenArms", str(len(have)), len(have), "arms", "systems with an R7 energy value")
    put("RsevenBlocks", str(camp["blocks"]), camp["blocks"], "blocks", "counterbalanced blocks over every arm")
    put("RsevenClips", str(camp["clips"]), camp["clips"], "clips", "utterances of the timing subset every arm decodes")
    put("RsevenCleanMin", str(min(n_clean)), min(n_clean), "blocks", "fewest core-clean blocks behind one arm's median")
    put("RsevenCleanMax", str(max(n_clean)), max(n_clean), "blocks", "most core-clean blocks behind one arm's median")
    for name, tag in (("live", "Live"), ("zip", "Zip")):
        a = camp["anchors"][name]
        dev = 100.0 * abs(a["j_per_audio_s"] / a["r6"] - 1.0)
        put(f"RsevenAnchor{tag}Pct", fmt(dev, 1), dev, "%",
            f"full-panel {name} anchor ({a['clips']} utterances, {a['audio_seconds']} audio-s) vs R6, absolute deviation")
    pair = {r["variant"]: r for r in have if r["variant"] in R7_PAIR}
    if len(pair) == 2:
        live, zipf = (float(pair[v]["energy_idle_subtracted"]) for v in R7_PAIR)
        put("RsevenLiveEnergy", fmt(live, 3), live, "J per audio-s", "Isosyll live, median over core-clean blocks")
        put("RsevenZipEnergy", fmt(zipf, 3), zipf, "J per audio-s", "Zipformer INT8, median over core-clean blocks")
        put("RsevenEnergyPct", fmt(100.0 * live / zipf, 0), 100.0 * live / zipf, "%",
            "Isosyll live as % of Zipformer INT8 (ratio of the two R7 medians, corroborating the paired R6 ratio)")


def load_evidence(root: Path, rec: Records, out: dict[str, str]) -> list[dict]:
    """Write cold_dev.csv (every group, one line per row) and one record per printed quantity."""
    rows = read_rows(root)
    out["cold_dev.csv"] = csv_text(CSV_HEADER, [_csv_row(r) for r in rows])
    anchors = [r for r in rows if r.get("anchor")]
    for r in rows:
        if not r.get("anchor"):
            _add_row_records(rec, r)
    add_r7_records(root, rows, rec)
    group = [r for r in rows if (r["machine"], r["threads"]) == HEADLINE and not r.get("anchor")]
    drawn = [r for r in group if r["plot"]]
    first = rows[0] if rows else None
    common = dict(label="reported", workload=first["workload"] if first else "pending (no evidence rows yet)",
                  workload_digest=first["workload_digest"] if first else "pending",
                  source="docs/paper/data/evidence/*.jsonl")
    # a host-agreement check needs a quiet second measurement: a row measured on a busy host says nothing about it
    pairs = [r for r in anchors if r["anchor_ratio"] is not None and not r["loaded"]]
    ratios = sorted(r["anchor_ratio"] for r in pairs)
    quiet_host = MACHINES[HEADLINE[0]]["primary_host"]
    if not ratios:
        sentence = "No system has a quiet second measurement yet, so there is no host-agreement check."
    else:
        span = fmt(ratios[0], 2) + (f" to {fmt(ratios[-1], 2)}" if fmt(ratios[0], 2) != fmt(ratios[-1], 2) else "")
        hosts = {r["host"] for r in pairs}
        one = len(pairs) == 1
        if len(hosts) == 1 and quiet_host not in hosts and all(r["anchor_base_host"] == quiet_host for r in pairs):
            who = (pairs[0].get("text") or pairs[0]["short"]) if one else f"{len(pairs)} systems"
            sentence = (f"{who}, measured on both hosts, {'costs' if one else 'cost'} {span} times as much CPU on the "
                        f"shared host {HOST_NAMES.get(hosts.pop(), '?')} as on the quiet host "
                        f"{HOST_NAMES.get(quiet_host, quiet_host)}.")
        else:
            sentence = (f"{len(pairs)} row{' was' if one else 's were'} measured a second time (another host or "
                        f"session); {'its' if one else 'their'} CPU cost is {span} times that of the row printed here.")
    interim = [r for r in drawn if r["interim"]]
    quiet = HOST_NAMES.get(MACHINES[HEADLINE[0]]["primary_host"], MACHINES[HEADLINE[0]]["primary_host"])
    other = HOST_NAMES.get(interim[0]["host"], interim[0]["host"]) if interim else ""
    if interim and len(interim) == len(drawn):
        host = (f"Interim: every point was measured on host {other}, a shared host that ran other jobs; rows from the "
                f"quiet host {quiet} replace them as they arrive.")
    elif interim:
        names = ", ".join(r["short"] for r in interim[:-1]) + (" and " if len(interim) > 1 else "") + interim[-1]["short"]
        host = (f"{names} {'is' if len(interim) == 1 else 'are'} still interim: measured on host {other}, a shared host "
                f"that ran other jobs and makes every system slower, until their rows from the quiet host {quiet} arrive.")
    else:
        host = (f"Every point was measured on host {quiet}, one system at a time on the otherwise idle machine."
                if drawn else "No measured rows yet.")
    if interim and len(interim) == len(drawn):
        short = f"Interim: every point was measured on the shared host {other}; quiet-host ({quiet}) rows replace them."
    elif interim:
        short = (f"Interim ({len(interim)} systems): measured on the shared host {other} "
                 f"until their quiet-host ({quiet}) rows arrive.")
    else:
        short = f"Every point was measured on host {quiet}, one system at a time." if drawn else "No measured rows yet."
    minutes = {round(r["audio_seconds"] / 60.0, 1) for r in rows if r.get("audio_seconds")}
    if len(minutes) == 1:
        rec.add("ColdAudioMinutes", fmt(minutes.pop(), 1), unit="minutes of audio", configuration="duration of the "
                "staged audio every measured row decodes", **common)
    for rid, value, what in (
        ("ColdRowsCount", len(drawn), "systems drawn in Figure 1 (one pinned Zen 2 core; 0 = pending)"),
        ("ColdRowsStreaming", sum(r["class"] == "streaming" for r in drawn), "live streaming systems in Figure 1"),
        ("ColdRowsWhole", sum(r["class"] == "whole_utterance" for r in drawn), "whole-utterance systems in Figure 1"),
        ("ColdRowsEnergy", sum(r["watts"] is not None for r in drawn),
         "Figure 1 systems with a measured energy row (0 = watts pending)"),
        ("ColdRowsLag", sum(r.get("lag_p50_ms") is not None for r in drawn),
         "Figure 1 systems with a measured word lag (0 = lag pending)"),
        ("ColdRowsUndrawn", sum(not r["plot"] and r["role"] == "comparator" for r in group),
         "one-core Zen 2 comparator rows excluded because their session holds no row of ours"),
        ("ColdRowsOffFrontier", sum(r["plot"] and not r["frontier"] for r in group),
         "drawn rows kept out of the frontier line because their session's anchor is more than 10% off"),
        ("ColdAnchorCount", len(anchors), "cross-check rows (a variant measured on a second host or session)"),
        ("ColdRowsInterim", len(interim), "systems drawn in Figure 1 from a row measured away from the quiet host"),
        ("ColdRowsFaded", sum(r["faded"] for r in drawn), "interim systems drawn lighter beside quiet-host rows"),
        ("ColdRowsSubset", sum(r["timing_panel"] != "full" for r in drawn),
         "systems drawn in Figure 1 that are timed on the 73-utterance subset of the panel"),
    ):
        rec.add(rid, str(value), raw=value, unit="rows", configuration=f"number of {what}", **common)
    rec.add("ColdAnchorSentence", sentence, unit="sentence", configuration="generated from the cross-check rows", **common)
    rec.add("ColdHostSentence", host, unit="sentence", configuration="generated from the hosts of the rows", **common)
    rec.add("ColdHostShort", short, unit="sentence", configuration="generated from the hosts of the rows (short form "
            "for the caption of Figure 1)", **common)
    if interim:
        brief = (f"{len(interim)} of the {len(drawn)} systems are still measured on a shared host; quiet-host rows "
                 f"replace them as they arrive.")
    else:
        brief = "\\relax"
    rec.add("ColdHostAbstract", brief, unit="sentence", configuration="generated from the hosts of the rows (the "
            "abstract's form)", **common)
    return rows
