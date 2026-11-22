"""Outputs of the measured comparison rows: cold_dev.csv (read by the figures) and one record per number."""

from __future__ import annotations

from pathlib import Path

from .common import MACHINES, csv_text, fmt
from .evidence import ANCHOR_LIMIT_PCT, HEADLINE, derived, read_rows, spell
from .records import Records

# cold_dev.csv is read by pgfplots, which splits on every comma: no free-text column except `short`.
# Columns are append-only (figures depend on the names). `plot` = 1 when the lead's rule lets the row be
# drawn: a primary, non-ablation row whose session also holds a row of ours.
CSV_HEADER = ["id", "variant", "short", "role", "class", "machine", "threads", "series", "cpu", "speed", "rtfx",
              "wer", "wer_clean", "wer_other", "rss_mb", "model_mb", "lag_p50_ms", "lag_p95_ms", "watts", "label",
              "host", "plot"]


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
            r["label"], r["host"], int(r["plot"])]


def _add_row_records(rec: Records, r: dict) -> None:
    d = derived(r)
    suffix = MACHINES[r["machine"]]["macro"] + ("" if r["threads"] == 1 else spell(str(r["threads"])))
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
    if d["lag_p50_ms"] is not None:
        lag = f"; {r['lag_method']}; lag workload {str(r['lag_workload_digest'])[:12]}"
        fields += [("LagPfifty", fmt(d["lag_p50_ms"], 0), d["lag_p50_ms"], "ms stable-word lag, p50", lag),
                   ("LagPninetyFive", fmt(d["lag_p95_ms"], 0), d["lag_p95_ms"], "ms stable-word lag, p95", lag)]
    if d["watts"] is not None:
        fields.append(("Watts", fmt(d["watts"], 2), d["watts"], "W per live stream (= J per audio-s)",
                       f"; energy method {r['energy_method']}"))
    for name, value, raw, unit, note in fields:
        rec.add(f"Cold{r['macro']}{name}{suffix}", value, raw=raw, unit=unit, configuration=conf + note, **common)


def load_evidence(root: Path, rec: Records, out: dict[str, str]) -> list[dict]:
    """Write cold_dev.csv (every group, one line per row) and one record per printed quantity."""
    rows = read_rows(root)
    out["cold_dev.csv"] = csv_text(CSV_HEADER, [_csv_row(r) for r in rows])
    anchors = [r for r in rows if r.get("anchor")]
    for r in rows:
        if not r.get("anchor"):
            _add_row_records(rec, r)
    group = [r for r in rows if (r["machine"], r["threads"]) == HEADLINE and not r.get("anchor")]
    drawn = [r for r in group if r["plot"]]
    first = rows[0] if rows else None
    common = dict(label="reported", workload=first["workload"] if first else "pending (no evidence rows yet)",
                  workload_digest=first["workload_digest"] if first else "pending",
                  source="docs/paper/data/evidence/*.jsonl")
    worst = max((r["anchor_pct"] for r in anchors), default=0.0)
    if not anchors:
        sentence = "No system has been re-measured in a second session yet, so there is no host-agreement check."
    else:
        sentence = (f"{len(anchors)} anchor rows re-measured in a second session differ from their primary rows by at "
                    f"most {fmt(worst, 1)}\\% in CPU cost"
                    + ("." if worst <= ANCHOR_LIMIT_PCT else "; sessions beyond 10\\% stay out of the frontier line."))
    for rid, value, what in (
        ("ColdRowsCount", len(drawn), "systems drawn in Figure 1 (one pinned Zen 2 core; 0 = pending)"),
        ("ColdRowsStreaming", sum(r["class"] == "streaming" for r in drawn), "live streaming systems in Figure 1"),
        ("ColdRowsWhole", sum(r["class"] == "whole_utterance" for r in drawn), "whole-utterance systems in Figure 1"),
        ("ColdRowsEnergy", sum(r["watts"] is not None for r in drawn),
         "Figure 1 systems with a measured energy row (0 = watts pending)"),
        ("ColdRowsLag", sum(r.get("lag_p50_ms") is not None for r in drawn),
         "Figure 1 systems with a measured word lag (0 = lag pending)"),
        ("ColdRowsUndrawn", sum(not r["plot"] and r["role"] == "comparator" for r in group),
         "one-core Zen 2 comparator rows not drawn because their session holds no row of ours"),
        ("ColdRowsOffFrontier", sum(r["plot"] and not r["frontier"] for r in group),
         "drawn rows kept out of the frontier line because their session's anchor is more than 10% off"),
        ("ColdAnchorCount", len(anchors), "anchor rows (a variant re-measured in a second session)"),
    ):
        rec.add(rid, str(value), raw=value, unit="rows", configuration=f"number of {what}", **common)
    rec.add("ColdAnchorSentence", sentence, unit="sentence", configuration="generated from the anchor rows", **common)
    return rows
