"""Measured comparison rows: docs/paper/data/evidence/*.jsonl, one per (system, runtime, machine, threads).

Rows are grouped by (machine, threads); `class` separates live streaming from whole-utterance systems.
Every row shares one workload digest; a group has one cost scope and one headline energy method.
Lag and energy may arrive as sidecar lines keyed by the row id. Published numbers are refused here
(they live in data/reported.json). Format: docs/paper/data/README.md.
"""

from __future__ import annotations

from pathlib import Path
import json
import statistics

from .common import DATA, EVIDENCE_REQUIRED, MACHINES, NAME_RE, PaperDataError, read_json

ROLES = ("ours", "comparator", "ablation")  # ablation: our weights in another runtime; tables only
CLASSES = ("streaming", "whole_utterance")
PARALLEL = ("intra_op", "workers")
EVIDENCE_LABELS = ("exploratory", "qualified")
ENERGY_METHODS = ("core_msr_active", "core_msr_dynamic", "pkg_rapl_delta", "wall_meter")
LAG_FIELDS = ("lag_p50_ms", "lag_p95_ms", "lag_method", "lag_workload_digest", "lag_receipt_sha256")
ENERGY_FIELDS = ("joules_per_audio_s", "energy_method", "energy_receipt_sha256")
PRESENTATION = ("system", "short", "macro", "cite", "order", "plot")  # data/systems.json may set these per variant
HEADLINE = ("zen2", 1)  # the group Figure 1 draws: one pinned Zen 2 core
MIN_REPS = 3
SHORT_MAX = 22
ANCHOR_LIMIT_PCT = 10.0  # a session whose anchor is further off than this stays out of the frontier
_DIGITS = "Zero One Two Three Four Five Six Seven Eight Nine".split()


def spell(text: str) -> str:
    """'d2e-16457' -> 'DTwoeOneSixFourFiveSeven': a letters-only macro stem."""
    return "".join(_DIGITS[int(c)] if c.isdigit() else c for part in text.split("-") for c in part.capitalize())


def _median_ok(value, reps) -> bool:
    med = statistics.median(float(x) for x in reps)
    return abs(med - float(value)) <= 1e-9 * max(1.0, med)


def _check_row(r: dict, where: str) -> None:
    miss = [k for k in EVIDENCE_REQUIRED if r.get(k) in (None, "", [])]
    if miss:
        raise PaperDataError(f"{where}: evidence row missing {miss}")
    if r["label"] == "reported":
        raise PaperDataError(f"{where}: a reported (published) number cannot be a measured row; use data/reported.json")
    for key, allowed in (("label", EVIDENCE_LABELS), ("role", ROLES), ("class", CLASSES), ("parallel", PARALLEL),
                         ("machine", tuple(MACHINES))):
        if r[key] not in allowed:
            raise PaperDataError(f"{where}: bad {key} {r[key]!r}; allowed: {', '.join(allowed)}")
    if not isinstance(r["threads"], int) or r["threads"] < 1:
        raise PaperDataError(f"{where}: threads must be a positive integer")
    base = f"{r['variant']}-{r['machine']}-t{r['threads']}"
    if r["id"] != base and not (r.get("anchor") and r["id"].startswith(base + "-anchor")):
        raise PaperDataError(f"{where}: id must be <variant>-<machine>-t<threads> (an anchor row appends "
                             f"-anchor-<host>), got {r['id']!r}")
    for value, reps in (("cpu_s_per_audio_s", "cpu_reps"), ("wall_s_per_audio_s", "wall_reps")):
        if len(r[reps]) < MIN_REPS:
            raise PaperDataError(f"{where}: {reps} needs at least {MIN_REPS} repetitions")
        if not _median_ok(r[value], r[reps]):
            raise PaperDataError(f"{where}: {value} is not the median of {reps}")
    lag = [k for k in LAG_FIELDS if r.get(k) not in (None, "")]
    if lag and r["class"] == "whole_utterance":
        raise PaperDataError(f"{where}: a whole_utterance row reports throughput only; it cannot carry a word lag")
    if lag and len(lag) != len(LAG_FIELDS):
        raise PaperDataError(f"{where}: lag fields are all-or-none; missing {sorted(set(LAG_FIELDS) - set(lag))}")
    r["short"] = r.get("short") or r["system"]
    if len(r["short"]) > SHORT_MAX or set(r["short"]) & set(',"{}\\$'):
        raise PaperDataError(f"{where}: `short` (the plot label) must be at most {SHORT_MAX} plain characters, "
                             f"no comma: {r['short']!r}")
    if "," in r["session"] or "," in r["host"]:
        raise PaperDataError(f"{where}: session and host must not contain a comma")
    r["macro"] = r.get("macro") or spell(r["variant"])
    if not NAME_RE.match(r["macro"]):
        raise PaperDataError(f"{where}: give the row a letters-only `macro` (derived {r['macro']!r} is not one)")


def _energy(target: dict, src: dict, where: str) -> None:
    """Attach one energy measurement (inline fields or a sidecar line) to its row, keyed by method."""
    have = [k for k in ENERGY_FIELDS if src.get(k) not in (None, "")]
    if not have:
        return
    if len(have) != len(ENERGY_FIELDS):
        raise PaperDataError(f"{where}: energy fields are all-or-none; missing {sorted(set(ENERGY_FIELDS) - set(have))}")
    method = src["energy_method"]
    if method not in ENERGY_METHODS:
        raise PaperDataError(f"{where}: bad energy_method {method!r}; allowed: {', '.join(ENERGY_METHODS)}")
    watts = src.get("watts_per_stream")
    if watts is not None and abs(float(watts) - float(src["joules_per_audio_s"])) > 1e-9:
        raise PaperDataError(f"{where}: watts_per_stream must equal joules_per_audio_s (the same quantity)")
    if method in target.setdefault("energy", {}):
        raise PaperDataError(f"{where}: energy by {method} is given twice for row {target['id']}")
    target["energy"][method] = float(src["joules_per_audio_s"])


def _read(root: Path) -> tuple[list[dict], list[dict], dict]:
    rows, sidecars, headline = [], [], {}
    edir = root / DATA / "evidence"
    for p in sorted(edir.glob("*.jsonl")) if edir.is_dir() else []:
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            r = json.loads(line)
            r["_where"] = f"{p.name}:{n}"
            kind = r.get("kind", "cost")
            if kind == "cost":
                rows.append(r)
            elif kind in ("lag", "energy"):
                sidecars.append(r)
            elif kind == "energy_headline":
                if headline.setdefault((r["machine"], r["threads"]), r["energy_method"]) != r["energy_method"]:
                    raise PaperDataError(f"{r['_where']}: a second energy headline for group {r['machine']}/t{r['threads']}")
            else:
                raise PaperDataError(f"{r['_where']}: unknown kind {kind!r}")
    return rows, sidecars, headline


def _merge(rows: list[dict], sidecars: list[dict]) -> None:
    by_id: dict[str, dict] = {}
    for r in rows:
        if r.get("id") in by_id:
            raise PaperDataError(f"duplicate evidence ids: {r['id']} ({by_id[r['id']]['_where']} and {r['_where']})")
        by_id[r.get("id")] = r
    for s in sidecars:
        target = by_id.get(s.get("row"))
        if target is None:
            raise PaperDataError(f"{s['_where']}: {s['kind']} sidecar points at unknown row {s.get('row')!r}")
        if s["kind"] == "energy":
            _energy(target, s, s["_where"])
            continue
        for k in LAG_FIELDS:
            if s.get(k) is not None and target.get(k) is not None:
                raise PaperDataError(f"{s['_where']}: {k} is given twice for row {target['id']}")
            target[k] = s.get(k, target.get(k))


def _group(group: tuple, members: list[dict], chosen: str | None) -> None:
    """One (machine, threads) group: energy headline, anchors against their primaries, what is drawn."""
    if len({r["scope"] for r in members}) > 1:
        raise PaperDataError(f"evidence rows mix scopes inside group {group}; a group shares one axis")
    methods = sorted({m for r in members for m in r.get("energy", {})})
    if len(methods) > 1 and chosen is None:
        raise PaperDataError(f"group {group} has energy by {methods}; add an energy_headline line naming one")
    chosen = chosen or (methods[0] if methods else None)
    primary = {r["variant"]: r for r in members if not r.get("anchor")}
    if len(primary) != sum(not r.get("anchor") for r in members):
        raise PaperDataError(f"group {group} has two primary rows for one variant; mark the re-measurement as anchor")
    ours = {r["session"] for r in members if r["role"] == "ours"}
    off = set()
    for r in members:
        r["watts"], r["energy_method"] = r.get("energy", {}).get(chosen), chosen
        if r.get("anchor"):
            if r["variant"] not in primary:
                raise PaperDataError(f"{r['_where']}: anchor row has no primary row of {r['variant']} in {group}")
            base = float(primary[r["variant"]]["cpu_s_per_audio_s"])
            r["anchor_pct"] = 100.0 * abs(float(r["cpu_s_per_audio_s"]) - base) / base
            if r["anchor_pct"] > ANCHOR_LIMIT_PCT:
                off.add(r["session"])
    stems: dict[str, str] = {}
    for r in members:
        # drawn only beside a row of ours from the same session; never an anchor or an ablation
        r["plot"] = bool(not r.get("anchor") and r["role"] != "ablation" and r.get("plot", True)
                         and (r["role"] == "ours" or r["session"] in ours))
        r["frontier"] = r["plot"] and r["session"] not in off
        if not r.get("anchor") and stems.setdefault(r["macro"], r["id"]) != r["id"]:
            raise PaperDataError(f"rows {stems[r['macro']]} and {r['id']} share the macro stem {r['macro']}")


def read_rows(root: Path) -> list[dict]:
    """Validated rows, sidecars merged, sorted by group; adds `watts`, `plot`, `frontier`, `anchor_pct`."""
    rows, sidecars, headline = _read(root)
    _merge(rows, sidecars)
    names = root / DATA / "systems.json"
    present = read_json(root, str(DATA / "systems.json"))["variants"] if names.is_file() else {}
    for r in rows:
        extra = present.get(r.get("variant"), {})
        if set(extra) - set(PRESENTATION):
            raise PaperDataError(f"systems.json may only set {PRESENTATION} for {r['variant']}")
        r.update(extra)
        _check_row(r, r["_where"])
        _energy(r, r, r["_where"])
    if len({r["workload_digest"] for r in rows}) > 1:
        raise PaperDataError("evidence rows mix workloads; every measured row must use one staged audio set")
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["machine"], r["threads"]), []).append(r)
    for group, members in sorted(groups.items()):
        _group(group, members, headline.get(group))
    return sorted(rows, key=lambda r: (r["machine"], r["threads"], r.get("order", 999), r["id"]))


def derived(r: dict) -> dict:
    """Printable quantities of one row. MB = 10^6 bytes. speed = audio-seconds per CPU-second."""
    return {
        "cpu": float(r["cpu_s_per_audio_s"]), "speed": 1.0 / float(r["cpu_s_per_audio_s"]),
        "rtfx": 1.0 / float(r["wall_s_per_audio_s"]), "wer": float(r["wer_pct"]),
        "wer_clean": float(r["wer_clean_pct"]), "wer_other": float(r["wer_other_pct"]),
        "rss_mb": r["peak_rss_bytes"] / 1e6, "model_mb": r["artifact_bytes"] / 1e6,
        "lag_p50_ms": r.get("lag_p50_ms"), "lag_p95_ms": r.get("lag_p95_ms"), "watts": r.get("watts"),
    }
