"""Measured comparison rows: docs/paper/data/evidence/*.jsonl, one per (system, runtime, machine, threads).

Rows are grouped by (machine, threads); `class` separates live streaming from whole-utterance systems.
Every row shares one workload digest; a group has one cost scope and one headline energy method.
Lag and energy may arrive as sidecar lines keyed by the row id. Published numbers are refused here
(they live in data/reported.json). Row format and checks: schema.py and docs/paper/data/README.md.
"""

from __future__ import annotations

from pathlib import Path
import json

from .common import DATA, MACHINES, PaperDataError, read_json
from .schema import (CROSS_CHECK_METHODS, LAG_KEYS, LAG_PROOF, PRESENTATION, attach_energy, check_row,  # noqa: F401
                     spell)  # spell: re-exported for cold.py

HEADLINE = ("zen2", 1)  # the group Figure 1 draws: one pinned Zen 2 core
ANCHOR_LIMIT_PCT = 10.0  # a session whose anchor is further off than this stays out of the frontier


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
            if "energy_j_per_audio_s" in r:  # matched-campaign energy line (R7): Figure 1 colour only
                kind = "r7_energy"
            if kind == "r7_energy":
                sidecars.append(r)
            elif kind == "cost":
                rows.append(r)
            elif kind in ("lag", "energy"):
                sidecars.append(r)
            elif kind == "energy_headline":  # without machine and threads it names the method of every group
                group = (r["machine"], r["threads"]) if "machine" in r or "threads" in r else "*"
                if headline.setdefault(group, r["energy_method"]) != r["energy_method"]:
                    raise PaperDataError(f"{r['_where']}: a second energy headline for group {group}")
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
        target = by_id.get(s.get("row", s.get("id")))
        if s["kind"] == "r7_energy":
            if target is None:
                raise PaperDataError(f"{s['_where']}: energy line points at unknown row {s.get('id', s.get('row'))!r}")
            if s.get("energy_method") != "core_idle_subtracted":
                raise PaperDataError(f"{s['_where']}: energy_method must be core_idle_subtracted")
            target["energy_idle_subtracted"] = float(s["energy_j_per_audio_s"])
            continue
        if target is None:
            raise PaperDataError(f"{s['_where']}: {s['kind']} sidecar points at unknown row {s.get('row')!r}")
        if s["kind"] == "energy":
            attach_energy(target, s, s["_where"])
            if s.get("energy_method") == "core_msr_active":  # Figure 1 says which rows still include the idle floor
                target["energy_includes_idle"] = "idle-subtracted" not in str(s.get("energy_tool", ""))
            continue
        for k in LAG_KEYS:
            if s.get(k) is not None and target.get(k) is not None:
                raise PaperDataError(f"{s['_where']}: {k} is given twice for row {target['id']}")
            target[k] = s.get(k, target.get(k))


def _group(group: tuple, members: list[dict], chosen: str | None) -> None:
    """One (machine, threads) group: energy headline, cross-checks against primary rows, what is drawn."""
    if len({r["scope"] for r in members}) > 1:
        raise PaperDataError(f"evidence rows mix scopes inside group {group}; a group shares one axis")
    methods = sorted({m for r in members for m in r.get("energy", {}) if m not in CROSS_CHECK_METHODS})
    if chosen in CROSS_CHECK_METHODS:
        raise PaperDataError(f"group {group}: {chosen} is a cross-check method and cannot be the energy headline")
    if len(methods) > 1 and chosen is None:
        raise PaperDataError(f"group {group} has energy by {methods}; add an energy_headline line naming one")
    chosen = chosen or (methods[0] if methods else None)  # None: no same-run energy value in this group yet
    quiet = MACHINES[group[0]]["primary_host"]
    seen: dict[tuple, dict] = {}
    for r in sorted(members, key=lambda r: (r["loaded"], r["session"], r["id"])):
        if not r.get("anchor") and quiet in (None, r["host"]):
            if r["loaded"] and quiet is not None:
                r["anchor"] = True  # too many other busy cores on the quiet host: a cross-check, never primary
            elif seen.setdefault((r["variant"], r["parallel"]), r) is not r:
                r["anchor"] = True  # measured again in a later session on the same host: a cross-check
    plain = [r for r in members if not r.get("anchor")]
    home = [r for r in plain if quiet in (None, r["host"])]
    here = {(r["variant"], r["parallel"]) for r in home}
    for r in plain:
        if r not in home and (r["variant"], r["parallel"]) in here:
            r["anchor"] = True  # the same variant exists on the quiet host: this row is its cross-check
    drawable = [r for r in plain if not r.get("anchor")]  # the quiet-host row of each variant, else its other row
    primary = {(r["variant"], r["parallel"]): r for r in drawable}
    if len(primary) != len(drawable):
        raise PaperDataError(f"group {group} has two primary rows for one variant; mark the re-measurement as anchor")
    for r in members:
        r["interim"] = r in drawable and r not in home  # measured only away from the quiet host so far
        r["faded"] = r["interim"] and bool(home)        # drawn lighter once quiet-host rows share the figure
    # rows measured under the quiet host's timing protocol (one arm at a time on an otherwise idle machine) are
    # comparable whatever their session; elsewhere a comparator needs a row of ours from its own session
    ours = {r["session"] for r in members if r["role"] == "ours"}
    home_ours = quiet is not None and any(r["role"] == "ours" and r["host"] == quiet and not r.get("anchor")
                                          for r in members)
    if home_ours:
        ours |= {r["session"] for r in members if r["host"] == quiet}
    off = set()
    for r in members:
        r["watts"], r["energy_method"] = r.get("energy", {}).get(chosen), chosen
        if r.get("anchor"):
            base = primary.get((r["variant"], r["parallel"]))
            if base is None and not r["loaded"]:
                raise PaperDataError(f"{r['_where']}: anchor row has no primary row of {r['variant']} in {group}")
            r["anchor_ratio"] = float(r["cpu_s_per_audio_s"]) / float(base["cpu_s_per_audio_s"]) if base else None
            r["anchor_base_host"] = base["host"] if base else None
            if base and not r["loaded"] and abs(r["anchor_ratio"] - 1.0) > ANCHOR_LIMIT_PCT / 100.0:
                off.add(r["session"])
    for r in drawable:  # a lag is a property of the system on the host it was measured on, not of one cost row
        if r.get("lag_p50_ms") is None and r.get("lag_design_p50_ms") is None:
            other = [a for a in members if a.get("anchor") and a["variant"] == r["variant"]
                     and (a.get("lag_p50_ms") is not None or a.get("lag_design_p50_ms") is not None)]
            if other:
                r.update({k: other[0].get(k) for k in LAG_KEYS})
    stems: dict[tuple, str] = {}
    for r in members:
        # drawn only beside a row of ours from the same session (or explicitly admitted interim row beside quiet ours)
        admitted_interim = bool(r["faded"] and home_ours and r.get("plot") is True)
        r["plot"] = bool(r in drawable and r["role"] != "ablation" and r.get("plot", True)
                         and (r["role"] == "ours" or r["session"] in ours or admitted_interim))
        # the frontier line joins rows of one host: the quiet host as soon as it has rows
        r["frontier"] = r["plot"] and r["session"] not in off and not r["faded"]
        if not r.get("anchor") and stems.setdefault((r["macro"], r["parallel"]), r["id"]) != r["id"]:
            raise PaperDataError(f"rows {stems[r['macro']]} and {r['id']} share the macro stem {r['macro']}")


def presentation(root: Path) -> dict:
    """data/systems.json: paper-side names per variant and the headline row of ours."""
    return read_json(root, str(DATA / "systems.json")) if (root / DATA / "systems.json").is_file() else {}


def read_rows(root: Path) -> list[dict]:
    """Validated rows, sidecars merged, sorted by group; adds `watts`, `plot`, `frontier`, `interim`,
    `faded`, and `anchor_ratio` on cross-check rows."""
    rows, sidecars, headline = _read(root)
    _merge(rows, sidecars)
    present = presentation(root).get("variants", {})
    for r in rows:  # ours: every file the engine's bundle installs (raw/bundle), not only the INT8 runtime files
        bundle = root / DATA / "raw" / "bundle" / f"{r.get('variant')}.files.json"
        if bundle.is_file():
            r["installed_bytes"] = sum(f["bytes"] for f in json.loads(bundle.read_text(encoding="utf-8"))["files"].values())
        extra = present.get(r.get("variant"), {})
        if set(extra) - set(PRESENTATION):
            raise PaperDataError(f"systems.json may only set {PRESENTATION} for {r['variant']}")
        r.update(extra)
        check_row(r, r["_where"])
        attach_energy(r, r, r["_where"])
    if len({r["workload_digest"] for r in rows}) > 1:
        raise PaperDataError("evidence rows mix workloads; every measured row must use one staged audio set")
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["machine"], r["threads"]), []).append(r)
    for group, members in sorted(groups.items()):
        _group(group, members, headline.get(group, headline.get("*")))
    return sorted(rows, key=lambda r: (r["machine"], r["threads"], r.get("order", 999), r["id"]))


def derived(r: dict) -> dict:
    """Printable quantities of one row. MB = 10^6 bytes. speed = audio-seconds per CPU-second."""
    return {
        "cpu": float(r["cpu_s_per_audio_s"]), "speed": 1.0 / float(r["cpu_s_per_audio_s"]),
        "rtfx": 1.0 / float(r["wall_s_per_audio_s"]), "wer": float(r["wer_pct"]),
        "wer_clean": float(r["wer_clean_pct"]), "wer_other": float(r["wer_other_pct"]),
        "rss_mb": r["peak_rss_bytes"] / 1e6, "model_mb": r["artifact_bytes"] / 1e6,
        "installed_mb": r["installed_bytes"] / 1e6 if r.get("installed_bytes") else None,
        "watts": r.get("watts"), **{k: r.get(k) for k in LAG_KEYS if k not in LAG_PROOF + ("lag_host",)},
    }
