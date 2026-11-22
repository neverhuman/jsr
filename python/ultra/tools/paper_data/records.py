"""The record registry (id, value, unit, label, configuration, workload, digest, source) and curated.json."""

from __future__ import annotations

from pathlib import Path

from .common import DATA, LABELS, NAME_RE, PaperDataError, REQUIRED, fmt, read_json, thousands


class Records:
    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}

    def add(self, rid: str, value, *, unit: str, label: str, configuration: str, workload: str,
            workload_digest: str, source: str, raw=None, line=None, quote=None) -> None:
        if not NAME_RE.match(rid):
            raise PaperDataError(f"macro name must be letters only: {rid!r}")
        if rid in self.rows:
            raise PaperDataError(f"duplicate record id {rid}")
        if label not in LABELS:
            raise PaperDataError(f"{rid}: label {label!r} not in {LABELS}")
        self.rows[rid] = {
            "id": rid, "value": value, "raw": raw, "unit": unit, "label": label,
            "configuration": configuration, "workload": workload, "workload_digest": workload_digest,
            "source": source, "line": line, "quote": quote,
        }

    def raw(self, rid: str) -> float:
        r = self.rows[rid]["raw"]
        if r is None:
            raise PaperDataError(f"{rid} has no numeric raw value")
        return float(r)

    def pending(self) -> list[str]:
        return sorted(rid for rid, r in self.rows.items() if r.get("pending"))

    def missing(self) -> list[str]:
        out = []
        for rid, r in self.rows.items():
            if r.get("pending"):
                continue
            if any(r.get(k) in (None, "") for k in REQUIRED):
                out.append(rid)
        return sorted(out)


# ---------------------------------------------------------------- inputs → records


def add_curated(root: Path, rec: Records) -> dict:
    cur = read_json(root, str(DATA / "curated.json"))
    for rid in sorted(cur["metrics"]):
        e = cur["metrics"][rid]
        if "text" in e:
            value, raw = e["text"], None
        elif e.get("raw") is None:
            value, raw = None, None
        else:
            raw = e["raw"]
            value = thousands(fmt(float(raw), int(e["decimals"])))
        rec.add(
            rid, value, raw=raw, unit=e["unit"], label=e["label"], configuration=e.get("configuration"),
            workload=e.get("workload"), workload_digest=e.get("workload_digest"),
            source=f"{e['source']} \u00a7 {e['section']}", line=e.get("line"), quote=e.get("quote"),
        )
        if e.get("pending"):
            rec.rows[rid]["pending"] = e["pending"]
    return cur


# Codex Evidence: Rust-rescored DEV accuracy rows (schema jsr.paper-evidence/v1), pinned in raw/.
