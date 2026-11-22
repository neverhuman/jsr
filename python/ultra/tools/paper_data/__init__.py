"""Regenerate every number and figure table of the ULTRA paper from repo-tracked sources.

Standard library only. Deterministic: sorted keys, fixed formatting, LF line endings.

Inputs (all under the repo root):
    docs/paper/data/raw/**           small raw bench artifacts, sha256-pinned in raw/MANIFEST.txt
    docs/paper/data/curated.json     numbers that live only in docs, each with its exact source
    docs/paper/data/evidence/*.jsonl matched cold-DEV comparator rows (format: data/README.md)
    research/curves/index.json, research/curves/<run>/{train,eval}.csv

Outputs (docs/paper/data/ unless --out):
    metrics.tex      one \\newcommand per record; a missing value prints a red ?? and fails
    records.jsonl    every record: id, value, unit, label, configuration, workload, digest, source
    SOURCES.md       macro -> value -> label -> source table
    *.csv            figure tables read by docs/paper/figures/*.tex

Usage (from python/ultra):
    ../../.venv/bin/python -m tools.paper_data            # write docs/paper/data
    ../../.venv/bin/python -m tools.paper_data --check    # exit 1 if committed files differ
"""


from __future__ import annotations

from pathlib import Path

from .common import (
    ABAB_WORKLOAD,
    CANDIDATES,
    DATA,
    EVIDENCE_REQUIRED,
    FRONT_DENSE,
    LABELS,
    LADDER_RUNS,
    LADDER_WORKLOAD,
    MATCHED_RUNG,
    NAME_RE,
    PANEL_DIGEST,
    RAW,
    REPO,
    REQUIRED,
    PaperDataError,
    csv_text,
    fmt,
    read_csv,
    read_json,
    sci_tex,
    sha256_file,
    thousands,
    verify_manifest,
    write_manifest,
)
from .derive import add_derived
from .jobg import add_job_g
from .figures import accuracy_table, add_curves, add_sweeps, add_tables
from .macros import metrics_tex, records_jsonl, sources_md
from .records import Records, add_curated
from .rehearsal import add_rehearsal
from .sources import add_cluster_gap, add_codex_evidence, add_panel5, load_evidence


def build(root: Path = REPO) -> tuple[dict[str, str], list[str]]:
    """Return ({relative file name: content}, missing record ids)."""
    verify_manifest(root)
    rec = Records()
    out: dict[str, str] = {}
    cur = add_curated(root, rec)
    add_codex_evidence(root, rec)
    add_panel5(root, rec)
    add_cluster_gap(root, rec)
    add_derived(rec)
    add_rehearsal(root, rec, out)
    add_job_g(root, rec, out)
    add_curves(root, rec, out)
    add_sweeps(root, rec, out)
    add_tables(cur, out)
    accuracy_table(rec, out)
    load_evidence(root, rec, out)
    out["metrics.tex"] = metrics_tex(rec)
    out["SOURCES.md"] = sources_md(rec)
    out["records.jsonl"] = records_jsonl(rec)
    return dict(sorted(out.items())), rec.missing()
