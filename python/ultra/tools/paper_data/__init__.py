"""Regenerate every number and figure table of the Isosyll paper from repo-tracked sources.

Standard library only. Deterministic: sorted keys, fixed formatting, LF line endings.

Inputs (all under the repo root):
    docs/paper/data/raw/**           small raw bench artifacts, sha256-pinned in raw/MANIFEST.txt
    docs/paper/data/curated.json     minimal curated constants (each with exact source quote)
    docs/paper/data/evidence/*.jsonl measured comparison rows, schema v2 (only release + kept comparators)
    docs/paper/data/systems.json     names and order of the measured variants; the headline operating points

Outputs (docs/paper/data/ unless --out):
    metrics.tex      one \\newcommand per record; a missing value prints a red ?? and fails
    records.jsonl    every record: id, value, unit, label, configuration, workload, digest, source
    SOURCES.md       macro -> value -> label -> source table (only macros emitted for the paper)
    cold_dev.csv, money_frontier.csv, money_layout.tex, money_readme.md, table_measured.tex, table_hardware.tex
"""

from __future__ import annotations

from pathlib import Path

from .common import (
    DATA,
    EVIDENCE_REQUIRED,
    LABELS,
    MACHINES,
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
from .cold import load_evidence
from .compare import add_comparison
from .evidence import presentation
from .facts import add_facts
from .macros import metrics_tex, records_jsonl, sources_md
from .money import add_money, hero_rows, nemotron_timing_note
from .publish import publish_figure, readme_block
from .records import Records, add_curated
from .release import add_release
from .reported import add_reported
from .tables import hardware_table, measured_table


def build(root: Path = REPO) -> tuple[dict[str, str], list[str]]:
    """Return ({relative file name: content}, missing record ids)."""
    verify_manifest(root)
    rec = Records()
    out: dict[str, str] = {}
    add_curated(root, rec)
    extras = add_release(root, rec)
    rows = load_evidence(root, rec, out)
    headline = presentation(root).get("headline", {})
    add_comparison(rows, headline, rec, extras)
    facts = add_facts(root, rows, headline, rec, extras)
    hero = hero_rows(rows)
    nemotron_note = nemotron_timing_note(rows)
    watts = add_money(rows, out, rec, facts["comparison"], hero, nemotron_note)
    out["money_readme.md"] = readme_block(rows, facts, watts, extras, focused=bool(hero), nemotron_note=nemotron_note,
                                          subtitle=rec.rows["ColdSubtitle"]["value"] if "ColdSubtitle" in rec.rows else "")
    out["table_measured.tex"] = measured_table(rows, extras)
    out["table_hardware.tex"] = hardware_table(rows)
    out["metrics.tex"] = metrics_tex(rec)
    out["SOURCES.md"] = sources_md(rec)
    out["records.jsonl"] = records_jsonl(rec)
    return dict(sorted(out.items())), rec.missing()
