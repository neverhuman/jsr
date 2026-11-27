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
    cold_dev.csv, money_frontier.csv, money_layout.tex, money_readme.md, table_measured.tex, table_versus.tex
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
from .macros import metrics_tex, records_jsonl, sources_md, used_macros
from .money import add_money, hero_rows, hero_sentences, nemotron_timing_note
from .words import md, tex
from .publish import publish_figure, readme_block, v4_block
from .records import Records, add_curated
from .release import add_release
from .reported import add_reported
from .tables import measured_table
from .versus import _paired, add_versus
from .training import add_training
from .candidate import add_candidate
from .look41 import add_look41
from .look42 import add_final_pass, add_look42
from .v42_rows import add_cost42


def build(root: Path = REPO) -> tuple[dict[str, str], list[str]]:
    """Return ({relative file name: content}, missing record ids)."""
    verify_manifest(root)
    rec = Records()
    out: dict[str, str] = {}
    add_curated(root, rec)
    extras = add_release(root, rec)
    add_candidate(root, rec, out)
    add_look41(root, rec)
    add_look42(root, rec)
    add_final_pass(root, rec)
    add_cost42(root, rec)
    rows = load_evidence(root, rec, out)
    headline = presentation(root).get("headline", {})
    add_comparison(rows, headline, rec, extras)
    facts = add_facts(root, rows, headline, rec, extras)
    hero = hero_rows(rows)
    if hero:  # Figure 1 headlines the v4.2.0 final pass and the unchanged v4.1.0 live point
        sentences = hero_sentences(hero, rec)
        rec.rows["MoneyHeadline"].update(value=tex(sentences["headline"]), configuration="Figure 1: what to see (v4.2.0 final and v4.1.0 live measurements)",
                                         source="; ".join(f"docs/paper/data/evidence/{r['_where']} id={r['id']}" for r in hero.values()))
        facts["figure"] = md(sentences["headline"])
        rec.add("HeroSubtitle", tex(sentences["subtitle"]).replace("\u00b7", "$\\cdot$"), unit="sentence", label="exploratory",
                configuration="README summary line of the v4.1.0 foreground", workload=hero["Zip"]["workload"],
                workload_digest=hero["Zip"]["workload_digest"], source=rec.rows["MoneyHeadline"]["source"])
    nemotron_note = nemotron_timing_note(rows)
    watts = add_money(rows, out, rec, facts["comparison"], hero, nemotron_note)
    pres = presentation(root)
    versus = add_versus(root, rows, pres.get("versus", {}), rec)
    prior = {q["mode"]: _paired(root, q["paired"]) for q in pres.get("versus", {}).get("prior", [])}  # v3.0.0's own record
    out["table_versus.tex"] = versus["table"]
    add_training(root, rec, out)
    out["money_readme.md"] = readme_block(rows, facts, watts, extras, focused=bool(hero), nemotron_note=nemotron_note,
                                          subtitle=(rec.rows["HeroSubtitle"]["value"] if hero else rec.rows["ColdSubtitle"]["value"] if "ColdSubtitle" in rec.rows else ""),
                                          versus=versus, rec_rows=rec.rows, release=pres.get("release", "v3.0.0"), hero=hero, prior=prior)
    out["v4_readme.md"] = v4_block(rec.rows)
    out["table_measured.tex"] = measured_table(rows, extras)
    paper = root / "docs" / "paper"
    used = None  # a bare data tree (tests) has no paper sources: emit every macro
    if (paper / "paper.tex").is_file():
        texts = [(paper / "paper.tex").read_text(encoding="utf-8")]
        texts += [p.read_text(encoding="utf-8") for p in sorted((paper / "sections").glob("*.tex")) if p.name != "fallbacks.tex"]
        texts += [p.read_text(encoding="utf-8") for p in sorted((paper / "figures").glob("*.tex"))]
        # the CEO/CTO briefs \input metrics.tex too, so their macros must survive the pruning
        texts += [p.read_text(encoding="utf-8") for p in sorted((paper.parent / "briefs" / "src").glob("*.tex"))]
        texts += [v for k, v in out.items() if k.endswith(".tex")]
        used = used_macros(rec, texts)
    out["metrics.tex"] = metrics_tex(rec, used)
    out["SOURCES.md"] = sources_md(rec, used)
    out["records.jsonl"] = records_jsonl(rec)
    return dict(sorted(out.items())), rec.missing()
