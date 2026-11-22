# H18: IEEE paper draft (docs/paper)

**Status:** draft. Every number is EXPLORATORY until `bench-freeze-v1`.

## Question
Can the project's evidence be written up as a two-column IEEE paper without overclaiming, with every number traceable to a file and regenerable?

## Setup
- **Coordination:** `~/jsr/agent_paper.md` (gitignored, append-only).
- **Claude paper lanes** (worktree `lane/paper`):
  - paper-data: `python/ultra/tools/paper_data.py` → `docs/paper/data/{metrics.tex,records.jsonl,SOURCES.md}` plus pgfplots figures;
  - paper-related: `refs.bib`, with 50 entries each verified against arXiv, Crossref, PMLR or the model card;
  - paper-writer: `paper.tex` and the sections.
- **Codex paper team:** matched cold-boundary comparator measurements on H0 core 46, plus manuscript critique and release checks.
- **Reviews:** runtime-lane's auditor, grok-lane's page-1 critique and the Codex manuscript review.
- **Build:** `just paper` runs pdflatex and bibtex. It fails on undefined references, overfull boxes, non-Type-1 fonts, or "Fig. 1." missing from page 1.

## Rules enforced
- R1 accuracy comes from DEV only; LibriSpeech test stays sealed.
- Cost and density are printed beside the same checkpoint's same-harness WER.
- The abstract leads with the WER gap.
- No literal result numbers appear in the sections: a test rejects bare decimals, and every value is a macro with a source record (id, configuration, workload digest, source).
- Systems we did not run are reported only in a separate "reported" group.
- Fig. 1 is the cold-boundary comparison, shown as PENDING until the matched receipts exist. The warm rehearsal (H19) appears in Results with its own scope label.

## Results so far
- A 10-page draft: about 8 pages of body plus references.
- The abstract states 6.65% / 16.54% full-dev WER for our best model against 2.85% for Zipformer. It then gives warm per-CCX density of 144 (MR) and 96–112 (dense) against 32 for Zipformer, and 7,300/7,300 cross-ISA identical decodes.
- **Pairing errors the reviewers caught and fixed in the repo docs:**
  - the README paired 259 MB RSS with a run that measured 646 MB;
  - a cross-run "about 4×" INT8-vs-ORT ratio, now removed;
  - an M3 number from an older checkpoint;
  - a related-work row that paired step-60k WER with a step-5k cost.

## Decision / consequence
- The paper is pushed as a labelled draft.
- After the xbabe2 window and the sealed test read, the qualified numbers replace the exploratory ones by regenerating `metrics.tex`. No text edits are needed.

## Pointers
`docs/paper/` (`paper.tex`, `paper.pdf`, `data/`, `figures/`, `sections/`), `python/ultra/tools/paper_data.py`, H19 for the warm rehearsal data.
