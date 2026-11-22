# ULTRA paper (IEEE two-column, double-blind)

`paper.tex` is the main file; `paper.pdf` is committed on purpose. **Every result in the current
draft is exploratory.** Qualified results replace them after `bench-freeze-v1` by regenerating
`data/metrics.tex`; the prose does not change.

The [six-page IEEE checkpoint edition](jsr-cpu-asr.pdf) gives the
preliminary DEV results for prepared INT8 D2MR at step 50,000. The portable
build runs with `make -C docs/paper/checkpoint check` and writes the PDF to
`target/paper-checkpoint/`. Its source, pinned evidence, bibliography and
figure generator are in [checkpoint/](checkpoint/). The matched cold plot
populates from completed three-repetition measurements. The paced rehearsal
has its own measured figure and table.

## Build

```bash
just paper     # from the repository root
```

The recipe:
1. regenerates `data/metrics.tex` and `figures/*.tex` with `python -m tools.paper_data` (in
   `python/ultra`), when that tool exists;
2. runs `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` in `docs/paper` with `SOURCE_DATE_EPOCH`
   set to the last commit time;
3. fails on undefined references or citations, any overfull hbox, a non-Type-1 font, or a missing
   `Fig. 1.` on page 1.

Needs TeX Live with IEEEtran, pgfplots, cleveref, microtype, booktabs, draftwatermark and
poppler-utils (`pdffonts`, `pdftotext`, `pdfinfo`). siunitx is not used.

## Layout

| Path | Owner | What |
|---|---|---|
| `paper.tex` | paper-writer | preamble, title, section order, protocol constants |
| `sections/*.tex` | paper-writer (except `related.tex`: paper-related) | prose, tables, Fig. 1 float |
| `sections/fallbacks.tex` | paper-writer | `\providecommand` fallback for every metric macro the sections use |
| `data/metrics.tex`, `data/SOURCES.md` | paper-data (generated) | one macro per number, with provenance |
| `figures/*.tex` | paper-data | pgfplots bodies: `money.tex` (Fig. 1: speed on one core against WER, every system measured here), `frontier_warm.tex` (streams per core), `density.tex` (warm ladder) |
| `data/evidence/*.jsonl` | the agent that measured the row | measured comparison rows (schema: `data/README.md`) |
| `refs.bib` | paper-related | bibliography |

## Rules

- **No literal result numbers in `sections/`.** Use a macro from `data/metrics.tex`. A test rejects
  bare decimals in sections; design constants (12.5 Hz, gate margins) live in `paper.tex`.
- A macro that `metrics.tex` does not define yet prints a red **??** (`\PH`). After adding a macro
  to a section, add it to `sections/fallbacks.tex` (the list is every `\CapitalisedMacro` used in
  `sections/*.tex` except LaTeX's own).
- Every speed number is printed beside the WER **of the same checkpoint**. Warm (in-process,
  single harness) and cold (process tree, model load included) cost boundaries are never mixed.
- The `PRELIMINARY` watermark and the red banner stay until `bench-freeze-v1`.
