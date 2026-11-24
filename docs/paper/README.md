# Isosyll paper (IEEE two-column, double-blind)

`paper.tex` is the main file; `paper.pdf` is committed on purpose. All numbers are LibriSpeech DEV
measurements generated deterministically from SHA-256-pinned receipts in `docs/paper/data/`.

## Build

```bash
just paper          # from the repository root
just paper-refresh  # the same, plus docs/paper/money.png, the README block and the paper tests
```

The recipe:
1. regenerates `data/metrics.tex`, `data/cold_dev.csv`, `data/money_layout.tex`, and `data/table_*.tex`
   with `python -m tools.paper_data` (in `python/ultra`);
2. runs `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` in `docs/paper` with `SOURCE_DATE_EPOCH`
   set to the last commit time;
3. fails on undefined references or citations, any overfull hbox, a non-Type-1 font, or a missing
   `Fig. 1.` on page 1.

Needs TeX Live with IEEEtran, pgfplots, cleveref, microtype, booktabs and poppler-utils
(`pdffonts`, `pdftotext`, `pdfinfo`). `siunitx` is not used.

## Layout

| Path | What |
|---|---|
| `paper.tex` | preamble, title, section order, design constants |
| `sections/*.tex` | prose, tables, Fig. 1 float |
| `sections/fallbacks.tex` | `\providecommand` fallback for every metric macro used in `sections/*.tex` |
| `data/metrics.tex`, `data/SOURCES.md` | one generated macro per number, with provenance |
| `figures/money.tex`, `figures/money_standalone.tex` | Fig. 1 pgfplots body (single-core throughput vs pooled WER) |
| `data/evidence/*.jsonl` | measured comparison rows (schema v2) |
| `data/systems.json` | plotted systems (our single release artifact in live + final modes + top 4 comparators) |
| `refs.bib` | bibliography |

## Rules

- **Measured and reported never mix.** Published third-party numbers (`data/reported.json`) never enter
  any measured CSV, figure, or table.
- **No literal result numbers in `sections/`.** Every result number comes from `data/metrics.tex`.
- **Single release artifact.** Only our latest release model (live and utterance-segmented final modes
  on the same INT8 weights) is plotted and tabulated; historical checkpoints stay in `research/` and
  release records.
