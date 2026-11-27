# H302: rare-word candidate repair (stopped at the headroom gate)

Watcher request (08:48): on gated low-margin utterances, build a TRAIN-only hyp→ref confusion lexicon for rare words (train-960 count < 100) and add swap variants scored by a forced acoustic score plus H261/t2. Apply on the H298 H255@2000 B48 train30 lists.

## Stage 0 diagnostics (DEV gated rows used only to measure the ceiling; nothing tuned)
`tools/coverage.py`, H298 H255@2000 B48 mixed list, 520 gated rows:
- 303 rare-word substitutions in the top-1. 110 have the ref word somewhere in the 48-best; 193 are absent from every candidate.
- TRAIN lexicon (H276/TT H255@2000 + H286/TT F3, all 16-best hypotheses aligned to refs): 18,904 pairs, 8,089 rare targets.
- Of the 193 absent misses: 4 have the exact (top-1 word → ref) pair in the lexicon, 18 have the ref word as any target, and 100 have the ref word in train-960 at all.

`tools/coverage_spell.py`: TRAIN-free spelling neighbours from the train-960 rare vocabulary (82,814 words).

| neighbour set | absent misses reached | variants per suspect word |
|---|---|---|
| char edit distance ≤ 1 | 18 / 193 | 6.1 |
| char edit distance ≤ 2 | 36 / 193 | 94.6 |

## Verdict
Stopped before any decode or fit.
- With a perfect scorer, the TRAIN lexicon reaches about 4 edits (≈ −0.008 pp). Spelling neighbours reach at most 18–36 edits (−0.035 to −0.07 pp).
- Spelling neighbours also need a forced RNN-T pass over 6–95 variants per suspect word, far beyond the +5% process budget.
- Rare words in LibriSpeech DEV are book-specific names, and MLS TRAIN confusions do not transfer. Recovery needs session- or book-level context (name biasing from earlier committed text, as in the H77 name-cache), not post-hoc swaps.
