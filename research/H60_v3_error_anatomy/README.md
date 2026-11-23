# H60: where v3.0.0 makes its errors (S / D / I anatomy)

**Status: EXPLORATORY, LibriSpeech DEV only.** claude-lead ran the analysis at the owner's request (2026-11-23, about 20:11Z; chat post 20:13:30Z), and paper-lane landed it here.
- It reads full dev-clean + dev-other: 5,567 utterances, 106,054 reference words.
- No test-set number appears. The test-set disclosure in [`docs/ultra/FREEZE.md`](../../docs/ultra/FREEZE.md) §4 applies.
- No selection was made on these numbers.

## Question
Where does v3.0.0 make its errors? The breakdown covers:
- error type (substitution / deletion / insertion);
- position in the sentence;
- utterance length;
- confusion class;
- utterance and speaker.

And which fixes should the v3.1 program aim at?

## Setup
- **System:** v3.0.0 is RC1 EMA INT8 (`weights.int8` edac19d8…) plus the release decoder (`ultra-engine-commit` 9a142e8e…). See the [release record](../../docs/ultra/releases/v3.0.0.md).
  - **Final:** the committed text of the commit pass. It is utterance-segmented: the LibriSpeech segment boundaries serve as endpoints.
  - **Live:** the provisional greedy chunk-4 text, which needs no endpoint.
- **Inputs:** eval-lane's RC1 release-row score inputs, `/mnt/xbabe3-external/jsr/evals/eval-lane/rc/RC1/release-decoder/release-decoder.{committed,provisional}.score-input.jsonl` (sha256 e293e4fd… / 434803dd…).
- **Method 1, the official counts:** `ultra-score` (sha256 d784f8ca…) with the canonical normaliser (`ultra-en-wer-norm/v1`, rule table 988c59ac…).
  - Reports: [`results/official_committed.json`](results/official_committed.json) and [`results/official_provisional.json`](results/official_provisional.json).
  - paper-lane re-scored both inputs with the pinned scorer, and the per-split counts are identical.
- **Method 2, the anatomy:** a raw-text alignment by `sdi_analysis.py`, which stays on the data drive (see [MANIFEST](MANIFEST)). It produces the position, length, confusion and concentration tables below.
- **Caveat:** method 2 aligns raw whitespace tokens, without the scorer's normaliser.
  - It counts 5,826 edits over 105,350 words, against the official 5,808 over 106,054.
  - Its S / D / I shares agree with the official counts (80 / 12 / 9% against 78.7 / 11.8 / 9.6% for the final text).
  - Read its tables as shapes and ratios. The official figures are the ones in the first table.

## Results
### Official S / D / I (method 1)
| text | split | S | D | I | edits | reference words | WER |
|---|---|---:|---:|---:|---:|---:|---:|
| final | dev-clean | 1,278 | 177 | 143 | 1,598 | 54,691 | 2.922% |
| final | dev-other | 3,290 | 506 | 414 | 4,210 | 51,363 | 8.197% |
| **final** | **pooled** | **4,568 (78.7%)** | **683 (11.8%)** | **557 (9.6%)** | **5,808** | 106,054 | **5.476%** |
| live | dev-clean | 1,730 | 221 | 197 | 2,148 | 54,691 | 3.928% |
| live | dev-other | 4,621 | 609 | 520 | 5,750 | 51,363 | 11.195% |
| **live** | **pooled** | **6,351 (80.4%)** | **830 (10.5%)** | **717 (9.1%)** | **7,898** | 106,054 | **7.447%** |

**What the commit pass removes** (official net counts, live → final):

| | S | D | I | all edits |
|---|---:|---:|---:|---:|
| live → final | 6,351 → 4,568 | 830 → 683 | 717 → 557 | 7,898 → 5,808 |
| removed, net | **28%** | **18%** | **22%** | 26% |

Method 2 splits the net change into about 2,578 edits fixed and 521 introduced.

### Where in the sentence (method 2)
Errors per 100 reference words. An insertion counts at the next word.

| position | live | final | share the final pass fixes |
|---|---:|---:|---:|
| **first word** | 10.69 | **8.06** | 25% |
| words 2-5 | 8.55 | 6.28 | 27% |
| words 6-15 | 7.14 | 5.45 | 24% |
| words 16+ (not the last 3) | 6.38 | **4.64** | 27% |
| last 3 words | 7.76 | 5.47 | 29% |
| last word | 8.95 | 6.25 | 30% |

| relative position (decile) | 0-10% | 10-20% | 20-30% | 30-40% | 40-50% | 50-60% | 60-70% | 70-80% | 80-90% | 90-100% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| live | 9.50 | 7.97 | 7.03 | 7.46 | 7.40 | 7.20 | 6.83 | 6.67 | 6.99 | 7.32 |
| final | 7.00 | 5.69 | 5.58 | 5.74 | 5.59 | 5.41 | 5.08 | 4.85 | 4.93 | 5.00 |

### By utterance length (method 2)
| length | utterances | live WER | final WER |
|---|---:|---:|---:|
| 1-9 words | 1,320 | 9.97% | **7.69%** |
| 10-19 words | 2,110 | 8.37% | 6.15% |
| 20-34 words | 1,520 | 7.07% | 5.15% |
| 35+ words | 617 | 6.35% | **4.73%** |

### Confusion classes (final text, method 2)
- **Substitutions are about 79% of all errors.** About 73% of them hit content words: 3,389 content against 1,248 function-word substitutions.
  - **Names:** ALEKSANDROVNA → ALEXANDROVNA 18, BOZZLE → BOSWELL 10, SERAFIMA → FEMALE 10, LELECHKA → LACHKA 10, EDISON → ADDISON 6.
  - **Morphology:** MAN → MEN 11, IS → WAS 8.
  - **Spelling variants:** O → OH 13, HONOR → HONOUR 6.
- **The largest single confusions are reduced function words that only syntax can decide:** IN ↔ AND 123 (64 + 59), THE ↔ A 114 (62 + 52), AN ↔ AND 28 (17 + 11).
- **Deletions and insertions are mostly short function words.**
  - Deletions: A 49, THE 38, IN 22, TO 19 (338 content / 332 function).
  - Insertions: THE 53, A 42, TO 16, AND 16 (248 content / 271 function).
- **Word splits and merges:** 839 adjacent S + D or S + I pairs look like splits or merges, such as ANY ONE against ANYONE.

### Concentration (final text, method 2)
- 3,060 of 5,567 utterances (55.0%) have no error.
- **The worst 10% of utterances hold 48.8% of all edits.**
- The worst speakers sit at 15-24% WER: 4572 23.7%, 1650 23.1%, 1585 17.7%, 6467 15.7%, 7697 15.3%, 6455 15.1%.

## What surprised us
- **Left context is worth about 40% relative.**
  - The first word errs at 8.1 per 100 words, against 4.6 from word 16 on.
  - Utterances of 1-9 words reach 7.7% WER, against 4.7% for 35+ words.
  - The rate falls as the sentence grows: 7.0 in the first decile, 4.85 at 70-80%.
  - The start of each utterance and short utterances are where most of the loss is.
- **The commit pass fixes about a quarter of the errors at every position** (24-30%). It does not target the start of the utterance. That is expected, because it re-decodes the same closed segment with no context from before it.
- **The last word bumps back up** to 6.3 per 100, against 4.6 in the middle. That is consistent with the pseudo-label tail truncation that v3.0.0 was trained with (see the release record's Limits and H52 / H58). It is to be checked again on the relabelled arm.

## Decision / consequence: fixes ranked by where the errors are
All screens run on the selection half S only. The confirmation half C and the test sets stay unused.
1. **Cross-utterance (session) context** for the first words and short utterances. Feed the previously committed *hypothesis* (never the reference) into the commit pass's LM or neural-rescorer context. This changes only the decoder. Report it as a labelled "session mode" row, apart from the per-utterance benchmark. Owner: Codex H55.
2. **A bidirectional neural rescorer for the function-word confusions** (L2R + R2L, −0.40 pp on S; runtime-lane). Stack it with the merged CTC rescoring + word-MBR (#115).
3. **A session cache / biasing LM for repeated names:** a name decoded well once gets a boost later in the session. It is cheap, so fold it into (1).
4. **A growing-window live re-pass:** re-decode the open utterance with beam + LM every 1-2 s, so the live text improves as the sentence grows instead of only at the endpoint. Target: live 7.45% → toward 5.5%, at a CPU cost. Owner: runtime-lane-wave1d.
5. **Hard speakers and robustness:** the PL2 relabelling and Libri-Light tier A (the S4a fork; H58).

## Pointers
- Receipts and source files with full sha256: [MANIFEST](MANIFEST).
- The analysed artifact: [`docs/ultra/releases/v3.0.0.md`](../../docs/ultra/releases/v3.0.0.md) and [H54](../H54_v3_release_qualification/README.md) (rows 13-15).
- The same split of errors on the original baseline B0: [H43's WER error anatomy](../H43_v3_baseline_and_measurement/results/wer_error_analysis_A1-49000.md).
- Coordination: `AGENT_CHAT.md` (local), claude-lead 2026-11-23T20:13:30Z.
