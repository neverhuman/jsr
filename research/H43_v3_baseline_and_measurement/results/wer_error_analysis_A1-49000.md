# Where B0 (A1-49000 EMA) loses WER on LibriSpeech DEV

Everything here was computed read-only on 2026-11-23, from per-utterance outputs already on the data drive. No test-set file was read. Rows are for 160 ms chunks and greedy decoding unless stated otherwise. Scoring uses the frozen Rust normaliser and aligner (`ultra-en-wer-norm/v1`), re-implemented in Python. It reproduces every `score.json` count exactly: 4.480 / 12.373 / 8.302.

**Error categories come from heuristic rules and carry roughly ±2–3 points of uncertainty.** Homophones are found from the dev forced-alignment phone sequences. The positions of deletions and insertions depend on the aligner's tie-breaking rule; both bounds are given where it matters.

## 1. Error counts

| Run | dev-clean WER (S / D / I) | dev-other WER (S / D / I) | Pooled WER |
|---|---|---|---:|
| A1, 160 ms, greedy | 4.480 (1991 / 224 / 235) | 12.373 (5085 / 665 / 605) | 8.302 |
| A1, 320 ms | 4.290 (1918 / 212 / 216) | 11.870 (4882 / 635 / 580) | 7.961 |
| A1, 640 ms | 4.143 (1851 / 202 / 213) | 11.493 (4717 / 607 / 579) | 7.703 |
| A1, 160 ms, beam 4 | 4.487 (1998 / 219 / 237) | 12.201 (5044 / 589 / 634) | 8.223 |
| A1, INT8 engine | 4.461 (1985 / 225 / 230) | 12.385 (5057 / 669 / 607) | 8.272 |
| H22ctl, beam 8 + M5 + LODR (LM reference point) | 3.736 | 10.545 | 7.033 |

- **Mix:** substitutions are about 80% of errors in every run.
- **Where:** dev-other holds 72% of the substitutions.
- **Check:** the INT8 engine matches PyTorch to −0.030 pp [−0.064, +0.003].

## 2. Five error groups, plus end-of-stream truncation

| # | Group | Share of 8,805 errors | Most likely cause | Best lever |
|---|---|---:|---|---|
| 1 | Function words: 1,543 swapped, 436 deleted, 404 inserted (76% in dev-other) | 27.1% | Unstressed words, with only 2 tokens of predictor context | LM (−22 to −27% on H22ctl), KD, the commit pass |
| 2 | Content words with a very different sound: 1,791 substituted, 262 deleted, 307 inserted | 26.8% | Acoustic capacity and robustness; concentrated in dev-other and on hard speakers | More data (Libri-Light), KD, capacity |
| 3 | Close-sounding content words: 1,262 near-homophones, 495 morphology (-s, -ed, tense), 214 homophones | 22.4% | Choice from context, and weak word endings | LM (−37% of near-homophones), KD, longer chunks |
| 4 | Rare words and names: 1,125 substitutions | 13.3% | Unseen spellings ("the model hears the name but cannot spell it") | Word-level LM or biasing; partly unfixable |
| 5 | Writing conventions: 400 contractions / apostrophes, 317 splits / merges, 205 spelling variants | 10.5% | LibriSpeech conventions plus the frozen normaliser | Diagnostic only |
| X | End-of-stream truncation; overlaps the groups above | ~1.7% | Too little audio after the last word | Pad the end of the stream with ≥ 0.3 s; fix beam finalisation |

**Most frequent substitutions:** IN→AND 99, AND→IN 98, THE→A 74, A→THE 59, AND→AN 25, IS→AS 18, THIS→THE 17, AS→IS 17. The top 40 pairs cover only 10% of substitutions, and 91% of those are function-word swaps.

## 3. End-of-stream truncation: the cheapest fix

The last word is cut off 4× more often than a word in the middle of an utterance (PRIVILEGE → PRIVIL, ELEVEN → ELEVE, PREFECT → PREFEC). It depends strongly on how much silence follows the last word:

| Silence after the last word | Utterances | Last-word error at 160 ms | 320 ms | 640 ms |
|---|---:|---:|---:|---:|
| under 0.1 s | 346 | 23.4% (cut-off 9.5, deleted 3.8) | 18.2% | 15.9% |
| 0.1–0.2 s | 946 | 13.4% | 10.3% | 9.3% |
| 0.2–0.3 s | 1,277 | 8.5% | 8.0% | 7.6% |
| 0.3–0.5 s | 2,700 | 8.1% | 7.9% | 7.8% |
| 0.5–1.0 s | 298 | 5.7% | 6.0% | 5.7% |

- **Cost:** 23% of utterances have under 0.2 s of trailing silence, which costs ~105 errors (1.2% of all errors).
- **Beam 4 is worse:** 164 cut-off last words, against 98 for greedy.
- **Likely cause:**
  - the native engine's `pad_tail` adds 40 ms (`crates/ultra-stream/src/native_engine.rs`);
  - the sherpa path pads 0.8 s;
  - the Python evaluator adds none.
- **Fix (H44):** flush with ≥ 0.3 s of silence and fix beam finalisation.

**The first word is not an onset problem.** First-word error does not depend on leading silence, so the cause is missing left context. The LM cuts first-word errors by 19%.

## 4. Where the errors are concentrated

- **Speakers:**
  - dev-other WER by speaker ranges from 3.6 to 33.0 (median 9.9);
  - pooled, the worst 8 of 73 speakers produce 32.7% of errors from 11.5% of the words;
  - their error mix is the same as everyone's, only more frequent.
- **Utterances:**
  - 42% are error-free;
  - the worst 10% by WER hold 33% of the errors;
  - 137 utterances with WER ≥ 50% hold 8% of the errors.
- **Length:** WER falls with utterance length (dev-other: 17.8% at 1–4 words, 10.3% at 50 words or more). Utterances over 20 s are not worse, so the 5.12 s attention window is not a limit.
- **Word frequency:**
  - words seen fewer than 10 times in train are 2.2% of the words but 17% of substitution + deletion errors (never seen: 85.6% error);
  - words that are not a single token of the 500-piece vocabulary are 38% of the words and 61% of S+D errors.

## 5. Where competitors win (730-utterance panel)

| System | dev-clean | dev-other | Pooled |
|---|---:|---:|---:|
| A1 | 4.40 | 11.91 | 7.49 |
| Parakeet TDT v3 | 2.43 | 4.78 | 3.40 |
| Nemotron 560 ms | 2.61 | 4.93 | 3.56 |
| Zipformer (R0) | 3.03 | 8.43 | 5.25 |
| H22ctl + LM | 4.05 | 10.28 | 6.61 |

- **How many of our errors they get right:** Parakeet 70% of A1's panel errors; Zipformer 47.5%; Moonshine Medium 65%.
- **Their advantage is broad, not in one category:** sound, close-sound and function words alike.
- **An LM alone won't close it:** of the errors Parakeet fixes, the H22ctl + LM run fixes only about 22%. **Most of the gap is acoustic or capacity,** which points to more data, KD and a larger model.
- **The floor:** 20% of A1's panel errors (~1.35 pp, mostly rare words) are wrong in every system.
- **The gap grows with speaker difficulty:** 1.96 pp on dev-clean, 5.83 pp on dev-other, 16.95 pp on the worst 4 dev-other speakers.

## 6. What larger chunks, beam and LM fix

- **640 ms instead of 160 ms:** −7.2% of errors. By category:
  - last-word errors −15.6%;
  - function-word deletions −12.4%;
  - near-homophones −10.4%;
  - function-word swaps −9.9%;
  - rare words only −1.2%.

  So streaming costs context and the last word, which is the case for the commit pass.
- **Beam 4:** −1.0% overall. Deletions fall, but insertions and cut-off last words rise.
- **LM, on H22ctl** (beam 4 → beam 8 + M5 + LODR): −19.5% of errors.
  - Near-homophones −37%, function-word swaps −27%, very-different-sound −20.5%, rare words −15.5%.
  - Deletions rise 10%, so tune a word-insertion bonus.
  - **No A1 + LM decode exists yet;** it is the first H46 run.

## 7. Levers in expected order of payoff
1. LM fusion. The A1 run is pending (H46).
2. KD, more data and capacity. They address 70% of the errors Parakeet fixes (H49, H52, H53).
3. Longer chunks via the commit pass: −7.2% at 640 ms (H45).
4. End padding and beam finalisation: about −1.2% of errors (H44).
5. Biasing or a larger vocabulary for names.
6. Normalisation, diagnostic only.

## Sources
Per-utterance outputs, competitor scores, word frequencies, alignments and the LM vocabulary:
- `/mnt/xbabe3-external/jsr/evals/eval-lane/curve/A1-49000.ema.c{4,8,16}.ms8.*.jsonl`, `.../A1-49000.ema.c4.ms8.b4.beam.*`
- `/mnt/xbabe3-external/jsr/evals/eval-lane/full-dev/H22ctl-79800.*`
- `/mnt/xbabe3-external/jsr/lm/work/results/hyps/*`
- `target/paper-evidence/cold/round-1/*/scores.jsonl`
- `/mnt/xbabe3-external/jsr/tokenizers/librispeech_train_text.txt`
- `/mnt/xbabe3-external/jsr/corpora/librispeech/mfa_alignments_zenodo2619474/`

Independent cross-checks: runtime-lane's forensic attribution (81% / 9% / 9% S / D / I) and codex-eval's reproduction of the INT8 counts.
