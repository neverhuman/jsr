# H42: n-gram language model for the streaming transducer ("+LM" rows)

**Status:** PRE-REGISTERED on 2026-11-22 at about 20:40Z, before any scored run of this wave. At that time the only numbers that existed for the acoustic model were the trainer's dev-clean400 greedy screens (4.95% EMA). No N-best list, no rescoring and no fusion decode had been run. Results are appended below the line "Results" later; amendments are dated and written before the runs they affect.

Authority: owner ruling **O17** (`docs/ultra/RULINGS.md`): the LibriSpeech language-model text (OpenSLR 11) may be used for a decoding language model; results that use it are reported in separate "+LM" rows. Every number of this wave is **exploratory**.

## Question
How much does an n-gram language model lower the word error rate of our streaming transducer on the development sets, (1) as N-best rescoring with the official word 4-gram and (2) as first-pass shallow fusion with a token-level n-gram that the streaming engine could run (a table lookup per token, no neural LM)?

## Hypothesis and expected size (written before looking)
- The transducer saw only the 960 h transcripts (9.4 M words); the LM text has about 800 M words. An n-gram LM should lower WER, more on dev-other than on dev-clean in absolute terms.
- **Expected, pooled over dev-clean + dev-other (baseline about 9%):**
  - word 4-gram rescoring of the beam-8 list: **4–8% relative** (about −0.45 pp);
  - token n-gram fusion without LODR: **3–7% relative** (about −0.4 pp);
  - token n-gram fusion with LODR: **5–10% relative** (about −0.6 pp).
- Expected N-best oracle (beam 8): 25–35% below the 1-best WER. Rescoring cannot do better than the oracle.
- A null or negative result is reported as such, with the numbers.

## Fixed setup
- **Acoustic model:** H22ctl step 79,800 (`ckpt-00079800.pt`, sha256 `097331eb7d41f34ba6cd1c7bdadef96f3518ab58cb17bbcf997bb3e4f936713c`), **EMA weights**. Dense S Conformer transducer, pruned RNN-T + CTC head, stateless predictor (context 2), 500-piece BPE of the R0 Zipformer (`bpe.model`, sha256 `c53433de083c4a6ad12d034550ef22de68cec62c4f58932a7b6b8b2f1e743fa5`).
- **Decode:** trained attention chunk 4 (160 ms), trained left context, `train.beam.modified_beam_search` (frame-synchronous, at most one non-blank token per frame, equal token sequences merged by log-sum-exp), no length normalisation. Encoder outputs are computed once with batch 8 in length order (the `train.evaluate` defaults) and reused by every decode of this wave, so all arms see identical acoustics.
- **Data:** full dev-clean (2,703 utterances) and dev-other (2,864). The test sets are sealed and never read. Dev transcripts are used for scoring and for the cross-validated choice of decode weights, nothing else.
- **LM data:** OpenSLR 11 only: `4-gram.arpa.gz` (official word 4-gram), the two pruned word 3-grams, `librispeech-lm-norm.txt.gz` (text for the token LM). The LODR low-order LM is trained on the 960 h **training transcripts only**. Downloads are sha256-locked in `docs/ultra/CANDIDATES.lock`.
- **Metric:** WER from `ultra.wer.corpus_wer` (lower case, punctuation stripped; (S+D+I)/reference words), for dev-clean, dev-other, and pooled (errors and reference words summed over both sets).

## Arms
- **Baseline (no LM):** beam 4, blank penalty 0 (the frozen DEV decode point of H12). Greedy and beam 8 without LM are reported for reference.
- **Control (no LM, tuned blank penalty):** beam 4 with the blank penalty chosen by the same cross-validation as the fusion arms, so that a gain from the penalty alone is not credited to the LM.
- **Phase 1, N-best rescoring:** final beam of beam 8 (up to 8 distinct token sequences with their model scores). New score = model score + w · ln P_LM(words, with `<s>` and `</s>`) + b · (number of words). LM = official word 4-gram through KenLM. Grid: w ∈ {0, 0.05, …, 1.0}, b ∈ {−2, −1.5, …, 4}. Secondary: the same with the pruned 3-grams (smaller files). The **oracle WER** of the list (reference-chosen best candidate) is reported as the ceiling.
- **Phase 2, first-pass shallow fusion (streaming-compatible):** a token-level n-gram over the same 500 pieces, estimated with KenLM `lmplz` (modified Kneser-Ney) on the tokenised LM text. For a non-blank token k that extends hypothesis h: score += λ1 · ln P_tok(k | last n−1 tokens of h) − λ2 · ln P_bi(k | last token of h); blank gets no LM term. The LM starts in the `<s>` state; **no end-of-sentence term** is added (a streaming decoder does not know where the utterance ends). P_bi is a token bigram from the 960 h training transcripts (LODR). Same beam (4) as the baseline.
  - Fusion without LODR: λ1 ∈ {0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5} × blank penalty ∈ {0, 0.25, 0.5, 1.0}.
  - Fusion with LODR: λ1 ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6} × λ2/λ1 ∈ {0.25, 0.5, 0.75} × blank penalty ∈ {0, 0.25, 0.5}.
  - **Primary token LM: 5-gram.** Secondary arms on the same text, if time allows: a smaller model (3-gram or a heavily pruned 5-gram, a few tens of MB) and a longer one (order 6–8), to show size against WER for the engine decision. The exact pruning thresholds are chosen from the n-gram counts and file sizes only (never from WER) and are recorded here before the first fusion decode.
- **Grid rule:** if a selected value sits on the edge of its grid, the grid is extended by further steps in that direction and the new points enter the same selection. No other points are added.

## Tuning protocol (both phases)
- **Two folds over SPEAKERS.** Within each set the speaker ids are sorted numerically; speakers at even positions (0, 2, …) form fold A, the others fold B. An utterance belongs to its speaker's fold.
- **Primary (one setting for the product):** the weights are chosen on fold A of dev-clean + dev-other together (fewest total word errors; ties go to the smallest LM weight, then the smallest absolute bonus or penalty) and applied to fold B; and the reverse. The reported number is the **held-out** WER: fold B scored with A's weights plus fold A scored with B's weights, given for dev-clean, dev-other and pooled, next to the baseline on the same utterances.
- Any number whose weights were tuned on the utterances it is scored on is labelled **"tuned on itself"** and is never the headline.

## Statistics and adoption rule (fixed now)
- **Paired bootstrap** over utterances, 10,000 resamples, seed 20261122, resampling within each set (the pooled statistic resamples dev-clean and dev-other separately and sums). Statistic: WER(arm) − WER(baseline) on the resampled utterances. 95% interval = 2.5th to 97.5th percentile. A speaker-level bootstrap (resampling whole speakers) is reported as a robustness check.
- **A "+LM" row is reported as a gain if** the pooled full-dev held-out WER is lower than the baseline's **and** the paired 95% interval of the difference excludes zero. Otherwise the row is recorded as "no measurable gain" with its numbers.
- **Decode cost** of the Python beam with and without fusion is reported for information only (engine speed is measured elsewhere).

## Amendment 1 (2026-11-22, about 21:18Z): written after Phase 1, before any fusion or blank-penalty decode
What existed when this was written: the Phase 1 results below (beam 4 and beam 8 without LM, 4-gram rescoring). No decode with an LM inside the beam and no decode with a blank penalty had been run.
1. **Sweep implementation.** One reference beam-4 pass over both dev sets costs 15 CPU-minutes in `train.beam` (1 ms per frame of Python and torch call overhead); the pre-registered grids are 85 passes per LM. The grid passes therefore use `train.lm_fusion.FastBeam`, a NumPy re-implementation of the same search (same pruning, same merging, float32 running scores). Measured before use, without LM: **identical token output on 5,567 / 5,567 utterances** at beam 4, and beam-8 N-best lists identical in content and order on 5,567 / 5,567; largest score difference 1e-4; 4.9× faster at beam 4. **Every reported held-out row is decoded again with `train.beam.modified_beam_search` + the fusion hook** at the chosen weights, so reported numbers come from the reference search; if a re-decoded row differs from its sweep value, both are shown.
2. **One token per frame.** `modified_beam_search` (and `FastBeam`) emit at most one token per frame. The lead measured the cap at 0.07 pp on greedy (21:06Z). The baseline and every LM row carry it equally. If an uncapped beam lands in time, the chosen configs are decoded with it as an extra labelled row, without re-tuning.
3. **Second scorer for the final table.** The pre-registered metric stays `ultra.wer`. The final rows are also scored with the pinned `ultra-score` binary that eval-lane uses (54,691 / 51,363 reference words), labelled as such, so the "+LM" rows can sit next to eval-lane's table.
4. **Decode cost.** Reported as the sum of the per-utterance wall time of the search call, one thread per worker, 14 workers in parallel on H0 cores 16-29, with the reference beam, for the baseline and for the chosen fusion configs. Informational only.
5. **Token LM pruning** is fixed in Amendment 2, from n-gram counts and file sizes only, before the first fusion decode.

## Amendment 2 (2026-11-22, 21:24Z): wider Phase 2 at the lead's direction (21:21Z), written before any fusion decode
What existed when this was written: Phase 1 (rescoring of the beam-8 list with the word 4-gram and the two pruned word 3-grams) and the no-LM blank-penalty control. **No decode with an LM inside the beam had been run.** The lead asked for the size/WER trade over n-gram order 3–5 and pruning (one model under 50 MB, one under 200 MB), beam 4 and 8, and a larger rescoring list.

1. **Token LM set, fixed from n-gram counts and KenLM's own size estimates only.** Counts from `lmplz` on the tokenised LM text (1,577,847,566 tokens, 40,418,261 lines): distinct n-grams of order 1–5 = 500 / 194,853 / 14,346,832 / 127,835,812 / 379,564,581; seen at least twice: 10.05 M 3-grams, 59.1 M 4-grams, 114.4 M 5-grams; at least 3 times: 40.3 M 4-grams; at least 4 times: 46.8 M 5-grams. **Size** means the KenLM trie with 8-bit quantised probabilities and backoffs and compressed pointers (`build_binary -a 22 -q 8 -b 8 trie`); the fusion decodes query that same file, so every WER belongs to the stated size. Classes: S ≤ 50 MB, M ≤ 200 MB, L larger (reference only).

   | name | order | `lmplz --prune` (counts ≤ value dropped) | size expected from the counts |
   |---|---:|---|---:|
   | S3 | 3 | none | ~32 MB |
   | S4 | 4 | 0 0 4 15 | ~42 MB |
   | S5 | 5 | 0 0 6 30 60 | ~40 MB |
   | M4 | 4 | 0 0 0 1 | ~182 MB |
   | **M5 (primary)** | 5 | 0 0 1 4 9 | ~162 MB |
   | L5a | 5 | 0 0 1 2 3 | 290 MB (KenLM estimate) |
   | L5b | 5 | 0 0 0 1 1 | 523 MB (KenLM estimate) |

   If a build misses its class limit, its thresholds are raised once from the counts and it is rebuilt. No threshold is chosen from WER. `lmplz` only prunes by count; a smarter (entropy) pruning might do better at the same size and is not tried here.
2. **Grids.** M5 gets the full pre-registered grids at beam 4. Every other LM gets a reduced grid at beam 4: fusion λ1 ∈ {0.1, 0.2, 0.3, 0.4, 0.5} × blank penalty ∈ {0, 0.25}; fusion + LODR λ1 ∈ {0.2, 0.3, 0.4, 0.5, 0.6}, λ2 = 0.5·λ1, blank penalty ∈ {0, 0.25}. **Beam 8:** M5 with the reduced grid, reported against the beam-4 baseline and against beam 8 without LM. The edge-extension rule applies to every grid. Folds, held-out scoring and bootstrap are unchanged; each (LM, beam, with or without LODR) is its own family with its own held-out choice of weights.
3. **Reading the size table.** Each LM's row is its own held-out number. Choosing the best LM from that table would be a choice made on the scored utterances, so the table is reported as a trade-off and **the headline "+LM" rows stay M5**, fixed here before any fusion decode.
4. **Larger rescoring lists (post hoc).** Beam-16 and beam-32 lists rescored with the word 4-gram, same grid and folds. This was added after seeing that the beam-8 oracle (6.51 pooled) is far below what rescoring reached (8.38), and is labelled post hoc.

## Results
All numbers are **exploratory**. Scorer `ultra.wer` (54,402 / 50,948 reference words) unless a row says otherwise. Every row carries the one-token-per-frame cap of the Python beam.

### Phase 1: N-best rescoring with the official word 4-gram (2026-11-22, 21:14Z)
- Encoder cache: `train.lm_decode encode`, EMA weights, 6 single-threaded workers taking every 6th `train.evaluate` batch. Its greedy hypotheses are **identical to eval-lane's EMA greedy dump on 5,567 / 5,567 utterances** (`/mnt/xbabe3-external/jsr/evals/eval-lane/full-dev/H22ctl-79800.ema.c4.ms8.*`).
- N-best: final beam of beam 8, mean 7.26 candidates per utterance. LM: `4-gram.arpa.gz` (sha256 `f2b2d150…`) as a KenLM trie.

| row | dev-clean | dev-other | pooled |
|---|---:|---:|---:|
| greedy, max 8 symbols per frame (eval-lane's row, same hypotheses) | 4.893 | 13.323 | 8.970 |
| beam 4, no LM (**baseline**) | 4.803 | 12.972 | 8.754 |
| beam 8, no LM (1-best of the list) | 4.777 | 12.988 | 8.748 |
| **beam-8 list + word 4-gram, held-out weights** | **4.652** | **12.362** | **8.381** |
| difference to baseline, paired bootstrap 95% (utterances) | −0.151 [−0.253, −0.049] | −0.610 [−0.734, −0.484] | **−0.373 [−0.452, −0.291]** |
| difference to baseline, paired bootstrap 95% (whole speakers) | [−0.266, −0.035] | [−0.784, −0.433] | [−0.478, −0.268] |
| relative change | −3.1% | −4.7% | −4.3% |
| weights tuned per set, held-out (secondary) | 4.673 | 12.368 | — |
| weights **tuned on itself** (not a result) | 4.638 | 12.364 | 8.374 |
| N-best oracle (ceiling for rescoring this list) | 3.000 | 10.267 | 6.514 |

- **Folds:** dev-clean A 20 speakers / 1,257 utterances / 26,490 words, B 20 / 1,446 / 27,912; dev-other A 17 / 1,487 / 26,969, B 16 / 1,377 / 23,979.
- **Chosen weights:** fold A chose w = 0.30, b = 1.0 (scored on fold B); fold B chose w = 0.35, b = 1.5 (scored on fold A). Tuned on all utterances: w = 0.35, b = 1.5.
- **Verdict by the pre-registered rule:** pooled held-out WER is lower and the paired interval excludes zero (utterance and speaker level): **the word 4-gram rescoring row is a gain**, 4.3% relative, inside the 4–8% expected. The oracle is 25.5% below the list's 1-best (expected 25–35%); rescoring recovers about one sixth of that.
- Error counts (S / D / I): baseline 7,466 / 862 / 894; rescored 7,112 / 852 / 865. The gain is almost entirely fewer substitutions.
