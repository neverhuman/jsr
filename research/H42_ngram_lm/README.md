# H42: n-gram language model for the streaming transducer ("+LM" rows)

**Status:** RESULTS (exploratory); PRE-REGISTERED on 2026-11-22, committed at 20:34:04Z (commit 2c48b91; after the rebase onto main it is 4f60479, same author date), before any scored run of this wave. At that time the only numbers that existed for the acoustic model were the trainer's dev-clean400 greedy screens (4.95% EMA). No N-best list, no rescoring and no fusion decode had been run. Results are appended below the line "Results" later; amendments are dated and written before the runs they affect.

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
All numbers are **exploratory**.

**Summary** (H22ctl-79800 EMA, chunk 160 ms; full dev-clean / dev-other / pooled; weights held out by 2-fold speaker cross-validation; paired bootstrap 95% on the pooled difference; every row carries the Python beam's one-token-per-frame cap):

| | LM size | dev-clean | dev-other | pooled | Δ pooled [95%] |
|---|---:|---:|---:|---:|---|
| beam 4, no LM | – | 4.803 | 12.972 | 8.754 | |
| word 4-gram rescoring of the beam-16 list (second pass, not streaming) | 1,556 MB | 4.586 | 12.240 | 8.288 | −0.466 [−0.549, −0.382] |
| **beam 4 + token 5-gram M5 + LODR** (streaming) | 179 MB | **3.895** | **11.027** | **7.344** | **−1.410 [−1.535, −1.286]** |
| beam 8 + M5 + LODR (streaming) | 179 MB | 3.768 | 10.668 | 7.105 | −1.649 [−1.779, −1.519] |
| beam 4 + token 5-gram S5 + LODR (streaming, device size) | 48 MB | 4.097 | 11.421 | 7.639 | −1.114 [−1.231, −1.002] |
| oracle of the beam-8 list (ceiling for rescoring) | – | 3.000 | 10.267 | 6.514 | |

The Rust engine (#49) reproduces the fusion gain on the 730-utterance panel: −1.134 pp for the engine (INT8) against −1.099 pp for Python, with the same weights. The LM text contains 42 of the 3,719 long dev transcripts verbatim; the gain is the same without them. Scorer `ultra.wer` (54,402 / 50,948 reference words) unless a row says otherwise. Every row carries the one-token-per-frame cap of the Python beam.

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

### Phase 1, continued: list size, smaller word LMs, and the no-LM control (2026-11-22, 21:19–21:27Z)
Same protocol (held-out speaker folds, beam-4 baseline 4.803 / 12.972 / 8.754). Size = KenLM trie as queried. "w/b" = LM weight / word bonus chosen on fold A (scored on B), then on fold B (scored on A).

| rescoring arm | LM MB | mean list | dev-clean | dev-other | pooled | Δ pooled [95%, utterances] | oracle pooled | w/b chosen |
|---|---:|---:|---:|---:|---:|---|---:|---|
| word 3-gram pruned 3e-7, beam-8 list | 21.6 | 7.26 | 4.884 | 12.948 | 8.784 | +0.030 [-0.029, +0.092] | 6.514 | 0.15/1.0, 0.1/-0.5 |
| word 3-gram pruned 1e-7, beam-8 list | 45.3 | 7.26 | 4.816 | 12.803 | 8.679 | -0.075 [-0.140, -0.008] | 6.514 | 0.2/1.0, 0.15/-0.5 |
| word 4-gram (official), beam-8 list | 1,556 | 7.26 | 4.652 | 12.362 | 8.381 | -0.373 [-0.452, -0.291] | 6.514 | 0.3/1.0, 0.35/1.5 |
| word 4-gram, beam-16 list (post hoc) | 1,556 | 14.48 | 4.586 | 12.240 | 8.288 | -0.466 [-0.549, -0.382] | 5.766 | 0.25/0.5, 0.25/0.5 |
| word 4-gram, beam-32 list (post hoc) | 1,556 | 28.9 | 4.610 | 12.136 | 8.250 | -0.504 [-0.596, -0.410] | 5.158 | 0.25/0.5, 0.3/1.0 |

- **Lists:** beam-16 and beam-32 lists were decoded with `FastBeam` (identical to `train.beam` without an LM; the beam-8 lists of both matched on 5,567 / 5,567). Doubling the list gains 0.09 pp, then 0.04 pp, while the oracle keeps falling (6.51 → 5.77 → 5.16): the 4-gram cannot tell most of the better candidates apart.
- **Small word LMs** buy almost nothing in rescoring: the 45 MB pruned 3-gram gives −0.075 pp (interval just excludes zero), the 22 MB one nothing.

| control | dev-clean | dev-other | pooled | Δ pooled [95%] | chosen |
|---|---:|---:|---:|---|---|
| beam 4, no LM, blank penalty chosen by the same folds (control) | 4.796 | 12.948 | 8.738 | -0.015 [-0.057, +0.027] | penalty 0.25 on both folds |

- The blank penalty alone does not lower WER, so a fusion gain is not a penalty effect.

### Phase 2: token n-gram fusion in the beam, all LMs (FINAL, 2026-11-23)
Token LMs (all on the OpenSLR 11 text, 1,577,847,566 tokens; sizes = the quantised KenLM trie each decode queries):

| name | order | `--prune` | n-grams (orders 2..N) | trie MB | file, sha256 prefix |
|---|---:|---|---|---:|---|
| S3 | 3 | none | 194,853 / 14,346,832 | 31.4 | `tok3.p000.trie.bin` 9ddfe279dd9ac17f |
| S4 | 4 | 0 0 4 15 | 194,853 / 6,441,413 / 9,565,946 | 47.4 | `tok4.p00415.trie.bin` 3175b9b14b806f26 |
| S5 | 5 | 0 0 6 30 60 | 194,853 / 5,466,300 / 5,345,531 / 2,295,257 | 48.2 | `tok5.p0063060.trie.bin` 7fffb51895e660bc |
| M4 | 4 | 0 0 0 1 | 194,853 / 14,346,832 / 59,140,270 | 187.7 | `tok4.p0001.trie.bin` c2f9a33faf196b16 |
| **M5** | 5 | 0 0 1 4 9 | 194,853 / 10,054,238 / 25,723,048 / 16,787,323 | 179.1 | `tok5.p00149.trie.bin` aaec33c6bacdc198 |
| L5a | 5 | 0 0 1 2 3 | 194,853 / 10,054,238 / 40,283,631 / 46,781,615 | 305.0 | `tok5.p00123.trie.bin` e6ab121cf8e0b00e |
| L5b | 5 | 0 0 0 1 1 | 194,853 / 14,346,832 / 59,140,270 / 114,371,177 | 548.5 | `tok5.p00011.trie.bin` 4068f44bb82530fa |

Every LM met its size class with the thresholds fixed in amendment 2 (no rebuild was needed). LODR: token bigram of the 960 h transcripts (18,252,156 tokens; 131,573 bigrams; 2.4 MB).

**Size against WER at beam 4** (held-out speaker folds, baseline 4.803 / 12.972 / 8.754; "decode" says whether the row is the reference search (`train.beam` + hook) or the sweep (`FastBeam`), and where the edge rule ended):

| LM | order | MB | arm | dev-clean | dev-other | pooled | Δ pooled [95%, utterances] | [95%, speakers] | rel. | weights chosen (held out) | decode |
|---|---:|---:|---|---:|---:|---:|---|---|---:|---|---|
| S3 | 3 | 31.4 | fusion | 4.643 | 12.348 | **8.369** | -0.384 [-0.471, -0.297] | [-0.463, -0.309] | -4.4% | A→B: λ1 0.3, pen 1; B→A: λ1 0.3, pen 1 | train.beam edge rule converged |
| S3 | 3 | 31.4 | fusion + LODR | 4.634 | 12.228 | **8.307** | -0.447 [-0.540, -0.352] | [-0.554, -0.345] | -5.1% | A→B: λ1 0.4, λ2 0.2, pen 0.5; B→A: λ1 0.5, λ2 0.25, pen 0.5 | train.beam edge rule converged |
| S4 | 4 | 47.4 | fusion | 4.294 | 11.757 | **7.903** | -0.850 [-0.964, -0.737] | [-1.007, -0.696] | -9.7% | A→B: λ1 0.5, pen 1.5; B→A: λ1 0.4, pen 1.25 | train.beam edge rule converged |
| S4 | 4 | 47.4 | fusion + LODR | 4.184 | 11.474 | **7.710** | -1.044 [-1.159, -0.928] | [-1.205, -0.884] | -11.9% | A→B: λ1 0.6, λ2 0.3, pen 0.5; B→A: λ1 0.5, λ2 0.25, pen 0.5 | train.beam edge rule converged |
| S5 | 5 | 48.2 | fusion | 4.285 | 11.673 | **7.858** | -0.896 [-1.008, -0.784] | [-1.045, -0.752] | -10.2% | A→B: λ1 0.4, pen 1.5; B→A: λ1 0.5, pen 1.5 | train.beam edge rule stopped (round 6 decoded, not tuned; row = round 5) |
| S5 | 5 | 48.2 | fusion + LODR | 4.097 | 11.421 | **7.639** | -1.114 [-1.231, -1.002] | [-1.287, -0.954] | -12.7% | A→B: λ1 0.6, λ2 0.3, pen 0.5; B→A: λ1 0.6, λ2 0.3, pen 0.5 | train.beam edge rule converged |
| M4 | 4 | 187.7 | fusion | 4.193 | 11.673 | **7.810** | -0.944 [-1.059, -0.830] | [-1.100, -0.794] | -10.8% | A→B: λ1 0.5, pen 1.25; B→A: λ1 0.4, pen 1 | train.beam edge rule incomplete (11 points pending) |
| M4 | 4 | 187.7 | fusion + LODR | 4.149 | 11.378 | **7.645** | -1.109 [-1.233, -0.986] | [-1.281, -0.941] | -12.7% | A→B: λ1 0.6, λ2 0.3, pen 0.75; B→A: λ1 0.7, λ2 0.35, pen 0.75 | train.beam edge rule converged |
| M5 | 5 | 179.1 | fusion | 4.061 | 11.361 | **7.591** | -1.163 [-1.280, -1.047] | [-1.337, -0.994] | -13.3% | A→B: λ1 0.4, pen 1.5; B→A: λ1 0.5, pen 1.5 | train.beam edge rule converged |
| M5 | 5 | 179.1 | fusion + LODR | 3.895 | 11.027 | **7.344** | -1.410 [-1.535, -1.286] | [-1.595, -1.227] | -16.1% | A→B: λ1 0.6, λ2 0.45, pen 0; B→A: λ1 0.7, λ2 0.525, pen 0 | train.beam edge rule converged |
| L5a | 5 | 305.0 | fusion | 4.116 | 11.698 | **7.783** | -0.971 [-1.072, -0.872] | [-1.094, -0.858] | -11.1% | A→B: λ1 0.3, pen 0.25; B→A: λ1 0.3, pen 0.25 | sweep reduced grid; edge rule incomplete (stopped) |
| L5a | 5 | 305.0 | fusion + LODR | 3.834 | 10.980 | **7.290** | -1.464 [-1.589, -1.339] | [-1.656, -1.280] | -16.7% | A→B: λ1 0.6, λ2 0.3, pen 0.25; B→A: λ1 0.6, λ2 0.3, pen 0.25 | sweep reduced grid; edge rule incomplete (stopped) |
| L5b | 5 | 548.5 | fusion | 4.103 | 11.681 | **7.767** | -0.986 [-1.090, -0.884] | [-1.105, -0.875] | -11.3% | A→B: λ1 0.3, pen 0.25; B→A: λ1 0.3, pen 0.25 | sweep reduced grid; edge rule not run (12 points pending) |
| L5b | 5 | 548.5 | fusion + LODR | 3.873 | 11.047 | **7.342** | -1.411 [-1.532, -1.293] | [-1.601, -1.230] | -16.1% | A→B: λ1 0.5, λ2 0.25, pen 0.25; B→A: λ1 0.6, λ2 0.3, pen 0.25 | sweep reduced grid; edge rule not run (12 points pending) |

- **Order matters more than size:** at about 180 MB the 5-gram (M5, 7.344) beats the 4-gram (M4, 7.645) by 0.30 pp. Under 50 MB, the heavily pruned 5-gram (S5, 7.639) and 4-gram (S4, 7.710) are close, and the unpruned 3-gram (S3, 31 MB, 8.307) is far behind. The 305 / 548 MB 5-grams are no better than M5 (reduced grids, edge rule not completed; reference only).
- **LODR helps every LM:** 0.06-0.25 pp pooled on top of fusion alone for the LMs whose grids converged (S3 0.06, M4 0.17, S4 0.19, S5 0.22, M5 0.25). The 0.43-0.49 pp for L5a / L5b is inflated: their fusion-only grids stopped at blank penalty 0.25.
- **Ship (lead, 23:35Z):** M5 for servers, S5 for devices.
- **Reference search for the table rows:** all chosen configs of S3, S4, S5, M4 and M5 at beam 4, and of M5 at beam 8, were decoded again with `train.beam` + hook. 21 of 22 gave token output identical to the sweep on all 5,567 utterances: `S3_b4_l0.50_r0.50_p0.50` differs on 1 of 5567 utterances (a float32 near-tie); its family's held-out WER is the same either way (8.307 with `train.beam`, 8.307 in the sweep). Rows marked `train.beam` are those re-decodes; the beam-8 rows below are re-decodes too. L5a / L5b are sweep decodes (`FastBeam`).
- **Python decode time with the LM** (sum of per-utterance search time, same-batch no-LM run, informational): M5 (H0): no LM 861 s, with LM 1169-1197 s (+36 to +39%); S5 (xbabe2): no LM 678 s, with LM 1035-1140 s (+53 to +68%); S4 (xbabe2): no LM 702 s, with LM 985-1115 s (+40 to +59%); S3 (xbabe2): no LM 746 s, with LM 864-1152 s (+16 to +54%); M4 (xbabe2): no LM 746 s, with LM 1030-1248 s (+38 to +67%). M5 ran on H0 (10 workers), the others on xbabe2 (16 workers on 8 cores and their siblings), so the ratios differ by host. Almost all of the extra time is the Python KenLM row (499 calls per new context).

**M5 at beam 8** (pre-registered in amendment 2: reduced grid plus the edge rule, held-out folds; both families' chosen configs decoded again with `train.beam` + hook, identical on 5,567 / 5,567).

| arm | dev-clean | dev-other | pooled | Δ vs beam 4 no LM [95%] | Δ vs beam 8 no LM [95%] | weights chosen (held out) |
|---|---:|---:|---:|---|---|---|
| beam 8, no LM | 4.777 | 12.988 | 8.748 | | | |
| beam 8 + M5 | 3.890 | 10.846 | **7.254** | -1.500 [-1.627, -1.373] | -1.494 [-1.620, -1.370] | A→B: λ1 0.5, pen 1.25; B→A: λ1 0.5, pen 1.5 |
| beam 8 + M5 + LODR | 3.768 | 10.668 | **7.105** | -1.649 [-1.779, -1.519] | -1.643 [-1.771, -1.514] | A→B: λ1 0.5, λ2 0.25, pen 0.25; B→A: λ1 0.7, λ2 0.35, pen 0.75 |

- **Beam 8 against beam 4:** with M5 + LODR, beam 8 is 0.24 pp better (7.105 against 7.344), almost all of it on dev-other. Without an LM, beam 8 changes nothing (8.748 against 8.754). So the wider beam pays only when an LM can re-rank the extra hypotheses.
- **Edge rule at beam 8:** both families converged (LODR after round 3, fusion only after round 6, its blank penalty rising from 0.25 to 1.25-1.5).
- **Cost of beam 8 with the LM:** 21.4 new LM contexts per audio second (3.97 per output token), against 13.3 at beam 4. Python decode time on xbabe2 was 1,205 s without an LM and 2,281-2,638 s with M5 (+89 to +119%, same batch).


### Overlap of the LM text with dev (2026-11-22, 23:10Z; a check I owed, not pre-registered)
- **Why:** my first chat post said the dev and test books are "excluded by construction" from the OpenSLR 11 LM text. The OpenSLR page says nothing of the kind (it describes the corpus only as "14500 public domain books, used as training material for the LibriSpeech's LM"), so I tested it.
- **Test:** every dev transcript with at least 12 words (1,866 dev-clean, 1,853 dev-other) searched as a fixed string in the normalised LM text (`grep -F` over `librispeech-lm-norm.txt.gz`). The test sets were not checked (sealed).
- **Found: 42 of 3,719 occur verbatim** (11 dev-clean, 31 dev-other). 41 of them occur once and one (a famous quotation) 9 times. 23 come from dev-other speaker 3915 and 7 from speaker 7976, so the source text of those dev chapters is in the LM corpus in some edition. Shorter utterances were not tested; the list is in `/mnt/xbabe3-external/jsr/lm/work/results/leak_uids.json`.
- **Effect on the result (M5, held-out weights):** relative pooled gain of fusion + LODR on all utterances −16.1%; without the 42 utterances −16.1%; without speakers 3915 and 7976 entirely (146 utterances) −16.0%. Fusion alone: −13.3% / −13.2% / −13.1%. The matched utterances themselves improve more (4.70 → 3.64), but they are too few to move the totals. **The overlap does not drive the gain.** The same corpus underlies every published LibriSpeech "+LM" number.

### Phase 2 headline: M5 fused in the beam, FINAL (2026-11-22, 23:52Z; reference search)
M5 = token 5-gram, `lmplz --prune 0 0 1 4 9`, 179.1 MB quantised trie. Full pre-registered grid at beam 4, edge rule run to completion (rounds 0-2; every chosen value interior). Both folds' choices were then decoded again with `train.beam.modified_beam_search` + `train.lm_fusion.NgramFusion`. **Token output was identical to the sweep decode on 5,567 / 5,567 utterances for all four chosen configs**, so the sweep's held-out numbers are the reference numbers.

| arm (beam 4, held-out speaker folds) | dev-clean | dev-other | pooled | Δ pooled [95%, utterances] | [95%, speakers] | relative |
|---|---:|---:|---:|---|---|---:|
| no LM (baseline) | 4.803 | 12.972 | 8.754 | | | |
| **M5 fusion** | **4.061** | **11.361** | **7.591** | −1.163 [−1.280, −1.047] | [−1.337, −0.994] | −13.3% |
| **M5 fusion + LODR** | **3.895** | **11.027** | **7.344** | −1.410 [−1.535, −1.286] | [−1.595, −1.227] | −16.1% |

- **Chosen weights.** Fusion: fold A chose λ1 0.4 with blank penalty 1.5 (scored on B); fold B chose λ1 0.5 with penalty 1.5. Fusion + LODR: fold A chose λ1 0.6, λ2 0.45, penalty 0; fold B chose λ1 0.7, λ2 0.525, penalty 0. Tuned on all dev utterances (the setting an engine would ship, not a held-out number): λ1 0.6, λ2 0.45, penalty 0 → 7.320.
- **Verdict by the pre-registered rule:** pooled held-out WER is lower and the paired interval excludes zero (utterance and speaker level): **both M5 rows are "+LM" gains.** The size is 2-3× what I pre-registered (fusion 3-7% expected, 13.3% measured; with LODR 5-10% expected, 16.1% measured).
- **Python decode cost (informational).** Sum of per-utterance search wall time over both dev sets, 10 single-threaded workers on H0 cores 20-29, the no-LM run in the same batch: no LM 860.9 s; M5 fusion 1,173.8-1,191.7 s; M5 + LODR 1,168.8-1,197.4 s, i.e. **+36-39%**. Almost all of the extra time is the Python KenLM row: 499 `BaseScore` calls per new context.
- **LM work per utterance (engine-relevant).** A config needs **90 distinct LM contexts per utterance = 13.3 per audio second = 2.47 per output token** (beam 4, all hypotheses counted; contexts repeat within an utterance and are counted once). A naive engine scores a full row per context (499 lookups: ~6,600 lookups per audio second); runtime-lane's exact upper-bound pruning skips most of them.

### Engine parity: the Rust engine reproduces the gain (2026-11-22, 23:57Z)
`ultra-engine-eval` built from this branch merged with main (#49: `ultra.lm-ngram/v1`, exact upper-bound pruning; exe sha256 `28531bde…`), H22ctl-79800 EMA **INT8** (prepared weights sha256 `f4133df6…`), the 730-utterance `dev_selection_panel.json`, 16 streams, beam 4, max 1 symbol per frame (the same cap as the Python beam), `--qualifying`. LMs packed with `ultra-lm-pack` (q8, LODR bigram inside). Each LM uses its weights chosen on all dev utterances (M5: λ1 0.6, λ2 0.45, penalty 0; S4 and S5: λ1 0.6, λ2 0.3, penalty 0.5), so the panel WERs are a parity check, not held-out numbers. The Python side is the FP32 EMA model, same configs, same 730 utterances. Scorer `ultra.wer` for both; `train.lm_parity`, `results/engine_parity.json`.

| panel (730), beam 4 | engine INT8: clean / other / pooled | Python FP32: clean / other / pooled | identical token sequences | engine Δ from the LM | Python Δ |
|---|---|---|---:|---:|---:|
| no LM | 5.075 / 12.062 / 7.944 | 5.099 / 11.943 / 7.909 | 680 / 730 | | |
| + M5 + LODR (179 MB trie, 334 MB pack) | 4.154 / 10.622 / **6.810** | 4.072 / 10.740 / **6.810** | 681 / 730 | **−1.134** | −1.099 |
| + S5 + LODR (48 MB trie, 95 MB pack) | 4.367 / 10.927 / 7.060 | 4.320 / 10.757 / 6.963 | 689 / 730 | −0.884 | −0.946 |
| + S4 + LODR (47 MB trie, 82 MB pack) | 4.343 / 11.028 / 7.088 | 4.272 / 10.944 / 7.012 | 682 / 730 | −0.856 | −0.897 |

- The engine's gain from each LM matches the Python gain within 0.1 pp. The hypotheses that differ (≈ 50 of 730) differ without an LM too: INT8 against FP32, not the fusion.
- **Engine cost** (one shared H0 core each, plain beam 4, exploratory): 0.146 CPU-s per audio-s without LM; 0.162 / 0.159 / 0.161 with M5 / S5 / S4 (**+9-11%**). Peak RSS 342 MiB without LM; +357 / +90 / +79 MiB (the pack is held in RAM).
- **Small models against M5 (ship rule "within ~0.2 pp"):** S5 +0.25 pp (engine, panel) / +0.30 pp (Python, full dev, held out); S4 +0.28 / +0.37. Neither meets it.

### What a streaming engine needs (the next step; runtime-lane's #49 already does most of it)
- **Model files.** One token n-gram over the model's 500 output tokens plus the LODR bigram. Tokens are written as decimal output indices "1".."500" (SentencePiece id + 1; 0 is blank and never queried). Sources:
  - ARPA (gzipped) under `/mnt/xbabe3-external/jsr/lm/models/token/arpa/`: M5 `tok5.p00149.arpa.gz`, S5 `tok5.p0063060.arpa.gz`, S4 `tok4.p00415.arpa.gz`; LODR `token/lodr2.train960.arpa`;
  - the KenLM tries these numbers were decoded with under `models/token/`, and the engine packs (`ultra.lm-ngram/v1`, q8) under `models/engine/`. Hashes in `models/SHA256SUMS`.
- **Size.** M5 is 179 MB as KenLM's quantised trie and 334 MB as an engine pack; S5 is 48 / 95 MB; S4 47 / 82 MB; LODR ≈ 2.4 MB (1 MB as a dense 501×501 table). The size is per process: every stream shares one read-only copy (mmap-able).
- **Per hypothesis:** the last 4 tokens (5-gram context), plus the cached backoff weights if the engine walks reverse tries.
- **Lookups.** 13.3 new LM contexts per audio second (2.47 per output token, beam 4). Scoring a full row per context (the Python reference) is 499 lookups per context, about 6,600 per audio second. The exact upper bound (ln P_LM ≤ 0) lets the engine look up only tokens that could still enter the beam. On a shared core the engine's measured cost of fusion was +9-11% CPU at beam 4.
- **Weights to ship (M5, beam 4):** λ1 0.6, λ2 0.45, blank penalty 0 (chosen on all dev utterances; held-out folds chose λ1 0.6-0.7 with λ2 = 0.75·λ1).
- **Formula:** see "Phase 2" above: natural log, no term on blank, `<s>` start state, no `</s>`.
- **Not streaming:** word 4-gram rescoring needs the final N-best list, so it can only be a second pass at the end of an utterance. It is also the smaller gain (−0.37 to −0.50 pp against −1.41 pp).
