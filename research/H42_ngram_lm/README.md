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

## Results
(none yet)
