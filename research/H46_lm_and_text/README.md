# H46: language model and text for B0 ("+LM" and "+text" rows)

**Status:** PRE-REGISTERED, 2026-11-23. This file is committed before any decode of this wave
that uses a language model, a beam, or a blank penalty. When it was written, the only results
that existed for B0 were:
- eval-lane's greedy and beam-4 full-dev decodes of A1-49000 EMA (canonical `ultra-score`
  8.302 / 8.223 pooled);
- the INT8 engine full-dev row (8.272);
- Codex's retained-control receipts.

No A1 decode with an LM had ever been run. The encoder pass of item 1 (below) was running.
Results are added under "Results" later. Amendments are dated, and each is written before the
runs it affects.

**Lane:** h46-lm (branch `h46/lm`, worktree `lead-gate`). **Owners** (lead's DECONFLICT post,
05:58Z):
- h46-lm runs the experiments, the offline analysis and the Python training.
- runtime-lane's swarm implements the Rust `TokenLm` trait and the FFNN runtime (`lm/ffnn.rs`).
- Codex owns `decoder.rs` / `beam.rs`, and so the Rust predictor-context change that +text
  needs.
- trainer-lane owns `trainer.py`, and so lands the +text training patch.

**Authority:**
- Owner ruling O17 lets the LibriSpeech LM text (OpenSLR 11) serve as a decoding language
  model. Results that use it are reported in separate **"+LM"** rows.
- The owner's v3 approval (2026-11-23 05:4xZ) lets that text train the predictor. Those results
  are labelled **"+text"**.
- Every number of this wave is **exploratory** and comes from **DEV only**. The test sets are
  sealed and never read: `train.evaluate.check_split` and the engine's sealed guard refuse
  `test-*`.

## Questions
1. **LM gain on B0.** How much does H42's token 5-gram M5 + LODR, fused in the beam, lower the
   full-dev WER of B0 (A1-49000 EMA, 160 ms)? Weights are held out by speaker folds, at beam 4
   and beam 8. Does the Rust engine reproduce the gain at H42's ship weights? **This result
   decides whether the LM branch (items 2-3) continues.**
2. **Pack parity.** Do the `ultra.lm-ngram/v1` and the bit-packed `v2` packs (PR #70,
   `ultra-lm-pack --repack`) give identical engine output: 730 / 730 panel utterances?
3. **Compact LM byte ladder.** At 0.5 / 2 / 8 MiB, which of three candidates keeps the most of
   the full LM's gain?
   - a pruned v2 n-gram;
   - a small neural token LM (FFNN);
   - a residual student.
4. **+text.** Can hashed 2-4-token context tables, added to the stateless predictor through a
   zero-initialised projection and trained with a text-only loss on the LM text, fix
   function-word swaps without an external LM? Swaps such as in↔and and a↔the are 20.5% of B0's
   substitutions.

Items 2-4 get their exact arms in dated amendments, written before their runs. The gates and
kill rules below are fixed now.

## Item 1: first A1 + LM decode on full dev

### Hypothesis and expected size (written before looking)
On H22ctl (H42) M5 + LODR gave:
- beam 4: −1.41 pp pooled, −16.1% relative;
- beam 8: −1.65 pp, −18.8% relative.

B0 is a better acoustic model trained on the same 960 h transcripts, so it has a similar
internal LM. Expected for B0, pooled over full dev, held out:
- **beam 4 + M5 + LODR: −10% to −16% relative** (about −0.8 to −1.3 pp from 8.22);
- **beam 8 + M5 + LODR:** a further 0.1-0.3 pp below beam 4 + M5 + LODR;
- **M5 fusion without LODR:** 0.1-0.3 pp behind fusion + LODR;
- **blank-penalty control (no LM):** no gain beyond noise, as on H22ctl (−0.015 pp).

A null or negative result is reported as such, with the numbers.

### Fixed setup
**Acoustic model:** A1 step 49,000 EMA weights.
- Source `xbabe5:~/jsr-data/runs/train/A1_S_r0bpe_960h/ckpt-00049000.pt`, sha256
  `2a3dd9fb6c7856126b3455fb65babc4295e27b10e011d18a8806b8be5181e146`.
- The decode copy is cut from eval-lane's EMA-as-model file `A1-49000-ema-as-model.pt` (sha256
  `4055c989…`, which records that source) by `train/ops/h46_slim_from_derived.py`, giving
  `A1-49000-ema.slim.pt`, sha256 `73de4f55…`.
- The original file is also copied from xbabe5. Its EMA tensors are compared with the decode
  copy (`torch.equal` on every tensor) and the result is recorded in `MANIFEST`.
- Architecture: dense S Conformer (384-d, 16 blocks, chunk 4, left context 128); stateless
  predictor with context 2; joiner 320 tanh; the R0 500-piece BPE (`bpe.model` sha256
  `c53433de…`).

**Code:** `python/ultra` at main `e61121e`, frozen as a `git archive` copy for every run of
this item. The tools are H42's: `train.lm_decode`, `train.lm_fusion`, `train.lm_tune` and the
`train/ops/lm_*.sh` drivers, unmodified. The H46 drivers are added next to them, and the frozen
tree is re-cut from the commit that adds them:
- `train/ops/h46_encode.sh`: the encoder pass;
- `train/ops/h46_sweep.sh`: H42's `lm_sweep.sh` on a given grid, with the edge rule driven by
  the selection-half choice;
- `train/h46_tune.py`: selection-half choice, then confirmation-half and full-dev scoring;
- `train/ops/h46_slim_from_derived.py`: cuts the decode copy.

**Torch:** 2.7.0+cpu with MKL 2024.2, copied from `xbabe2:~/jsr-data/venvs/torch270-cpu` into
`/dev/shm` (`libtorch_cpu.so` sha256 `2700b1d4…`). This is the build eval-lane's exports use.
- H0's system torch (Lambda 2.7.0, `BLAS_INFO=generic`, Eigen) runs single-thread SGEMM at
  5 GFLOPS against 65 GFLOPS for this build. The first encoder attempt with it ran at 1.9 CPU-s
  per audio-s and was stopped at 06:16Z before it wrote anything.
- `FastBeam` uses NumPy (OpenBLAS) either way.

**Encoder pass:** `train.lm_decode encode`.
- EMA weights, batch 8 in length order, trained chunk 4 (160 ms) and left context: the
  `train.evaluate` defaults.
- 14 single-threaded workers, then `merge`.
- **Check:** its greedy output (max 8 symbols per frame) must give token text identical to
  eval-lane's `A1-49000.ema.c4.ms8` dumps on 5,567 / 5,567 utterances. Otherwise nothing below
  runs until the difference is explained.

**Search:**
- Sweeps use `train.lm_fusion.FastBeam`: the NumPy twin of `train.beam.modified_beam_search`,
  identical on 5,567 / 5,567 in H42.
- The configs that a fold chooses are decoded again with the reference search
  (`train.beam.modified_beam_search` + `NgramFusion`), as in H42 amendment 1. Reported rows come
  from the reference search. If they differ from the sweep, both are shown.
- At most one token per frame (the cap of both searches); no length normalisation; no
  end-of-sentence term.

**Language models** (H42's, unchanged; sha256 in `/mnt/xbabe3-external/jsr/lm/models/SHA256SUMS`):
- **M5** `tok5.p00149.trie.bin`, `aaec33c6…`: token 5-gram, `lmplz --prune 0 0 1 4 9`, 179.1 MB
  quantised KenLM trie.
- **LODR** `lodr2.train960.bin`, `00b5588e…`: token bigram of the 960 h training transcripts.

**Data:** full dev-clean (2,703 utterances) and dev-other (2,864).

### Arms
- **No-LM reference rows:**
  - greedy (from the encoder pass);
  - **beam 4** (`b4`, the baseline of every beam-4 row);
  - **beam 8** (`b8`, the no-LM comparator of the beam-8 rows).
- **C4, control.** Beam 4 with no LM; blank penalty ∈ {0, 0.25, 0.5, 1.0}, chosen by the same
  folds. A fusion gain is not credited to the penalty.
- **L4, primary.** Beam 4 + M5 + LODR, on H42's full LODR grid:
  - λ1 ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
  - λ2/λ1 ∈ {0.25, 0.5, 0.75};
  - blank penalty ∈ {0, 0.25, 0.5}.
- **F4, secondary.** Beam 4 + M5 fusion without LODR, on H42's full fusion grid:
  - λ1 ∈ {0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5};
  - blank penalty ∈ {0, 0.25, 0.5, 1.0}.

  F4 and L4 are decoded together in one pass: `lm_tune.make_grid(kind="full")`, the same 82
  configs as H42's `M5_b4`.
- **L8, primary.** Beam 8 + M5 + LODR:
  - λ1 ∈ {0.3, 0.4, 0.5, 0.6, 0.7};
  - λ2/λ1 ∈ {0.5, 0.75};
  - blank penalty ∈ {0, 0.25, 0.5}.

  This grid is centred on H42's beam-8 choices (λ1 0.5-0.7, ratio 0.5, penalty 0.25-0.75). Its
  F8 fusion-only twin is not run.
- **T4, transfer row.** Beam 4 + M5 + LODR at H42's ship weights: λ1 0.6, λ2 0.45, penalty 0.
  These weights were tuned on H22ctl over the same dev utterances, not on A1. This config is a
  member of L4's grid; the row is reported separately with that label and is never the
  headline.
- **Grid rule** (H42's, `lm_tune extend`): a chosen value on the edge of its axis adds one more
  step in that direction, crossed with the family's other axes, until no choice is on an edge,
  at most 10 rounds. Axes with a single value are not extended. No other points are added. In
  the primary protocol the rule is driven by the **selection-half choice only** (below).

### Tuning protocol
**Primary: eval-lane's fixed dev speaker split v2** (06:00Z post,
`/mnt/xbabe3-external/jsr/evals/eval-lane/h43/dev_split_v2.json`, sha256 `002ac7ae…`), adopted
for every wave:
- **Selection half:** 20 + 16 speakers, 2,782 utterances.
  - Each family's weights are chosen on it alone: the fewest word errors under `ultra.wer`, ties
    to the earliest config in `lm_tune`'s preference order (smallest LM weight, then LODR
    weight, then penalty).
  - The edge rule runs on that choice until it converges.
  - The result is **one finalist per family, named before the confirmation half is scored.**
- **Confirmation half:** 20 + 17 speakers, 2,785 utterances. Each finalist is scored on it once,
  against its no-LM comparator, with the paired speaker-cluster bootstrap. This is the decision
  statistic.
- **Full dev at the finalist's weights** is the reported endpoint. It is labelled "weights tuned
  on the selection half (one look per family)".

**Secondary: H42's two-fold cross-validation**, kept for comparability with the H42 table. It
runs only after every primary finalist is named, and its own edge extensions come after that.
- **Folds:** within each dev set the speaker ids are sorted numerically; speakers at even
  positions form fold A, the others fold B.
- **Choice:** weights are chosen on fold A and scored on fold B, and the reverse.
- **Reported number:** fold B scored with A's weights plus fold A scored with B's.

**"Tuned on itself"** labels any number whose weights were chosen on the utterances it is scored
on. Such a number is never a headline.

### Scoring and statistics (fixed now)
- **Headline scorer:** the canonical Rust `ultra-score` (`ultra.score-report/v1`, normaliser
  `ultra-en-wer-norm/v1`).
  - It is built from `e61121e`, and its sha256 is recorded.
  - It must reproduce eval-lane's 8.302 on the greedy dump before use; eval-lane's pinned copy
    (sha256 `d784f8ca…`) is run alongside.
  - It reports pooled WER (errors / reference words over both sets: 106,054 words on full dev),
    per set, and macro.
- **Interval:** `ultra-score`'s paired speaker-cluster bootstrap of arm − comparator: 10,000
  replicates, seed 20261122, `--alpha 0.025`. The bounds are the 2.5th and 97.5th percentiles,
  a two-sided 95% interval. That is stricter than the one-sided 95% bound of the release gate.
- **Secondary:** H42's paired bootstrap over utterances and over whole speakers on `ultra.wer`
  counts: 10,000 resamples, seed 20261122, resampled within each set.
- **"+LM" gain.** A "+LM" row counts as a gain if both hold on the confirmation half:
  - its pooled WER (canonical) is below its no-LM comparator's;
  - the canonical paired 95% interval lies below zero.

  Otherwise the row is "no measurable gain".

### Decision rule for the LM branch (fixed now)
The LM branch continues, with items 2-3 and the engine-side LM work, if the L4 finalist
(beam 4 + M5 + LODR) meets both conditions on the confirmation half (canonical):
- its pooled gain over `b4` is **larger than 0.17 pp** (2× the four-run spread);
- its paired 95% interval lies below zero.

Otherwise items 2-3 stop and the result is posted as a null.

**Controlling gates.** codex-lead's 06:03Z release decisions, which the lead adopted at 06:03:48Z,
apply wherever they are stricter:
- **Compact LM:** ≥ 70% of the same-checkpoint LM gain, within +5% CPU and +8 MiB.
- **Release:**
  - ≥ 10% relative canonical full-dev improvement over the retained control (≤ 7,895 edits);
  - paired upper bound below zero;
  - split upper bounds ≤ +0.20 pp.

  The "+LM" row of the final matrix is judged by these.

### Rust engine path (parity)
- **Binary and model:** `ultra-engine-eval` built from `e61121e` (release; sha256 recorded),
  with eval-lane's prepared INT8 directory of A1-49000 EMA
  (`/mnt/xbabe3-external/jsr/evals/eval-lane/export/A1-49000-ema/int8/`,
  `weights.int8.safetensors` sha256 `4701787c…`).
- **Decode:** 16 streams, 20 ms packets, beam 4, `--beam-max-symbols 1` (the same cap as the
  Python beam).
- **Runs:**
  - no LM;
  - `--lm M5.ulm` (H42's `ultra.lm-ngram/v1` q8 pack with LODR inside, sha256 `72a621c2…`)
    `--lm-weight 0.6 --lodr-weight 0.45 --blank-penalty 0`, H42's ship weights.
- **Utterances:** full dev, cut into per-core npz shards that keep each split's name. The shard
  files are byte slices of the cached dev npz; their hashes are recorded.
- **Python comparators:** the Python FP32 decodes `b4` and T4, the same weights on the same
  utterances.
- **Reported:**
  - canonical WER of both sides;
  - each side's change from its own no-LM run;
  - identical token sequences and texts;
  - engine CPU-s per audio-s and peak RSS, informational only (timing claims come only from
    xbabe0).
- **Parity criterion:** the engine's LM change is within 0.15 pp (pooled) of the Python
  change. The INT8 against FP32 difference without an LM is shown next to it.

### Compute
- H0 cores 16-29: 14 single-threaded workers, nice 10, as posted in `AGENT_CHAT.md` at 05:56Z.
- Estimate:
  - encoder pass, about 1 CPU-h;
  - L4 + F4 sweep, about 25 CPU-h with its extensions;
  - L8, about 15 CPU-h;
  - reference re-decodes, about 10 CPU-h;
  - engine runs, about 4 CPU-h.

  Together, about 4 h wall. No GPU is used.

## Item 2: v1 / v2 pack parity
**Gate:** 730 / 730 identical token sequences and identical texts on the 730-utterance DEV panel
(`dev_selection_panel.json`, H42's copy). Engine at beam 4, `--qualifying`, the ship weights,
the v1 pack against its `--repack`ed v2 twin. M5, S5 and S4 packs are compared.

**Reported:** sizes, load time, peak RSS. A miss on any utterance is a bug report to the pack's
owner (runtime-lane), with the utterance and both outputs.

## Item 3: compact LM byte ladder
**Rungs:** 0.5, 2 and 8 MiB of LM bytes as loaded by the engine. A pack or weight file counts
in full.

**Candidates per rung:**
- **(a) pruned token n-gram,** v2 pack. Pruning is chosen from n-gram counts and sizes only,
  never from WER.
- **(b) small feed-forward neural token LM,** INT8. FFNN-6 for 2 MiB: `Embedding(502, 128)`, 6
  tokens concatenated, `Linear(768, 1024)`, ReLU, `Linear(1024, 500)`. Smaller and larger
  variants fit the other rungs.
- **(c) residual student:** a small network trained to reproduce the fused correction
  λ1 ln P_M5 − λ2 ln P_LODR, with blank kept at exactly 0.

**Training data:** the OpenSLR 11 LM text tokenised as in H42. **The 42 dev transcripts that
occur verbatim in it** (H42's `leak_uids.json`) are removed from the training text of (b) and
(c); perplexity is reported with and without them.

**Gain retention:** (WER gain of the candidate) / (WER gain of full M5 + LODR). Both are held
out with the same folds, at the same beam, on canonical counts.

**Promotion gate (fixed now):**
- retention ≥ 70%;
- ≤ 8 MiB;
- ≤ 5% extra CPU. CPU is measured later on xbabe0 through bench-open; a MAC-count pre-screen is
  reported first.

**Kill rule:** if item 1 fails its decision rule, item 3 does not start.

## Item 4: +text predictor tables
**Design:** in an amendment before any training run, written with trainer-lane.
- Hashed 2-, 3- and 4-token context tables enter the stateless predictor through a projection
  initialised at zero, so the start equals B0 exactly.
- They are trained with a text-only loss on the LM text, with B0's audio losses kept.
- They target the function-word swaps.

**Rows:** DEV only, labelled "+text", against a continuation control with the same steps and no
text loss.

**Adoption:** > 2× the four-run spread (0.17 pp), with both seeds agreeing.

**Engine:** the Rust predictor context is fixed at 2 tokens today. Engine support is a design
note for Codex; no engine code comes from this lane.

## Amendment 1 (2026-11-23, 06:42Z): written before any engine output and before any item-2 run
What existed when this was written:
- the encoder pass (dev-clean done, dev-other running);
- the engine full-dev runs of item 1, started 06:38Z, none finished.

No LM decode, beam decode or pack comparison had produced a number.

1. **Engine host.** H0 is crowded: about 60% busy, 23 GB swapped, and memory-bound torch ops
   run ~5× slower than on xbabe2. So the item-1 engine runs (`nolm_b4`, `M5ship_b4`) run on
   **xbabe2 cores 14-27**, nice 10. Everything is identical to H0 except the host:
   - the same `ultra-engine-eval` binary, sha256 `a9b552e9…`;
   - the same prepared INT8 directory and M5 pack (hashes checked after the copy);
   - the same dev npz, sha256 `bfb27f58…` / `2944dbff…`.

   Both hosts are Threadripper 3995WX running the `avx2fma` kernels; the ISA is recorded in
   each `engine_eval.json`.
   - **Shards:** full dev is cut by `train/ops/h46_npz_shards.py` into 14 shards per split
     (utterance i goes to shard i mod 14; byte-identical slices; hashes in
     `results/engine_shards.jsonl`).
   - **Runs:** `train/ops/h46_engine_fulldev.sh` runs them with 16 streams, beam 4 and
     `--beam-max-symbols 1`.
   - **Isolation:** each shard is an independent run, so streams never mix utterances of
     different shards.
2. **Item 2, exact arms (gate unchanged: 730 / 730 identical token sequences and texts):**
   - **Packs:** each v1 pack (H42's M5, S5 and S4: `72a621c2…`, `9e2d3f98…`, `73d82f7a…`) is
     repacked with `ultra-lm-pack --repack` (the binary built from `e61121e`, sha256
     `208fe3ce…`) into `ultra.lm-ngram/v2`.
   - **Engine runs:** `ultra-engine-eval --panel dev_selection_panel.json` (sha256
     `199c6843…`), `--qualifying`, A1-49000 EMA INT8, 16 streams, beam 4,
     `--beam-max-symbols 1`.
   - **Weights:**
     - M5: λ1 0.6, λ2 0.45, penalty 0;
     - S5 and S4: λ1 0.6, λ2 0.3, penalty 0.5 (H42's engine-parity weights).
   - **Compared:** v1 against v2 for each LM, per utterance: tokens, text and score. Also
     reported: file size, load time, peak RSS and CPU-s per audio-s, informational only.
   - **On a miss:** a bug report to runtime-lane (PR #70's owner) with the utterances.

## Amendment 2 (2026-11-23, 07:45Z): item 3 arms, written before any item-3 decode or LM build
What existed when this was written:
- **Item 1:** L4 confirmation half −1.328 pp; full dev −1.292 pp.
- **Item 2:** passed.
- The tokenised LM text was being built on xbabe1; no compact LM had been trained, built or
  decoded.
- runtime-lane's PR #79 (open) defines the neural runtime format `ultra.lm-ffnn/v1`:
  E [V, d_e] → ctx slots → one hidden layer (ReLU / SiLU) → V outputs, symmetric per-row int8.

1. **Bytes.** A rung's budget covers everything the engine loads for fusion: the LM file plus,
   for a "+LODR" family, the LODR bigram. The LODR bigram as a v2 pack is 287,296 B, sha256 of
   `packs/lodr2.v2.ulm` recorded. Rungs: ≤ 0.5 MiB (524,288 B), ≤ 2 MiB, ≤ 8 MiB. A family
   counts for the smallest rung its total fits.
2. **Candidates:**
   - **(a) Count-pruned token n-grams** (`lmplz --prune`, the H42 text and tokenisation), packed
     as `ultra.lm-ngram/v2` with `ultra-lm-pack`.
     - Order 3 for each rung, sized for the LM-only bytes (fusion) and the LM + LODR bytes
       (+LODR).
     - Order 4 for the 8 MiB rung.
     - Thresholds are the smallest that keep the pack ≤ 95% of the rung, read from exact
       bigram / trigram count-of-counts on the token stream (and from `lmplz`'s reported
       counts for order 4).
     - One rebuild with raised thresholds is allowed if a pack misses its rung; no threshold is
       chosen from WER or perplexity.
   - **(b) Feed-forward neural token LMs** (`train.nlm`, PR #79's shape at vocabulary 501):
     - `0.5m`: ctx 6, d_e 96, d_h 384, ReLU, 476,224 B;
     - `2.0m`: ctx 6, d_e 192, d_h 1,120, SiLU, 1,977,152 B;
     - `4m`: ctx 8, d_e 192, d_h 2,048, SiLU, 4,290,428 B.

     Training, fixed now:
     - the OpenSLR 11 text minus every line holding one of H42's 42 verbatim dev transcripts;
     - every 1000th line held out;
     - AdamW, lr 2e-3, 1,000 warm-up steps then cosine, weight decay 0.01, batch 65,536
       positions, seed 1234;
     - steps: 20,000 for `0.5m` and `2.0m`, 30,000 for `4m`;
     - the last 10% of steps through the exporter's symmetric per-row int8 round trip.

     The exported int8 file (PR #79's `export_ffnn_v1`) is the one scored. The Python beam
     evaluates the same int8 tables through `train.nlm.NeuralFusion` on the QAT model, whose
     parity with the exported file is reported.
   - **(c) Residual student** (time permitting, after (a) and (b)):
     - R̂(c, v) = b_v + f(c)ᵀ e_v, where f embeds 4 context tokens (64-d each) and runs an
       MLP 256 → 128 → 64, with e_v of 64-d.
     - It is fitted by MSE over all 500 tokens to R = 0.8 ln P_M5 − 0.6 ln P_LODR (L4's
       finalist weights) on 4 M contexts from the LM text.
     - It needs no separate LODR. Fused as α · R̂ with α ∈ {0.6, 0.8, 1.0, 1.2} and penalty
       ∈ {0, 0.25}.
3. **Weights per candidate and family** (beam 4 only), edge rule as in item 1:
   - fusion: λ1 ∈ {0.2 … 0.7, step 0.1} × penalty ∈ {0, 0.5, 1.0, 1.5};
   - +LODR: λ1 ∈ {0.4 … 0.9, step 0.1} × λ2/λ1 ∈ {0.5, 0.75} × penalty ∈ {0, 0.25}.
4. **Choosing and scoring:**
   - Each family's finalist is chosen on the selection half and re-decoded with the reference
     search.
   - The rung's representative is the eligible finalist with the fewest selection-half errors.
     It is named before any confirmation-half look.
   - **Retention** = Δ(candidate) / Δ(L4) against `b4`, on the confirmation half (decision) and
     on full dev (endpoint), canonical counts.
5. **Gate** (unchanged; codex-lead's 06:03Z version):
   - retention ≥ 70%;
   - ≤ 8 MiB total;
   - ≤ +5% CPU. A MAC / lookup count per audio second against the encoder is reported now; the
     measured CPU comes later on xbabe0 through bench-open.

## Amendment 3 (2026-11-23, 08:15Z): item 4 design ("+text" predictor tables), written before any training run
This fixes the design. The training patch is written in this lane, under the exception in
`python/ultra/EXCEPTION.md`, and handed to trainer-lane, which owns `trainer.py`, launches every
training run, and fixes seeds and steps in its own amendment. Engine support is a design note for
Codex (`decoder.rs` / `beam.rs`); no engine code comes from this lane.

1. **What B0 lacks.** The stateless predictor sees only the last 2 tokens, its "language model"
   trained on 9.4 M transcript words.
   - The error analysis (05:47Z) puts 20.5% of substitutions on function-word swaps: in↔and
     197, a↔the 133.
   - Item 1 shows a token 5-gram over 800 M words removes 16% of all edits.
   - +text moves part of that context into the model itself, so it costs no external LM pass.
2. **Tables.** For n ∈ {2, 3, 4}, the exact id of the last n emitted tokens (oldest first,
   0-padded before the utterance start) is g_n = Σ c_i · 501^(n−1−i) (< 501⁴ < 2⁶³, so no
   overflow). It indexes a table T_n with P_n rows of d = 64 by i_n = g_n mod P_n, where P_n is
   a prime:
   - P₂ = 16,381; P₃ = 16,369; P₄ = 16,363.

   The three rows are concatenated and projected by a **zero-initialised** W_t (192 → 320, no
   bias), and the result is added to the predictor output:
   - pred_p' = proj(E[c₃], E[c₄]) + W_t · [T₂; T₃; T₄].
   - At initialisation pred_p' equals B0's pred_p bit for bit, so every arm starts at B0
     exactly.
   - Size: 3 × 16,37x × 64 + 61,440 ≈ 3.2 M parameters (3.2 MB as int8 rows).
3. **Losses.**
   - **Audio:** B0's recipe, unchanged (pruned RNN-T + CTC 0.3 + the adopted delay penalty).
     It trains all parameters, the tables included.
   - **Text:** batches of 256 × 64-token windows from the leak-free OpenSLR 11 stream
     (`train.nlm tokens`, sha256 `8e249254…`).
     - Logits: z = joiner.out(act(ē + pred_p')), where ē is the mean `joiner.enc_proj` output
       over 2,000 TRAIN utterances' frames, computed once from B0 and frozen.
     - Loss: cross-entropy over the 500 non-blank tokens (log-softmax renormalised without
       blank), weight λ_text.
     - Gradient: only into T₂₋₄ and W_t. The predictor embedding / projection and the joiner
       see the text loss through a stop-gradient, so the acoustic calibration and the blank
       margin are trained by audio alone.
4. **Arms**, for trainer-lane's amendment; seeds 1234 and 4321:
   - **T0:** continuation control. B0 → +N steps on A1's recipe + dp 0.005; no tables, no text.
   - **T1:** tables + text loss (λ_text 0.3).
   - **T1a (ablation):** tables, audio only.
   - N = the H48a A1L budget (8k steps), so T0 = A1L's recipe.
5. **Rows.** DEV only, labelled "+text": full-dev greedy and beam 4, canonical scorer, paired
   against T0 on both seeds. Also "+text +LM": T1 with item 1's M5 + LODR (weights re-chosen on
   the selection half), against T0 + LM.
6. **Gate:** > 2× the four-run spread (0.17 pp) against T0, with both seeds agreeing. Also a
   function-word swap count against T0, so the stated mechanism is checked.
7. **Kill rule:** after 2k steps, T1's dev-clean400 greedy WER above T0's by more than 0.3 pp on
   both seeds.
8. **Engine note for Codex.** The predictor context grows from `[usize; 2]` to `[usize; 4]` for
   the table ids only; the dense path still reads the last 2.
   - Per new hypothesis context: three exact ids, three `mod` operations, three 64-byte
     (int8) row loads, and one 192 × 320 product added to `pred_p`.
   - Beam state carries 4 tokens.
   - The H44 beam-reset fix must clear all 4.

## Results
All numbers are **exploratory**, from DEV only. "+LM" rows are separate rows.

### Item 1: B0 + M5 + LODR (2026-11-23, 07:39Z)
**Inputs and checks:**
- **Encoder cache** (`train.lm_decode encode`, 14 workers on H0, torch 2.7.0+cpu MKL). Greedy
  output equals eval-lane's `A1-49000.ema.c4.ms8` dumps on 5,566 / 5,567.
  - The exception is `6319-64726-0000` at frame 196: 'THE' here against 'A' in the dump, with
    a top-2 log-prob margin of 0.0016. The next smallest margin on that utterance is 0.064.
  - The cache matches the reference transcript there. eval-lane's curve passes run on H0's
    GPU queue, so a near-tie flip is expected (`results/item1_encoder_check.json`).
- **Search:** `FastBeam` = `train.beam` without an LM on 5,567 / 5,567 at beam 4 and beam 8.
  Every finalist's `train.beam` re-decode equals its sweep twin on 5,567 / 5,567.
- **Scorers:** the canonical `ultra-score` (h46 build `f5596785…`, eval-lane's `d784f8ca…`)
  gives 8.3024% on eval-lane's greedy dump with both builds.
- **Weights:** chosen on the selection half of eval-lane's split v2 (`002ac7ae…`). The beam-4
  sweep converged in 4 rounds (105 configs). Finalists were named at 07:36:47Z, before any
  confirmation-half score.

**Beam 4, canonical `ultra-score`.** Intervals are the paired speaker-cluster bootstrap,
10,000 replicates, two-sided 95% (`results/item1_confirm_M5_b4.json`).

| row | weights (chosen on selection half) | dev-clean | dev-other | full dev pooled | Δ full [95%] | Δ confirmation half [95%] |
|---|---|---:|---:|---:|---|---|
| no LM (`b4`) | – | 4.489 | 12.201 | 8.224 | | |
| C4: blank penalty only | penalty 0 (= b4) | 4.489 | 12.201 | 8.224 | 0 | 0 |
| F4: M5 fusion | λ1 0.5, penalty 1.5 | 3.752 | 10.632 | 7.084 | −1.140 [−1.291, −0.991] | −1.168 [−1.401, −0.943] |
| **L4: M5 + LODR** | **λ1 0.8, λ2 0.6, penalty 0** | **3.717** | **10.356** | **6.932** | **−1.292 [−1.482, −1.108]** | **−1.328 [−1.619, −1.053]** |
| T4: M5 + LODR at H42's ship weights | λ1 0.6, λ2 0.45, penalty 0 | 3.734 | 10.457 | 6.990 | −1.234 [−1.429, −1.046] | −1.268 [−1.570, −0.985] |

- **Decision:** L4's confirmation-half gain is 1.328 pp (−16.3% relative). That is more than
  0.17 pp, and the 95% interval lies below zero. **The LM branch continues.**
- **Size of the gain:**
  - Full dev with the LM: S / D / I 5,809 / 845 / 698 (7,352 edits, against 8,722 without).
  - The relative gain (−15.7% on full dev) matches H22ctl's (−16.1%); the better acoustic model
    kept the whole LM gain.
  - LODR adds 0.15 pp on top of fusion.
  - On the `ultra.wer` counts, the speaker-level bootstrap of L4 on the confirmation half is
    [−1.584, −1.031].
- **Python decode cost** (informational): sum of per-utterance search time with the reference
  beam, MKL torch, H0 / xbabe2.
  - b4: 275 s; L4: 411 s; T4: 563 s.
  - L4 needs 12.5 distinct LM contexts per audio second.

**Rust engine parity at H42's ship weights** (amendment 1; xbabe2; `results/item1_engine_parity.json`):

| full dev, beam 4 | no LM | + M5 + LODR (λ1 0.6, λ2 0.45) | Δ [95%] |
|---|---:|---:|---|
| engine INT8 (`ultra-engine-eval` a9b552e9…, M5 v1 pack) | 8.211 | 6.996 | −1.214 [−1.403, −1.029] |
| Python FP32 (`train.beam` + `NgramFusion`) | 8.224 | 6.990 | −1.234 [−1.429, −1.046] |

- The engine's LM change is within 0.02 pp of Python's (criterion ≤ 0.15 pp): **parity holds.**
- Identical token sequences, engine against Python: 5,143 / 5,567 without the LM and 5,138 / 5,567
  with it. The differences are INT8 against FP32, not the fusion.
- Engine cost (non-quiet xbabe2, decode thread): 0.213 → 0.247 CPU-s per audio-s; peak RSS
  224 → 686 MiB.

### Item 2: v1 / v2 pack parity (2026-11-23, 06:58Z): PASSED
730-utterance panel, `--qualifying`, beam 4, A1-49000 EMA INT8
(`results/item2_pack_parity.json`):

| LM | v1 → v2 bytes | identical tokens | identical text | errors v1 / v2 | peak RSS v1 → v2 |
|---|---|---:|---:|---:|---|
| M5 | 334,427,094 → 254,718,038 (−23.8%) | 730 / 730 | 730 / 730 | 936 / 936 | 686 → 586 MiB |
| S5 | 94,503,208 → 70,200,308 (−25.7%) | 730 / 730 | 730 / 730 | 964 / 964 | 410 → 410 MiB |
| S4 | 81,876,459 → 61,285,991 (−25.1%) | 730 / 730 | 730 / 730 | 960 / 960 | 402 → 401 MiB |

- v2 sha256: M5 `58ef726b…`, S5 `185381f1…`, S4 `3a9b6885…`.
- Panel WER without an LM is 7.316%. With M5 + LODR at the ship weights it is 6.478%
  (−0.84 pp).
