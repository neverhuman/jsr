# H48d: speculative onset peek (design note for Codex and runtime-lane)

**Status:** design only, 2026-11-23. Nothing is implemented. Engine changes belong to Codex (H44 engine owner). Runtime-lane owns the hub and the event contract. Author: h48-latency.

**Goal (plan, H48d):**
- Show a *provisional* first token 40–60 ms earlier at 160 ms chunks, measured on the end-of-first-word clock;
- retract at most 25% of previews;
- leave **committed text bit-for-bit unchanged**.

## 1. Where the wait comes from

**Chunk timing.** Chunk `c` covers encoder frames `[Cc, Cc + C − 1]` (`C` = 4 at 160 ms). It can run only once log-mel frame `4(Cc + C − 1)` is complete. That frame ends at sample `640(Cc + C − 1) + 400`, so:
- the chunk closes at **`40·C·(c + 1) − 15` ms** of audio (145 ms for the first 160 ms chunk);
- a token the model emits at frame `t` is published at the close of chunk `t // C`, plus compute.

**Mean wait.** For a token with a fixed emission frame, the wait for its chunk to close averages `(C − 1)/2` frames: 60 ms at `C = 4` and 20 ms at `C = 2`.

**Two levers, which add:**
- Training (H48b / H48c) moves the emission frame earlier.
- The peek removes the wait for the chunk to close, for provisional text only.

**The peek.** At a frame boundary `f ∈ {1, 2, 3}` frames into the open chunk, as soon as the audio for those frames exists, the engine:
1. runs the encoder on the `f` available frames as a partial chunk, reading the committed caches;
2. decodes greedily from the committed decoder state;
3. publishes the result as a **preview**.

The committed step for the full chunk runs later, exactly as today.

## 2. The invariant: committed text unchanged

- **Signature.** `EngineStream::peek(&self, pcm_tail: &[i16]) -> Option<Preview>` takes an *immutable* borrow.
  - It reads the committed state:
    - the stem conv cache (2 log-mel frames);
    - per-layer K/V (the 128-frame left context);
    - per-layer depthwise-conv caches (`kernel − 1` frames);
    - the predictor context (2 tokens);
    - the best committed hypothesis.
  - It writes only scratch that the call owns: K/V for ≤ 3 frames per layer, conv outputs, joiner logits and preview tokens.
  - No `Cell` / `RefCell` / `Mutex` may be reachable from `peek`. With that, the borrow checker makes mutating committed state impossible.
- **Witness tests (fail first):**
  - The committed event stream with peeks on equals the stream with peeks off, byte for byte. Checked on full dev at `C ∈ {2, 4}`, greedy and beam 4, and on a seeded random-PCM property test.
  - A test that peeks after every packet still yields the same committed stream.
- **Decoding rules:**
  - **Beam:** the peek decodes greedily from the best committed hypothesis only. It never expands or reorders the beam.
  - **LM:** the peek uses the same fusion as committed decoding, or none; the receipt records which.

## 3. Event contract (runtime-lane)

- **New event:** `preview {seq, base_commit_seq, through_sample, text}`.
  - `text` is the committed text up to `base_commit_seq` plus the preview suffix.
  - `through_sample` is the last sample the peek's log-mel frames consumed.
- **Client rules:**
  - Each preview replaces the previous one.
  - Any committed partial with `seq > base_commit_seq` supersedes every older preview.
  - Final and immutable events are unchanged.
- **Confirmed vs retracted.** A non-empty preview suffix is *confirmed* if, within the next 2 committed steps (≤ 2 chunks), the committed text gains that suffix as a prefix of its new text. Otherwise it is *retracted*. Empty previews are neither.
- **Scoring (§9 of the final engineering plan):**
  - Previews are provisional text and are scored separately from committed text.
  - A retracted preview never counts as a correct word, for first-correct-word or for correct-by-deadline.
  - Revision burden counts retracted preview characters.

## 4. When to peek (bounding the cost)

- **Only in the onset window:** from a speech onset until the segment's first committed non-blank token, and for at most 4 chunks.
- **Onset sources:**
  - **(a) The H47 wake gate's Listening → Waking transition.**
    - The wake replay already processes the 1 s pre-roll in one batched pass (H47 one-pass replay).
    - The trailing partial chunk of that replay is the natural first peek.
  - **(b) In continuous mode, a cue that costs nothing.** For example, the blank log-prob at the last committed frame falls below a threshold, or the CTC head's non-blank mass rises.
    - Both are already computed.
    - The threshold is set on the TRAIN split, never on dev.
- **Cadence:** at most one peek per 40 ms frame boundary.
  - `f ∈ {1, 2, 3}` for models trained with 1-frame chunks (the arm-10 family `{1, 2, 4, 8, 16}`).
  - `f = 2` only for the `{2, 4, 8, 16}` family (H48b / H48c).
  - A1 / A1L were trained at `{4, 8, 16}`, so a partial chunk is out of distribution for them. Measure it before relying on it.
- **Never in non-speech.** Silence and noise blocks of the standard always-on benchmark must show zero peeks beyond those inside a wake event.

## 5. Cost model

**Measured basis:**
- **runtime-lane's matched profile (chat 04:37Z):** a step's cost is dominated by streaming the 56.5 MB INT8 weights from DRAM once per step (88–91% of the step), so it barely depends on the frame count.
- **bench-open's quiet-xbabe0 rows (TIMING 14br / 14d, chat 06:43Z; one core, greedy, cold):**

| Step size | CPU-s per audio-s | Steps per s | ms per step | Core draw while running |
|---|---:|---:|---:|---:|
| 4 frames (160 ms) | 0.0351 | 6.25 | 5.6 | ~10 W |
| 2 frames (80 ms) | 0.0605 | 12.5 | 4.8 | – |
| 1 frame (40 ms) | 0.1188 | 25 | 4.75 | ~6 W |

**So a peek of `f ≤ 3` frames costs about 5 ms of CPU, ~85–90% of a committed step.** Its energy is ~0.5–0.6× a committed step's, because one-row steps wait on memory and draw less. The greedy decode of ≤ 3 frames is negligible.

**Peeks per onset.** `K` = peeks per onset ≈ `W × p`:
- `W` = chunks from the onset to the first committed token. With a first word of ~300 ms and a ~200 ms first-token delay, that is about 3 at `C = 4`.
- `p` = peeks per chunk: 3 at every boundary, or 1 at mid-chunk.

**Added CPU ≈ `r_on × K × 5 ms`**, compared with B0's 0.0351 CPU-s/s of continuous speech at `C = 4`:

| Onsets per min | K | Added CPU-s per audio-s | vs continuous speech (100% duty) | vs a sleep-gated 15%-duty stream (active 0.0053 CPU-s/s) |
|---:|---:|---:|---:|---:|
| 1 | 3 | 0.00025 | +0.7% | +5% |
| 6 | 3 | 0.0015 | +4.3% | +28% |
| 6 | 9 | 0.0045 | +12.8% | +85% |

Energy overheads are about 0.6× these CPU figures.

**Reading the table:**
- In always-on mode the peek is affordable only with trigger (b) or (a) and `K ≤ 3`.
- **Multi-stream servers are different:** concurrent streams' peeks share one weight pass in a batched step. The 8-stream batched figure at chunk 1 was 0.136 CPU-s per stream-second against 1.9 unbatched (both on loaded H0), so batched peeks cost little more than their arithmetic.
- Report energy paced (`uclamp`) and unpaced, on quiet xbabe0 or the owner's metered device.

**Expected gain = P(confirmed) × E[saving].**
- With a peek at every boundary, `E[saving] ≈ 40·(C − 1)/2 − 5 ≈ 55 ms` at `C = 4`.
- With one mid-chunk peek (`f = 2`):
  - tokens emitted in the first two frames of the chunk (half of them) gain ~80 ms;
  - the mean gain is ≈ 40 ms.
- Retracted previews gain nothing and add flicker.
- Meeting the plan's −40 to −60 ms needs P(confirmed) ≥ 0.75 and peeks at most boundaries.

## 6. Offline probe first (hours, no engine change)

**The PyTorch model reproduces a peek exactly:**
1. Truncate the log-mel input to the frames available at the peek: log-mel frame `4(t0 + f − 1)` is the last one encoder frame `t0 + f − 1` needs.
2. Run the standard chunked forward at chunk `C`.
   - The last chunk is then partial (`f` frames).
   - By the causality contract in `train/model.py`, every frame before `t0` is bit-identical to the committed computation.
3. Greedy decoding over the truncated output gives the committed tokens before `t0` plus the preview tokens.

**Procedure.** For each dev utterance and each peek position in its onset window, one truncated forward yields the preview. Compare it with the full decode to get:
- the retraction rate;
- preview latency on both clocks;
- first-correct-word and correct-by-150/300/500 ms (provisional), with coverage.

Break the results down by chunk size, `f`, and model (A1L, H48b, H48c, arm 10). That is ~10 truncated forwards × 5,567 utterances: minutes on one GPU. The latency clocks use eval-lane's H43 metric code; dev MFA timings are used for evaluation only.

**The engine path (Codex) starts only if the probe passes K1 and K3 below.**

## 7. Pre-registered kill rules

| Rule | Condition | Action |
|---|---|---|
| K1 | Retracted > 25% of non-empty previews on full dev at the target chunk (probe or engine) | Stop |
| K2 | Committed text differs from peeks-off on any utterance | Blocking defect; nothing ships until fixed |
| K3 | Mean first-visible-preview gain < 20 ms (end-of-first-word clock) at 160 ms | Stop: not worth the cost |
| K4 | CPU overhead > +10% of continuous speech compute at 100% duty, or any peek outside a wake / onset window on non-speech | Stop, or tighten the trigger and re-run |
| K5 | First-correct-word p50 (provisional) or correct-by-300 ms (provisional) not improved | Stop |

**Pass (plan gate):**
- −40 to −60 ms mean first visible preview at 160 ms, end-of-first-word clock;
- ≤ 25% retracted;
- committed text unchanged;
- plus the controlling latency rules: lower median, paired p95 ≤ +20 ms, first-word / coverage bounds within 0.50 pp.

## 8. Relation to the other latency levers

- **H48b / H48c (training).** Earlier committed emission. It also helps the peek: a peek can only show what the model has already decided on the available frames.
- **The trained dense 40 / 40 / 80 → 160 ms ramp (controlling plan).**
  - It changes committed computation: smaller committed chunks after an onset. That changes committed text, so it needs its own WER qualification.
  - It costs ~3 steps for the first 160 ms of audio instead of 1, because each step costs about the same.
  - The peek keeps committed text and spends steps only on previews in the onset window.
  - The two combine: peek during the ramp's 160 ms phase.
- **Dual-clock preview (§8.2).** A trained fast head would give more accurate previews, but it is a model change.
