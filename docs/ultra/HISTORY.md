# ULTRA results history (exploratory)

Earlier headline sections of the top-level README (moved here on 2026-11-23), kept as they were for
traceability; only their relative links were adjusted to this folder. They describe older checkpoints,
harnesses and hosts. Current numbers are in the README's generated block and in the paper
([`docs/paper/paper.pdf`](../paper/paper.pdf)), both regenerated from `docs/paper/data/`.

## Snapshot (2026-11-22, exploratory)

- **Stream density, single harness** (H19, xbabe0 Zen 2; one run harness for every system; p99 ≤ 800 ms):

  | System | Checkpoint | Streams per CCX (4 cores) | Per core | DEV-clean panel WER |
  |---|---|---:|---:|---:|
  | **R1 multirate S_MR, prepared INT8** | D2MR, step 50k | **144** | **36** | 11.561% greedy |
  | R1 dense S, prepared INT8 | D2 average, steps 55k/60k/65k | 96 | 24 | 7.444% greedy, 6.962% beam 4 |
  | R0 Zipformer INT8 (sherpa-onnx) | Pretrained | 32 | 8 | 2.952% |

  Multirate R1 passed the 20-second ladder rung at **4.5× as many streams per CCX** as Zipformer (bracket 3.1–5.0× across failed rungs). The WER column uses a separate 400-utterance DEV-clean run with the same benchmark harness. On the pooled 730-utterance DEV panel, multirate R1 scored 15.93% WER against 5.11% for Zipformer, a **10.8 percentage-point gap** (95% CI 9.9–11.7). In the eight-block single-stream CPU comparison, multirate D2MR-50k used **0.629×** the CPU of Zipformer and dense D2-45k used **0.667×**. See [H19](../../research/H19_single_harness_rehearsal/README.md). Paper draft: [`docs/paper/paper.pdf`](../paper/paper.pdf) (receipts in `docs/paper/data/raw/`); six-page checkpoint edition: [`docs/paper/jsr-cpu-asr.pdf`](../paper/jsr-cpu-asr.pdf).
- **Best accuracy so far:** D2 (dense S), an average of the snapshots at 60k/65k/70k with beam 4: **6.34% dev-clean / 15.89% dev-other** on the full dev sets (10.96% pooled). D2 continues to step 102k (at 76k, epoch 11, on 2026-11-22).
- **Production line:** D2c, a warm start from D2 using the pruned RNN-T loss, scores 6.56% on the 400-utterance dev-clean screen (EMA weights) after 2.5k A100 steps. It is moving to multi-GPU data-parallel training (exact global-batch sharding, native NCCL overlap) across the A100 nodes.
- **Training fleet:** 10 hosts: 3090, A10 and 6× A100 GPUs, an M3 Ultra and CPU nodes. Each GPU is capped at a 90% 5-minute mean by `ultra/fleet_guard.py` and the guard watchdog. A WebSocket dashboard (`just dashboard`) shows live loss, WER and audio-hours per hour.

The tables below are the earlier speed and accuracy record (2026-11-21). They are kept for traceability.

## Earlier results (2026-11-21, exploratory)

### Accuracy: D2
- **D2:** dense S (d_model 384, 16 Conformer blocks, streaming, 160 ms chunks), trained on LibriSpeech 960 h.
- **Latest full-DEV checkpoint:** step 70,000. Training continues to 102k steps. The paper draft cites the earlier step 60,000, **epoch 8** (0-indexed, i.e. the 9th pass over the data). Full curves: [`research/curves/`](../../research/curves/README.md).

| Full dev sets (2,703 / 2,864 utterances), streaming, step 70k | dev-clean | dev-other |
|---|---:|---:|
| single checkpoint, greedy | 7.10% | 17.26% |
| single checkpoint, beam 4 | 6.61% | 16.16% |
| **average of 60k/65k/70k, beam 4** | **6.34%** | **15.89%** |

Beam search and checkpoint averaging need no retraining (H12). The Rust engine's beam 4 is token-identical to the Python reference (H16); the CPU timing rows below use greedy decoding.

- **Other runs:**
  - multirate S_MR (D2MR, final 50k, epoch 5): 11.09% full dev-clean / 22.90% full dev-other;
  - S_MR with int8 stem (D3MRb): 15.0% at step 20k, epoch 2 (stopped by the divergence guard; fixed as D3MRc);
  - ternary S_MR (R4T, 100 h): 17.98%;
  - larger M_MR (D4M): early.
- **Reference points:** R0 Zipformer (icefall, pretrained) scores 2.85% dev-clean, and Nemotron 0.6B 2.16% on the same DEV400. **We are not yet competitive on accuracy.**

### Speed: whole-service CPU cost
- **Same build, same session, same DEV400 audio,** observer on a separate CCX.
- **Host:** H0, AMD TR PRO 3995WX (Zen 2, AVX2).
- **Metric:** CPU-seconds per audio-second; lower is better.

**Run A: same retained executable** (`docs/ultra/reviews/2026-11-21-fused-d2-cpu.md`, build 6df5c8f). 400 dev-clean utterances, eight unpaced streams, one inference thread, 20 ms packets. The R1 rows are the early D2 **step-5,000** checkpoint.

| System | Path | WER on this run | CPU-s/s | Sampled peak RSS |
|---|---|---:|---:|---:|
| **R1 (ours), ultra-engine prepared INT8** | pure Rust, per-row activations | 21.80%* | **0.0413** | 646 MB |
| R0 Zipformer int8 (alternative) | sherpa-onnx | 2.85% | 0.0782 | 239 MB |
| R1 (ours), ultra-engine FP32 | pure Rust | 21.88%* | 0.1036 | 478 MB |

**Run B: FP32 runtime control, same executable, same checkpoint** (same review). This is a separate run:

| System | CPU-s/s | Sampled peak RSS | Transcripts |
|---|---:|---:|---|
| R1 ONNX Runtime FP32 | 0.1747 | 776 MB | 400/400 identical |
| R1 ultra-engine fused FP32 | 0.0947 | 478 MB | 400/400 identical |

**Reference:** Nemotron 0.6B Q8_0 (NeMo-Speech.cpp, child process) scored 2.16% WER at 0.472 CPU-s/s and 1.28 GB on the same DEV400. That's a different cost scope (cold, includes model load), so it's not directly comparable.

- \* Cost per audio-second does not depend on the checkpoint's accuracy. The current checkpoint is far more accurate (above). Speed and WER for the *current* checkpoints come from the qualifying run.
- **Roughly:**
  - in Run A, engine INT8 uses **about 1.9× less CPU than R0**, but more memory (646 vs 239 MB);
  - in Run B, the fused FP32 engine uses **1.85× less CPU than ONNX Runtime** on the same model, with identical transcripts.
  - The two runs are not combined into a single INT8-vs-ORT ratio.

**Single stream, engine INT8:**

| Host | Dense S | Multirate S_MR | Per core |
|---|---:|---:|---|
| Zen 2 (H0) | 0.040 CPU-s/s | — | — |
| Apple M3 Ultra (xbabe4) | 0.0132 | 0.0105 | about 60 (S) / 84 (S_MR) live streams per core in our engine dev tool |

**Stream density:** the earlier cross-harness density comparison was retracted after an independent review (`research/H10…`). The density figures in the snapshot above come from the single-harness rehearsal (H19). The qualifying number comes only from the FREEZE §4 run.

### Caveats
- Exploratory, shared hosts: a same-day A/A replay varied by up to +25%.
- LibriSpeech test sets were read before the freeze (disclosed). Selection uses DEV only.
- Full receipts: `docs/ultra/reviews/2026-11-21-fused-d2-cpu.md` and `2026-11-21-service-cpu-control.md`; independent reviews in `docs/ultra/reviews/`.
