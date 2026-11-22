# JSR: ULTRA, an event-clocked, integer-exact streaming speech recogniser for CPUs

[![jankurai](https://github.com/neverhuman/jsr/actions/workflows/jankurai.yml/badge.svg)](https://github.com/neverhuman/jsr/actions/workflows/jankurai.yml)

This is a work in progress. Our own streaming transducer (R1) is training on LibriSpeech 960 h, and a pure-Rust CPU engine runs it. **Every number below is exploratory**: shared hosts, not a qualifying window. The qualifying protocol is `docs/ultra/FREEZE.md`. Current state: `docs/ultra/STATUS.md` and `docs/ultra/PLAN_TO_V1.md`. Research history: [`research/`](research/README.md) (waves H1…).

**New contributors and external agents: start with [`CONTRIBUTING.md`](CONTRIBUTING.md).** It covers the rules, how to get data, and the open workstreams.

The controlling specification is `docs/ultra/specs/ULTRA_ASR_MASTER_ENGINEERING_SPEC.md`. `docs/ultra/specs/cadence_ultra_v3_0_master_engineering_specification.md` is a companion, used for its AVX2 lookup-table kernels and the integer PCEN front-end arm.

## Snapshot (2026-11-22, exploratory)

- **Stream density, single harness** (H19, xbabe0 Zen 2; one run harness for every system; p99 ≤ 800 ms):

  | System | Streams per CCX (4 cores) | Per core | Its WER in the same run |
  |---|---:|---:|---|
  | **R1 multirate S_MR (ours), INT8 engine** | **144** | **36** | D2MR, step 50k |
  | R1 dense S (ours), INT8 engine, greedy and beam 4 | 96 | 24 | D2, average of 55/60/65k |
  | R0 Zipformer int8 (sherpa-onnx) | 32 | 8 | R0 (pretrained) |

  That's **4.5× the streams of Zipformer per CCX** (bracket 3.1–5.0× across failed rungs), plus 1.50× lower single-stream CPU in a counterbalanced ABAB test. WER for each row is measured in the same run on the 400 dev-clean panel utterances (see the paper's Fig. 1). The multirate model is still **10.8 pp worse than R0** (pooled, CI 9.9–11.7), so we are not yet competitive on accuracy. Paper draft: [`docs/paper/paper.pdf`](docs/paper/paper.pdf) (receipts in `docs/paper/data/raw/`); six-page checkpoint edition: [`docs/paper/jsr-cpu-asr.pdf`](docs/paper/jsr-cpu-asr.pdf).
- **Best accuracy so far:** D2 (dense S), an average of the snapshots at 60k/65k/70k with beam 4: **6.34% dev-clean / 15.89% dev-other** on the full dev sets (10.96% pooled). D2 continues to step 102k (at 76k, epoch 11, on 2026-11-22).
- **Production line:** D2c, a warm start from D2 using the pruned RNN-T loss, scores 6.56% on the 400-utterance dev-clean screen (EMA weights) after 2.5k A100 steps. It is moving to multi-GPU data-parallel training (exact global-batch sharding, native NCCL overlap) across the A100 nodes.
- **Training fleet:** 10 hosts: 3090, A10 and 6× A100 GPUs, an M3 Ultra and CPU nodes. Each GPU is capped at a 90% 5-minute mean by `ultra/fleet_guard.py` and the guard watchdog. A WebSocket dashboard (`just dashboard`) shows live loss, WER and audio-hours per hour.

The tables below are the earlier speed and accuracy record (2026-11-21). They are kept for traceability.

## Earlier results (2026-11-21, exploratory)

### Accuracy: D2
- **D2:** dense S (d_model 384, 16 Conformer blocks, streaming, 160 ms chunks), trained on LibriSpeech 960 h.
- **Latest full-DEV checkpoint:** step 70,000. Training continues to 102k steps. The paper draft cites the earlier step 60,000, **epoch 8** (0-indexed, i.e. the 9th pass over the data). Full curves: [`research/curves/`](research/curves/README.md).

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

## Quick start

```bash
just fast          # fmt, clippy, tests
just venv          # Python trainer env (python/ultra)
just required      # full push gate, including security and jankurai audit
```

## Layout

| Path | What |
|---|---|
| `crates/ultra-core` | numeric contract (§5), output events and ledger (§3), lifecycle, receipts |
| `crates/ultra-kernels` | scalar reference, AVX2 and AVX-512 ternary and int8 kernels, bit-exact property tests |
| `crates/ultra-engine` | pure-Rust R1 inference engine: FP32, per-row INT8, S/S_MR/ternary, AVX2 and NEON |
| `crates/ultra-frontend` | float log-mel control (trainer-parity, portable path) and integer Cochlea |
| `crates/ultra-gp` | R1 bundle loader, greedy RNN-T state, `(asr …)` program tools |
| `crates/ultra-model`, `crates/ultra-stream` | runtime backends (sherpa R0, ORT R1, engine R1), scheduler, grid simulation, placement |
| `crates/ultra-eval` | normaliser, WER, delays, paced replay, process-tree CPU accounting, bootstrap, `ultra-measure` |
| `crates/ultra-bench` | the cost-law benchmark (`wstream` port) and kernel bake-off |
| `crates/ultra-bridge` | whole-service measured benchmark harness (M1/M2) |
| `crates/ultra-adapters`, `crates/ultra-cli` | external recogniser adapters; command-line tool and `ultra bench-suite` |
| `python/ultra/train/{speed_pack,diloco}.py` | GPU input pipeline and DiLoCo multi-node outer optimiser (merged, not yet wired into training; see `CONTRIBUTING.md`) |
| `python/ultra` | PyTorch trainer (`train/`, including `train/ops/` run scripts), data prep, waves |
| `docs/ultra` | STATUS, PLAN_TO_V1, FREEZE, DECISIONS, RULINGS, ledgers, specs, reviews |
| `python/ultra/tools/fleet_dashboard` | real-time training dashboard (aiohttp + WebSocket), `just dashboard` |
| `docs/paper` | IEEE two-column paper: sources, generated metrics, committed PDFs (`just paper`) |
| `research/` | research waves H1…Hn: experiment, results and pointers (summaries only) |

## Gates

`just required` runs fmt, clippy (`-D warnings`), the tests, pytest, the security lane and a jankurai audit. It must pass before every push.

Data, checkpoints and build output never go in git. On the maintainers' fleet they live on the data drive (`/mnt/xbabe3-external/jsr`); elsewhere, set `JSR_DATA` (see `CONTRIBUTING.md`).
