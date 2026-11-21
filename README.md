# JSR: ULTRA, an event-clocked, integer-exact streaming speech recogniser for CPUs

[![jankurai](https://github.com/neverhuman/jsr/actions/workflows/jankurai.yml/badge.svg)](https://github.com/neverhuman/jsr/actions/workflows/jankurai.yml)

This is a work in progress. Our own streaming transducer (R1) is training on LibriSpeech 960 h, and a pure-Rust CPU engine runs it. **Every number below is exploratory**: shared hosts, not a qualifying window. The qualifying protocol is `docs/ultra/FREEZE.md`. Current state: `docs/ultra/STATUS.md` and `docs/ultra/PLAN_TO_V1.md`. Research history: [`research/`](research/README.md) (waves H1…).

The controlling specification is `docs/ultra/specs/ULTRA_ASR_MASTER_ENGINEERING_SPEC.md`. `docs/ultra/specs/cadence_ultra_v3_0_master_engineering_specification.md` is a companion, used for its AVX2 lookup-table kernels and the integer PCEN front-end arm.

## Latest results (2026-11-21, exploratory)

### Accuracy: our best R1 so far
- **D2:** dense S (d_model 384, 16 Conformer blocks, streaming, 160 ms chunks), trained on LibriSpeech 960 h.
- **Best checkpoint:** step 50,000, **epoch 7** (0-indexed, i.e. the 8th pass over the data). Training continues to 102k steps.

| Set (400 utterances, greedy, streaming) | WER |
|---|---:|
| dev-clean | **8.52%** |
| dev-other | **19.60%** |

- **Other runs:**
  - multirate S_MR (D2MR): 11.3% dev-clean at step 45k, epoch 4;
  - S_MR with int8 stem (D3MRb): 15.0% at step 20k, epoch 2 (stopped by the divergence guard; fixed as D3MRc);
  - ternary S_MR (R4T, 100 h): 17.98%;
  - larger M_MR (D4M): early.
- **Reference points:** R0 Zipformer (icefall, pretrained) scores 2.85% dev-clean, and Nemotron 0.6B 2.16% on the same DEV400. **We are not yet competitive on accuracy.**

### Speed: whole-service CPU cost
- **Same build, same session, same DEV400 audio,** observer on a separate CCX.
- **Host:** H0, AMD TR PRO 3995WX (Zen 2, AVX2).
- **Metric:** CPU-seconds per audio-second; lower is better.

| System | Path | WER on this run | CPU-s/s | Peak RSS |
|---|---|---:|---:|---:|
| **R1 (ours), ultra-engine INT8** | pure Rust, prepared int8, per-row activations | 21.80%* | **0.041** | 259 MB |
| R0 Zipformer int8 (alternative) | sherpa-onnx | 2.85% | 0.078 | 239 MB |
| R1 (ours), ultra-engine FP32 | pure Rust | 21.88%* | 0.104 | 478 MB |
| R1 (ours), ONNX Runtime FP32 | ORT | 21.88%* | 0.175 | 776 MB |
| Nemotron 0.6B Q8_0 (alternative) | NeMo-Speech.cpp, child process | 2.16% | 0.472† | 1.28 GB |

- \* That speed run used the early D2 snapshot (step 5,000). Cost per audio-second does not depend on the checkpoint's accuracy. The current checkpoint is 8.52% (above).
- † A different cost scope (cold, child process including model load), so it is not directly comparable.
- **Roughly,** our engine runs R1 at **about 1.9× less CPU than R0** and **about 4× less than ONNX Runtime** running the same model.

**Single stream, engine INT8:**

| Host | Dense S | Multirate S_MR | Per core |
|---|---:|---:|---|
| Zen 2 (H0) | 0.040 CPU-s/s | — | — |
| Apple M3 Ultra (xbabe4) | 0.0132 | 0.0105 | about 60 (S) / 84 (S_MR) live streams per core in our engine dev tool |

**Stream density:** not yet claimed. The cross-harness density comparison was retracted after an independent review (`research/H10…`). The headline density ratio will come only from the single-harness qualifying run in FREEZE §4.

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
| `python/ultra` | PyTorch trainer (`train/`, including `train/ops/` run scripts), data prep, waves |
| `docs/ultra` | STATUS, PLAN_TO_V1, FREEZE, DECISIONS, RULINGS, ledgers, specs, reviews |
| `research/` | research waves H1…Hn: experiment, results and pointers (summaries only) |

## Gates

`just required` runs fmt, clippy (`-D warnings`), the tests, pytest, the security lane and a jankurai audit. It must pass before every push.

Data, checkpoints and build output live on the data drive (`/mnt/xbabe3-external/jsr`), never in git.
