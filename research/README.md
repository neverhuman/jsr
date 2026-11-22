# Research waves (H1, H2, …)

This is a chronological record of the experiments behind JSR/ULTRA: what was tried, why, what was measured, and what it changed. It is written so that a future agent or outside reviewer can rebuild the reasoning without reading thousands of chat lines.

## Convention
- **One directory per wave:** `research/H<N>_<short-slug>/README.md`. N increments and is never reused.
- **A new wave starts** whenever a question gets its own experiment (training run, benchmark, kernel study, review).
- **Each README holds:**
  1. **Question**: what we wanted to know.
  2. **Setup**: code commits, data, hosts, settings.
  3. **Results**: the numbers, with labels. *Exploratory* means not from a qualifying window.
  4. **What went wrong or surprised us**: failed attempts stay visible.
  5. **Decision / consequence**: what changed in the plan or code.
  6. **Pointers**: commits, docs, reviews, and data-drive paths for large artifacts.
- **What may be committed:** small summaries only (tables, small JSON). **No models, checkpoints, audio or large logs.** Those stay on `/mnt/xbabe3-external/jsr/...` and are referenced by path and sha256.
- **Status labels:** `EXPLORATORY` (shared host, not pre-registered), `QUALIFYING` (FREEZE protocol, quiet host), `FAILED` (kept on purpose), `SUPERSEDED`.
- **Where the authorities live:** `docs/ultra/` holds the plans, freeze, decisions and ledgers (`EXPERIMENTS.jsonl`, `LOOKS.jsonl`). `docs/ultra/reviews/` holds the independent reviews. Waves summarise and link to them; they don't replace them.

## Index

| Wave | Topic | Status | Main result |
|---|---|---|---|
| [H1](H1_stream_density_R0/README.md) | Stream density of the pretrained Zipformer (R0); SP-A batching | EXPLORATORY | 192 live streams machine-wide; batching lift 1.49× (NO_LIFT); L3 per CCX is the limit |
| [H2](H2_baselines_nemotron_contrast/README.md) | Baselines: Nemotron 0.6B, Moonshine, Whisper | EXPLORATORY | Nemotron 2.6% test-clean at ~0.45–0.7 CPU-s/s; cost law W01 PASS |
| [H3](H3_r1_first_training/README.md) | First R1 training (100 h): collapse → fix | FAILED → fixed | Post-norm collapse; SpecAugment plateau; delayed SpecAugment → 29.8% |
| [H4](H4_int8_batch_invariance/README.md) | INT8 export and batch invariance | resolved | Dynamic INT8 output depends on co-batched streams; static encoder-only INT8 +0.066 |
| [H5](H5_whole_service_cpu_benchmark/README.md) | Whole-service CPU benchmark and measurement hygiene | EXPLORATORY | Observer placement inflated CPU; A/A replay ±25% on a shared host |
| [H6](H6_pure_rust_engine/README.md) | Pure-Rust inference engine (x86 + ARM) | EXPLORATORY | Same-weight FP32 control: Rust 0.0947 vs ORT 0.1747 CPU-s/audio-s, 400/400 identical hypotheses. Prepared INT8 used 0.0413 in a separate run. |
| [H7](H7_multirate_S_MR/README.md) | Multirate encoder (S_MR) vs dense | EXPLORATORY | Same or better WER (18.0% vs 29.8% at 100 h); about 25% less CPU per stream |
| [H8](H8_ternary_R4/README.md) | Ternary (R4) QAT pilot | EXPLORATORY | −0.6 WER cost, 14 MiB weights; no faster than INT8 at batch (compute-bound) |
| [H9](H9_960h_training_and_divergence/README.md) | 960 h training, divergence and the qkv cap | ongoing | D2 dense 9.99% dev-clean; D3 runs diverged (qkv runaway) → cap + guards |
| [H10](H10_benchmark_methodology_review/README.md) | Independent methodology review → FREEZE | ongoing | 7 blocking findings; the cross-harness density claim was retracted; FREEZE rewritten |
| [H11](H11_cross_isa_portable_logmel/README.md) | Cross-ISA bit-exact portable log-mel (G4) | EXPLORATORY | **7,300/7,300 (100%)** identical PCM→tokens x86 AVX2 == M3 NEON across 5 models × FP32/INT8 (R4T 636→730/730); 5/5 INT8 gates PASS |
| [H12](H12_accuracy_diagnostics/README.md) | Accuracy diagnostics: MR padding fix, D2 error sources, DEV decode point | EXPLORATORY | MR pad-before-pair fixed; D2MR impact ≤0.03 pp. D2 beam 4 −0.65/−1.63 pp; last-3 average + beam 4 stack (full dev 7.003/16.603). **DEV sweep → freeze beam 4, no blank penalty** (D2 avg-3 panel 7.258/15.941; D2MR 10.126/20.312). k2 pruned loss exact vs ours |
| [H13](H13_vocab_and_pruned_loss/README.md) | SentencePiece-512 (both heads) and the pruned RNN-T loss | ONGOING | H13a (960 h, 512 vocab, exact loss, EMA) training on xbabe2: dev-clean400 20.98 at 10k. Pruned option integrated and parity-tested. H13b–d matched 100 h arms queued for one GPU (CPU route < 1 audio-h/h) |
| [H15](H15_engine_non_gemm_and_kernels/README.md) | Engine non-GEMM cuts and kernel ideas from the six specs | EXPLORATORY | Bit-identical RoPE table + fused norm: attention −3–10%, total within noise; unsigned-offset 7-bit tile FAILS on Zen 2 (0.82–0.94×); S_MR padding safe |
| [H16](H16_engine_beam_search/README.md) | Bounded beam search in the Rust engine (B6); D2 avg-3 INT8 gate; left-context sweep | EXPLORATORY | Beam 4 = beam.py 730/730 token-identical, −1.43 pp on D2MR panel, INT8 gate PASS; left context 64 costs +0.09 pp |
| [H17](H17_spec_corrections/README.md) | Spec claims our data contradicts; 3 qualification bugs (grok-lane) | note | Double fleet offset, grid error spin (runtime), ternary+act7 width switch (engine); trainer resume/CMVN facts |
| [H18](H18_ieee_paper_draft/README.md) | IEEE paper draft (docs/paper), every number sourced and generated | draft | Abstract leads with the WER gap; pairing errors in repo docs caught and fixed |
| [H19](H19_single_harness_rehearsal/README.md) | Single-harness rehearsal: R0 vs dense vs MR, per-CCX ladder + 8-block ABAB | EXPLORATORY | Density 32 / 96–112 / 144 per CCX (MR 4.5×, reproduced 3×); single-stream CPU 1.50× / 1.59× lower (p<1e-5); WER gap stated |
| [H20](H20_matched_cpu_frontier/README.md) | Common-boundary streaming ASR CPU and DEV WER | EXPLORATORY, ongoing | The first 730-utterance repetition has verified timing and retained hypotheses. Three-repeat rotation underway. |

Reserved: H14 dual-mode / chunk-size sampling. Next free: **H21**.
