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
| [H6](H6_pure_rust_engine/README.md) | Pure-Rust inference engine (x86 + ARM) | EXPLORATORY | Engine INT8 0.041 vs ORT FP32 0.175 CPU-s/s, same transcripts; bit-identical across ISAs |
| [H7](H7_multirate_S_MR/README.md) | Multirate encoder (S_MR) vs dense | EXPLORATORY | Same or better WER (18.0% vs 29.8% at 100 h); about 25% less CPU per stream |
| [H8](H8_ternary_R4/README.md) | Ternary (R4) QAT pilot | EXPLORATORY | −0.6 WER cost, 14 MiB weights; no faster than INT8 at batch (compute-bound) |
| [H9](H9_960h_training_and_divergence/README.md) | 960 h training, divergence and the qkv cap | ongoing | D2 dense 9.99% dev-clean; D3 runs diverged (qkv runaway) → cap + guards |
| [H10](H10_benchmark_methodology_review/README.md) | Independent methodology review → FREEZE | ongoing | 7 blocking findings; the cross-harness density claim was retracted; FREEZE rewritten |
| [H11](H11_cross_isa_portable_logmel/README.md) | Cross-ISA bit-exact portable log-mel (G4) | EXPLORATORY | **7,300/7,300 (100%)** identical PCM→tokens x86 AVX2 == M3 NEON across 5 models × FP32/INT8 (R4T 636→730/730); 5/5 INT8 gates PASS |
| [H12](H12_accuracy_diagnostics/README.md) | Accuracy diagnostics: MR padding fix, D2 error sources | EXPLORATORY | MR pad-before-pair fixed; D2MR impact ≤0.03 pp (no fine-tune). D2: beam 4 −0.65/−1.63 pp (search error); untrained chunk 16 −0.39/−0.90; no length or final-word effect; last-3 average −0.41/−1.64 (panel); k2 pruned loss exact vs ours (≤1e-13), 10× faster loss on CPU |
| [H17](H17_spec_corrections/README.md) | Spec claims our data contradicts; 3 qualification bugs (grok-lane) | note | Double fleet offset, grid error spin (runtime), ternary+act7 width switch (engine); trainer resume/CMVN facts |

Reserved: H13 recipe parity, H14 dual-mode, H15 engine speed, H16 R0 distillation (owner sign-off). Next free: **H18**.
