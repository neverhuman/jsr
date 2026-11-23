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
| [H13](H13_vocab_and_pruned_loss/README.md) | SentencePiece-512 (both heads) and the pruned RNN-T loss | ONGOING | Pruned loss 3.8× faster per step than exact at V=4097 (A100); 100 h arms at 10.8k steps: pruned-512 23.0 vs pruned-4096 27.1 dev-clean400; exact-4096 behind on steps; H13a (960 h, 512) training |
| [H14](H14_chunk2_finetune/README.md) | **80 ms attention chunk:** 4k-step fine-tunes from D2e final, chunk 2 vs chunk 4 control | EXPLORATORY (written after results) | Chunk 2 costs +0.26 pp raw / +0.22 pp EMA on dev-clean400; latency measurement open |
| [H15](H15_engine_non_gemm_and_kernels/README.md) | Engine non-GEMM cuts and kernel ideas from the six specs | EXPLORATORY | Bit-identical RoPE table + fused norm: attention −3–10%, total within noise; unsigned-offset 7-bit tile FAILS on Zen 2 (0.82–0.94×); S_MR padding safe |
| [H16](H16_engine_beam_search/README.md) | Bounded beam search in the Rust engine (B6); D2 avg-3 INT8 gate; left-context sweep | EXPLORATORY | Beam 4 = beam.py 730/730 token-identical, −1.43 pp on D2MR panel, INT8 gate PASS; left context 64 costs +0.09 pp |
| [H17](H17_spec_corrections/README.md) | Spec claims our data contradicts; 3 qualification bugs (grok-lane) | note | Double fleet offset, grid error spin (runtime), ternary+act7 width switch (engine); trainer resume/CMVN facts |
| [H18](H18_ieee_paper_draft/README.md) | IEEE paper draft (docs/paper), every number sourced and generated | draft | Abstract leads with the WER gap; pairing errors in repo docs caught and fixed |
| [H19](H19_single_harness_rehearsal/README.md) | Single-harness rehearsal: R0 vs dense vs MR, per-CCX ladder + 8-block ABAB | EXPLORATORY | Density 32 / 96–112 / 144 per CCX (MR 4.5×, reproduced 3×); single-stream CPU 1.50× / 1.59× lower (p<1e-5); WER gap stated |
| [H20](H20_matched_cpu_frontier/README.md) | Common-boundary streaming ASR CPU and DEV WER | EXPLORATORY, ongoing | The first 730-utterance repetition has verified timing and retained hypotheses. Three-repeat rotation underway. |
| [H21](H21_long_final_recipe/README.md) | Long final-recipe runs, ~30 epochs of 960 h (L1 dense S, L2 S_MR, L3 M_MR) | ONGOING (pre-registered) | SPM-512, pruned loss, EMA, speed perturbation, qkv cap; process loader (bit-identical batches) lifted A100 runs to 2.5–3.2k audio-h/h |
| [H22](H22_r0_distillation/README.md) | Distillation from R0 (owner-approved; TRAIN only; labelled "distilled from R0") | PRE-REGISTERED | R0 teacher cache (top-8 joiner log-probs per frame, 3 speeds) building on xbabe0; H22-ctl running; H22-kd after the cache |
| [H23](H23_optimizer_bakeoff/README.md) | Optimiser bake-off at matched 100 h exposure (AdamW vs ScaledAdam+Eden vs Schedule-Free AdamW vs Muon; AdamW + adaptive clipping side arm) | ONGOING | Code in train/optim; H23a AdamW 16.20% @ 24k, H23b ScaledAdam diverged @ 4.1k, H23c–e running on xbabe6 |
| [H24](H24_codebook_scan_and_fwht_kernels/README.md) | **JSR v2 Kernel Gate:** 4-bit gain–shape codebook-scan (`vpshufb`/`vqtbl1q`) + `FWHT16` + `ReLU²` sparse scan on Zen 2 AVX2 & M3 Ultra NEON | EXPLORATORY (**PASS**) | **Zen 2 PASS:** `384→1536` **2.68×** (`M=4`), **3.36×** (`M=1`), **3.75×** (`80%` sparse); `768→3072` **3.01×** (`4.69×` sparse); `3072→768` **2.60×** (`3.70×` sparse); `FWHT16` **483 ns** (`<0.1%`); 0-ULP scalar==AVX2==NEON |
| [H25](H25_runtime_hygiene_and_kv_compression/README.md) | **JSR v2 Engine Hygiene:** zero-alloc `pool.rs`/`layers.rs`, `beam.rs` prefix arena + tropical `logsumexp`, `M=1/2` tiles, INT8 K/V cache (3.5× state cut), H17 fixes | PRE-REGISTERED | Target `≥ 10%` exact whole-service CPU drop + `6.70 → 1.92 MiB/stream` INT8 KV (`≤ +0.05 pp`) |
| [H26](H26_kd_audit_smbr_and_cr_ctc/README.md) | **JSR v2 Accuracy (0 inference cost):** H22 KD exact-path & delay audit + sMBR top-8 sequence KD + CR-CTC auxiliary loss | PRE-REGISTERED | Gate: `≥ 5%` rel `dev-other` WER cut per component at matched GPU hours |
| H27 | **JSR v2 Horizon Ladder:** deterministic `Preview`/`Commit`/`Final` tiers, Speculative Finalization (`<100 ms`), `SU-Live`/`SU-File` tracks, H14/MCR-RNNT 3-mask training | PLANNED | Target `≥ 15%` rel Commit WER cut vs Preview (`≤ 0.35` CPU-s/s) |
| H28 | **JSR v2 Model Family (`V2-S/M/L`):** `25→12.5→6.25 Hz` 0-lookahead U-Net + GQA (`6→2`/`8→2`) + `FWHT16` + `ReLU²` + Pruned TDT + matched causal Zipformer control | PLANNED | Gate: `6.25 Hz` within `+0.3 pp` at `≤ 70%` MACs; TDT `≥ 2×` faster beam-4/8 |
| H29 | **JSR v2 Codebook-Scan QAT & Precision Ladder:** `1 b/w` FFN, `2 b/w` Attn/Conv, Stage-1 Preview / Stage-1+2 Commit & Final (`ultra.r2-codebook/v0`) | PLANNED | Gate: `≤ +0.15 pp` vs FP32 parent, 730/730 AVX2==NEON==PyTorch, `≥ 1.8×` engine speedup (or 300M `V2-L` at 54M cost) |
| H30 | **JSR v2 Semantic Verifier:** Oracle headroom assay + VoC-gated Confusion-Sausage DAG Prefill Verifier (`L=24..64` GEMM + ILME) + compact LODR LM | PLANNED | Gate: Oracle `≥ 10%` headroom; verifier `≥ 8%` rel WER cut within `+20%` CPU |
| H31 | **JSR v2 Biasing & Personalization:** L1 Double-Array Contextual Trie (`<32 KB`, `<8 ns`/step) + opt-in causal LHUC/codebook adaptation (`1.4 MB`) | PLANNED | Gate: `≥ +15 pp` entity recall (`<+0.1 pp` false ins); `≥ 5%` rel speaker WER drop, `0` speakers `>+0.5 pp` worse |
| H32 | **JSR v2 Expanded Data Track (Owner-Gated):** Libri-Light `unlab-60k` MVQ-KD + GigaSpeech multi-domain + long-form sessions (separate from Controlled 960 h) | PLANNED | Target `≤ 1.9% / 4.2%` DEV clean/other + strong OOD |
| H33 | **JSR v2 Qualifying Freeze & Paper v2:** quiet-host `FREEZE.md` across `CCX Density` (`V2-S`), `SU-Live` (`V2-M/L`), and `SU-File` (`≥ 1,500×` RT) | PLANNED | Full SHA-256 receipts, cross-ISA 100% bit-exact, `jankurai ≥ 85` |
| [H34](H34_delay_penalty/README.md) | **Streaming first-preview latency:** pruned RNN-T `delay_penalty` λ ∈ {0.0015, 0.004} vs λ=0, 3k-step fine-tunes from D2c final | **NEGATIVE** | No WER cost, but first-preview packets identical across λ (0 ms); the floor is the 160 ms chunk |
| [H37](H37_trunks_a1_f1/README.md) | **Production trunks:** A1 (dense S, R0 BPE-500, high-LR restart from H22ctl, chunk sampling {4,8,16}, 4-rank DDP) and F1 (S_MR, R0 BPE-500, from scratch) | RUNNING | A1 expected below 4.95% dev-clean400 late in its cosine |
| [H41](H41_replay_gates/README.md) | **Replay gates:** fork H22ctl step 60,000, replay 8k steps; control, A/A, Muon, chunk sampling | PRE-REGISTERED (running) | Adopt only above 2× the A/A spread on full dev-clean + dev-other |
| [H39](H39_energy/README.md) | Energy per live stream: independent check of the per-core method, core register against the package counter, paced against unpaced | EXPLORATORY, ongoing | Protocol fixed before measuring; results in the wave README |

H14 is the chunk-2 fine-tune; dual-mode / 3-mask tier training is executed in H27. Next free: **H42**.
