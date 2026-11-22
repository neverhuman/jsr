# DECISIONS

## 2026-11-20: ULTRA v1.0 is the controlling spec
- **Alternatives:** CADENCE-ULTRA v3.
- **Evidence:** ULTRA's ladder (R0–R5) gives one mechanism per rung, and it has the more exact numeric contract.
- **Consequence:** we borrow v3's AVX2 T-LUT kernels and its isqrt-PCEN arm.
- **Revisit:** if the owner prefers v3.

## 2026-11-20: AVX2 first
- **Evidence:** H0 is Zen 2 and has no AVX-512.
- **Consequence:** the AVX-512 kernels are written but stay UNTESTED until a machine with AVX-512 is available.
- **Revisit:** when such a machine is available.

## 2026-11-20: Cargo target on the data drive
- **Evidence:** root NVMe is at 95%, and the disk-steward guard refuses builds above 85%.
- **Consequence:** `.cargo/config.toml` points `target-dir` at `/mnt/xbabe3-external/jsr/cargo-target/main`.

## 2026-11-20: Spec v1.0-B sets the order of work; first proof is stream density (SP-A)
- **Evidence:** v1.0-B arrived. The owner asked for early, shocking proof of speed and quality, approved pretrained weights and chose stream density.
- **Consequence:**
  - SP-A runs the icefall LibriSpeech streaming Zipformer through our cross-stream batcher, compared with stock sherpa-onnx on the same pinned cores.
  - The DSL parser is deferred.
  - Kernels and the integer contract continue as a research lane.
- **Revisit:** after the SP-A verdict.

## 2026-11-20: LibriSpeech from the HF mirror
- **Evidence:** OpenSLR served this host at under 1 MB/s; the HF parquet mirror `openslr/librispeech_asr` serves it at about 25 MB/s.
- **Consequence:** shards are recorded with SHA-256 in `MANIFEST.sha256`.

## 2026-11-20: Contract schemas stay Rust-first (jankurai HLT-007 accepted as a known finding)
- **Evidence:** jankurai 1.7.6 clears HLT-007 only when the schemas are the source and the types are generated from them. Our Rust types are the source of truth. `crates/ultra-core/tests/schema_contract.rs` checks every serialised event and receipt against the schemas, and it has already caught real receipt drift.
- **Alternatives:** schema-first codegen.
- **Consequence:** the pre-commit diff-audit flags these two files whenever a commit touches them; such commits bypass the hook and say why.
- **Revisit:** if a later jankurai release accepts a drift test as evidence.

## 2026-11-20: W01 on H0 passes; SP-A batching lift is small on the ORT runtime
- **Evidence:** in `docs/ultra/costlaw/W01_H0.md`:
  - out-of-cache ternary T-LUT2 has a = 9.7 ps/weight against int8's 51.6 (0.18×);
  - c ≈ 14–17 ps/MAC;
  - R² ≥ 0.988.
- **SP-A screen:** sherpa-onnx int8 Zipformer batching N=64 gives only 1.46× streams per core over single decoding. That runtime is compute-bound, not fetch-bound.
- **Consequence:** density beyond about 18 streams per core needs our own runtime/kernels or a smaller, gated model (Track B). The ORT path is still the R0 baseline.

## 2026-11-21 — lead: R1-MR enters the main freeze (deviation from BENCH_PLAN §0a)
BENCH_PLAN §0a held R1-MR to an add-on block after the D2 window. R1-MR (S_MR) is now the density design, and it has 960 h checkpoints (D2MR, D3MRb) finishing before the window. It is in the main freeze, labelled, with the same gates. The independent review (2026-11-21-independent-freeze-review.md B7) flagged the inconsistency, and this records it. The 5% CPU rerun tolerance and the two 1 h runs from §0a are restored in FREEZE §4.

## 2026-11-21 — owner: the qualifying host is xbabe2 (deviation from H0)
No Zen 2 host was quiet: H0, xbabe0 and xbabe2 all carry other projects' unpinned jobs. The owner chose **xbabe2** (same SKU: TR PRO 3995WX) as the qualifying host. The owner pauses the other projects' jobs there for the window, and JSR pauses its own xbabe2 GPU job (R4T-960 or any successor) during the window as well. Rows from that window carry host=xbabe2. HARDWARE.json gets an xbabe2 entry, verified from the window's own host snapshot.

## 2026-11-22: owner approves R0 Zipformer as a distillation teacher
The owner approved using R0 (the pretrained icefall streaming Zipformer, trained on LibriSpeech 960 h only) as a teacher for knowledge distillation of R1. No new data enters training: the teacher sees only LibriSpeech training audio. The teacher's outputs are cached as training targets.

**Rules:**
- The teacher runs only on LibriSpeech **train** splits, never on dev or test.
- Every R1 checkpoint trained with distillation is labelled "distilled from R0" in the paper and in research waves.
- Accuracy is still selected on the DEV panel only.

Tracked as research wave H22.

## 2026-11-22: owner widens the comparison set and asks for watts (paper v2)
The owner asked for every recogniser in a published CPU comparison to be measured on our own hardware, for the paper's first-page figure to become processing speed against WER, and for inference power to be measured against the others. Rulings O16–O18 record the approvals.

**Rules:**
- One protocol for every system: the frozen 730-utterance DEV panel, `docs/paper/run-cold.sh`, one pinned core and eight pinned cores, three repetitions. No test-set audio.
- Live streaming systems and whole-utterance systems are reported as separate classes. Vendor-published numbers stay in a "reported" group and never share an axis with measured ones.
- Each system is also measured on a machine where its fast path exists (AVX-512 VNNI for Photon; Apple silicon), with our engine beside it on the same cores.
- Watts come from CPU energy counters, as joules per audio-second, with the method and its noise stated.

Tracked as research waves H39 (energy) and H40 (external benchmark and paper v2).

## 2026-11-22: production moves to the 500-token line; tests move to replay gates
- **Evidence:** the 4,096-token D2 line reached 5.29% on the 400-utterance dev-clean screen after about 40k audio-hours. H22ctl (dense S, the Zipformer's 500-token BPE, one A100) reached 4.95% after 25k, and L3 (M_MR) 4.88% after 21.5k.
- **Decision:** the D2 line ends with D2f. Two trunks replace it: A1 (dense S, 500 tokens, restarted from H22ctl at a high learning rate, chunks sampled from 160/320/640 ms) and F1 (multi-rate S_MR, 500 tokens, from scratch). The M_MR line stops after L3, and D4M is stopped.
- **Why a high learning rate:** the D2 restarts summed to a learning-rate total of about 8, against about 60 for H22ctl's single run; they could not move the model.
- **Tests:** a change is tested by replaying the tail of H22ctl from its step-60,000 weights with a control and a repeated control, scored on the full dev sets with a paired bootstrap. It is adopted only above twice the spread between the two controls. Two identical 100 h runs differed by 0.57 points, so 100 h runs from scratch no longer decide optimiser changes.
- **Multi-GPU:** five A100s in step-synchronised training delivered 3.9–4.2k audio-hours per hour, against 2.7k for one unshared A100, because shared GPUs make every rank wait and sleep. Trunks start on the unshared GPUs; shard sizes will follow each GPU's speed.
- **Sharing:** another project runs small jobs on four A100s and the A10. We share and never touch its processes; our throttles hold the combined load under the cap.

Tracked as research waves H36 (multi-node efficiency), H37 (trunks) and H41 (replay gates).

