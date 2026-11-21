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
