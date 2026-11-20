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
