# Changelog

Versions follow the workspace `version` in `Cargo.toml`. Model bundles carry their own content hashes. ULTRA model releases (`v3.0.0`, ...) name one artifact, weights + decoder + scorer, and each has a release record under [`docs/ultra/releases/`](docs/ultra/releases/). They do not change the crate versions.

## Unreleased
- **ULTRA v3.2.0** (runtime-only release, staged 2026-11-25): the v3.1.0 weights and LMs with new decoders built at `6d95f37d`.
  - **Added:** Z1, an opt-in whole-segment RMS gain in the final commit pass (`ultra-engine-commit --rms-norm`, `ultra stream --commit-pass --commit-rms-norm`), default off; with it off the output is byte-identical to v3.1.0's. Also the fail-closed side-feature export guard.
  - **Measured:** S-half final M5 -0.067 pp (2,838 to 2,803 edits); C-half confirmation -0.022 pp at native level and -0.17 / -0.94 pp at -20 / -30 dB synthetic attenuation; relative CPU upper bounds +0.21% (`ultra-engine-commit`) and +0.31% (`ultra stream`), measured on H0.
  - **Unchanged:** headline WER, speed and energy are v3.1.0's, and v3.1.0's registered gate misses still apply. Energy was not re-measured.
  - **Record:** [`docs/ultra/releases/v3.2.0.md`](docs/ultra/releases/v3.2.0.md). **Pin:** [`docs/ultra/releases/pinned.json`](docs/ultra/releases/pinned.json).
- **Fix: `ultra transcribe` / `ultra stream` printed only the last segment of audio longer than 30 s.** The stream coordinator's 30 s segment hard cap closes the open segment as `SegmentIncomplete`, and the printed text came from the ledger's immutable text, so the first 30 s were dropped. The CLI now prints the capped segment's last displayed snapshot followed by the next segment (presentation only; the wire events and the receipt's `commit` hash are unchanged, and `final_` hashes the printed text). Cancellations, timeouts and gaps are never recovered. Release `ultra` vs `ultra-cold-wav` on the 730-utterance DEV panel: 730 / 730 identical, up from 727 / 730. The signed v3.0.1 decoder `ultra-engine-commit` was never affected.
- **ULTRA v3.0.1** (model release, 2026-11-23): the v3.0.0 artifact, re-issued as a **signed** release with its files attached. It is the same commit and the same bytes; no model or code change.
  - **Manifest:** `SHA256SUMS` sha256 `7612d5beb2b577893b649033073d44c6198c0e83a275e3980b3ede38e0162ef7`; INT8 weights `edac19d8…`.
  - **Measured:**
    - LibriSpeech DEV (5,567 utterances): final text 5.476% (utterance-segmented) and live stream 7.447%;
    - one quiet Zen 2 core: 14.3× real time with the full release decoder;
    - final text 211 / 586 ms (p50 / p95) after the endpoint.
  - **Signed** with the ed25519 release key `SHA256:Z6YS73tC3B1XPvdmAWA9z+IM2vc2Tmch7AZmiyX0nqA`.
  - **Record:** [`docs/ultra/releases/v3.0.1.md`](docs/ultra/releases/v3.0.1.md). **Pin:** [`docs/ultra/releases/pinned.json`](docs/ultra/releases/pinned.json).
- ULTRA v3.0.0 (model release, 2026-11-23): S2's RC1 EMA weights + the release decoder. Release record: [`docs/ultra/releases/v3.0.0.md`](docs/ultra/releases/v3.0.0.md). Its qualification receipts are in [`research/H54_v3_release_qualification/`](research/H54_v3_release_qualification/README.md).
- Workspace bootstrap; `ultra-core` contracts (numeric, events, lifecycle, receipts).
- Spec v1.0-B adopted as the order of work; SP-A stream-density proof in progress.
