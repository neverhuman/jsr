# Changelog

Workspace crates and published speech-recognition packages track their own
versions. Model bundles carry content hashes. Qualification and signed asset
details are in the [release records](docs/ultra/releases/).

## v4.1.0 · 2026-11-27

- New model and live decoder settings, same 60 MB INT8 architecture and the
  same `isosyll` commands. Live text: 160 ms chunks, beam 12 with a 4.0 score
  margin, blank penalty 0.24 and a larger compact live LM (`M5.v2`, 243 MiB).
  Final text keeps its settings (commit chunk 16, beam 16, M5 LM).
- DEV accuracy (730-utterance panel): live 5.08% WER (3.08% dev-clean),
  down from 5.54%; final 5.02%. Full DEV: live 5.32%, final 5.09%
  (dev-clean: live 2.84%, final 2.73%).
  On the unselected DEV half, live is not significantly different from
  Zipformer INT8 (+0.09 [−0.12, +0.30] pp) and final is lower (−0.26 pp).
- One quiet Zen 2 core: live 31.5× real time at 0.331 J per audio-second,
  2.22× Zipformer INT8's throughput at 57% of its energy (paired R8
  blocks; AMD Threadripper PRO 3995WX, idle-subtracted core
  registers). Final: 17.0× real time at 0.656 J per audio-second (113% of
  Zipformer INT8's). Model archive sha256 `c5181748…`; all files in the signed
  `SHA256SUMS`. [Release record](docs/ultra/releases/v4.1.0.md)

## v4.0.0 · 2026-11-26

- Signed `isosyll` CLI for Linux x86-64, Linux arm64, and Apple silicon macOS.
  It bundles live and final transcription commands and verifies the model
  manifest. [Install and quick start](docs/install.md)
- The timed Linux x86-64 glibc build reproduced the reference transcripts and
  passed the registered relative CPU gates. The static musl fallback missed
  the final-mode CPU gate and is available for systems without glibc 2.35.
  [Release qualification](docs/ultra/releases/v4.0.0.md#qualification)

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
