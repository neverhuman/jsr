# Changelog

Versions follow the workspace `version` in `Cargo.toml`. Model bundles carry their own content hashes. ULTRA model releases (`v3.0.0`, ...) name one artifact, weights + decoder + scorer, and each has a release record under [`docs/ultra/releases/`](docs/ultra/releases/). They do not change the crate versions.

## Unreleased
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
