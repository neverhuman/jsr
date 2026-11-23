# Changelog

Versions follow the workspace `version` in `Cargo.toml`. Model bundles carry their own content hashes. ULTRA model releases (`v3.0.0`, ...) name one artifact, weights + decoder + scorer, and each has a release record under [`docs/ultra/releases/`](docs/ultra/releases/). They do not change the crate versions.

## Unreleased
- ULTRA v3.0.0 release record (draft): [`docs/ultra/releases/v3.0.0.md`](docs/ultra/releases/v3.0.0.md). Its qualification receipts are in [`research/H54_v3_release_qualification/`](research/H54_v3_release_qualification/README.md).
- Workspace bootstrap; `ultra-core` contracts (numeric, events, lifecycle, receipts).
- Spec v1.0-B adopted as the order of work; SP-A stream-density proof in progress.
