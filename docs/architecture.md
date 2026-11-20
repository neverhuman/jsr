# Architecture

- **Controlling plan:** `docs/ultra/specs/ULTRA_ASR_MASTER_ENGINEERING_SPEC_v1.0-B.md`.
- **Design depth:** `docs/ultra/specs/ULTRA_ASR_MASTER_ENGINEERING_SPEC.md`.

## Boundaries
- **Rust (`crates/`) owns everything that ships or decides:** the streaming runtime, kernels, the output event contract, scoring, accounting, receipts and release gates.
- **Python (`python/ultra/`) is a dated exception** for training and data preparation (`python/ultra/EXCEPTION.md`). It produces bundles; it never scores a release.
- **External recognisers** run behind process-isolated adapters (`crates/ultra-adapters`). They receive audio, never references.
- **Data and checkpoints** live on the data drive (`/mnt/xbabe3-external/jsr`), never in git.

## Proof lanes
See `agent/test-map.json` for the command per path. `just required` is the push gate.
