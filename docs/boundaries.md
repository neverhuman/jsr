# Boundaries

The machine-readable manifest is `agent/boundaries.toml`; ownership is
`agent/owner-map.json`. This page explains them.

| Boundary | Rule | Enforced by |
|---|---|---|
| `crates/ultra-core` is pure | No filesystem, environment, network, wall clock, randomness or logging. It holds the numeric contract (§5), the output events and ledger (§3.1), the lifecycle (§3.5) and receipts (Appendix D) | `[rust] domain_paths` in `agent/boundaries.toml`, checked by the audit; `cargo test -p ultra-core` |
| Process boundary: output events | Every event that leaves the runtime is `ultra.transcript-event/v1` | `contracts/events/transcript-event.v1.schema.json`, `just contract-drift` |
| Process boundary: receipts | Every reported number carries an `ultra.receipt/v1` receipt that `Receipt::validate_as_evidence` accepts | `contracts/events/receipt.v1.schema.json`, `just contract-drift` |
| External recognisers | Run in a separate process behind `crates/ultra-adapters`; they receive audio, never references or labels | adapter crate tests |
| Python | Only under `python/ultra/`, training and data preparation; produces bundles, never scores or gates | `[python]` in `agent/boundaries.toml`; `docs/exceptions/2026-11-20-python-ultra-training.md`; `just pytest` |
| Data and checkpoints | On the data drive (`/mnt/xbabe3-external/jsr`), never in git | `.gitignore`; gitleaks in `just security` |
| Sealed evaluation data | Read only through a pre-registered look (`docs/ultra/LOOKS.jsonl`) | spec §0.4 decision rights |

The jankurai scaffold's web, API, database and queue boundaries do not apply; see
`docs/exceptions/2026-11-20-profile-cells.md`.
