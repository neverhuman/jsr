# Generated and checked zones

`agent/generated-zones.toml` lists files that are written by a command or kept in
step with a source by a check. Never hand-edit a `read_only` zone; rerun its
command.

| Path | Source | Policy | Command |
|---|---|---|---|
| `contracts/events/transcript-event.v1.schema.json` | `crates/ultra-core/src/events.rs` | `reviewed_manual` | `cargo test -p ultra-core --test schema_contract` |
| `contracts/events/receipt.v1.schema.json` | `crates/ultra-core/src/receipt.rs` | `reviewed_manual` | `cargo test -p ultra-core --test schema_contract` |
| `Cargo.lock` | `Cargo.toml` | `lockfile` | `cargo metadata --locked` (cargo updates it) |
| `.jankurai/score-history.jsonl` | `agent/audit-policy.toml` | `auditor_output` | `just score` |

The two schemas are written by hand, and the Rust types are the source of truth.
The `schema_contract` test serialises every event kind and both measured and
unmeasured receipts, validates them against the schemas, checks the schemas refuse
bad input, and checks the receipt schema's `required` list equals the Rust
non-optional fields. When you change a type in `ultra-core`, update the schema in
the same commit; the test fails until they agree.

Audit reports (`target/jankurai/`) are build output, not zones; `just score`
regenerates them.
