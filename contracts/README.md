# Contracts

`contracts/events/` holds the JSON schemas for every type that crosses a process boundary: transcript events, receipts. The Rust types in `crates/ultra-core` are the source. `cargo test -p ultra-core --test schema_contract` checks that serialised Rust values validate against these schemas (`just contract-drift`).
