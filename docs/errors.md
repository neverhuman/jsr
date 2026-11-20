# Errors and repair hints

Every refusal in the core contracts is an `ultra_core::CoreError`
(`crates/ultra-core/src/lib.rs`). Each variant names the rule it enforces. The
machine-readable catalogue is `agent/exceptions.toml`: one entry per variant with
`purpose`, `reason`, `common_fixes`, `docs_url` and `repair_hint` (the command to
rerun once the cause is fixed). `cargo test -p ultra-core --test error_catalogue`
fails when a variant has no entry here or there.

## Convention

- A refusal names its rule in the message (`rule N1`, `§3.1`, `Appendix D`). Do not
  replace it with a generic error.
- No silent fallback. A refused weight, event or receipt stops that operation; the
  caller reports it, and nothing downstream sees a repaired value.
- Repair loop: read the message, find the variant below, apply one of its common fixes, then run
  the `repair_hint` command. When it passes, run the lane for the path you touched
  (`agent/test-map.json`).
- Evidence is a receipt (`ultra.receipt/v1`, `contracts/events/receipt.v1.schema.json`).
  A receipt that `Receipt::validate_as_evidence` refuses is not evidence; re-run the
  measurement instead of editing the file. Audit and gate reports land in
  `target/jankurai/` and are reproduced by their rerun command, never hand-edited.

## Kinds

### BadShift
`requant` received a shift outside `1..=31` (spec §5). The shift comes from the
calibrated scale of the layer. Rerun: `cargo test -p ultra-core --test numeric_contract`.

### Shape
An operator received a slice of the wrong length; the message names the operator
(`rms_norm`, `softmax_q15`, `pack_t1 length`). Compare the bundle manifest shapes
with the call site. Rerun: `cargo test -p ultra-core`, then the owning crate's tests.

### Overflow
Rule N2: a row is accepted only if `fan_in * 127 * max|w| < 2^31`, so the int32
accumulator cannot overflow. Fix at export (split the row or rescale), never by
widening the accumulator silently. Rerun: `cargo test -p ultra-core --test numeric_contract`.

### WeightRange
Rule N1: int8 weights are in `[-127, 127]` (no `-128`), ternary weights in
`{-1, 0, 1}`. Re-run the quantisation export; do not hand-edit weight files.

### ActivationRange
Rule N1: A7 activations are in `0..=127`. Usually the preceding requantisation did
not saturate, or the A7z zero point (64) was applied twice.

### BadTernaryCode
A packed 2-bit ternary field held code 3. The file is corrupt or not a T1 pack:
check the bundle with `sha256sum -c SHA256SUMS` and re-export.

### Contract
`events::Ledger` refused an event that breaks the output contract (spec §3.1):
non-increasing `event_sequence`, text after `Final`, a second open segment,
non-increasing `revision_id`, a correction outside committed text, an empty gap, or a
payload above 4096 bytes. Rerun: `just contract-drift`.

### Lifecycle
The stream lifecycle (spec §3.5) forbids the call in the current state: push after
`finish_input`, `finish_input` on a cancelled or failed stream, or `complete`
without a flush. Rerun: `cargo test -p ultra-core --test output_contract`.

### Receipt
`Receipt::validate_as_evidence` refused a receipt: wrong schema, not complete,
CPU-only not verified, a null where a measurement belongs, or zero audio. Re-run the
producing bench or eval; a number without a valid receipt is not reported.
