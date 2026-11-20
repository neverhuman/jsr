# Release process

No release has been made. A release is a model bundle plus a runtime build that pass
the release gates of spec v1.0-B §38.2 on H0
(`docs/ultra/specs/ULTRA_ASR_MASTER_ENGINEERING_SPEC_v1.0-B.md`). JSR ships a
library and a command-line tool that run on the user's own CPU; there is no hosted
service, account system or server to operate.

## Steps

1. **Version source.** The workspace `version` in `Cargo.toml` versions the runtime. A bundle is `models/sha256-<manifest-digest>/`.
2. **Gate.** `just required` must be green on the release commit. The receipts for every reported number sit under `docs/ultra/receipts/`. `Receipt::validate_as_evidence` must accept each one.
3. **Changelog.** Add an entry to `CHANGELOG.md` naming the bundle digest, hardware, workload, WER and delay. No "best", "leader" or "Nx" without all four.
4. **Integrity and provenance.** Record `SHA256SUMS`, the training commit and the data manifest hash in the bundle `manifest.json`. Pin the tag to the commit.
5. **Rollback.** The previous qualified bundle stays installed. Rolling back means pointing the runtime at that bundle digest (verify it first with `sha256sum -c SHA256SUMS` in the bundle; `ultra-cli bundle verify <dir>` is planned and not built yet). No data migration exists, so rollback is a one-step file switch.

## Launch gates

Each release gate below must have its evidence in place before a tag. Items marked
*required* name what must exist; nothing here is claimed as done until its evidence
is committed.

| Gate | What it means for JSR | Evidence |
|---|---|---|
| Correctness and contract | Spec §38.1 test families and §38.2 items 1-9 pass on H0; transcript, Gap and cancellation invariants hold | `just required` transcript on the release commit; receipts under `docs/ultra/receipts/` |
| Security | No secrets in the tree, no RustSec advisories, linted workflows and scripts; external recognisers run only behind process-isolated adapters (`crates/ultra-adapters`) and receive audio, never references | `just security` (`tools/security-lane.sh`), also run in CI |
| Integrity | The runtime must refuse a bundle whose hashes, shapes, schema, ISA, tokenizer or front-end do not match (spec §38.1 "Model loader"); packed ternary weights with an invalid code are already refused at unpack (`CoreError::BadTernaryCode`) | `SHA256SUMS` in the bundle (`sha256sum -c`); *required:* the runtime loader checks and `ultra-cli bundle verify <dir>` (planned, not built yet) |
| Backup and restore | Bundles are content-addressed and immutable; the previous qualified bundle is kept next to the new one. Training checkpoints and data live on the data drive, never in git; restore is re-pointing to a kept bundle digest | bundle digests in `CHANGELOG.md`; *required:* the kept rollback bundle named in the release entry |
| Rollback | Spec §38.2 item 10: a rollback bundle and a known-limitations model card exist | release entry names the rollback digest and model card |
| Monitoring | The output contract (implemented in `ultra-core`) carries the runtime's own state: `Status` events for mode changes, shed steps, skipped passes and endpoint decisions, and `Gap` events for lost input (spec §3.1). Receipts record process-tree CPU, peak RSS, wakeups, cap hits and chunk-latency percentiles | `ultra.receipt/v1` receipts; `Status`/`Gap` events in the event log |
| Abuse and overload controls | Input must be bounded at the boundary: unsupported or malformed audio formats are rejected (spec §14.1); each event payload is capped at 4096 bytes (`MAX_PAYLOAD_BYTES`, enforced today by `events::Ledger`); overload sheds optional work first, then returns bounded backpressure or an explicit `Gap` (spec §15.4). There is no network endpoint, so request rate limiting does not apply | ledger tests (`just contract-drift`); *required:* §38.1 long-session and overload tests before the first release |
| Rights | Code, model-weight, data and derived-label rights resolved for the distribution (spec §38.2 item 8) | `docs/ultra/DATA.lock`, `docs/ultra/CANDIDATES.lock` |
| Reproduction | A second run reproduces the report within the declared tolerance (spec §38.2 item 9) | two receipts per reported number |
