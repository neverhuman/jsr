# Release process

A release is a model bundle and a runtime build that pass the release tests of spec v1.0-B §38 on H0.

1. **Version source.** The workspace `version` in `Cargo.toml` versions the runtime. A bundle is `models/sha256-<manifest-digest>/`.
2. **Gate.** `just required` must be green on the release commit. The receipts for every reported number sit under `docs/ultra/receipts/`. `Receipt::validate_as_evidence` must accept each one.
3. **Changelog.** Add an entry to `CHANGELOG.md` naming the bundle digest, hardware, workload, WER and delay. No "best", "leader" or "Nx" without all four.
4. **Integrity and provenance.** Record `SHA256SUMS`, the training commit and the data manifest hash in the bundle `manifest.json`. Pin the tag to the commit.
5. **Rollback.** The previous qualified bundle stays installed. Rolling back means pointing the runtime at that bundle digest (`ultra-cli bundle verify <dir>` first). No data migration exists, so rollback is a one-step file switch.
