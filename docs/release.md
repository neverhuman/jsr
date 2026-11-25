# Release process

A release is a model bundle plus a runtime build that pass the release gates of
spec v1.0-B §38.2 on H0 (`docs/ultra/specs/ULTRA_ASR_MASTER_ENGINEERING_SPEC_v1.0-B.md`).
JSR ships a library and a command-line tool that run on the user's own CPU; there is
no hosted service, account system or server to operate.

**Releases so far:**
- **v3.1.0** is the pinned release ([`docs/ultra/releases/pinned.json`](ultra/releases/pinned.json)): the RC2 INT8 model with its live and final decoders, both LMs and the scorer, signed, published 2026-11-24. It was released by owner decision with its registered gate misses stated. Record: [`v3.1.0.md`](ultra/releases/v3.1.0.md).
- **v3.0.1** is the previous release and the rollback target: the v3.0.0 artifact, signed, with its files attached. Record: [`v3.0.1.md`](ultra/releases/v3.0.1.md).
- **v3.0.0** is the same artifact as v3.0.1, with an unsigned tag. Record: [`v3.0.0.md`](ultra/releases/v3.0.0.md).

## Signing
- **Tags:** release tags are SSH-signed annotated tags. The tagger is `NeverHuman <219957145+neverhumanbot@users.noreply.github.com>`.
  ```bash
  git -c user.name=NeverHuman -c user.email=219957145+neverhumanbot@users.noreply.github.com \
      -c gpg.format=ssh -c user.signingkey=<private key path> tag -s vX.Y.Z <commit> -F <message>
  ```
  Never tag with a placeholder identity. v3.0.0's tag used one and stays unsigned.
- **Files:**
  - `SHA256SUMS` lists every release file;
  - `SHA256SUMS` and each large file get `ssh-keygen -Y sign -n file` signatures (`.sig`).
- **The key:**
  - ed25519, fingerprint `SHA256:Z6YS73tC3B1XPvdmAWA9z+IM2vc2Tmch7AZmiyX0nqA`;
  - the trust anchor is [`docs/ultra/releases/allowed_signers`](ultra/releases/allowed_signers);
  - the private key never enters the repository.
- **To publish:**
  1. Create the GitHub release as a draft with every file.
  2. Check each uploaded size.
  3. Verify `SHA256SUMS` and its signature from a fresh download.
  4. Only then publish.

## Install the pinned release
Run from the repository root. Release files go to `models/sha256-<SHA256SUMS digest>/`, which git ignores.
The example runs in a subshell with `set -eu`. It stops at the first failed check, so nothing is unpacked or run after a bad signature, digest or hash, and your own shell is left as it was.
```bash
(
  set -eu
  P=docs/ultra/releases/pinned.json
  TAG=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["release"])' "$P")
  DIR=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["install_dir"])' "$P")
  mkdir -p "$DIR"
  gh release download "$TAG" -R neverhuman/jsr -D "$DIR" --skip-existing
  cd "$DIR"
  # 1. The signature, against the committed trust anchor
  ssh-keygen -Y verify -f ../../docs/ultra/releases/allowed_signers \
    -I 219957145+neverhumanbot@users.noreply.github.com -n file -s SHA256SUMS.sig < SHA256SUMS
  # 2. The checksum list is the pinned one: its digest names this directory
  test "sha256-$(sha256sum SHA256SUMS | cut -c1-64)" = "$(basename "$PWD")"
  # 3. Every file matches the checksum list
  sha256sum -c SHA256SUMS
  # 4. Unpack the model bundle and check its contents
  tar -xf ultra-v3.1.0-int8-bundle.tar
  sha256sum -c ultra-v3.1.0-int8-bundle.SHA256SUMS
  chmod +x ultra-cold-wav ultra-engine-commit ultra-score
  # 5. Run: utts.jsonl in this directory has one {"uid", "wav"} row per utterance (16 kHz mono WAV)
  ./ultra-engine-commit --model int8 --manifest utts.jsonl --out commit.jsonl \
    --commit-chunk 16 --commit-beam 16 --max-sym-per-frame 1 --tail-pad-ms 320 --threads 1 \
    --commit-lm M5.ulm --lm-scale 0.80 --lodr-scale 0.60 --token-bonus 0.0
  # Live text only (160 ms chunk + 34.5 MiB LM) over ./wav-dir, which must hold only regular files named u00000.wav, u00001.wav, ... (any other entry is rejected):
  # ./ultra-cold-wav --chunk 4 --beam 4 --beam-max-symbols 1 --beam-margin 0.0 --lm c4_p0_0_10_40.v2.ulm \
  #   --lm-weight 0.70 --lodr-weight 0.50 --blank-penalty 0.0 r1-engine-int8 int8 ./wav-dir ./live.jsonl
)
```
- **Output:** `commit.jsonl` rows hold `uid`, the live stream text (`provisional`), the final text (`committed`) and timing fields, followed by one summary row without `uid`. The live decoder writes `{"id", "hyp"}` rows.
- **Without `gh`:** download each file from the `download_base` URL in `pinned.json`.
- **Rollback:** point at the previous install directory. For v3.1.0 that is v3.0.1's (`models/sha256-7612d5be…`), installed from its own record.

## Steps

1. **Version source.** The workspace `version` in `Cargo.toml` versions the runtime. A bundle is `models/sha256-<manifest-digest>/`.
2. **Gate.** `just required` must be green on the release commit. The receipts for every reported number sit under `docs/ultra/receipts/`. `Receipt::validate_as_evidence` must accept each one.
3. **Changelog.** Add an entry to `CHANGELOG.md` naming the bundle digest, hardware, workload, WER and delay. No "best", "leader" or "Nx" without all four.
4. **Integrity and provenance.** Record `SHA256SUMS`, the training commit and the data manifest hash in the bundle `manifest.json`. Attach the runtime SBOMs from `just sbom` (CycloneDX 1.5, `target/sbom/`, with their own `SHA256SUMS`; CI uploads them as the `sbom` artifact). Pin the tag to the commit.
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
