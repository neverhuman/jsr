# jankurai Repo Score

- Standard: `jankurai`
- Auditor: `1.7.6`
- Schema: `1.9.0`
- Paper edition: `2026.05-ed8`
- Target stack ID: `rust-ts-vite-react-postgres-bounded-python`
- Target stack: `Rust core + TypeScript/React/Vite + PostgreSQL + generated contracts + exception-only Python AI/data service`
- Repo: `.`
- Run ID: `1795180987`
- Started at: `1795180987`
- Elapsed: `495` ms
- Scope: `full`
- Raw score: `85`
- Final score: `85`
- Decision: `advisory`
- Minimum score: `85`
- Caps applied: `none`

## Hard Rule Caps

| Rule | Max Score | Applied |
| --- | ---: | --- |
| `no-root-agent-instructions` | 75 | no |
| `no-one-command-setup-or-validation` | 70 | no |
| `no-deterministic-fast-lane` | 65 | no |
| `no-security-lane-on-high-risk-repo` | 60 | no |
| `generated-contracts-or-public-api-drift-untested` | 80 | no |
| `python-direct-product-truth-or-db-ownership` | 72 | no |
| `no-secret-or-dependency-scanning-in-ci` | 78 | no |
| `no-jankurai-audit-lane-in-ci` | 82 | no |
| `jankurai-required-tool-ci-evidence-gap` | 88 | no |
| `non-optimal-product-language-found` | 74 | no |
| `too-much-python-in-product-surface` | 72 | no |
| `boundary-reclassification-evidence-gap` | 72 | no |
| `vibe-placeholders-in-product-code` | 68 | no |
| `fallback-soup-in-product-code` | 70 | no |
| `future-hostile-dead-language-in-product-code` | 64 | no |
| `severe-duplication-in-product-code` | 70 | no |
| `generated-zone-mutation-risk` | 76 | no |
| `direct-db-access-from-wrong-layer` | 66 | no |
| `missing-web-e2e-lane` | 82 | no |
| `missing-rendered-ux-qa-lane` | 84 | no |
| `prompt-injection-risk` | 78 | no |
| `overbroad-agent-agency` | 65 | no |
| `secret-like-content-detected` | 60 | no |
| `false-green-test-risk` | 76 | no |
| `destructive-migration-risk` | 70 | no |
| `authz-or-data-isolation-gap` | 78 | no |
| `input-boundary-gap` | 78 | no |
| `agent-tool-supply-chain-gap` | 78 | no |
| `release-readiness-gap` | 80 | no |
| `missing-rust-property-or-integration-tests` | 82 | no |
| `no-agent-friendly-exception-pattern` | 76 | no |
| `missing-agent-readable-docs` | 80 | no |
| `streaming-runtime-drift` | 78 | no |
| `rust-bad-behavior` | 72 | no |
| `sql-bad-behavior` | 72 | no |
| `typescript-bad-behavior` | 72 | no |
| `docker-bad-behavior` | 72 | no |
| `python-bad-behavior` | 72 | no |
| `ci-bad-behavior` | 70 | no |
| `git-bad-behavior` | 70 | no |
| `gittools-bad-behavior` | 70 | no |
| `release-bad-behavior` | 70 | no |
| `web-security-bad-behavior` | 68 | no |
| `repo-rot-bad-behavior` | 88 | no |
| `comment-hygiene-dangerous-residue` | 72 | no |
| `ci-local-parity` | 70 | no |

## Copy-Code Redundancy

- Status: `review` hard=`0` warning=`8` files=`43`
- Policy: min-lines=`10` min-tokens=`100` max-findings=`50` include-tests=`false` strict=`false`
- Duplicate volume: lines=`21` tokens=`89` bytes=`672`

- Notes:
  - hard classes are limited to exact active-source file matches and substantial exact same-name units
  - warning classes include same-body different-name units and token/block duplication
  - tests, fixtures, stories, config, Docker, and migrations are omitted unless --include-tests is set

| Kind | Severity | Language | Lines | Tokens | Instances | Reason |
| --- | --- | --- | ---: | ---: | --- | --- |
| `ExactUnitSameName` | `Warning` | `rust` | 14 | 64 | `crates/ultra-kernels/src/x86_avx2.rs:78-92, crates/ultra-kernels/src/x86_avx512.rs:57-71` | `same-name semantic unit copied across multiple files` |
| `ExactUnitSameName` | `Warning` | `rust` | 1 | 8 | `crates/ultra-kernels/src/matrix.rs:77-78, crates/ultra-kernels/src/matrix.rs:175-176, crates/ultra-kernels/src/tlut2.rs:161-162` | `same-name semantic unit copied across multiple files` |
| `ExactUnitDifferentName` | `Warning` | `rust` | 1 | 2 | `crates/ultra-bench/src/fit.rs:191-192, crates/ultra-bench/src/fit.rs:201-202, crates/ultra-bench/src/fit.rs:213-214` | `same body appears under different names across files` |
| `ExactUnitSameName` | `Warning` | `rust` | 1 | 2 | `crates/ultra-kernels/src/matrix.rs:60-61, crates/ultra-kernels/src/matrix.rs:157-158, crates/ultra-kernels/src/tlut2.rs:142-143` | `same-name semantic unit copied across multiple files` |
| `ExactUnitSameName` | `Warning` | `rust` | 1 | 2 | `crates/ultra-kernels/src/matrix.rs:63-64, crates/ultra-kernels/src/matrix.rs:160-161, crates/ultra-kernels/src/tlut2.rs:145-146` | `same-name semantic unit copied across multiple files` |
| `ExactUnitSameName` | `Warning` | `rust` | 1 | 6 | `crates/ultra-kernels/src/matrix.rs:69-70, crates/ultra-kernels/src/matrix.rs:166-167` | `same-name semantic unit copied across multiple files` |
| `ExactUnitDifferentName` | `Warning` | `rust` | 1 | 4 | `crates/ultra-kernels/src/matrix.rs:145-146, crates/ultra-kernels/src/tlut2.rs:120-121` | `same body appears under different names across files` |
| `ExactUnitSameName` | `Warning` | `rust` | 1 | 1 | `crates/ultra-kernels/src/matrix.rs:66-67, crates/ultra-kernels/src/matrix.rs:163-164` | `same-name semantic unit copied across multiple files` |

## Dimensions

| Dimension | Weight | Score | Weighted | Evidence |
| --- | ---: | ---: | ---: | --- |
| Ownership and navigation surface | 13 | 100 | 13.00 | root `AGENTS.md` present; owner map present |
| Contract and boundary integrity | 13 | 73 | 9.49 | contract surface found; generated contract artifacts found |
| Proof lanes and test routing | 12 | 100 | 12.00 | one-command setup/validation lane found; deterministic fast lane found |
| Security and supply-chain posture | 12 | 100 | 12.00 | lockfile present; secret or dependency scan tooling found |
| Code shape and semantic surface | 12 | 80 | 9.60 | largest authored code file: crates/ultra-eval/src/procfs.rs (434 LOC); most code files stay under 300 LOC |
| Data truth and workflow safety | 8 | 90 | 7.20 | no adopter product DB surface; standards/tooling repo classification is explicit |
| Observability and repair evidence | 8 | 80 | 6.40 | observability libraries or patterns found; ops/observability directory present |
| Context economy and agent instructions | 7 | 100 | 7.00 | root `AGENTS.md` present; root `AGENTS.md` stays short |
| Jankurai tool adoption and CI replacement | 7 | 17 | 1.19 | control-plane files present; applicable=15 |
| Python containment and polyglot hygiene | 4 | 100 | 4.00 | no Python files in scope |
| Build speed signals | 4 | 90 | 3.60 | build acceleration markers found; targeted test/build commands found |

## Reference Profile Structure

- Applicable cells: `3` canonical=`2` noncanonical=`1` guidance missing=`1`

| Cell | Status | Canonical | Detected | Aliases | Guidance | Owner | Proof lane | Agent fix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `web` | `not_applicable` | `apps/web/` | `-` | `frontend/, ui/, packages/web/, packages/ui/` | `not_required` | `apps/web` | `rendered UX / Playwright` | `no action` |
| `api` | `not_applicable` | `apps/api/` | `-` | `api/, server/, backend/` | `not_required` | `apps/api` | `edge handler / contract tests` | `no action` |
| `domain` | `not_applicable` | `crates/domain/` | `-` | `domain/, core/` | `not_required` | `crates/domain` | `unit / property tests` | `no action` |
| `application` | `not_applicable` | `crates/application/` | `-` | `application/, usecases/, use-cases/` | `not_required` | `crates/application` | `use-case / authz tests` | `no action` |
| `adapters` | `not_applicable` | `crates/adapters/` | `-` | `adapters/, infra/, integrations/` | `not_required` | `crates/adapters` | `adapter integration tests` | `no action` |
| `workers` | `not_applicable` | `crates/workers/` | `-` | `workers/, jobs/, scheduler/, queue/` | `not_required` | `crates/workers` | `workflow / replay tests` | `no action` |
| `contracts` | `canonical` | `contracts/` | `contracts` | `openapi/, protobuf/, json-schema/, generated/` | `present` | `contracts` | `generation / drift checks` | `keep `contracts/AGENTS.md` aligned with owns / forbidden / proof lane guidance` |
| `db` | `not_applicable` | `db/` | `-` | `migrations/, constraints/, sql/` | `not_required` | `db` | `migration / constraint tests` | `no action` |
| `python-ai` | `noncanonical` | `python/ai-service/` | `python` | `python/, ai-service/, evals/, embeddings/, model/` | `missing` | `python/ai-service` | `eval / contract tests` | `migrate the detected Python AI/data surface to `python/ai-service/` or document a dated advanced-ML/data exception with owner, proof lane, expiry, and migration plan` |
| `ops` | `canonical` | `ops/` | `.github, .github/workflows, ops` | `.github/, .github/workflows/, ci/, release/, observability/, security/` | `present` | `ops` | `security lane / workflow lint` | `keep `ops/AGENTS.md` aligned with owns / forbidden / proof lane guidance` |

## Rendered UX QA

- Web surface: `false`
- Layered UX lane: `true`
- Missing: `none`

## Tool Adoption

- Control plane present: `true`
- Applicable tools: `15`
- Configured: `5`
- CI evidence: `0`
- Artifact verified: `0`
- Replaced count: `0`
- Missing CI evidence: `audit-ci, proof-routing, proofbind, proofmark-rust, copy-code, security, ci-bad-behavior, git-bad-behavior, release-bad-behavior, contract-drift, rust-witness, authz-matrix, agent-tool-supply, release-readiness, cost-budget`

| Tool | Category | Mode | Status | Replaced | Artifacts |
| --- | --- | --- | --- | --- | --- |
| `audit-ci` | `audit` | `auto` | `configured` | `manual repo scoring, ad hoc score gates` | `.jankurai/repo-score.json, .jankurai/repo-score.md` |
| `proof-routing` | `proof` | `auto` | `configured` | `ad hoc proof lane selection, manual proof receipts` | `.jankurai/repo-score.json, .jankurai/repo-score.md, target/jankurai/repair-queue.jsonl` |
| `proofbind` | `proof` | `auto` | `missing` | `manual changed-surface routing, ad hoc proof obligation lists` | `target/jankurai/proofbind/surface-witness.json, target/jankurai/proofbind/obligations.json` |
| `proofmark-rust` | `proof` | `auto` | `missing` | `line-only coverage review, manual in-diff mutation review` | `target/jankurai/proofmark/proofmark-receipt.json, target/jankurai/proofmark/proof-receipt.json` |
| `copy-code` | `audit` | `auto` | `missing` | `ad hoc copy-code review, manual duplication triage` | `target/jankurai/copy-code.json, target/jankurai/copy-code.md` |
| `security` | `security` | `auto` | `configured` | `gitleaks, dependency review, SBOM/provenance` | `target/jankurai/security/evidence.json` |
| `ci-bad-behavior` | `security` | `auto` | `missing` | `mutable workflow refs, secret echo/debug workflow checks, non-blocking security scans` | `target/jankurai/language-bad-behavior.log` |
| `git-bad-behavior` | `audit` | `auto` | `missing` | `destructive git automation, force-push release scripts, hidden stash-based state` | `target/jankurai/language-bad-behavior.log` |
| `release-bad-behavior` | `release` | `auto` | `missing` | `manual release checklist, ad hoc tag and artifact review, manual provenance review` | `target/jankurai/language-bad-behavior.log` |
| `ux-qa` | `ux` | `auto` | `not_applicable` | `playwright, axe-core, visual baselines` | `target/jankurai/ux-qa.json` |
| `db-migration-analyze` | `db` | `auto` | `not_applicable` | `manual migration review` | `target/jankurai/migration-report.json` |
| `contract-drift` | `contract` | `auto` | `configured` | `handwritten contract drift checks, openapi diff` | `.jankurai/repo-score.json, .jankurai/repo-score.md` |
| `rust-witness` | `rust` | `auto` | `configured` | `manual witness graphing` | `target/jankurai/rust/witness-graph.json` |
| `vibe-coverage` | `audit` | `auto` | `not_applicable` | `manual vibe-coding coverage spreadsheet` | `target/jankurai/vibe-coverage.json, target/jankurai/vibe-coverage.md` |
| `coverage-evidence` | `proof` | `auto` | `not_applicable` | `manual coverage report review, ad hoc mutation survivor review` | `target/jankurai/coverage/coverage-audit.json, target/jankurai/coverage/coverage-audit.md` |
| `authz-matrix` | `security` | `auto` | `missing` | `manual authz matrix review` | `.jankurai/repo-score.json, .jankurai/repo-score.md` |
| `input-boundary` | `security` | `auto` | `not_applicable` | `manual unsafe sink review` | `.jankurai/repo-score.json, .jankurai/repo-score.md` |
| `agent-tool-supply` | `security` | `auto` | `missing` | `manual MCP/tool trust review` | `.jankurai/repo-score.json, .jankurai/repo-score.md` |
| `release-readiness` | `release` | `auto` | `missing` | `manual launch checklist` | `.jankurai/repo-score.json, .jankurai/repo-score.md` |
| `cost-budget` | `release` | `auto` | `missing` | `manual spend review` | `.jankurai/repo-score.json, .jankurai/repo-score.md` |

## Boundary Reclassifications

No audited runtime boundary reclassifications declared.

## Findings

1. `medium` `shape` `.`
   Rule: `HLT-001-DEAD-MARKER`
   Check: `HLT-001-DEAD-MARKER:shape` `soft` confidence `0.76`
   Route: TLR `Entropy`, lane `fast`, owner `tools`
   Docs: `docs/audit-rubric.md#future-hostile-language-rule`
   Reason: `Code shape and semantic surface` scored 80 below the standard floor of 85
   Fix: split large or ambiguous authored code into smaller semantic modules with focused tests
   Rerun: `just fast`
   Fingerprint: `sha256:eca4a7a354838b7c0baef0a4178e218ad87124af3f00304fb31422429a0cad1a`
   Evidence: largest authored code file: crates/ultra-eval/src/procfs.rs (434 LOC), most code files stay under 300 LOC, copy-code advisory classes found: 8 (advisory only, no score impact), rust bad-behavior advisory signals: 199
2. `medium` `boundary` `agent/boundaries.toml`
   Rule: `HLT-007-HANDWRITTEN-CONTRACT`
   Check: `HLT-007-HANDWRITTEN-CONTRACT:boundary` `soft` confidence `0.76`
   Route: TLR `Contracts/data`, lane `contract`, owner `agent`
   Docs: `docs/audit-rubric.md#known-vibe-coding-insults`
   Reason: `Contract and boundary integrity` scored 73 below the standard floor of 85
   Fix: add generated contracts and boundary checks for public APIs, data access, and cross-runtime seams
   Rerun: `just fast`
   Fingerprint: `sha256:6b0c42073e9966a0a9974db6b1d4614736c65798b81b87c159f7e6f4974cebb9`
   Evidence: contract surface found, generated contract artifacts found, boundary manifest present, Rust typed boundary helpers found
3. `high` `boundary` `contracts/events/receipt.v1.schema.json:1`
   Rule: `HLT-007-HANDWRITTEN-CONTRACT`
   Check: `HLT-007-HANDWRITTEN-CONTRACT:boundary` `hard` confidence `0.88`
   Route: TLR `Contracts/data`, lane `contract`, owner `contracts`
   Docs: `docs/audit-rubric.md#known-vibe-coding-insults`
   Reason: contract source `contracts/events/receipt.v1.schema.json` has no generated zone entry
   Fix: add a `[[zone]]` in `agent/generated-zones.toml`: contract-first, set `source` to this contract plus the `command` and `path` it generates; code-first, set `path` to this contract with `source` = the typed crate, `command` = the regenerate command, `read_only = true` and `write_policy = "generated_output"`, and prove the command in a drift test or gate lane
   Rerun: `just fast`
   Fingerprint: `sha256:cbc18930da56d499ca785ac10d81f3e785ba5f4a5e274ed876799355ee7236ba`
   Evidence: contract source `contracts/events/receipt.v1.schema.json` has no generated zone entry — handwritten drift is likely
4. `high` `boundary` `contracts/events/transcript-event.v1.schema.json:1`
   Rule: `HLT-007-HANDWRITTEN-CONTRACT`
   Check: `HLT-007-HANDWRITTEN-CONTRACT:boundary` `hard` confidence `0.88`
   Route: TLR `Contracts/data`, lane `contract`, owner `contracts`
   Docs: `docs/audit-rubric.md#known-vibe-coding-insults`
   Reason: contract source `contracts/events/transcript-event.v1.schema.json` has no generated zone entry
   Fix: add a `[[zone]]` in `agent/generated-zones.toml`: contract-first, set `source` to this contract plus the `command` and `path` it generates; code-first, set `path` to this contract with `source` = the typed crate, `command` = the regenerate command, `read_only = true` and `write_policy = "generated_output"`, and prove the command in a drift test or gate lane
   Rerun: `just fast`
   Fingerprint: `sha256:161a1fea644837b6b9ced5a1df18c8914a085485924e60bea0ce7f39f27b1905`
   Evidence: contract source `contracts/events/transcript-event.v1.schema.json` has no generated zone entry — handwritten drift is likely
5. `medium` `observability` `docs/testing.md`
   Rule: `HLT-017-OPAQUE-OBSERVABILITY`
   Check: `HLT-017-OPAQUE-OBSERVABILITY:observability` `soft` confidence `0.76`
   Route: TLR `Repair`, lane `observability`, owner `standard`
   Docs: `agent/JANKURAI_STANDARD.md#repair-receipts`
   Reason: `Observability and repair evidence` scored 80 below the standard floor of 85
   Fix: add structured errors, telemetry, and repair receipts that tell the next agent where to rerun proof
   Rerun: `just score`
   Fingerprint: `sha256:65f3b47174cc80ae87728fb2987add4bead4b427d73f35dceca1b39297a2f661`
   Evidence: observability libraries or patterns found, ops/observability directory present, repair receipts or raw artifact language found, agent-friendly exception pattern found
6. `medium` `release` `docs/testing.md`
   Rule: `HLT-026-COST-BUDGET-GAP`
   Check: `HLT-026-COST-BUDGET-GAP:release` `soft` confidence `0.88`
   Route: TLR `Verification`, lane `release`, owner `standard`
   Docs: `docs/testing.md`
   Matched term: `budget`
   Reason: unbounded paid work needs budgets and stop conditions
   Fix: add explicit budgets, quotas, stop conditions, and kill-switch evidence for paid or unbounded operations
   Rerun: `just check`
   Fingerprint: `sha256:edd248b7afc24b644107205fa5b84a88103ac4b622009ff9f19b779de8798f59`
   Evidence: cost surface found without budget/stop-condition policy
7. `medium` `context` `python`
   Rule: `HLT-038-REFERENCE-PROFILE-STRUCTURE-GAP`
   Check: `HLT-038-REFERENCE-PROFILE-STRUCTURE-GAP:context` `soft` confidence `0.88`
   Route: TLR `Context/setup`, lane `fast`, owner `agent`
   Docs: `docs/audit-rubric.md#required-shape`
   Reason: reference-profile cell `python-ai` is detected at a noncanonical path
   Fix: migrate the detected Python AI/data surface to `python/ai-service/` or document a dated advanced-ML/data exception with owner, proof lane, expiry, and migration plan
   Rerun: `just fast`
   Fingerprint: `sha256:1497a3a9e78cbdc7dd6920ded6b2b9675eb086f9a21d83c0127041289572d6d5`
   Evidence: canonical_path=python/ai-service/, detected_paths=python, aliases=python/, ai-service/, evals/, embeddings/, model/, guidance_status=missing, owner=python/ai-service, proof_lane=eval / contract tests
8. `medium` `context` `python/ai-service/`
   Rule: `HLT-038-REFERENCE-PROFILE-STRUCTURE-GAP`
   Check: `HLT-038-REFERENCE-PROFILE-STRUCTURE-GAP:context` `soft` confidence `0.88`
   Route: TLR `Context/setup`, lane `fast`, owner `training`
   Docs: `docs/audit-rubric.md#required-shape`
   Reason: reference-profile cell `python-ai` lacks local AGENTS.md guidance
   Fix: add `python/ai-service/AGENTS.md` with owns / forbidden / proof lane guidance
   Rerun: `just fast`
   Fingerprint: `sha256:0ef63cc49a79fa6429f49b2ad91ad77e618c24c1cbab3ed509fe238ed3d81389`
   Evidence: canonical_path=python/ai-service/, detected_paths=python, guidance_status=missing, owner=python/ai-service, proof_lane=eval / contract tests

## Policy

- Policy file: `./agent/audit-policy.toml`
- Minimum score: `85`
- Fail on: `critical, high`

## Agent Fix Queue

1. `high` `HLT-007-HANDWRITTEN-CONTRACT` `contracts/events/receipt.v1.schema.json` - add a `[[zone]]` in `agent/generated-zones.toml`: contract-first, set `source` to this contract plus the `command` and `path` it generates; code-first, set `path` to this contract with `source` = the typed crate, `command` = the regenerate command, `read_only = true` and `write_policy = "generated_output"`, and prove the command in a drift test or gate lane
   Route: `Contracts/data`/`contract`
2. `high` `HLT-007-HANDWRITTEN-CONTRACT` `contracts/events/transcript-event.v1.schema.json` - add a `[[zone]]` in `agent/generated-zones.toml`: contract-first, set `source` to this contract plus the `command` and `path` it generates; code-first, set `path` to this contract with `source` = the typed crate, `command` = the regenerate command, `read_only = true` and `write_policy = "generated_output"`, and prove the command in a drift test or gate lane
   Route: `Contracts/data`/`contract`
3. `medium` `HLT-007-HANDWRITTEN-CONTRACT` `agent/boundaries.toml` - add generated contracts and boundary checks for public APIs, data access, and cross-runtime seams
   Route: `Contracts/data`/`contract`
4. `medium` `HLT-026-COST-BUDGET-GAP` `docs/testing.md` - add explicit budgets, quotas, stop conditions, and kill-switch evidence for paid or unbounded operations
   Route: `Verification`/`release`
5. `medium` `HLT-017-OPAQUE-OBSERVABILITY` `docs/testing.md` - add structured errors, telemetry, and repair receipts that tell the next agent where to rerun proof
   Route: `Repair`/`observability`
6. `medium` `HLT-038-REFERENCE-PROFILE-STRUCTURE-GAP` `python` - migrate the detected Python AI/data surface to `python/ai-service/` or document a dated advanced-ML/data exception with owner, proof lane, expiry, and migration plan
   Route: `Context/setup`/`fast`
7. `medium` `HLT-038-REFERENCE-PROFILE-STRUCTURE-GAP` `python/ai-service/` - add `python/ai-service/AGENTS.md` with owns / forbidden / proof lane guidance
   Route: `Context/setup`/`fast`
8. `medium` `HLT-001-DEAD-MARKER` `.` - split large or ambiguous authored code into smaller semantic modules with focused tests
   Route: `Entropy`/`fast`
