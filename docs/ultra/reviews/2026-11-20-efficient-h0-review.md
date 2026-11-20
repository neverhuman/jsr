# EFFICIENT/H0 independent qualification review — 2026-11-20

**Disposition: experimental; v1.0.0 qualification is incomplete.** SP-A's paced
instrument has received fixes through `404c58c`, with remaining qualification
and policy limitations described below. There is
no qualified density result, final ≥3× verdict, or supported microphone-domain
claim. The approximate 1.5× unpaced screen remains exploratory.

Latest worktree validation: **17/17 independent synthetic diagnostics pass;
`just required` passes; audit 85 with zero caps.** The lead's atomic core/schema/
eval commit is pending. This establishes instrument progress, while the runtime,
qualification evidence and release-package dependencies below remain open.

## Scope and identities

The user-supplied “Coordinate benchmarks → qualified local v1.0.0” takeover plan
controls this review. Qualify EFFICIENT on H0 first, compare the existing
Zipformer and Nemotron, and keep training and additional recognizers outside the
release critical path. The technical authority is
[spec v1.0-B](../specs/ULTRA_ASR_MASTER_ENGINEERING_SPEC_v1.0-B.md), especially
§§9.4, 12, 30, 33, 34, and 38. `tips/` is absent and is not a dependency.

The active checkout is `/home/ubuntu/jsr`, `main`. Entry commit: `97612db`.
Follow-up commits reviewed: `1505371` (Python instrument repair) and `ba8f5bb`
(security scan repair), `1a09202` (fixed-clock follow-up), and `404c58c`
(cutoff and validation follow-up). The Nemotron lane
was reviewed from local Git objects at `787d24a`. No other checkout was entered or edited. Owners retain
their product paths and the lead retains integration; review findings and
dependencies were posted with locked, append-only UTC entries in `AGENT_CHAT.md`.

Diagnostic work uses shared cores `48–55,112–119`, nice 10, and at most eight build
jobs, avoiding the runtime lane's reported cores 56 and 60. All observations below
are synthetic correctness probes or repository checks, not benchmark timings.
No public-test audio was opened by this review. The review uses no GPU.

## Findings and verification

| ID | Finding | Owner | Status |
|---|---|---|---|
| M1 | Audio accepted before capture deadline | Lead/Python harness | First-packet case fixed in `1505371`; short packets, staggering and overload still need coverage |
| M2 | Duration cutoff loses accepted unfinished work | Lead/Python harness | Cutoff catch-up and drain verified in `404c58c`; authoritative backend processing evidence remains a runtime dependency |
| M3 | Stale or missing workers inflate capacity | Lead/Python harness | Identity/count/invalid-run and malformed-row fixes verified through `404c58c`; effective-config binding remains unverified |
| M4 | Lateness clock resets after overload; cutoff and backlog checks | Lead/Python harness | Historical repros fixed through `404c58c`; growth tolerance and grid rounding must be recorded before confirmation |
| C1 | Ledger implements the older suffix contract rather than §33 snapshots | Lead/core, runtime and eval consumers | Migration in progress; draft probes partly verified; atomic commit/consumer checks pending |
| R1 | Invalid numeric measurements and incomplete receipts pass evidence validation | Lead/core | Draft hardening passes independent mutation probes; committed integration pending |
| Q1 | Human-checked timing panel and EFFICIENT CQ/FC qualification evidence absent | Lead/eval | **Missing evidence** |
| S1 | Required gate scans nested lane worktrees and flags candidate manifests | Lead/security | **Fixed and independently verified in `ba8f5bb`** |
| N1 | Nemotron reuses cached audio by length, accepts incorrect shard sets, and lacks release accounting/rescoring fields | Nemotron lane | **Open at `787d24a`; findings sent to owner** |

### M1–M3: baseline failures and partial repair

The baseline instrument probes load the exact `97612db` Python source through
`rtk git show` into an in-memory module, replace the recognizer and clocks, and
mock fleet file/process operations. They use no real model or dataset.

| Fixture | Observed at `97612db` |
|---|---|
| One 20 ms packet | Accepted at 0 ms; its capture deadline is 20 ms |
| Cutoff before the first utterance completes | `ZeroDivisionError` after unfinished audio is excluded from the denominator |
| One requested worker, two prior output rows | Two workers and 24 sustained streams, exit 0 |
| Two requested workers, one output row | One worker and 12 sustained streams, exit 0 |
| Worker exits 9 but a result row exists | Still reports 12 sustained streams; fleet exits 1 |

At `1505371`, the first packet waits until 20 ms, cutoff streams are flushed,
accepted audio is counted, and worker output has run and worker identities.
The two new fleet tests cover ordinary aggregation and combined
missing/stale/failed-result rejection. This resolves useful parts of the initial
review but does not establish an instrument PASS.

At `1505371`, the fleet derives the expected worker count from `len(cores)` after core
selection, rather than requiring `args.cores` workers. `summarise` does not require
exactly one exit code per worker, validate the full effective configuration, or
check processed/drained work and backlog growth. A failed worker may remain in
the reported sustained count even when the overall run is marked invalid.
`1a09202` corrects those count and invalid-run-capacity cases. All eleven Python
tests pass independently after that follow-up.

### M4: overload hidden by rebasing to actual arrival

At `1505371`, `arrival_t = max(due, perf_counter())` followed by
`lateness = completion - arrival_t` removes scheduling lateness accumulated
before the packet was accepted. This measures service time rather than total
lateness from the source clock.

A deterministic fixture uses one second of zero PCM, 20 ms packets, 20 ms decode
grid, a one-second cutoff, no tail padding, and a fake recognizer requiring
100 ms per decode. Observed:

| Quantity | Value |
|---|---:|
| First packet acceptance | 0.020 s |
| Last packet acceptance | 0.920 s |
| Audio accepted before cutoff | 0.200 s |
| Last accepted packet's capture deadline | 0.200 s |
| Last packet's arrival lateness | 0.720 s |
| Reported p99 and last lateness | 0.100 s |
| Total wall time including drain | 1.120 s |

The worker's reported lateness is below the 320 ms bound despite a growing
source backlog and severely reduced offered load. Its accepted/completed counts
do not expose oldest unprocessed age or processed-through samples. Duration
cutoff makes this selection especially misleading: unfinished utterances are
excluded from WER, so their quality cannot be used to claim unchanged WER.

Required resolution: retain fixed sample-derived capture/decode deadlines, retain
all offered work when processing falls behind, and record offered, accepted and
required-processed watermarks. Separate steady arrival/load observations from
drain. Require non-growing oldest-unprocessed age and successful required
processing before a worker contributes to capacity. Test this fixture together
with staggered starts, short final packets, inference crashes and duration cutoff.

#### Follow-up at `1a09202`

Fixed source-clock deadlines and catch-up acceptance replace the clock rebase.
The owner's new deterministic overload fixture exposes growing lateness. The
first instrument repair is therefore partly verified, with these remaining
independently reproduced failures:

| Fixture | Observed at `1a09202` |
|---|---|
| Duration 5 ms, 20 ms packets | Accepts 20 ms of audio, beyond cutoff |
| Duration 30 ms, 20 ms packets | Accepts 40 ms of audio, beyond cutoff |
| One second cutoff, fake 500 ms decode per 320 ms chunk | Accepts only 0.82 s, then drains; capture backlog due before cutoff is discarded |
| First-third lateness 10 ms, last-third 290 ms, tail 300 ms | `sustains` returns true despite growth |
| Result row with negative CPU | Valid, contributes ten sustained streams |
| Result row with negative infinite lateness | Valid, contributes ten sustained streams |
| Result is JSON array / omits `wer` | `AttributeError` / `KeyError` instead of an invalid-run diagnostic |

The grid also rounds `grid_ms / packet_ms`: 37 ms transport gives a 333 ms
effective grid for requested 320 ms. Record the effective schedule and verify
packetization invariance. `k_dec` advances even when no backend work is ready;
it is not an authoritative processed-through watermark. These findings remain
with the owner, and confirmation stays blocked pending their resolution.

Reproduce the cutoff and growth cases from immutable source, without changing
the active checkout or reading real evaluation data:

```sh
rtk proxy taskset -c 48-55,112-119 nice -n 10 \
  env PYTHONPATH=python/ultra .venv/bin/python - <<'PY'
import json, subprocess, sys, types

def frozen(name, path):
    m = types.ModuleType(name)
    m.__file__ = path
    sys.modules[name] = m
    src = subprocess.check_output(['rtk', 'proxy', 'git', 'show', '1a09202:' + path], text=True)
    exec(compile(src, path + '@1a09202', 'exec'), m.__dict__)
    return m

bench = frozen('ultra.sp_a', 'python/ultra/ultra/sp_a.py')
fleet = frozen('review_fleet', 'python/ultra/ultra/sp_a_fleet.py')
fx = frozen('review_fixture', 'python/ultra/tests/test_sp_a_sim.py')
for duration, cost in [(0.005, 0.1), (0.03, 0.1), (1.0, 0.5)]:
    ft = fx.FakeTime()
    out = bench.simulate(fx._items([30.0]), fx.FakeRec(ft, 0.0, cost),
        fx._opt(streams=1, grid_ms=20, duration_s=duration), ft.clock())
    print(json.dumps(dict(duration_s=duration, accepted_audio_s=out['audio_accepted_s'])))
print('growing_backlog_sustains:', fleet.sustains(dict(drained=True, lateness_ms=dict(
    p99=300.0, last=300.0, first_third_mean=10.0, last_third_mean=290.0)), 320.0))
PY
```

Expected diagnostic output at `1a09202`: accepted audio 0.020, 0.040 and 0.820 s,
then `growing_backlog_sustains: True`. These are bug reproductions, not passing
qualification fixtures.

#### Follow-up at `404c58c`

Fifteen Python tests pass independently. Re-running the review fixtures on this
commit gives accepted audio 0.000, 0.020 and 1.000 s for the three durations above;
all accepted work drains. The growing-backlog fixture is rejected. Negative CPU,
array-shaped results and missing WER yield structured invalid runs with zero
capacity. Nonfinite latency yields zero capacity but still leaves `valid=true`;
it should be explicitly identified as instrument failure.

This revision records effective grid size and renames the scheduler's decoded
packet count to `offered_to_decode_packets`. That count remains distinct from a
backend's required-processed watermark. The new growth rule allows last-third
minus first-third mean lateness up to 25% of the 320 ms bound (80 ms). Record this
DEV-selected policy and the grid change as preregistration deviations, and freeze
them before confirmation. They do not establish the absence of sustained backlog
growth by themselves.

### C1: normative public event contract

At entry, `TranscriptEvent` lacked schema, state epoch, observed monotonic timestamp
and model configuration identity. Preview had only an increasing revision ID;
Commit/Final append supplied text without base-revision, full-snapshot,
immutable-byte-offset or prefix-hash checks. There is no SegmentIncomplete event
or required-processed/flush watermark check before Final.

The initial standalone Rust probe confirmed four behavioral failures on the
legacy `ultra-core`. The [current probe](qualification_probe.rs) retains their
regression cases against the new API:

- Identical same-sequence Commit and Final retries return errors rather than
  acting as no-ops.
- Sequence 1 followed by sequence 3 is applied without detecting the missing
  predecessor.
- A correction range starting inside the two-byte encoding of `é` is accepted.

The old schema and its drift tests agree with the old Rust type. Passing them
does not prove conformance to §33. The lead must update the unreleased Rust
contract, schema, runtime emitters and eval replay together. Retain explicit
successful versus incomplete closure in both WER and finalization scoring.

The in-progress migration adds `Body`, full `TextUpdate` snapshots,
`ProcessingEvidence`, incomplete closures, duplicate detection, staged atomic
application and bounded recent duplicate retention. The adapted reviewer probe
confirms duplicate no-ops, missing-sequence/UTF-8 refusals, rejection atomicity,
and rejection of unfinished/unflushed Final. These draft observations are not a
committed integration pass. Acceptance of a required-processed watermark beyond
accepted input was reported and corrected in the
draft. All seventeen current diagnostics now observe the required behavior,
including ignoring delayed retransmissions in eval snapshots. The narrow
`cargo test -p ultra-core -p ultra-eval` lane also passes with 18 output-contract,
seven receipt-evidence and six schema-contract tests, plus the eval suites.

### R1: evidence validation is weaker than its documentation

The standalone probe starts with a synthetic control receipt that passes
`validate_as_evidence`, then mutates one property at a time. The entry validator
accepted negative, NaN and infinite CPU; NaN audio duration; negative chunk
time; zero configured/observed threads; empty identities and hashes; and null
event/wakeup/cap counters.

Reject invalid/nonfinite measurements, impossible counts/quantiles and missing
required identities. Keep schema, validator and negative fixtures aligned.
Receipt acceptance must accompany actual CPU-only verification, complete service
accounting and artifact validation. It cannot replace those checks.

The owner's draft hardening now rejects all eight independent mutation cases.
This is verified on the draft source, pending the atomic commit and contract
tests before marking the finding resolved.

### N1: committed Nemotron adapter review

At `787d24a`, `nemotron_adapter.write_wavs` reuses an anonymized WAV when its
length equals `44 + 2 * samples`, without checking its contents. A synthetic
equal-length/different-PCM fixture confirms that no rewrite occurs. Use immutable
fresh run directories or content-verified cache identities.

`nemotron_merge.merge` marks shard set `0/2, 3/2` complete because it checks only
the number of distinct shard labels. It also selects the latest row for each
shard without requiring one frozen run/manifest/runtime identity. Require the
exact expected shard set and matching unique input identities before pooling.

Per-utterance records have `uid/seconds/ref/hyp` but omit `id/speaker/dataset`
required for direct Rust `ScoreRecord` rescoring. `wait4` and per-child peak RSS
do not measure the harness plus the full service tree, aggregate concurrent
memory/threads or orphan drain. Failed GPU queries return an empty PID set.
Treat unavailable verification as unavailable, and use the Rust whole-service
instrument before qualification. The suite's per-block lease and concurrent
arms qualify only as exploratory timing under the adopted host-lease protocol.

### S1: required-gate repair verified

Initial `just required` passed formatting, clippy, workspace tests, pytest and
contract drift, then failed gitleaks on seven `generic-api-key` findings under
`.agent/worktrees/{nemotron,runtime}/docs/ultra/candidates/*.yaml`. Only redacted
path/line/rule metadata was inspected; this review did not open those checkouts.

`ba8f5bb` adds a configured scan excluding nested lane worktrees/build output and
an allowlist for an exact SHA-256 value on a hash-named YAML field. Independent
stdin canaries confirm that a digest is accepted (exit 0) and a generated
credential-shaped value is still detected (exit 1).

On `ba8f5bb`, `just required` completed successfully: workspace tests, four Python
tests, output/schema contracts, security and audit. Audit: **85, zero caps**.
This is a repository-health result; the open qualification defects above remain.

The mapped documentation proof was re-run on the in-progress migration tree:
`just score` exits 0 but reports **83, zero caps**, because the expanded
`events.rs` exceeds 500 lines. The lead has the finding; split the producer into
a focused submodule and recheck the final committed tree. Do not carry forward
the earlier 85 as a claim about the unfinished migration. After the producer and
ledger were split into focused modules and generated-zone routes updated, a
fresh full `just required` passed with audit **85, zero caps**. This latest result
still covers an uncommitted worktree and must be bound to the owner's commit.

## Evidence required before freezing and release

| Gate | Required artifact or decision | Reviewed checkout evidence |
|---|---|---|
| Instrument | Early-packet, staggered/overload/cutoff, stale/missing-worker and crash probes pass | Partial; M4 and full fleet validation open |
| Exact identities | Model/runtime/build/frontend/tokenizer/data/scorer/settings/hardware hashes frozen before confirmation | `CANDIDATES.lock` still lists unpinned candidates; `DATA.lock` incomplete; experiment/look ledgers empty |
| Implemented runtime | Reference-blind Zipformer/Nemotron, common lifecycle, Rust CLI and verified loader | Main's runtime/model/CLI are placeholders; lane/runtime implementation reported but no reviewable commit yet |
| CPU | Whole service ≤0.25 CPU-s/audio-s, actual CPU-only execution | Screening estimates only |
| Stable output | Correct-word stable p95 ≤2 s and CQ(2000 ms) noninferiority within 0.5 percentage points versus frozen reference | No human-checked word timing panel/results committed |
| Finalization | Conditional p95 ≤1.5 s and FC(1500 ms) ≥95%, using all reference episodes; no unprocessed accepted audio | Episode scorer/evidence absent on main |
| Resources | Aggregate service footprint ≤1.5 GiB, bounded buffers and no accumulating backlog | Qualifying whole-tree/long-session receipts absent |
| Quality | Rust-rescored actual output, frozen family/margins, critical/silence support and declared limitations | Public smoke and provisional Python screen only |
| Repetitions | SP-A three alternating confirmation blocks; release finalists ≥5 paired timing blocks; ≥1 hour sustained | Qualifying artifacts absent |
| Package | Verified content-addressed bundle, rights/provenance, SBOM, second reproduction, model card, rollback artifact | No qualified local bundle/rollback evidence in reviewed checkout |

Use the single host lease `BENCH_LEASE.host` plus an announced quiet window and
active-lane acknowledgements for qualifying timings. Per-block simultaneous
timings remain exploratory/coexistence evidence. Retain CPU loader/background
work in the recorded host conditions. Qualify host conditions before inspecting
the winning arm.

DEV selects settings; only then freeze exact candidate identities and analysis.
SP-A retains its ≥3× hypothesis and ±0.05 percentage-point WER rule. Its screen
used train-clean-100 instead of the preregistered DEV splits; record that deviation
and all prior test-clean access before any confirmation. Never promote the 1.5×
unpaced screen to a paced NO_LIFT verdict without the valid confirmation evidence.
Report INCONCLUSIVE or instrument failure until the evidence supports a verdict.

Training success and ≥3× lift are optional outcomes. Missing mandatory evidence
keeps the bundle experimental and blocks v1.0.0 tagging. Restrict eventual quality
claims to the evaluated population; microphone-domain performance remains
unqualified without supporting evidence.

## Reproduce the core diagnostic

Run in the active repository; the runner uses the configured Cargo target and
creates no separate checkout. Its output is diagnostic JSONL. Each `observed`
boolean reports whether the named required behavior occurred; any false result
exits nonzero. Exit 0 covers these synthetic probes, not release qualification.
The original twelve probes were all false on
the legacy API. The runner is now adapted to the in-progress §33 API and has
seventeen probes, all observed behaviors passing on the latest draft. Re-run
after the atomic migration commits.

```sh
rtk proxy taskset -c 48-55,112-119 nice -n 10 \
  bash docs/ultra/reviews/run-qualification-probe.sh
```

The mapped documentation proof lane is `rtk proxy just score`. The release gate
remains `rtk proxy taskset -c 48-55,112-119 nice -n 10 env CARGO_BUILD_JOBS=8 just required`.
