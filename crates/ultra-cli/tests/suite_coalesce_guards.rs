//! CLI policy guard tests for `--coalesce-ms` in `ultra bench-suite` (`#149` review fix):
//! 1. `--abab-blocks > 0` combined with `--coalesce-ms > 0` fails fast with `CliError::Config`
//!    and never silently writes an uncoalesced `abab_timing.json`.
//! 2. `--qualified` combined with `--coalesce-ms > 0` fails fast with `CliError::Config` so the
//!    frozen `FREEZE.md §3` qualified mode cannot silently accept an unregistered scheduler knob.
//! 3. `AbabTimingReport` explicitly serializes `effective_streams=1`, `effective_paced=false`,
//!    and `effective_coalesce_ms=0`.
//! 4. Default `--coalesce-ms 0` preserves 100% backwards-compatible behavior across all modes.

use ultra_cli::suite_abab::{AbabTimingReport, ABAB_TIMING_SCHEMA};
use ultra_cli::suite_cmd::validate_coalesce_policy;
use ultra_cli::{run_cli_args, CliError};

fn svec(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

#[test]
fn bench_suite_rejects_abab_with_nonzero_coalesce_ms_without_writing_file() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_dir = tmp.path().join("abab_coalesce_reject");
    let out_s = out_dir.to_string_lossy().into_owned();
    let err = run_cli_args(&svec(&[
        "bench-suite",
        "--candidates",
        "zipformer-int8,r1-engine",
        "--abab-blocks",
        "2",
        "--coalesce-ms",
        "10",
        "--out-dir",
        &out_s,
    ]))
    .expect_err("must reject --abab-blocks with --coalesce-ms > 0");
    assert!(
        matches!(err, CliError::Config(ref msg) if msg.contains("--abab-blocks") && msg.contains("--coalesce-ms")),
        "expected CliError::Config mentioning --abab-blocks and --coalesce-ms, got: {err}"
    );
    assert!(
        !out_dir.join("abab_timing.json").exists(),
        "must not write abab_timing.json on rejected combination"
    );
}

#[test]
fn bench_suite_rejects_qualified_with_nonzero_coalesce_ms() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_dir = tmp.path().join("qualified_coalesce_reject");
    let out_s = out_dir.to_string_lossy().into_owned();
    let err = run_cli_args(&svec(&[
        "bench-suite",
        "--candidates",
        "zipformer-int8,r1-engine",
        "--qualified",
        "--grid-ms",
        "320",
        "--max-chunks",
        "2",
        "--threads",
        "1",
        "--coalesce-ms",
        "80",
        "--out-dir",
        &out_s,
    ]))
    .expect_err("must reject --qualified with --coalesce-ms > 0");
    assert!(
        matches!(err, CliError::Config(ref msg) if msg.contains("--qualified") && msg.contains("--coalesce-ms")),
        "expected CliError::Config mentioning --qualified and --coalesce-ms, got: {err}"
    );
    assert!(
        !out_dir.exists(),
        "must fail before creating output directory"
    );
}

#[test]
fn coalesce_policy_allows_default_zero_and_exploratory_paced_runs() {
    // Legacy default coalesce_ms == 0 works across ABAB, qualified, and exploratory modes.
    validate_coalesce_policy(0, 0, false).expect("exploratory default");
    validate_coalesce_policy(0, 8, false).expect("exploratory abab default");
    validate_coalesce_policy(0, 8, true).expect("qualified abab default");
    validate_coalesce_policy(0, 0, true).expect("qualified default");
    // Exploratory paced runs (no ABAB, not qualified) accept coalesce_ms > 0.
    validate_coalesce_policy(80, 0, false).expect("exploratory coalescing");
    // Invalid combinations always fail.
    assert!(validate_coalesce_policy(10, 2, false).is_err());
    assert!(validate_coalesce_policy(10, 0, true).is_err());
    assert!(validate_coalesce_policy(10, 8, true).is_err());
}

#[test]
fn abab_timing_report_serializes_effective_single_stream_uncoalesced_policy() {
    let report = AbabTimingReport {
        schema: ABAB_TIMING_SCHEMA.into(),
        split: "dev-clean".into(),
        completion_status: "exploratory".into(),
        blocks_count: 2,
        baseline_candidate: "zipformer-int8".into(),
        grid_ms: 320,
        max_chunks: 2,
        effective_streams: 1,
        effective_paced: false,
        effective_coalesce_ms: 0,
        candidates: Vec::new(),
        paired_vs_baseline: Vec::new(),
    };
    let val = serde_json::to_value(&report).expect("serialize AbabTimingReport");
    assert_eq!(val["effective_streams"], 1);
    assert_eq!(val["effective_paced"], false);
    assert_eq!(val["effective_coalesce_ms"], 0);
}
