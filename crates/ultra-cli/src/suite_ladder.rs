//! Pre-registered §4.4 density ladder (`FREEZE.md §4.4` / re-review `N1`, `N6`):
//! fixed bounds (`4:48:4`, 120 s rungs, 3 repeats alternating candidates), **strictly monotone
//! stop-at-first-failure** (`N1`), append-only point log (`density_ladder.points.jsonl`, `N6`)
//! that records errors as failed points without aborting, and optional `3 × 300 s` confirmation
//! at `claimed_n` (dropping one rung if confirmation fails).

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::suite_ladder_point::run_one_density_point;
use crate::suite_manifest::write_json_create_new;
use crate::suite_runners::SuiteCtx;
use crate::CliError;

pub const DENSITY_LADDER_SCHEMA: &str = "ultra.density-ladder/v2";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LadderSpec {
    pub start: usize,
    pub end: usize,
    pub step: usize,
}

impl LadderSpec {
    /// Parse `"start:end:step"` (e.g. `"4:48:4"`) or `"freeze"` (`FREEZE.md §4.4` fixed `4:48:4`).
    pub fn parse(spec: &str) -> Result<Self, CliError> {
        if spec.eq_ignore_ascii_case("freeze") {
            return Ok(Self {
                start: 4,
                end: 48,
                step: 4,
            });
        }
        let parts: Vec<&str> = spec.split(':').collect();
        if parts.len() != 3 {
            return Err(CliError::Config(format!(
                "invalid --density-ladder `{spec}` (expected start:end:step or `freeze`, e.g. 4:48:4)"
            )));
        }
        let parse_u = |s: &str| {
            s.trim()
                .parse::<usize>()
                .map_err(|_| CliError::Config(format!("invalid integer `{s}` in ladder `{spec}`")))
        };
        let (start, end, step) = (parse_u(parts[0])?, parse_u(parts[1])?, parse_u(parts[2])?);
        if start == 0 || step == 0 || start > end {
            return Err(CliError::Config(format!(
                "invalid --density-ladder bounds `{spec}` (require 0 < start <= end, step > 0)"
            )));
        }
        Ok(Self { start, end, step })
    }

    pub fn values(self) -> Vec<usize> {
        let mut out = Vec::new();
        let mut n = self.start;
        while n <= self.end {
            out.push(n);
            n = n.saturating_add(self.step);
        }
        out
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LadderRepeatPoint {
    pub candidate: String,
    pub phase: String,
    pub repeat: usize,
    pub streams_per_worker: usize,
    pub total_streams: usize,
    pub duration_seconds: f64,
    pub all_sustained: bool,
    pub drained: bool,
    pub max_worker_p99_ms: Option<f64>,
    pub rtf_service_cpu: f64,
    pub peak_rss_bytes: u64,
    pub audio_processed_seconds: f64,
    pub audio_unfinished_seconds: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl LadderRepeatPoint {
    pub fn passed(&self) -> bool {
        self.error.is_none()
            && self.all_sustained
            && self.drained
            && self.audio_unfinished_seconds == 0.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateLadderResult {
    pub candidate: String,
    pub workers: usize,
    pub ladder_claimed_streams_per_worker: Option<usize>,
    pub claimed_streams_per_worker: Option<usize>,
    pub claimed_total_streams: Option<usize>,
    pub confirmation_passed: Option<bool>,
    pub repeats: Vec<LadderRepeatPoint>,
    pub confirmation_repeats: Vec<LadderRepeatPoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DensityLadderReport {
    pub schema: String,
    pub split: String,
    pub completion_status: String,
    pub grid_ms: u32,
    pub max_chunks: usize,
    pub workers: usize,
    pub step_seconds: f64,
    pub confirm_seconds: f64,
    pub repeats_required: usize,
    pub ladder_values: Vec<usize>,
    pub results: Vec<CandidateLadderResult>,
}

/// Strictly monotone §4.4 rule (`N1`): return the largest `N` on `ladder` such that **every**
/// ladder rung `<= N` passed ALL `repeats_required` repeats. Stops immediately at the first rung
/// where any repeat failed or errored (a pass at a larger N never counts after an earlier failure).
pub fn compute_claimed_n(
    ladder: &[usize],
    repeats_required: usize,
    points: &[LadderRepeatPoint],
) -> Option<usize> {
    let mut best = None;
    for &n in ladder {
        let matching: Vec<&LadderRepeatPoint> = points
            .iter()
            .filter(|p| p.streams_per_worker == n)
            .collect();
        if matching.len() >= repeats_required && matching.iter().all(|p| p.passed()) {
            best = Some(n);
        } else {
            break;
        }
    }
    best
}

fn append_point_jsonl(path: &Path, pt: &LadderRepeatPoint) -> Result<(), CliError> {
    let mut fh = OpenOptions::new().create(true).append(true).open(path)?;
    let line = serde_json::to_string(pt).map_err(|e| CliError::Internal(e.to_string()))?;
    writeln!(fh, "{line}")?;
    Ok(())
}

impl LadderRepeatPoint {
    pub(crate) fn from_r1_fleet(
        cand: &str,
        phase: &str,
        repeat: usize,
        streams: usize,
        w_cnt: usize,
        dur: f64,
        f: &ultra_stream::R1FleetResult,
    ) -> Self {
        Self::from_fields(
            cand,
            phase,
            repeat,
            streams,
            w_cnt,
            dur,
            f.all_sustained,
            &f.worker_summaries,
            f.rtf_service_cpu,
            f.peak_rss_bytes,
            f.audio_processed_seconds,
            f.audio_unfinished_seconds,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_fields(
        cand: &str,
        phase: &str,
        repeat: usize,
        streams_per_worker: usize,
        w_cnt: usize,
        duration_seconds: f64,
        all_sustained: Option<bool>,
        workers: &[ultra_stream::WorkerSummary],
        rtf_service_cpu: f64,
        peak_rss_bytes: u64,
        audio_processed_seconds: f64,
        audio_unfinished_seconds: f64,
    ) -> Self {
        let max_worker_p99_ms = workers
            .iter()
            .filter_map(|w| w.lateness_ms.as_ref().map(|l| l.p99))
            .max_by(f64::total_cmp);
        Self {
            candidate: cand.to_owned(),
            phase: phase.to_owned(),
            repeat,
            streams_per_worker,
            total_streams: w_cnt * streams_per_worker,
            duration_seconds,
            all_sustained: all_sustained == Some(true),
            drained: workers.iter().all(|w| w.drained),
            max_worker_p99_ms,
            rtf_service_cpu,
            peak_rss_bytes,
            audio_processed_seconds,
            audio_unfinished_seconds,
            error: None,
        }
    }
}

fn run_point_or_record_error(
    ctx: &SuiteCtx<'_>,
    cand: &str,
    n: usize,
    repeat: usize,
    duration_s: f64,
    phase: &str,
    log_path: &Path,
) -> Result<LadderRepeatPoint, CliError> {
    let w_cnt = ctx.density_workers.max(1);
    let pt = match run_one_density_point(ctx, cand, n, repeat, duration_s, phase) {
        Ok(p) => p,
        Err(e) => LadderRepeatPoint {
            candidate: cand.to_owned(),
            phase: phase.to_owned(),
            repeat,
            streams_per_worker: n,
            total_streams: w_cnt * n,
            duration_seconds: duration_s,
            all_sustained: false,
            drained: false,
            max_worker_p99_ms: None,
            rtf_service_cpu: 0.0,
            peak_rss_bytes: 0,
            audio_processed_seconds: 0.0,
            audio_unfinished_seconds: 0.0,
            error: Some(e.to_string()),
        },
    };
    append_point_jsonl(log_path, &pt)?;
    Ok(pt)
}

/// Execute the §4.4 alternating density ladder (stop-at-first-failure `N1`, error-resilient
/// append-only log `N6`, optional `3 × confirm_seconds` confirmation) and always write
/// `density_ladder.json`.
pub(crate) fn run_density_ladder(
    ctx: &SuiteCtx<'_>,
    candidates: &[&str],
    spec: LadderSpec,
    repeats: usize,
    confirm_seconds: f64,
) -> Result<DensityLadderReport, CliError> {
    let ladder_values = spec.values();
    let reps = repeats.max(1);
    let log_path = ctx.out_dir.join("density_ladder.points.jsonl");
    let mut per_cand: Vec<Vec<LadderRepeatPoint>> = vec![Vec::new(); candidates.len()];
    let mut stopped: Vec<bool> = vec![false; candidates.len()];

    for &n in &ladder_values {
        if stopped.iter().all(|&s| s) {
            break;
        }
        for r in 0..reps {
            let order: Vec<usize> = if r % 2 == 0 {
                (0..candidates.len()).collect()
            } else {
                (0..candidates.len()).rev().collect()
            };
            for ci in order {
                if stopped[ci] {
                    continue;
                }
                let pt = run_point_or_record_error(
                    ctx,
                    candidates[ci],
                    n,
                    r,
                    ctx.density_seconds,
                    "ladder",
                    &log_path,
                )?;
                let ok = pt.passed();
                per_cand[ci].push(pt);
                if !ok {
                    stopped[ci] = true;
                }
            }
        }
    }

    let w_cnt = ctx.density_workers.max(1);
    let mut results = Vec::with_capacity(candidates.len());
    for (i, &c) in candidates.iter().enumerate() {
        let ladder_n = compute_claimed_n(&ladder_values, reps, &per_cand[i]);
        let mut confirm_pts = Vec::new();
        let (final_n, confirm_ok) = if confirm_seconds > 0.0 {
            if let Some(n) = ladder_n {
                let mut all_ok = true;
                for r in 0..reps {
                    let pt = run_point_or_record_error(
                        ctx,
                        c,
                        n,
                        r,
                        confirm_seconds,
                        "confirm",
                        &log_path,
                    )?;
                    all_ok &= pt.passed();
                    confirm_pts.push(pt);
                }
                if all_ok {
                    (Some(n), Some(true))
                } else {
                    // FREEZE §4.4: confirmation failure drops the claim by one ladder rung.
                    let dropped = ladder_values
                        .iter()
                        .position(|&v| v == n)
                        .and_then(|idx| idx.checked_sub(1).map(|p| ladder_values[p]));
                    (dropped, Some(false))
                }
            } else {
                (None, None)
            }
        } else {
            (ladder_n, None)
        };
        results.push(CandidateLadderResult {
            candidate: c.to_owned(),
            workers: w_cnt,
            ladder_claimed_streams_per_worker: ladder_n,
            claimed_streams_per_worker: final_n,
            claimed_total_streams: final_n.map(|n| n * w_cnt),
            confirmation_passed: confirm_ok,
            repeats: per_cand[i].clone(),
            confirmation_repeats: confirm_pts,
        });
    }

    let report = DensityLadderReport {
        schema: DENSITY_LADDER_SCHEMA.into(),
        split: ctx.split.into(),
        completion_status: ctx.status_tag.into(),
        grid_ms: ctx.grid_ms,
        max_chunks: ctx.max_chunks,
        workers: w_cnt,
        step_seconds: ctx.density_seconds,
        confirm_seconds,
        repeats_required: reps,
        ladder_values,
        results,
    };
    write_json_create_new(&ctx.out_dir.join("density_ladder.json"), &report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ladder_parser_and_monotone_stop_at_first_failure_rule_work() {
        let s = LadderSpec::parse("4:10:2").expect("parse");
        assert_eq!(s.values(), vec![4, 6, 8, 10]);
        assert_eq!(
            LadderSpec::parse("freeze").expect("freeze").values(),
            vec![4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
        );
        assert!(LadderSpec::parse("10:4:2").is_err());
        assert!(LadderSpec::parse("0:4:2").is_err());

        let mk = |n, r, ok| LadderRepeatPoint {
            candidate: "r1".into(),
            phase: "ladder".into(),
            repeat: r,
            streams_per_worker: n,
            total_streams: n * 4,
            duration_seconds: 120.0,
            all_sustained: ok,
            drained: ok,
            max_worker_p99_ms: Some(250.0),
            rtf_service_cpu: 0.04,
            peak_rss_bytes: 1024,
            audio_processed_seconds: 120.0,
            audio_unfinished_seconds: 0.0,
            error: None,
        };
        let pts = vec![
            mk(4, 0, true),
            mk(4, 1, true),
            mk(4, 2, true),
            mk(6, 0, true),
            mk(6, 1, false),
            mk(6, 2, true),
            mk(8, 0, true),
            mk(8, 1, true),
            mk(8, 2, true),
        ];
        // N1: Strictly monotone — N=6 failed a repeat, so claimed N MUST stop at 4 even if N=8 passed.
        assert_eq!(compute_claimed_n(&[4, 6], 3, &pts), Some(4));
        assert_eq!(compute_claimed_n(&[4, 6, 8], 3, &pts), Some(4));
    }
}
