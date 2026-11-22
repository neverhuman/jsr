//! Pre-registered §4.4 density ladder (`FREEZE.md §4.4` rev 3 / pass-3 `B6(a–d)`, `N1`, `N6`):
//! - Frozen preset: `2..=16` step `2`, then `20..=48` step `4` (`120 s` rungs, `3` repeats,
//!   `3 × 300 s` confirmation), forced under `--qualified` (`B6(a)`);
//! - Strictly monotone stop-at-first-failure (`N1`);
//! - Append-only point log (`density_ladder.points.jsonl`, `N6`) recording errors as failed points;
//! - Confirmation failure steps down one rung at a time and **re-confirms** the dropped rung
//!   (`3 × 300 s`) rather than claiming an unconfirmed rung (`B6(b)`);
//! - Top-of-ladder censoring (`censored_at_top`, `claimed_display = ">=48"`, `B6(d)`) and
//!   bracket reporting (`claimed_bracket`, `B6(c)`).

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::suite_ladder_point::run_one_density_point;
use crate::suite_manifest::write_json_create_new;
use crate::suite_runners::SuiteCtx;
use crate::CliError;

pub const DENSITY_LADDER_SCHEMA: &str = "ultra.density-ladder/v3";

/// `FREEZE.md §4.4` rev 3 fixed ladder: `2..=16` step `2`, then `20..=48` step `4`.
pub const FREEZE_LADDER_RUNGS: &[usize] =
    &[2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 40, 44, 48];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LadderSpec {
    pub rungs: Vec<usize>,
    pub is_freeze_preset: bool,
}

impl LadderSpec {
    pub fn freeze_preset() -> Self {
        Self {
            rungs: FREEZE_LADDER_RUNGS.to_vec(),
            is_freeze_preset: true,
        }
    }

    /// Parse `"freeze"` (`FREEZE.md §4.4` rev 3), `"start:end:step"`, or comma-separated rungs.
    pub fn parse(spec: &str) -> Result<Self, CliError> {
        if spec.eq_ignore_ascii_case("freeze") {
            return Ok(Self::freeze_preset());
        }
        if spec.contains(',') && !spec.contains(':') {
            let mut rungs = Vec::new();
            for part in spec.split(',') {
                let v: usize = part.trim().parse().map_err(|_| {
                    CliError::Config(format!("invalid rung `{part}` in ladder `{spec}`"))
                })?;
                if v == 0 {
                    return Err(CliError::Config(format!(
                        "ladder rung must be > 0 in `{spec}`"
                    )));
                }
                rungs.push(v);
            }
            if rungs.is_empty() || rungs.windows(2).any(|w| w[0] >= w[1]) {
                return Err(CliError::Config(format!(
                    "ladder `{spec}` must be non-empty and strictly increasing"
                )));
            }
            let is_freeze_preset = rungs.as_slice() == FREEZE_LADDER_RUNGS;
            return Ok(Self {
                rungs,
                is_freeze_preset,
            });
        }
        let parts: Vec<&str> = spec.split(':').collect();
        if parts.len() != 3 {
            return Err(CliError::Config(format!(
                "invalid --density-ladder `{spec}` (expected `freeze`, `start:end:step`, or comma-separated rungs)"
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
        let mut rungs = Vec::new();
        let mut n = start;
        while n <= end {
            rungs.push(n);
            n = n.saturating_add(step);
        }
        let is_freeze_preset = rungs.as_slice() == FREEZE_LADDER_RUNGS;
        Ok(Self {
            rungs,
            is_freeze_preset,
        })
    }

    pub fn values(&self) -> Vec<usize> {
        self.rungs.clone()
    }
}

/// Enforce `FREEZE.md §4.4` rev 3 (`B6(a)`): under `--qualified`, the density ladder preset,
/// repeats (`3`), rung duration (`120 s`), and confirmation duration (`300 s`) cannot be overridden.
pub fn validate_qualified_ladder_preset(
    spec: &LadderSpec,
    repeats: usize,
    step_seconds: f64,
    confirm_seconds: f64,
) -> Result<(), CliError> {
    if !spec.is_freeze_preset {
        return Err(CliError::Config(format!(
            "--qualified forces --density-ladder freeze ({FREEZE_LADDER_RUNGS:?}); got {:?} (FREEZE.md §4.4 / B6)",
            spec.rungs
        )));
    }
    if repeats != 3 {
        return Err(CliError::Config(format!(
            "--qualified forces --density-repeats 3 per FREEZE.md §4.4 (got {repeats}; B6)"
        )));
    }
    if !step_seconds.is_finite() || (step_seconds - 120.0).abs() > 1e-6 {
        return Err(CliError::Config(format!(
            "--qualified forces --density-seconds 120 for ladder rungs per FREEZE.md §4.4 (got {step_seconds}; B6)"
        )));
    }
    if !confirm_seconds.is_finite() || (confirm_seconds - 300.0).abs() > 1e-6 {
        return Err(CliError::Config(format!(
            "--qualified forces --ladder-confirm-seconds 300 per FREEZE.md §4.4 (got {confirm_seconds}; B6)"
        )));
    }
    Ok(())
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
    /// True iff `claimed_streams_per_worker` passed the top rung of the ladder (`>= N`; `B6(d)`).
    pub censored_at_top: bool,
    /// Human-readable claim (`">=48"`, `"12"`, or `"none"`).
    pub claimed_display: String,
    /// `[claimed, next_rung]` capacity bracket when not censored at the top (`B6(c)`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub claimed_bracket: Option<[usize; 2]>,
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

/// Outcome of confirmation with downward fallback: `(claimed_n, confirmation_passed, points)`.
pub type ConfirmationOutcome = (Option<usize>, Option<bool>, Vec<LadderRepeatPoint>);

/// Run confirmation at `ladder_n`, stepping down one rung at a time and **re-confirming** any
/// dropped rung (`FREEZE.md §4.4` rev 3 / `B6(b)`). Never claims an unconfirmed rung.
pub fn resolve_confirmation_with_fallback<F>(
    ladder_values: &[usize],
    ladder_n: Option<usize>,
    reps: usize,
    mut run_one: F,
) -> Result<ConfirmationOutcome, CliError>
where
    F: FnMut(usize, usize, &str) -> Result<LadderRepeatPoint, CliError>,
{
    let Some(start_n) = ladder_n else {
        return Ok((None, None, Vec::new()));
    };
    let Some(mut idx) = ladder_values.iter().position(|&v| v == start_n) else {
        return Ok((None, Some(false), Vec::new()));
    };
    let mut confirm_pts = Vec::new();
    let mut is_fallback = false;
    loop {
        let n = ladder_values[idx];
        let phase = if is_fallback {
            "confirm_fallback"
        } else {
            "confirm"
        };
        let mut all_ok = true;
        for r in 0..reps {
            let pt = run_one(n, r, phase)?;
            let ok = pt.passed();
            confirm_pts.push(pt);
            if !ok {
                all_ok = false;
                break;
            }
        }
        if all_ok {
            return Ok((Some(n), Some(true), confirm_pts));
        }
        if idx == 0 {
            return Ok((None, Some(false), confirm_pts));
        }
        idx -= 1;
        is_fallback = true;
    }
}

fn append_point_jsonl(path: &Path, pt: &LadderRepeatPoint) -> Result<(), CliError> {
    let mut fh = OpenOptions::new().create(true).append(true).open(path)?;
    let line = serde_json::to_string(pt).map_err(|e| CliError::Internal(e.to_string()))?;
    writeln!(fh, "{line}")?;
    Ok(())
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
/// append-only log `N6`, `3 × confirm_seconds` confirmation with fallback re-confirmation `B6(b)`,
/// top censoring `B6(d)`) and always write `density_ladder.json`.
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
        let (final_n, confirm_ok, confirm_pts) = if confirm_seconds > 0.0 {
            resolve_confirmation_with_fallback(&ladder_values, ladder_n, reps, |n, r, phase| {
                run_point_or_record_error(ctx, c, n, r, confirm_seconds, phase, &log_path)
            })?
        } else {
            (ladder_n, None, Vec::new())
        };
        let censored_at_top = final_n.is_some() && final_n == ladder_values.last().copied();
        let claimed_display = match final_n {
            None => "none".into(),
            Some(n) if censored_at_top => format!(">={n}"),
            Some(n) => n.to_string(),
        };
        let claimed_bracket = final_n.and_then(|n| {
            let pos = ladder_values.iter().position(|&v| v == n)?;
            ladder_values.get(pos + 1).map(|&next| [n, next])
        });
        results.push(CandidateLadderResult {
            candidate: c.to_owned(),
            workers: w_cnt,
            ladder_claimed_streams_per_worker: ladder_n,
            claimed_streams_per_worker: final_n,
            claimed_total_streams: final_n.map(|n| n * w_cnt),
            censored_at_top,
            claimed_display,
            claimed_bracket,
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

    fn mk_pt(n: usize, r: usize, phase: &str, ok: bool) -> LadderRepeatPoint {
        LadderRepeatPoint {
            candidate: "r1".into(),
            phase: phase.into(),
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
        }
    }

    #[test]
    fn freeze_preset_monotone_rule_and_confirmation_fallback_work() {
        let freeze = LadderSpec::parse("freeze").expect("freeze");
        assert!(freeze.is_freeze_preset);
        assert_eq!(freeze.values(), FREEZE_LADDER_RUNGS);
        validate_qualified_ladder_preset(&freeze, 3, 120.0, 300.0).expect("qualified ok");
        assert!(validate_qualified_ladder_preset(&freeze, 1, 120.0, 300.0).is_err());
        assert!(validate_qualified_ladder_preset(&freeze, 3, 60.0, 300.0).is_err());
        assert!(validate_qualified_ladder_preset(&freeze, 3, 120.0, 0.0).is_err());
        let custom = LadderSpec::parse("4:48:4").expect("custom");
        assert!(!custom.is_freeze_preset);
        assert!(validate_qualified_ladder_preset(&custom, 3, 120.0, 300.0).is_err());

        // N1: Strictly monotone — N=6 failed a repeat, so claimed N stops at 4 even if N=8 passed.
        let pts = vec![
            mk_pt(4, 0, "ladder", true),
            mk_pt(4, 1, "ladder", true),
            mk_pt(4, 2, "ladder", true),
            mk_pt(6, 0, "ladder", true),
            mk_pt(6, 1, "ladder", false),
            mk_pt(6, 2, "ladder", true),
            mk_pt(8, 0, "ladder", true),
            mk_pt(8, 1, "ladder", true),
            mk_pt(8, 2, "ladder", true),
        ];
        assert_eq!(compute_claimed_n(&[4, 6, 8], 3, &pts), Some(4));

        // B6(b): Confirmation failure at 16 re-confirms 14 before claiming it.
        let (claimed, ok, cpts) =
            resolve_confirmation_with_fallback(FREEZE_LADDER_RUNGS, Some(16), 3, |n, r, ph| {
                Ok(mk_pt(n, r, ph, n <= 14))
            })
            .expect("confirm fallback");
        assert_eq!(claimed, Some(14));
        assert_eq!(ok, Some(true));
        assert!(cpts
            .iter()
            .any(|p| p.streams_per_worker == 16 && p.phase == "confirm"));
        assert_eq!(
            cpts.iter()
                .filter(|p| p.streams_per_worker == 14 && p.phase == "confirm_fallback")
                .count(),
            3
        );
    }
}
