//! Pre-registered §4.4 density ladder (`FREEZE.md §4.4`):
//! step N in `start..=end` by `step`, `repeats` alternating passes across candidates,
//! `claimed_n` = largest N where ALL workers pass A1 in ALL repeats.

use serde::{Deserialize, Serialize};
use ultra_core::receipt::HonestyLabel;
use ultra_stream::{
    run_sp_a_native_engine_shared_workers, run_sp_a_ort_r1_shared_workers,
    run_sp_a_shared_workers_with_observer, DecodeMode, EngineSpAConfig, R1SpAConfig, SpAConfig,
    DEFAULT_LATENESS_BOUND_MS,
};

use crate::runner::{resolve_bundle_or_zipformer, resolve_r1_bundle};
use crate::suite_engine::resolve_r1_weights_dir;
use crate::suite_manifest::write_json_create_new;
use crate::suite_runners::SuiteCtx;
use crate::CliError;

pub const DENSITY_LADDER_SCHEMA: &str = "ultra.density-ladder/v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LadderSpec {
    pub start: usize,
    pub end: usize,
    pub step: usize,
}

impl LadderSpec {
    /// Parse `"start:end:step"` (e.g. `"4:40:2"`).
    pub fn parse(spec: &str) -> Result<Self, CliError> {
        let parts: Vec<&str> = spec.split(':').collect();
        if parts.len() != 3 {
            return Err(CliError::Config(format!(
                "invalid --density-ladder `{spec}` (expected start:end:step, e.g. 4:40:2)"
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
    pub repeat: usize,
    pub streams_per_worker: usize,
    pub total_streams: usize,
    pub all_sustained: bool,
    pub drained: bool,
    pub max_worker_p99_ms: Option<f64>,
    pub rtf_service_cpu: f64,
    pub peak_rss_bytes: u64,
    pub audio_processed_seconds: f64,
    pub audio_unfinished_seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateLadderResult {
    pub candidate: String,
    pub workers: usize,
    pub claimed_streams_per_worker: Option<usize>,
    pub claimed_total_streams: Option<usize>,
    pub repeats: Vec<LadderRepeatPoint>,
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
    pub repeats_required: usize,
    pub ladder_values: Vec<usize>,
    pub results: Vec<CandidateLadderResult>,
}

/// Compute largest N on `ladder` where every repeat in `0..repeats_required` passed A1 (`all_sustained && drained`).
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
        if matching.len() >= repeats_required
            && matching
                .iter()
                .all(|p| p.all_sustained && p.drained && p.audio_unfinished_seconds == 0.0)
        {
            best = Some(n);
        }
    }
    best
}

fn run_one_density_point(
    ctx: &SuiteCtx<'_>,
    raw_cand: &str,
    streams_per_worker: usize,
    repeat: usize,
) -> Result<LadderRepeatPoint, CliError> {
    let w_cnt = ctx.density_workers.max(1);
    let dur = Some(ctx.density_seconds.max(1.0));
    if let Some(eng_res) = resolve_r1_weights_dir(raw_cand, ctx.precision) {
        let (weights_dir, prec) = eng_res?;
        let f = run_sp_a_native_engine_shared_workers(
            &EngineSpAConfig {
                weights_dir,
                precision: prec,
                threads: ctx.threads,
                npz_path: ctx.npz_path.to_path_buf(),
                split_name: ctx.split.into(),
                partition: ctx.partition,
                honesty_label: HonestyLabel::InCorpus,
                offset: 0,
                limit: None,
                selected_indices: ctx.selected_indices.map(<[usize]>::to_vec),
                duration_s: dur,
                streams: streams_per_worker,
                decode: DecodeMode::Batch,
                max_batch: 1024,
                max_chunks: ctx.max_chunks,
                packet_ms: 20,
                grid_ms: ctx.grid_ms,
                lateness_bound_ms: DEFAULT_LATENESS_BOUND_MS,
                paced: true,
                source_commit: ctx.commit.into(),
                hardware_id: "H0".into(),
                completion_status: ctx.status_tag.into(),
            },
            w_cnt,
            ctx.observer_cpus,
        )?;
        let max_p99 = f
            .worker_summaries
            .iter()
            .filter_map(|w| w.lateness_ms.as_ref().map(|l| l.p99))
            .max_by(f64::total_cmp);
        return Ok(LadderRepeatPoint {
            repeat,
            streams_per_worker,
            total_streams: w_cnt * streams_per_worker,
            all_sustained: f.all_sustained == Some(true),
            drained: f.worker_summaries.iter().all(|w| w.drained),
            max_worker_p99_ms: max_p99,
            rtf_service_cpu: f.rtf_service_cpu,
            peak_rss_bytes: f.peak_rss_bytes,
            audio_processed_seconds: f.audio_processed_seconds,
            audio_unfinished_seconds: f.audio_unfinished_seconds,
        });
    }
    if let Some(r1_res) = resolve_r1_bundle(raw_cand, ctx.precision) {
        let (bundle_dir, prec) = r1_res?;
        let f = run_sp_a_ort_r1_shared_workers(
            &R1SpAConfig {
                bundle_dir,
                precision: prec,
                threads: ctx.threads,
                npz_path: ctx.npz_path.to_path_buf(),
                split_name: ctx.split.into(),
                partition: ctx.partition,
                honesty_label: HonestyLabel::InCorpus,
                offset: 0,
                limit: None,
                selected_indices: ctx.selected_indices.map(<[usize]>::to_vec),
                duration_s: dur,
                streams: streams_per_worker,
                decode: DecodeMode::Batch,
                max_batch: 1024,
                packet_ms: 20,
                grid_ms: ctx.grid_ms,
                lateness_bound_ms: DEFAULT_LATENESS_BOUND_MS,
                paced: true,
                source_commit: ctx.commit.into(),
                hardware_id: "H0".into(),
                completion_status: ctx.status_tag.into(),
            },
            w_cnt,
            ctx.observer_cpus,
        )?;
        let max_p99 = f
            .worker_summaries
            .iter()
            .filter_map(|w| w.lateness_ms.as_ref().map(|l| l.p99))
            .max_by(f64::total_cmp);
        return Ok(LadderRepeatPoint {
            repeat,
            streams_per_worker,
            total_streams: w_cnt * streams_per_worker,
            all_sustained: f.all_sustained == Some(true),
            drained: f.worker_summaries.iter().all(|w| w.drained),
            max_worker_p99_ms: max_p99,
            rtf_service_cpu: f.rtf_service_cpu,
            peak_rss_bytes: f.peak_rss_bytes,
            audio_processed_seconds: f.audio_processed_seconds,
            audio_unfinished_seconds: f.audio_unfinished_seconds,
        });
    }
    if let Some(z_res) = resolve_bundle_or_zipformer(raw_cand, ctx.precision, ctx.threads, 0.8) {
        let (bundle_id, zcfg) = z_res?;
        let f = run_sp_a_shared_workers_with_observer(
            &SpAConfig {
                bundle_id: Some(bundle_id),
                npz_path: ctx.npz_path.to_path_buf(),
                split_name: ctx.split.into(),
                partition: ctx.partition,
                honesty_label: HonestyLabel::InCorpus,
                offset: 0,
                limit: None,
                selected_indices: ctx.selected_indices.map(<[usize]>::to_vec),
                duration_s: dur,
                streams: streams_per_worker,
                decode: DecodeMode::Batch,
                max_batch: 1024,
                packet_ms: 20,
                grid_ms: ctx.grid_ms,
                lateness_bound_ms: DEFAULT_LATENESS_BOUND_MS,
                paced: true,
                zipformer: zcfg,
                source_commit: ctx.commit.into(),
                hardware_id: "H0".into(),
            },
            w_cnt,
            ctx.observer_cpus,
        )?;
        let max_p99 = f
            .worker_summaries
            .iter()
            .filter_map(|w| w.lateness_ms.as_ref().map(|l| l.p99))
            .max_by(f64::total_cmp);
        return Ok(LadderRepeatPoint {
            repeat,
            streams_per_worker,
            total_streams: w_cnt * streams_per_worker,
            all_sustained: f.all_sustained == Some(true),
            drained: f.worker_summaries.iter().all(|w| w.drained),
            max_worker_p99_ms: max_p99,
            rtf_service_cpu: f.rtf_cpu,
            peak_rss_bytes: f.peak_rss_bytes,
            audio_processed_seconds: f.audio_processed_seconds,
            audio_unfinished_seconds: f.audio_unfinished_seconds,
        });
    }
    Err(CliError::Config(format!(
        "candidate `{raw_cand}` does not support paced density ladder"
    )))
}

/// Execute the §4.4 alternating density ladder across `candidates` and write `density_ladder.json`.
pub(crate) fn run_density_ladder(
    ctx: &SuiteCtx<'_>,
    candidates: &[&str],
    spec: LadderSpec,
    repeats: usize,
) -> Result<DensityLadderReport, CliError> {
    let ladder_values = spec.values();
    let reps = repeats.max(1);
    let mut per_cand: Vec<Vec<LadderRepeatPoint>> = vec![Vec::new(); candidates.len()];

    for &n in &ladder_values {
        for r in 0..reps {
            // Alternate candidate order across repeats (ABAB / BABA).
            let order: Vec<usize> = if r % 2 == 0 {
                (0..candidates.len()).collect()
            } else {
                (0..candidates.len()).rev().collect()
            };
            for ci in order {
                let pt = run_one_density_point(ctx, candidates[ci], n, r)?;
                per_cand[ci].push(pt);
            }
        }
    }

    let w_cnt = ctx.density_workers.max(1);
    let results: Vec<CandidateLadderResult> = candidates
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let claimed = compute_claimed_n(&ladder_values, reps, &per_cand[i]);
            CandidateLadderResult {
                candidate: c.to_owned(),
                workers: w_cnt,
                claimed_streams_per_worker: claimed,
                claimed_total_streams: claimed.map(|n| n * w_cnt),
                repeats: per_cand[i].clone(),
            }
        })
        .collect();

    let report = DensityLadderReport {
        schema: DENSITY_LADDER_SCHEMA.into(),
        split: ctx.split.into(),
        completion_status: ctx.status_tag.into(),
        grid_ms: ctx.grid_ms,
        max_chunks: ctx.max_chunks,
        workers: w_cnt,
        step_seconds: ctx.density_seconds,
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
    fn ladder_parser_and_all_repeats_pass_rule_work() {
        let s = LadderSpec::parse("4:10:2").expect("parse");
        assert_eq!(s.values(), vec![4, 6, 8, 10]);
        assert!(LadderSpec::parse("10:4:2").is_err());
        assert!(LadderSpec::parse("0:4:2").is_err());

        let mk = |n, r, ok| LadderRepeatPoint {
            repeat: r,
            streams_per_worker: n,
            total_streams: n * 4,
            all_sustained: ok,
            drained: ok,
            max_worker_p99_ms: Some(250.0),
            rtf_service_cpu: 0.04,
            peak_rss_bytes: 1024,
            audio_processed_seconds: 300.0,
            audio_unfinished_seconds: 0.0,
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
        // Largest N where ALL 3 repeats pass is 8; if only N=4,6 tested, 6 fails so 4 wins.
        assert_eq!(compute_claimed_n(&[4, 6], 3, &pts), Some(4));
        assert_eq!(compute_claimed_n(&[4, 6, 8], 3, &pts), Some(8));
    }
}
