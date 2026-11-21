//! Pre-registered §4.3 ABAB single-stream whole-service timing-block runner (`FREEZE.md §4.3`).
//!
//! Runs `blocks` (default 5 in qualifying window) alternating single-stream warm blocks across
//! in-process candidates (`R0`, `ORT-R1`, `engine-R1`), reports per-candidate median/min/max
//! `rtf_cpu`, and computes paired block differences (`cand - baseline`) with min/max and a
//! deterministic 10,000-resample paired bootstrap CI over blocks.

use serde::{Deserialize, Serialize};
use ultra_core::receipt::HonestyLabel;
use ultra_stream::{
    run_sp_a_native_engine, run_sp_a_ort_r1, run_sp_a_zipformer, DecodeMode, EngineSpAConfig,
    R1SpAConfig, SpAConfig, DEFAULT_LATENESS_BOUND_MS,
};

use crate::runner::{resolve_bundle_or_zipformer, resolve_r1_bundle};
use crate::suite_engine::resolve_r1_weights_dir;
use crate::suite_manifest::write_json_create_new;
use crate::suite_runners::SuiteCtx;
use crate::CliError;

pub const ABAB_TIMING_SCHEMA: &str = "ultra.abab-timing/v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbabBlockObservation {
    pub block: usize,
    pub order_index: usize,
    pub rtf_cpu: f64,
    pub cpu_seconds: f64,
    pub audio_seconds: f64,
    pub peak_rss_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbabCandidateSummary {
    pub candidate: String,
    pub cost_scope: String,
    pub median_rtf_cpu: f64,
    pub min_rtf_cpu: f64,
    pub max_rtf_cpu: f64,
    pub blocks: Vec<AbabBlockObservation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbabPairedComparison {
    pub baseline: String,
    pub candidate: String,
    pub median_diff_rtf_cpu: f64,
    pub min_diff_rtf_cpu: f64,
    pub max_diff_rtf_cpu: f64,
    pub median_ratio_cand_over_base: f64,
    pub bootstrap_ci95_diff: [f64; 2],
    pub bootstrap_replicates: usize,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbabTimingReport {
    pub schema: String,
    pub split: String,
    pub completion_status: String,
    pub blocks_count: usize,
    pub grid_ms: u32,
    pub max_chunks: usize,
    pub candidates: Vec<AbabCandidateSummary>,
    pub paired_vs_first: Vec<AbabPairedComparison>,
}

fn median_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

pub fn summarize_values(vals: &[f64]) -> (f64, f64, f64) {
    if vals.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut s = vals.to_vec();
    s.sort_by(f64::total_cmp);
    (median_sorted(&s), s[0], s[s.len() - 1])
}

/// Deterministic splitmix64-seeded paired bootstrap CI over block differences (`cand - base`).
pub fn paired_block_comparison(
    baseline_name: &str,
    cand_name: &str,
    base_rtfs: &[f64],
    cand_rtfs: &[f64],
    replicates: usize,
    seed: u64,
) -> AbabPairedComparison {
    let n = base_rtfs.len().min(cand_rtfs.len());
    let diffs: Vec<f64> = (0..n).map(|i| cand_rtfs[i] - base_rtfs[i]).collect();
    let (med_d, min_d, max_d) = summarize_values(&diffs);
    let (med_base, _, _) = summarize_values(&base_rtfs[..n]);
    let (med_cand, _, _) = summarize_values(&cand_rtfs[..n]);
    let ratio = if med_base > 0.0 {
        med_cand / med_base
    } else {
        0.0
    };
    let mut state = seed;
    let mut next_idx = |len: usize| -> usize {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) as usize) % len
    };
    let reps = replicates.max(1);
    let mut boot = Vec::with_capacity(reps);
    let mut sample = vec![0.0f64; n];
    if n > 0 {
        for _ in 0..reps {
            for dst in &mut sample {
                *dst = diffs[next_idx(n)];
            }
            sample.sort_by(f64::total_cmp);
            boot.push(median_sorted(&sample));
        }
        boot.sort_by(f64::total_cmp);
    }
    let ci_lo = if boot.is_empty() {
        0.0
    } else {
        boot[((boot.len() as f64) * 0.025) as usize]
    };
    let ci_hi = if boot.is_empty() {
        0.0
    } else {
        boot[(((boot.len() as f64) * 0.975) as usize).min(boot.len() - 1)]
    };
    AbabPairedComparison {
        baseline: baseline_name.to_owned(),
        candidate: cand_name.to_owned(),
        median_diff_rtf_cpu: med_d,
        min_diff_rtf_cpu: min_d,
        max_diff_rtf_cpu: max_d,
        median_ratio_cand_over_base: ratio,
        bootstrap_ci95_diff: [ci_lo, ci_hi],
        bootstrap_replicates: reps,
        seed,
    }
}

fn run_single_warm_block(
    ctx: &SuiteCtx<'_>,
    raw_cand: &str,
    block: usize,
    order_index: usize,
) -> Result<AbabBlockObservation, CliError> {
    if let Some(eng_res) = resolve_r1_weights_dir(raw_cand, ctx.precision) {
        let (weights_dir, prec) = eng_res?;
        let r = run_sp_a_native_engine(&EngineSpAConfig {
            weights_dir,
            precision: prec,
            threads: ctx.threads,
            npz_path: ctx.npz_path.to_path_buf(),
            split_name: ctx.split.into(),
            partition: ctx.partition,
            honesty_label: HonestyLabel::InCorpus,
            offset: 0,
            limit: ctx.limit,
            selected_indices: ctx.selected_indices.map(<[usize]>::to_vec),
            duration_s: None,
            streams: 1,
            decode: DecodeMode::Single,
            max_batch: 1,
            max_chunks: ctx.max_chunks,
            packet_ms: 20,
            grid_ms: ctx.grid_ms,
            lateness_bound_ms: DEFAULT_LATENESS_BOUND_MS,
            paced: false,
            source_commit: ctx.commit.into(),
            hardware_id: "H0".into(),
            completion_status: ctx.status_tag.into(),
        })?;
        return Ok(AbabBlockObservation {
            block,
            order_index,
            rtf_cpu: r.rtf_cpu,
            cpu_seconds: r.cpu_seconds,
            audio_seconds: r.audio_seconds,
            peak_rss_bytes: r.receipt.peak_rss_bytes,
        });
    }
    if let Some(r1_res) = resolve_r1_bundle(raw_cand, ctx.precision) {
        let (bundle_dir, prec) = r1_res?;
        let r = run_sp_a_ort_r1(&R1SpAConfig {
            bundle_dir,
            precision: prec,
            threads: ctx.threads,
            npz_path: ctx.npz_path.to_path_buf(),
            split_name: ctx.split.into(),
            partition: ctx.partition,
            honesty_label: HonestyLabel::InCorpus,
            offset: 0,
            limit: ctx.limit,
            selected_indices: ctx.selected_indices.map(<[usize]>::to_vec),
            duration_s: None,
            streams: 1,
            decode: DecodeMode::Single,
            max_batch: 1,
            packet_ms: 20,
            grid_ms: ctx.grid_ms,
            lateness_bound_ms: DEFAULT_LATENESS_BOUND_MS,
            paced: false,
            source_commit: ctx.commit.into(),
            hardware_id: "H0".into(),
            completion_status: ctx.status_tag.into(),
        })?;
        return Ok(AbabBlockObservation {
            block,
            order_index,
            rtf_cpu: r.rtf_cpu,
            cpu_seconds: r.cpu_seconds,
            audio_seconds: r.audio_seconds,
            peak_rss_bytes: r.receipt.peak_rss_bytes,
        });
    }
    if let Some(z_res) = resolve_bundle_or_zipformer(raw_cand, ctx.precision, ctx.threads, 0.8) {
        let (bundle_id, zcfg) = z_res?;
        let r = run_sp_a_zipformer(&SpAConfig {
            bundle_id: Some(bundle_id),
            npz_path: ctx.npz_path.to_path_buf(),
            split_name: ctx.split.into(),
            partition: ctx.partition,
            honesty_label: HonestyLabel::InCorpus,
            offset: 0,
            limit: ctx.limit,
            selected_indices: ctx.selected_indices.map(<[usize]>::to_vec),
            duration_s: None,
            streams: 1,
            decode: DecodeMode::Single,
            max_batch: 1,
            packet_ms: 20,
            grid_ms: ctx.grid_ms,
            lateness_bound_ms: DEFAULT_LATENESS_BOUND_MS,
            paced: false,
            zipformer: zcfg,
            source_commit: ctx.commit.into(),
            hardware_id: "H0".into(),
        })?;
        return Ok(AbabBlockObservation {
            block,
            order_index,
            rtf_cpu: r.rtf_cpu,
            cpu_seconds: r.cpu_seconds,
            audio_seconds: r.audio_seconds,
            peak_rss_bytes: r.receipt.peak_rss_bytes,
        });
    }
    Err(CliError::Config(format!(
        "candidate `{raw_cand}` is excluded from C-cost ABAB blocks (FREEZE.md §3: in-process warm scope only)"
    )))
}

/// Execute `blocks` alternating single-stream timing blocks (`FREEZE.md §4.3`) and write `abab_timing.json`.
pub(crate) fn run_abab_timing_blocks(
    ctx: &SuiteCtx<'_>,
    candidates: &[&str],
    blocks: usize,
) -> Result<AbabTimingReport, CliError> {
    if candidates.is_empty() {
        return Err(CliError::Config(
            "--abab-blocks requires >= 1 candidate".into(),
        ));
    }
    let mut per_cand: Vec<Vec<AbabBlockObservation>> =
        vec![Vec::with_capacity(blocks); candidates.len()];
    for b in 0..blocks {
        let order: Vec<usize> = if b % 2 == 0 {
            (0..candidates.len()).collect()
        } else {
            (0..candidates.len()).rev().collect()
        };
        for (pos, &ci) in order.iter().enumerate() {
            let obs = run_single_warm_block(ctx, candidates[ci], b, pos)?;
            per_cand[ci].push(obs);
        }
    }
    let summaries: Vec<AbabCandidateSummary> = candidates
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let rtfs: Vec<f64> = per_cand[i].iter().map(|o| o.rtf_cpu).collect();
            let (med, min_v, max_v) = summarize_values(&rtfs);
            AbabCandidateSummary {
                candidate: c.to_owned(),
                cost_scope: "in_process_warm_rusage_self_post_load".into(),
                median_rtf_cpu: med,
                min_rtf_cpu: min_v,
                max_rtf_cpu: max_v,
                blocks: per_cand[i].clone(),
            }
        })
        .collect();
    let base_rtfs: Vec<f64> = per_cand[0].iter().map(|o| o.rtf_cpu).collect();
    let paired_vs_first: Vec<AbabPairedComparison> = candidates
        .iter()
        .enumerate()
        .skip(1)
        .map(|(i, &c)| {
            let rtfs: Vec<f64> = per_cand[i].iter().map(|o| o.rtf_cpu).collect();
            paired_block_comparison(candidates[0], c, &base_rtfs, &rtfs, 10_000, 20_261_121)
        })
        .collect();

    let report = AbabTimingReport {
        schema: ABAB_TIMING_SCHEMA.into(),
        split: ctx.split.into(),
        completion_status: ctx.status_tag.into(),
        blocks_count: blocks,
        grid_ms: ctx.grid_ms,
        max_chunks: ctx.max_chunks,
        candidates: summaries,
        paired_vs_first,
    };
    write_json_create_new(&ctx.out_dir.join("abab_timing.json"), &report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abab_summary_and_paired_bootstrap_compute_exact_medians() {
        let base = [0.080, 0.078, 0.082, 0.079, 0.081];
        let cand = [0.041, 0.040, 0.042, 0.041, 0.040];
        let (med_b, min_b, max_b) = summarize_values(&base);
        assert!((med_b - 0.080).abs() < 1e-9);
        assert!((min_b - 0.078).abs() < 1e-9);
        assert!((max_b - 0.082).abs() < 1e-9);

        let cmp = paired_block_comparison("r0", "r1-engine", &base, &cand, 2_000, 42);
        assert!((cmp.median_diff_rtf_cpu - (-0.039)).abs() < 1e-9);
        assert!(cmp.bootstrap_ci95_diff[0] <= cmp.median_diff_rtf_cpu);
        assert!(cmp.bootstrap_ci95_diff[1] >= cmp.median_diff_rtf_cpu);
        assert!(cmp.bootstrap_ci95_diff[1] < 0.0);
        assert!((cmp.median_ratio_cand_over_base - (0.041 / 0.080)).abs() < 1e-9);
    }
}
