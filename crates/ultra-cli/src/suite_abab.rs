//! Pre-registered §4.3 / §5 ABAB single-stream whole-service timing-block runner (`FREEZE.md §4.3`, `§5`, `N10`).
//!
//! Runs `blocks` (`>= 8` required in `--qualified`) alternating single-stream warm blocks across
//! in-process candidates (`R0`, `ORT-R1`, `engine-R1`), requires an explicit or resolved `R0`
//! baseline (never arbitrary `candidates[0]`), and delegates paired log-ratio t-test + Holm
//! statistics to [`crate::suite_stats`].

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
pub use crate::suite_stats::{
    apply_holm_correction, paired_block_comparison, student_t_cdf, student_t_critical_975,
    summarize_values, AbabPairedComparison,
};
use crate::CliError;

pub const ABAB_TIMING_SCHEMA: &str = "ultra.abab-timing/v2";

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
pub struct AbabTimingReport {
    pub schema: String,
    pub split: String,
    pub completion_status: String,
    pub blocks_count: usize,
    pub baseline_candidate: String,
    pub grid_ms: u32,
    pub max_chunks: usize,
    pub candidates: Vec<AbabCandidateSummary>,
    pub paired_vs_baseline: Vec<AbabPairedComparison>,
}

pub fn is_r0_baseline_candidate(cand: &str) -> bool {
    let lower = cand.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "r0" | "r0-zipformer-int8"
            | "zipformer-int8"
            | "zipformer_en_int8_320"
            | "sherpa-onnx-streaming-zipformer-en"
            | "sherpa-onnx-streaming-zipformer-en-int8"
    ) || (lower.contains("zipformer") && !lower.contains("fp32"))
}

/// Validate the pre-registered `FREEZE.md §5` (`04d09f0`, `N10`) `--qualified` C-cost matrix:
/// exact 8 counterbalanced 3-arm blocks in order `R0 → R1-dense → R1-MR` (reverse on odd blocks),
/// producing `n = 8` pairs (`df = 7`) for the 2 Holm-adjusted contrasts against R0.
pub fn validate_qualified_ccost_arms(
    candidates: &[&str],
    blocks: usize,
    explicit_baseline: Option<&str>,
) -> Result<usize, CliError> {
    if blocks != 8 {
        return Err(CliError::Config(format!(
            "--qualified C-cost requires exactly 8 counterbalanced blocks per FREEZE.md §5 (got {blocks}; N10)"
        )));
    }
    if candidates.len() != 3 {
        return Err(CliError::Config(format!(
            "--qualified C-cost requires the exact 3-arm R0,R1-dense,R1-MR matrix per FREEZE.md §5 (got {} candidates: {candidates:?}; N10)",
            candidates.len()
        )));
    }
    if !is_r0_baseline_candidate(candidates[0]) {
        return Err(CliError::Config(format!(
            "--qualified C-cost requires R0 (Zipformer INT8) as arm 0 in `R0,R1-dense,R1-MR` order (got `{}`; FREEZE.md §5 / N10)",
            candidates[0]
        )));
    }
    if let Some(b) = explicit_baseline {
        if b != candidates[0] {
            return Err(CliError::Config(format!(
                "--qualified C-cost --abab-baseline `{b}` must match arm 0 R0 `{}` (N10)",
                candidates[0]
            )));
        }
    }
    if candidates[1] == candidates[2]
        || is_r0_baseline_candidate(candidates[1])
        || is_r0_baseline_candidate(candidates[2])
    {
        return Err(CliError::Config(format!(
            "--qualified C-cost arms 1 and 2 must be two distinct R1 finalists (got `{}` and `{}`; FREEZE.md §5 / N10)",
            candidates[1], candidates[2]
        )));
    }
    Ok(0)
}

/// Resolve the named R0 baseline index in `candidates` (`FREEZE.md §5` / `N10`).
pub fn resolve_abab_baseline_index(
    candidates: &[&str],
    explicit_baseline: Option<&str>,
    qualified: bool,
) -> Result<usize, CliError> {
    if candidates.is_empty() {
        return Err(CliError::Config(
            "--abab-blocks requires >= 1 candidate".into(),
        ));
    }
    if let Some(b) = explicit_baseline {
        if qualified && !is_r0_baseline_candidate(b) {
            return Err(CliError::Config(format!(
                "--qualified C-cost ABAB baseline must be R0 (Zipformer INT8), got `{b}` (FREEZE.md §5 / N10)"
            )));
        }
        return candidates.iter().position(|&c| c == b).ok_or_else(|| {
            CliError::Config(format!(
                "--abab-baseline `{b}` is not present in --candidates {candidates:?}"
            ))
        });
    }
    candidates
        .iter()
        .position(|&c| is_r0_baseline_candidate(c))
        .ok_or_else(|| {
            CliError::Config(format!(
                "ABAB timing requires an R0 baseline (`zipformer-int8` / `zipformer_en_int8_320`) in --candidates {candidates:?} or explicit --abab-baseline (FREEZE.md §5 / N10)"
            ))
        })
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
            hardware_id: ctx.hardware_id.into(),
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
            hardware_id: ctx.hardware_id.into(),
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
            hardware_id: ctx.hardware_id.into(),
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

/// Execute `blocks` alternating single-stream timing blocks (`FREEZE.md §4.3` / `§5`) and write `abab_timing.json`.
pub(crate) fn run_abab_timing_blocks(
    ctx: &SuiteCtx<'_>,
    candidates: &[&str],
    blocks: usize,
    baseline_idx: usize,
) -> Result<AbabTimingReport, CliError> {
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
    let base_name = candidates[baseline_idx];
    let base_rtfs: Vec<f64> = per_cand[baseline_idx].iter().map(|o| o.rtf_cpu).collect();
    let mut paired_vs_baseline: Vec<AbabPairedComparison> = candidates
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != baseline_idx)
        .map(|(i, &c)| {
            let rtfs: Vec<f64> = per_cand[i].iter().map(|o| o.rtf_cpu).collect();
            paired_block_comparison(base_name, c, &base_rtfs, &rtfs, 10_000, 20_261_121)
        })
        .collect();
    apply_holm_correction(&mut paired_vs_baseline, 0.05);

    let report = AbabTimingReport {
        schema: ABAB_TIMING_SCHEMA.into(),
        split: ctx.split.into(),
        completion_status: ctx.status_tag.into(),
        blocks_count: blocks,
        baseline_candidate: base_name.to_owned(),
        grid_ms: ctx.grid_ms,
        max_chunks: ctx.max_chunks,
        candidates: summaries,
        paired_vs_baseline,
    };
    write_json_create_new(&ctx.out_dir.join("abab_timing.json"), &report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abab_baseline_resolution_requires_r0() {
        assert_eq!(
            resolve_abab_baseline_index(&["r1-engine", "zipformer-int8"], None, true).expect("r0"),
            1
        );
        assert!(resolve_abab_baseline_index(&["r1-engine", "r1-d1"], None, true).is_err());
        assert!(resolve_abab_baseline_index(
            &["r1-engine", "zipformer-int8"],
            Some("r1-engine"),
            true
        )
        .is_err());
    }
}
