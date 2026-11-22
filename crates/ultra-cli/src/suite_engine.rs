//! Pure-Rust ONNX-free [`NativeEngineBackend`] (`ultra.r1-weights/v1` SafeTensors) candidate
//! resolver and `ultra bench-suite` runner (`E4` / `R-ACCEL-3`).

use std::path::{Path, PathBuf};

use ultra_core::receipt::HonestyLabel;
use ultra_engine::model::WEIGHTS_SCHEMA;
use ultra_engine::Precision;
use ultra_stream::{
    run_sp_a_native_engine, run_sp_a_native_engine_shared_workers, DecodeMode, EngineSpAConfig,
    DEFAULT_LATENESS_BOUND_MS,
};

use crate::suite_latency::{DensityPoint, FrontierRow};
use crate::suite_manifest::{
    current_loadavg_3, sanitize_slug, write_json_create_new, write_jsonl_create_new,
    write_receipt_create_new, write_scores_create_new,
};
use crate::suite_runners::{eval_backend_probes, SuiteCtx};
use crate::suite_slices::compute_slice_summaries;
use crate::{jsr_data_root, CliError};

fn peek_arch_schema(dir: &Path) -> Option<String> {
    let bytes = std::fs::read(dir.join("arch.json")).ok()?;
    let val: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    val.get("schema")
        .and_then(|s| s.as_str())
        .map(str::to_owned)
}

/// Resolve an `ultra.r1-weights/v1` SafeTensors directory (`arch.json`) or named `r1-engine*` alias.
pub fn resolve_r1_weights_dir(
    candidate_id: &str,
    precision_override: Option<&str>,
) -> Option<Result<(PathBuf, Precision), CliError>> {
    let data = PathBuf::from(jsr_data_root());
    let (candidate_dir, default_prec) = match candidate_id {
        "r1-engine" | "r1-ultra-engine" | "r1-d1-engine" | "r1-d1-engine-int8" => {
            let prep = data.join("runs/engine/weights-r1-d1-ckpt7800-v3-int8prep-723da8f");
            let fp32 = data.join("runs/engine/weights-r1-d1-ckpt7800-v3");
            let want_fp32 = precision_override == Some("fp32");
            let dir = if !want_fp32 && prep.join("arch.json").exists() {
                prep
            } else {
                fp32
            };
            (Some(dir), "int8")
        }
        "r1-engine-fp32" | "r1-d1-engine-fp32" => (
            Some(data.join("runs/engine/weights-r1-d1-ckpt7800-v3")),
            "fp32",
        ),
        "r1-d1mr-engine" | "r1-d1mr-engine-int8" => (
            Some(data.join("runs/engine/weights-d1mr-40k-int8prep")),
            "int8",
        ),
        "r1-r4t-engine" => (
            Some(data.join("runs/engine/r4t-12000/engine-weights")),
            "int8",
        ),
        _ => {
            let direct = Path::new(candidate_id);
            let named = data.join("bundles").join(candidate_id);
            let dir = if direct.join("arch.json").exists() {
                Some(direct.to_path_buf())
            } else if named.join("arch.json").exists() {
                Some(named)
            } else {
                None
            };
            (dir, "int8")
        }
    };
    let candidate_dir = candidate_dir?;
    if peek_arch_schema(&candidate_dir).as_deref() != Some(WEIGHTS_SCHEMA) {
        return None;
    }
    let prec_s = precision_override.unwrap_or(default_prec);
    Some(
        Precision::parse(prec_s)
            .map(|p| (candidate_dir, p))
            .map_err(|e| CliError::Config(e.to_string())),
    )
}

pub(crate) fn run_engine_candidate(
    ctx: &SuiteCtx<'_>,
    raw_cand: &str,
    weights_dir: PathBuf,
    prec: Precision,
) -> Result<FrontierRow, CliError> {
    let slug = sanitize_slug(raw_cand);
    let scores_path = ctx.out_dir.join(format!("{slug}.scores.jsonl"));
    let receipt_path = ctx.out_dir.join(format!("{slug}.receipt.json"));
    let jm_path = ctx.out_dir.join(format!("{slug}.job_manifest.json"));
    let load_before = current_loadavg_3();
    let mk_cfg = |st: usize, dec: DecodeMode, paced: bool, dur: Option<f64>| EngineSpAConfig {
        weights_dir: weights_dir.clone(),
        precision: prec,
        threads: ctx.threads,
        npz_path: ctx.npz_path.to_path_buf(),
        split_name: ctx.split.into(),
        partition: ctx.partition,
        honesty_label: HonestyLabel::InCorpus,
        offset: 0,
        limit: if dur.is_some() { None } else { ctx.limit },
        selected_indices: ctx.selected_indices.map(<[usize]>::to_vec),
        duration_s: dur,
        streams: st,
        decode: dec,
        max_batch: 1024,
        max_chunks: ctx.max_chunks,
        beam: ctx.beam,
        packet_ms: 20,
        grid_ms: ctx.grid_ms,
        lateness_bound_ms: DEFAULT_LATENESS_BOUND_MS,
        paced,
        source_commit: ctx.commit.into(),
        hardware_id: ctx.hardware_id.into(),
        completion_status: ctx.status_tag.into(),
    };
    let single = run_sp_a_native_engine(&mk_cfg(1, DecodeMode::Single, false, None))?;
    let batch = run_sp_a_native_engine(&mk_cfg(ctx.batch_streams, DecodeMode::Batch, false, None))?;
    let dens =
        if ctx.density_seconds > 0.0 && (ctx.density_workers > 1 || ctx.observer_cpus.is_some()) {
            let w_cnt = ctx.density_workers.max(1);
            let f = run_sp_a_native_engine_shared_workers(
                &mk_cfg(
                    ctx.density_streams,
                    DecodeMode::Batch,
                    true,
                    Some(ctx.density_seconds),
                ),
                w_cnt,
                ctx.observer_cpus,
            )?;
            Some(DensityPoint {
                workers: w_cnt,
                streams_per_worker: ctx.density_streams,
                total_streams: w_cnt * ctx.density_streams,
                sustained: f.all_sustained,
                drained: f.worker_summaries.iter().all(|w| w.drained),
                lateness_p99_ms: f
                    .worker_summaries
                    .iter()
                    .filter_map(|w| w.lateness_ms.as_ref().map(|l| l.p99))
                    .max_by(f64::total_cmp),
                peak_rss_bytes: f.receipt.peak_rss_bytes,
            })
        } else if ctx.density_seconds > 0.0 {
            let d = run_sp_a_native_engine(&mk_cfg(
                ctx.density_streams,
                DecodeMode::Batch,
                true,
                Some(ctx.density_seconds),
            ))?;
            Some(DensityPoint::from_sp_a(&d, ctx.density_streams))
        } else {
            None
        };
    let (lat, sil) = eval_backend_probes(ctx, raw_cand, &slug)?;
    let sust = if let Some(sec) = ctx.sustained_seconds {
        let sw = ctx.sustained_workers.max(1);
        let sn = ctx.sustained_streams.max(1);
        if sw > 1 || ctx.observer_cpus.is_some() {
            let f = run_sp_a_native_engine_shared_workers(
                &mk_cfg(sn, DecodeMode::Batch, true, Some(sec)),
                sw,
                ctx.observer_cpus,
            )?;
            write_receipt_create_new(
                &ctx.out_dir.join(format!("{slug}.sustained.receipt.json")),
                &f.receipt,
            )?;
            write_jsonl_create_new(
                &ctx.out_dir.join(format!("{slug}.sustained.series.jsonl")),
                &f.series,
            )?;
            Some(DensityPoint {
                workers: sw,
                streams_per_worker: sn,
                total_streams: sw * sn,
                sustained: f.all_sustained,
                drained: f.worker_summaries.iter().all(|w| w.drained),
                lateness_p99_ms: f
                    .worker_summaries
                    .iter()
                    .filter_map(|w| w.lateness_ms.as_ref().map(|l| l.p99))
                    .max_by(f64::total_cmp),
                peak_rss_bytes: f.receipt.peak_rss_bytes,
            })
        } else {
            let s = run_sp_a_native_engine(&mk_cfg(sn, DecodeMode::Batch, true, Some(sec)))?;
            write_receipt_create_new(
                &ctx.out_dir.join(format!("{slug}.sustained.receipt.json")),
                &s.receipt,
            )?;
            Some(DensityPoint::from_sp_a(&s, sn))
        }
    } else {
        None
    };
    let scope = "in_process_warm_rusage_self_post_load";
    let jm = ctx.host_snap.build_job_manifest(
        raw_cand,
        &batch.receipt.bundle,
        &batch.driver,
        scope,
        ctx.commit,
        ctx.split,
        ctx.npz_path,
        ctx.panel_manifest_sha256,
        ctx.grid_ms,
        ctx.max_chunks,
        Some(prec.as_str()),
        ctx.threads,
        load_before,
        ctx.cli_args,
    );
    write_json_create_new(&jm_path, &jm)?;
    write_scores_create_new(&scores_path, &batch.scored_rows)?;
    write_receipt_create_new(&receipt_path, &batch.receipt)?;
    Ok(FrontierRow {
        candidate: raw_cand.to_owned(),
        loaded_identity: batch.receipt.bundle.clone(),
        driver: batch.driver,
        cost_scope: scope.into(),
        honesty_label: batch.receipt.label,
        partition: ctx.partition,
        completion_status: batch.receipt.completion_status.clone(),
        utterances: batch.utterances,
        audio_seconds: batch.audio_seconds,
        wer_pct: batch.wer.wer_pct,
        single_rtf_cpu: single.rtf_cpu,
        single_streams_per_core: single.streams_per_core_at_full_load,
        batch_rtf_cpu: Some(batch.rtf_cpu),
        batch_streams_per_core: Some(batch.streams_per_core_at_full_load),
        chunk_ms: batch.receipt.chunk_ms.clone(),
        peak_rss_bytes: batch.receipt.peak_rss_bytes,
        cap_hits: batch.receipt.cap_hits,
        evidence_complete: batch.receipt.validate_as_evidence().is_ok(),
        density: dens,
        latency: lat,
        sustained_probe: sust,
        slices: compute_slice_summaries(&batch.scored_rows),
        silence_probe: sil,
        job_manifest_file: jm_path.to_string_lossy().into_owned(),
        scores_file: scores_path.to_string_lossy().into_owned(),
        receipt_file: receipt_path.to_string_lossy().into_owned(),
    })
}
