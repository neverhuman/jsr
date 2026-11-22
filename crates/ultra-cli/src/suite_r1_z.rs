//! Streaming `R1` (`OrtTransducerBackend`) and `R0` (`SherpaZipformerBackend`) candidate
//! runners for `ultra bench-suite`, supporting multi-worker CCX fleets and disjoint observer CPUs.

use std::path::PathBuf;

use ultra_core::receipt::HonestyLabel;
use ultra_gp::R1Precision;
use ultra_stream::{
    run_sp_a_ort_r1, run_sp_a_ort_r1_shared_workers, run_sp_a_shared_workers_with_observer,
    run_sp_a_zipformer, DecodeMode, R1SpAConfig, SherpaZipformerConfig, SpAConfig,
    DEFAULT_LATENESS_BOUND_MS,
};

use crate::suite_latency::{DensityPoint, FrontierRow};
use crate::suite_manifest::{
    current_loadavg_3, sanitize_slug, write_json_create_new, write_jsonl_create_new,
    write_receipt_create_new, write_scores_create_new,
};
use crate::suite_runners::{eval_backend_probes, SuiteCtx};
use crate::suite_slices::compute_slice_summaries;
use crate::CliError;

pub(crate) fn run_r1_candidate(
    ctx: &SuiteCtx<'_>,
    raw_cand: &str,
    bundle_dir: PathBuf,
    prec: R1Precision,
) -> Result<FrontierRow, CliError> {
    let slug = sanitize_slug(raw_cand);
    let scores_path = ctx.out_dir.join(format!("{slug}.scores.jsonl"));
    let receipt_path = ctx.out_dir.join(format!("{slug}.receipt.json"));
    let jm_path = ctx.out_dir.join(format!("{slug}.job_manifest.json"));
    let load_before = current_loadavg_3();
    let mk_cfg = |st: usize, dec: DecodeMode, paced: bool, dur: Option<f64>| R1SpAConfig {
        bundle_dir: bundle_dir.clone(),
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
        packet_ms: 20,
        grid_ms: ctx.grid_ms,
        lateness_bound_ms: DEFAULT_LATENESS_BOUND_MS,
        paced,
        source_commit: ctx.commit.into(),
        hardware_id: ctx.hardware_id.into(),
        completion_status: ctx.status_tag.into(),
    };
    let single = run_sp_a_ort_r1(&mk_cfg(1, DecodeMode::Single, false, None))?;
    let batch = run_sp_a_ort_r1(&mk_cfg(ctx.batch_streams, DecodeMode::Batch, false, None))?;
    let dens =
        if ctx.density_seconds > 0.0 && (ctx.density_workers > 1 || ctx.observer_cpus.is_some()) {
            let w_cnt = ctx.density_workers.max(1);
            let f = run_sp_a_ort_r1_shared_workers(
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
            let d = run_sp_a_ort_r1(&mk_cfg(
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
            let f = run_sp_a_ort_r1_shared_workers(
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
            let s = run_sp_a_ort_r1(&mk_cfg(sn, DecodeMode::Batch, true, Some(sec)))?;
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

pub(crate) fn run_zipformer_candidate(
    ctx: &SuiteCtx<'_>,
    raw_cand: &str,
    bundle_id: String,
    zcfg: SherpaZipformerConfig,
) -> Result<FrontierRow, CliError> {
    let slug = sanitize_slug(raw_cand);
    let scores_path = ctx.out_dir.join(format!("{slug}.scores.jsonl"));
    let receipt_path = ctx.out_dir.join(format!("{slug}.receipt.json"));
    let jm_path = ctx.out_dir.join(format!("{slug}.job_manifest.json"));
    let load_before = current_loadavg_3();
    let mk_z = |st: usize, dec: DecodeMode, paced: bool, dur: Option<f64>| SpAConfig {
        bundle_id: Some(bundle_id.clone()),
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
        packet_ms: 20,
        grid_ms: ctx.grid_ms,
        lateness_bound_ms: DEFAULT_LATENESS_BOUND_MS,
        paced,
        zipformer: zcfg.clone(),
        source_commit: ctx.commit.into(),
        hardware_id: ctx.hardware_id.into(),
    };
    let single = run_sp_a_zipformer(&mk_z(1, DecodeMode::Single, false, None))?;
    let mut batch = run_sp_a_zipformer(&mk_z(ctx.batch_streams, DecodeMode::Batch, false, None))?;
    batch.receipt.completion_status = ctx.status_tag.into();
    let dens =
        if ctx.density_seconds > 0.0 && (ctx.density_workers > 1 || ctx.observer_cpus.is_some()) {
            let w_cnt = ctx.density_workers.max(1);
            let f = run_sp_a_shared_workers_with_observer(
                &mk_z(
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
            let d = run_sp_a_zipformer(&mk_z(
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
            let mut f = run_sp_a_shared_workers_with_observer(
                &mk_z(sn, DecodeMode::Batch, true, Some(sec)),
                sw,
                ctx.observer_cpus,
            )?;
            f.receipt.completion_status = ctx.status_tag.into();
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
            let mut s = run_sp_a_zipformer(&mk_z(sn, DecodeMode::Batch, true, Some(sec)))?;
            s.receipt.completion_status = ctx.status_tag.into();
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
        Some(zcfg.precision.as_str()),
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
