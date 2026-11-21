//! Context, Nemotron external process-tree runner, offline sherpa runner, and probe helpers
//! for `ultra bench-suite`.

use std::path::{Path, PathBuf};
use std::time::Duration;

use ultra_adapters::{run_nemotron_batch, NemotronChunkMode, NemotronConfig};
use ultra_core::receipt::{HonestyLabel, Partition};
use ultra_stream::{run_offline_contrast, ContrastConfig};

use crate::runner::{instantiate_backend_with_precision, resolve_offline_config};
use crate::suite_latency::{measure_paced_latency, FrontierRow};
use crate::suite_manifest::{
    current_loadavg_3, sanitize_slug, write_json_create_new, write_receipt_create_new,
    write_scores_create_new, SuiteHostSnapshot,
};
use crate::suite_slices::{compute_slice_summaries, run_silence_probe_backend};
use crate::{jsr_data_root, CliError};

pub(crate) struct SuiteCtx<'a> {
    pub out_dir: &'a Path,
    pub npz_path: &'a Path,
    pub split: &'a str,
    pub partition: Partition,
    pub limit: Option<usize>,
    pub selected_indices: Option<&'a [usize]>,
    pub panel_manifest_sha256: Option<&'a str>,
    pub batch_streams: usize,
    pub grid_ms: u32,
    pub max_chunks: usize,
    pub density_workers: usize,
    pub density_streams: usize,
    pub density_seconds: f64,
    pub latency_limit: usize,
    pub sustained_seconds: Option<f64>,
    pub sustained_streams: usize,
    pub sustained_workers: usize,
    pub silence_probe_wav: Option<&'a (PathBuf, bool)>,
    pub observer_cpus: Option<&'a [usize]>,
    pub precision: Option<&'a str>,
    pub threads: i32,
    pub commit: &'a str,
    pub status_tag: &'a str,
    pub host_snap: &'a SuiteHostSnapshot,
    pub cli_args: &'a [String],
}

pub(crate) fn run_nemotron_candidate(
    ctx: &SuiteCtx<'_>,
    raw_cand: &str,
) -> Result<FrontierRow, CliError> {
    let slug = sanitize_slug(raw_cand);
    let scores_path = ctx.out_dir.join(format!("{slug}.scores.jsonl"));
    let receipt_path = ctx.out_dir.join(format!("{slug}.receipt.json"));
    let jm_path = ctx.out_dir.join(format!("{slug}.job_manifest.json"));
    let load_before = current_loadavg_3();
    let mode = if raw_cand.contains("160") {
        NemotronChunkMode::Live160
    } else {
        NemotronChunkMode::Live560
    };
    let mut ncfg = NemotronConfig::default_h0(mode, ctx.npz_path.to_path_buf(), ctx.split.into());
    ncfg.partition = ctx.partition;
    ncfg.limit = ctx.limit;
    ncfg.threads = ctx.threads.max(1) as u32;
    ncfg.source_commit = ctx.commit.into();
    ncfg.completion_status = ctx.status_tag.into();
    ncfg.timeout = Duration::from_secs(3600);
    ncfg.retain_dir = Some(ctx.out_dir.join(format!("{slug}_artifacts")));
    let single = run_nemotron_batch(&ncfg).map_err(|e| CliError::Internal(e.to_string()))?;
    let loaded_id = format!(
        "{}:bin={}:gguf={}",
        single.candidate_id,
        &single.binary_sha256[..12],
        &single.model_sha256[..12]
    );
    let (driver, scope) = (
        "rust-nemo-speech-adapter-v1",
        "post_prep_process_tree_service_cold_load",
    );
    let jm = ctx.host_snap.build_job_manifest(
        &single.candidate_id,
        &loaded_id,
        driver,
        scope,
        ctx.commit,
        ctx.split,
        ctx.npz_path,
        ctx.panel_manifest_sha256,
        ctx.grid_ms,
        ctx.max_chunks,
        ctx.precision,
        ctx.threads,
        load_before,
        ctx.cli_args,
    );
    write_json_create_new(&jm_path, &jm)?;
    write_scores_create_new(&scores_path, &single.scored_rows)?;
    write_receipt_create_new(&receipt_path, &single.receipt)?;
    Ok(FrontierRow {
        candidate: single.candidate_id,
        loaded_identity: loaded_id,
        driver: driver.into(),
        cost_scope: scope.into(),
        honesty_label: single.receipt.label,
        partition: ctx.partition,
        completion_status: single.receipt.completion_status.clone(),
        utterances: single.utterances,
        audio_seconds: single.audio_seconds,
        wer_pct: single.wer_pct.unwrap_or(0.0),
        single_rtf_cpu: single.rtf_cpu,
        single_streams_per_core: single.streams_per_core_at_full_load,
        batch_rtf_cpu: None,
        batch_streams_per_core: None,
        chunk_ms: single.receipt.chunk_ms.clone(),
        peak_rss_bytes: single.receipt.peak_rss_bytes,
        cap_hits: single.receipt.cap_hits,
        evidence_complete: single.receipt.validate_as_evidence().is_ok(),
        density: None,
        latency: None,
        sustained_probe: None,
        slices: compute_slice_summaries(&single.scored_rows),
        silence_probe: None,
        job_manifest_file: jm_path.to_string_lossy().into_owned(),
        scores_file: scores_path.to_string_lossy().into_owned(),
        receipt_file: receipt_path.to_string_lossy().into_owned(),
    })
}

pub(crate) fn run_offline_candidate(
    ctx: &SuiteCtx<'_>,
    raw_cand: &str,
) -> Result<FrontierRow, CliError> {
    if let Some(p) = ctx.precision.filter(|&p| p != "int8") {
        return Err(CliError::Config(format!(
            "--precision {p} unsupported for offline candidate `{raw_cand}`"
        )));
    }
    let slug = sanitize_slug(raw_cand);
    let scores_path = ctx.out_dir.join(format!("{slug}.scores.jsonl"));
    let receipt_path = ctx.out_dir.join(format!("{slug}.receipt.json"));
    let jm_path = ctx.out_dir.join(format!("{slug}.job_manifest.json"));
    let load_before = current_loadavg_3();
    let models_root = PathBuf::from(jsr_data_root()).join("models");
    let off_cfg = resolve_offline_config(raw_cand, &models_root, ctx.threads)?;
    let mut res = run_offline_contrast(&ContrastConfig {
        npz_path: ctx.npz_path.to_path_buf(),
        split_name: ctx.split.into(),
        partition: ctx.partition,
        honesty_label: HonestyLabel::UnknownOverlap,
        offset: 0,
        limit: ctx.limit,
        batch_size: 1,
        offline: off_cfg,
        source_commit: ctx.commit.into(),
        hardware_id: "H0".into(),
    })?;
    res.receipt.completion_status = ctx.status_tag.into();
    let sil = match ctx.silence_probe_wav {
        Some((wp, req_sha)) => {
            let b = instantiate_backend_with_precision(raw_cand, Some("int8"), ctx.threads)?;
            let sp_path = ctx.out_dir.join(format!("{slug}.silence.scores.jsonl"));
            Some(run_silence_probe_backend(
                &*b,
                wp,
                *req_sha,
                Some(&sp_path),
            )?)
        }
        None => None,
    };
    let (driver, scope) = ("rust-sherpa-offline-c-api", "in_process_offline_recognizer");
    let jm = ctx.host_snap.build_job_manifest(
        &res.candidate_id,
        &res.receipt.bundle,
        driver,
        scope,
        ctx.commit,
        ctx.split,
        ctx.npz_path,
        ctx.panel_manifest_sha256,
        ctx.grid_ms,
        ctx.max_chunks,
        Some("int8"),
        ctx.threads,
        load_before,
        ctx.cli_args,
    );
    write_json_create_new(&jm_path, &jm)?;
    write_scores_create_new(&scores_path, &res.scored_rows)?;
    write_receipt_create_new(&receipt_path, &res.receipt)?;
    Ok(FrontierRow {
        candidate: res.candidate_id.clone(),
        loaded_identity: res.receipt.bundle.clone(),
        driver: driver.into(),
        cost_scope: scope.into(),
        honesty_label: res.receipt.label,
        partition: ctx.partition,
        completion_status: res.receipt.completion_status.clone(),
        utterances: res.utterances,
        audio_seconds: res.audio_seconds,
        wer_pct: res.wer.wer_pct,
        single_rtf_cpu: res.rtf_cpu,
        single_streams_per_core: res.streams_per_core_at_full_load,
        batch_rtf_cpu: None,
        batch_streams_per_core: None,
        chunk_ms: res.receipt.chunk_ms.clone(),
        peak_rss_bytes: res.receipt.peak_rss_bytes,
        cap_hits: res.receipt.cap_hits,
        evidence_complete: res.receipt.validate_as_evidence().is_ok(),
        density: None,
        latency: None,
        sustained_probe: None,
        slices: compute_slice_summaries(&res.scored_rows),
        silence_probe: sil,
        job_manifest_file: jm_path.to_string_lossy().into_owned(),
        scores_file: scores_path.to_string_lossy().into_owned(),
        receipt_file: receipt_path.to_string_lossy().into_owned(),
    })
}

pub(crate) fn eval_backend_probes(
    ctx: &SuiteCtx<'_>,
    raw_cand: &str,
    slug: &str,
) -> Result<
    (
        Option<crate::suite_latency::SuiteLatencySummary>,
        Option<crate::suite_slices::SilenceProbeReport>,
    ),
    CliError,
> {
    if ctx.latency_limit == 0 && ctx.silence_probe_wav.is_none() {
        return Ok((None, None));
    }
    let b = instantiate_backend_with_precision(raw_cand, ctx.precision, ctx.threads)?;
    let lat = if ctx.latency_limit > 0 {
        Some(measure_paced_latency(
            &*b,
            ctx.npz_path,
            ctx.split,
            ctx.latency_limit,
        )?)
    } else {
        None
    };
    let sil = match ctx.silence_probe_wav {
        Some((wp, req_sha)) => {
            let sp_path = ctx.out_dir.join(format!("{slug}.silence.scores.jsonl"));
            Some(run_silence_probe_backend(
                &*b,
                wp,
                *req_sha,
                Some(&sp_path),
            )?)
        }
        None => None,
    };
    Ok((lat, sil))
}
