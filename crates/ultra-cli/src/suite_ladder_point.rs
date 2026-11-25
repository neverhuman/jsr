//! Per-point paced fleet execution helper for [`crate::suite_ladder`].

use ultra_core::receipt::HonestyLabel;
use ultra_stream::{
    run_sp_a_native_engine_shared_workers, run_sp_a_ort_r1_shared_workers,
    run_sp_a_shared_workers_with_observer, DecodeMode, EngineSpAConfig, R1SpAConfig, SpAConfig,
    DEFAULT_LATENESS_BOUND_MS,
};

use crate::runner::{resolve_bundle_or_zipformer, resolve_r1_bundle};
use crate::suite_engine::resolve_r1_weights_dir;
use crate::suite_ladder::LadderRepeatPoint;
use crate::suite_runners::SuiteCtx;
use crate::CliError;

pub(crate) fn run_one_density_point(
    ctx: &SuiteCtx<'_>,
    raw_cand: &str,
    streams_per_worker: usize,
    repeat: usize,
    duration_s: f64,
    phase: &str,
) -> Result<LadderRepeatPoint, CliError> {
    let w_cnt = ctx.density_workers.max(1);
    let dur = Some(duration_s.max(1.0));
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
                beam: ctx.beam,
                decode_config: ctx.decode_config.clone(),
                packet_ms: 20,
                grid_ms: ctx.grid_ms,
                coalesce_ms: ctx.coalesce_ms,
                lateness_bound_ms: DEFAULT_LATENESS_BOUND_MS,
                paced: true,
                source_commit: ctx.commit.into(),
                hardware_id: ctx.hardware_id.into(),
                completion_status: ctx.status_tag.into(),
            },
            w_cnt,
            ctx.observer_cpus,
        )?;
        return Ok(LadderRepeatPoint::from_r1_fleet(
            raw_cand,
            phase,
            repeat,
            streams_per_worker,
            w_cnt,
            duration_s,
            &f,
        ));
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
                hardware_id: ctx.hardware_id.into(),
                completion_status: ctx.status_tag.into(),
            },
            w_cnt,
            ctx.observer_cpus,
        )?;
        return Ok(LadderRepeatPoint::from_r1_fleet(
            raw_cand,
            phase,
            repeat,
            streams_per_worker,
            w_cnt,
            duration_s,
            &f,
        ));
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
                hardware_id: ctx.hardware_id.into(),
            },
            w_cnt,
            ctx.observer_cpus,
        )?;
        return Ok(LadderRepeatPoint::from_fields(
            raw_cand,
            phase,
            repeat,
            streams_per_worker,
            w_cnt,
            duration_s,
            f.all_sustained,
            &f.worker_summaries,
            f.rtf_cpu,
            f.peak_rss_bytes,
            f.audio_processed_seconds,
            f.audio_unfinished_seconds,
        ));
    }
    Err(CliError::Config(format!(
        "candidate `{raw_cand}` does not support paced density ladder"
    )))
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
