//! Single-process, shared-weight multi-worker SP-A runner (`--workers W`, `FREEZE.md §3`).
//!
//! Spawns `W` worker threads inside ONE OS process sharing ONE immutable `Arc<SherpaZipformerBackend>`
//! with optional disjoint-CCX `/proc` observer thread (`observer_cpu_seconds` separated from
//! `service_cpu_seconds`) and fail-closed L3/SMT placement verification.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use ultra_core::receipt::{Receipt, RECEIPT_SCHEMA};
use ultra_eval::score::ScoreRecord;

use crate::api::StreamError;
use crate::contrast::{
    build_event_hashes_and_chunks, current_affinity, load_selected_audio, read_loadavg,
    score_completed, self_cpu_seconds, WerSummary,
};
use crate::grid_sim::{simulate_grid, sustains_lateness, LatenessQuantiles, SimParams};
use crate::placement::{
    physical_first_cpus, pin_current_thread_verified, run_fleet_observer, verify_worker_l3_and_smt,
    FleetSeriesPoint,
};
use crate::scheduler::{RealClock, SherpaBank, SpAConfig};
use crate::sherpa_ffi::{verify_cpu_only_process, SherpaLib};
use crate::sherpa_online::SherpaZipformerBackend;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerSummary {
    pub worker_id: usize,
    pub pinned_cpus: Vec<usize>,
    #[serde(default)]
    pub effective_cpus: Vec<usize>,
    pub utterances: usize,
    pub audio_accepted_s: f64,
    #[serde(default)]
    pub audio_processed_s: f64,
    #[serde(default)]
    pub audio_unfinished_s: f64,
    pub wall_seconds: f64,
    pub accepted_packets: u64,
    pub drained: bool,
    pub sustained: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lateness_ms: Option<LatenessQuantiles>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedFleetResult {
    pub schema: String,
    pub driver: String,
    pub sherpa_version: String,
    pub onnxruntime_version: String,
    pub split: String,
    pub workers: usize,
    pub streams_per_worker: usize,
    pub total_streams_offered: usize,
    pub sustained_workers: usize,
    pub sustained_streams: usize,
    pub all_sustained: Option<bool>,
    pub decode: String,
    pub precision: String,
    pub threads_per_recognizer: i32,
    pub paced: bool,
    pub packet_ms: u32,
    pub grid_ms: u32,
    pub effective_grid_ms: u32,
    pub utterances: usize,
    pub audio_seconds: f64,
    pub audio_processed_seconds: f64,
    pub audio_unfinished_seconds: f64,
    pub cpu_seconds: f64,
    pub observer_cpu_seconds: f64,
    pub wall_seconds: f64,
    pub rtf_cpu: f64,
    pub total_streams_at_full_load: f64,
    pub streams_per_physical_core_equiv: f64,
    pub peak_rss_bytes: u64,
    pub wer: WerSummary,
    pub affinity: Vec<usize>,
    pub observer_cpus: Vec<usize>,
    pub effective_observer_cpus: Vec<usize>,
    pub load_before: [f64; 3],
    pub load_after: [f64; 3],
    pub worker_summaries: Vec<WorkerSummary>,
    pub series: Vec<FleetSeriesPoint>,
    pub receipt: Receipt,
    #[serde(skip)]
    pub scored_rows: Vec<ScoreRecord>,
}

pub fn run_sp_a_shared_workers(
    cfg: &SpAConfig,
    workers: usize,
) -> Result<SharedFleetResult, StreamError> {
    run_sp_a_shared_workers_with_observer(cfg, workers, None)
}

/// Unified R0 Zipformer shared-weight fleet with disjoint observer CPUs and L3/SMT verification.
pub fn run_sp_a_shared_workers_with_observer(
    cfg: &SpAConfig,
    workers: usize,
    observer_cpus: Option<&[usize]>,
) -> Result<SharedFleetResult, StreamError> {
    if workers == 0 {
        return Err(StreamError::Protocol("workers must be >= 1".into()));
    }
    let audio_utts = load_selected_audio(
        &cfg.npz_path,
        cfg.offset,
        cfg.limit,
        cfg.selected_indices.as_deref(),
    )?;
    if audio_utts.is_empty() {
        return Err(StreamError::Protocol("no audio utterances loaded".into()));
    }
    let lib = SherpaLib::load_default()?;
    let sherpa_ver = lib.version();
    let ort_ver = lib.onnxruntime_version();
    let engine = SherpaZipformerBackend::new(lib, cfg.zipformer.clone())?;

    let aff = physical_first_cpus(&current_affinity());
    let mut worker_cpus: Vec<Vec<usize>> = (0..workers).map(|_| Vec::new()).collect();
    if aff.len() >= workers {
        let per = aff.len() / workers;
        for (w, dst) in worker_cpus.iter_mut().enumerate() {
            let start = w * per;
            let end = if w + 1 == workers {
                aff.len()
            } else {
                start + per
            };
            dst.extend_from_slice(&aff[start..end]);
        }
    }
    verify_worker_l3_and_smt(&worker_cpus, observer_cpus, observer_cpus.is_some())?;

    let load_before = read_loadavg();
    let pid = std::process::id() as i32;
    let stop_obs = Arc::new(AtomicBool::new(false));
    let obs_cpus_vec: Vec<usize> = observer_cpus.map_or_else(Vec::new, <[usize]>::to_vec);
    let cpu0 = self_cpu_seconds()?;
    let wall_t0 = Instant::now();
    let barrier = Barrier::new(workers);

    let (outcomes, obs_res) = std::thread::scope(|scope| {
        let stop_ref = Arc::clone(&stop_obs);
        let obs_pin = obs_cpus_vec.clone();
        let obs_handle = scope.spawn(move || run_fleet_observer(pid, &obs_pin, &stop_ref));
        let mut handles = Vec::with_capacity(workers);
        for (w, cpus) in worker_cpus.iter().enumerate() {
            let eng = Arc::clone(&engine);
            let bar = &barrier;
            let utts_ref = &audio_utts;
            let cpus_clone = cpus.clone();
            handles.push(scope.spawn(move || {
                let pin_res = pin_current_thread_verified(&cpus_clone);
                let mut bank = SherpaBank::new(
                    utts_ref,
                    (w * 137) % utts_ref.len(),
                    eng,
                    cfg.streams,
                    cfg.zipformer.tail_pad_s,
                    None,
                );
                bar.wait();
                let eff_cpus = pin_res?;
                let mut clk = RealClock::new()?;
                let sim = simulate_grid(
                    &mut bank,
                    &SimParams {
                        streams: cfg.streams,
                        decode: cfg.decode,
                        max_batch: cfg.max_batch,
                        packet_ms: cfg.packet_ms,
                        grid_ms: cfg.grid_ms,
                        paced: cfg.paced,
                        duration_s: cfg.duration_s,
                    },
                    &mut clk,
                )?;
                Ok::<_, StreamError>((sim, eff_cpus))
            }));
        }
        let outs: Vec<_> = handles
            .into_iter()
            .map(|h| {
                h.join()
                    .map_err(|_| StreamError::Runtime("worker panicked".into()))?
            })
            .collect();
        stop_obs.store(true, Ordering::Relaxed);
        let obs_res = obs_handle
            .join()
            .map_err(|_| StreamError::Runtime("zipformer observer panicked".into()))
            .and_then(|r| r);
        (outs, obs_res)
    });
    let obs = obs_res?;
    let (peak_rss_b, max_thr, obs_cpu_s, eff_obs_cpus, series) = (
        obs.peak_rss_bytes,
        obs.max_threads,
        obs.observer_cpu_s,
        obs.effective_cpus,
        obs.series,
    );

    let proc_cpu_s = (self_cpu_seconds()? - cpu0).max(1e-6);
    let svc_cpu_s = (proc_cpu_s - obs_cpu_s).max(1e-6);
    let wall_s = wall_t0.elapsed().as_secs_f64().max(1e-6);
    let load_after = read_loadavg();
    let cpu_only = verify_cpu_only_process();
    drop(engine);

    let mut all_completed = Vec::new();
    let mut all_ticks = Vec::new();
    let mut total_audio_s = 0.0f64;
    let mut total_processed_s = 0.0f64;
    let mut total_unfinished_s = 0.0f64;
    let mut total_wakeups = 0u64;
    let mut eff_grid = cfg.grid_ms;
    let mut summaries = Vec::with_capacity(workers);
    let mut sustained_workers = 0usize;

    for (w, res) in outcomes.into_iter().enumerate() {
        let (sim, eff_cpus) = res?;
        if !sim.drained {
            return Err(StreamError::Runtime(format!("worker {w} did not drain")));
        }
        eff_grid = sim.effective_grid_ms;
        total_audio_s += sim.audio_accepted_s;
        total_processed_s += sim.audio_processed_s;
        total_unfinished_s += sim.audio_unfinished_s;
        total_wakeups += sim.wakeups;
        all_ticks.extend(sim.tick_compute_ms);
        let sust = sim
            .lateness_ms
            .as_ref()
            .map(|lat| sustains_lateness(lat, sim.drained, cfg.lateness_bound_ms));
        if sust == Some(true) {
            sustained_workers += 1;
        }
        summaries.push(WorkerSummary {
            worker_id: w,
            pinned_cpus: worker_cpus[w].clone(),
            effective_cpus: eff_cpus,
            utterances: sim.completed.len(),
            audio_accepted_s: (sim.audio_accepted_s * 1000.0).round() / 1000.0,
            audio_processed_s: (sim.audio_processed_s * 1000.0).round() / 1000.0,
            audio_unfinished_s: (sim.audio_unfinished_s * 1000.0).round() / 1000.0,
            wall_seconds: (sim.wall_seconds * 1000.0).round() / 1000.0,
            accepted_packets: sim.accepted_packets,
            drained: sim.drained,
            sustained: sust,
            lateness_ms: sim.lateness_ms,
        });
        all_completed.extend(sim.completed);
    }

    let audio_denom = if total_processed_s > 0.0 {
        total_processed_s
    } else {
        total_audio_s
    };
    if audio_denom <= 0.0 {
        return Err(StreamError::Protocol("zero audio processed".into()));
    }
    let model_id = match &cfg.bundle_id {
        Some(b) => b.clone(),
        None => format!(
            "sherpa-onnx-streaming-zipformer-en-2023-06-26-{}",
            cfg.zipformer.precision
        ),
    };
    let (hashes, chunk_ms) =
        build_event_hashes_and_chunks(&model_id, &audio_utts, &all_completed, &mut all_ticks)?;
    let eff_workers: Vec<&[usize]> = summaries
        .iter()
        .map(|s| s.effective_cpus.as_slice())
        .collect();
    let cfg_threads = (workers as u32) * (cfg.zipformer.num_threads.max(1) as u32);
    let receipt = Receipt {
        schema: RECEIPT_SCHEMA.into(),
        bundle: model_id,
        source_commit: cfg.source_commit.clone(),
        hardware: cfg.hardware_id.clone(),
        target: std::env::consts::ARCH.into(),
        configured_threads: cfg_threads,
        observed_threads: Some(max_thr.max(1) as u32),
        cpu_only_verified: cpu_only,
        audio_seconds: (audio_denom * 1000.0).round() / 1000.0,
        process_tree_cpu_seconds: Some((svc_cpu_s * 1000.0).round() / 1000.0),
        peak_rss_bytes: Some(peak_rss_b.max(1)),
        events: Some(all_completed.len() as u64),
        wakeups: Some(total_wakeups),
        cap_hits: Some(0),
        chunk_ms,
        status_events: vec![
            format!("observer_cpu_seconds:{obs_cpu_s:.6}"),
            format!("process_cpu_seconds:{proc_cpu_s:.6}"),
            format!("effective_observer_cpus:{eff_obs_cpus:?}"),
            format!("effective_worker_cpus:{eff_workers:?}"),
            format!("audio_accepted_s:{total_audio_s:.3}"),
            format!("audio_processed_s:{total_processed_s:.3}"),
            format!("audio_unfinished_s:{total_unfinished_s:.3}"),
        ],
        gaps: Vec::new(),
        hashes,
        label: cfg.honesty_label,
        partition: cfg.partition,
        completion_status: "complete".into(),
    };
    receipt.validate_as_evidence()?;

    let (wer_summary, scored_rows) = score_completed(
        &cfg.npz_path,
        &cfg.split_name,
        cfg.offset,
        cfg.limit,
        cfg.selected_indices.as_deref(),
        true,
        &all_completed,
    )?;

    let all_sustained = cfg.paced.then_some(sustained_workers == workers);
    let sustained_streams = if all_sustained == Some(true) {
        workers * cfg.streams
    } else {
        0
    };
    let pinned_cores = (aff.len().max(1)) as f64;
    let per_core_full_load = audio_denom / svc_cpu_s.max(1e-9);
    let wall_streams = audio_denom / wall_s.max(1e-9);

    Ok(SharedFleetResult {
        schema: "ultra.sp-a.shared-fleet/v1".into(),
        driver: "rust-sherpa-c-api-shared-weights".into(),
        sherpa_version: sherpa_ver,
        onnxruntime_version: ort_ver,
        split: cfg.split_name.clone(),
        workers,
        streams_per_worker: cfg.streams,
        total_streams_offered: workers * cfg.streams,
        sustained_workers,
        sustained_streams,
        all_sustained,
        decode: cfg.decode.as_str().into(),
        precision: cfg.zipformer.precision.clone(),
        threads_per_recognizer: cfg.zipformer.num_threads,
        paced: cfg.paced,
        packet_ms: cfg.packet_ms,
        grid_ms: cfg.grid_ms,
        effective_grid_ms: eff_grid,
        utterances: all_completed.len(),
        audio_seconds: (total_audio_s * 1000.0).round() / 1000.0,
        audio_processed_seconds: (total_processed_s * 1000.0).round() / 1000.0,
        audio_unfinished_seconds: (total_unfinished_s * 1000.0).round() / 1000.0,
        cpu_seconds: (svc_cpu_s * 1000.0).round() / 1000.0,
        observer_cpu_seconds: (obs_cpu_s * 1000.0).round() / 1000.0,
        wall_seconds: (wall_s * 1000.0).round() / 1000.0,
        rtf_cpu: ((svc_cpu_s / audio_denom.max(1e-9)) * 100_000.0).round() / 100_000.0,
        total_streams_at_full_load: ((per_core_full_load * pinned_cores) * 100.0).round() / 100.0,
        streams_per_physical_core_equiv: ((wall_streams / pinned_cores).min(per_core_full_load)
            * 100.0)
            .round()
            / 100.0,
        peak_rss_bytes: peak_rss_b.max(1),
        wer: wer_summary,
        affinity: aff,
        observer_cpus: obs_cpus_vec,
        effective_observer_cpus: eff_obs_cpus,
        load_before,
        load_after,
        worker_summaries: summaries,
        series,
        receipt,
        scored_rows,
    })
}
