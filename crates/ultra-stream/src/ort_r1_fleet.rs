//! CCX-pinned shared-weight multi-worker `R1` fleet (`run_sp_a_ort_r1_shared_workers`) with
//! optional disjoint-CCX `/proc` observer thread (`observer_cpu_seconds` separated from
//! `service_cpu_seconds` per the `2026-11-21` metrology freeze rule).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use ultra_core::receipt::{Receipt, RECEIPT_SCHEMA};

use crate::api::StreamError;
use crate::contrast::{
    build_event_hashes_and_chunks, current_affinity, load_selected_audio, read_loadavg,
    self_cpu_seconds,
};
use crate::fleet_inproc::WorkerSummary;
use crate::grid_sim::{simulate_grid, sustains_lateness, SimParams};
use crate::ort_r1::OrtTransducerBackend;
use crate::ort_r1_bank::{OrtR1SlotBank, R1SpAConfig};
use crate::placement::{
    physical_first_cpus, pin_current_thread_verified, run_fleet_observer, verify_worker_l3_and_smt,
    FleetSeriesPoint,
};
use crate::scheduler::RealClock;
use crate::sherpa_ffi::verify_cpu_only_process;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct R1FleetResult {
    pub schema: String,
    pub driver: String,
    pub bundle_id: String,
    pub workers: usize,
    pub streams_per_worker: usize,
    pub total_streams_offered: usize,
    pub sustained_workers: usize,
    pub all_sustained: Option<bool>,
    pub utterances: usize,
    pub audio_seconds: f64,
    #[serde(default)]
    pub audio_processed_seconds: f64,
    #[serde(default)]
    pub audio_unfinished_seconds: f64,
    pub process_cpu_seconds: f64,
    pub observer_cpu_seconds: f64,
    pub service_cpu_seconds: f64,
    pub wall_seconds: f64,
    pub rtf_service_cpu: f64,
    pub peak_rss_bytes: u64,
    pub forced_advances: u64,
    pub observer_cpus: Vec<usize>,
    pub effective_observer_cpus: Vec<usize>,
    pub worker_summaries: Vec<WorkerSummary>,
    #[serde(default)]
    pub series: Vec<FleetSeriesPoint>,
    pub receipt: Receipt,
}

pub fn run_sp_a_ort_r1_shared_workers(
    cfg: &R1SpAConfig,
    workers: usize,
    observer_cpus: Option<&[usize]>,
) -> Result<R1FleetResult, StreamError> {
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
    let engine = OrtTransducerBackend::load(&cfg.bundle_dir, cfg.precision, cfg.threads, true)?;
    let bundle_id = format!(
        "ultra-r1-{}-{}-{}",
        engine.bundle.family.as_str(),
        engine.bundle.precision.as_str(),
        &engine.bundle.manifest_sha256[..12]
    );

    let phys_aff = physical_first_cpus(&current_affinity());
    let mut worker_cpus: Vec<Vec<usize>> = (0..workers).map(|_| Vec::new()).collect();
    if phys_aff.len() >= workers {
        let per = phys_aff.len() / workers;
        for (w, dst) in worker_cpus.iter_mut().enumerate() {
            let start = w * per;
            let end = if w + 1 == workers {
                phys_aff.len()
            } else {
                start + per
            };
            dst.extend_from_slice(&phys_aff[start..end]);
        }
    }
    verify_worker_l3_and_smt(&worker_cpus, observer_cpus, observer_cpus.is_some())?;

    let _load_before = read_loadavg();
    let pid = std::process::id() as i32;
    let stop_obs = Arc::new(AtomicBool::new(false));
    let obs_cpus_vec: Vec<usize> = match observer_cpus {
        Some(s) => s.to_vec(),
        None => Vec::new(),
    };
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
                let mut bank = OrtR1SlotBank::with_start_offset(
                    eng,
                    utts_ref,
                    (w * 137) % utts_ref.len(),
                    cfg.streams,
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
                if let Some(err) = bank.last_error {
                    return Err(err);
                }
                Ok::<_, StreamError>((sim, bank.forced_advances, eff_cpus))
            }));
        }
        let outs: Vec<_> = handles
            .into_iter()
            .map(|h| {
                h.join()
                    .map_err(|_| StreamError::Runtime("R1 fleet worker panicked".into()))?
            })
            .collect();
        stop_obs.store(true, Ordering::Relaxed);
        let obs_res = obs_handle
            .join()
            .map_err(|_| StreamError::Runtime("R1 observer thread panicked".into()))
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
    let _load_after = read_loadavg();
    let cpu_only = verify_cpu_only_process();

    let mut all_completed = Vec::new();
    let mut all_ticks = Vec::new();
    let mut total_audio_s = 0.0f64;
    let mut total_processed_s = 0.0f64;
    let mut total_unfinished_s = 0.0f64;
    let mut total_wakeups = 0u64;
    let mut total_forced = 0u64;
    let mut summaries = Vec::with_capacity(workers);
    let mut sustained_workers = 0usize;

    for (w, res) in outcomes.into_iter().enumerate() {
        let (sim, fa, eff_cpus) = res?;
        if !sim.drained {
            return Err(StreamError::Runtime(format!("R1 worker {w} did not drain")));
        }
        total_forced += fa;
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
    let (hashes, chunk_ms) =
        build_event_hashes_and_chunks(&bundle_id, &audio_utts, &all_completed, &mut all_ticks)?;
    let eff_workers: Vec<&[usize]> = summaries
        .iter()
        .map(|s| s.effective_cpus.as_slice())
        .collect();
    let cfg_threads = (workers as u32) * (cfg.threads.max(1) as u32);
    let receipt = Receipt {
        schema: RECEIPT_SCHEMA.into(),
        bundle: bundle_id.clone(),
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
        cap_hits: Some(total_forced),
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
        completion_status: cfg.completion_status.clone(),
    };

    Ok(R1FleetResult {
        schema: "ultra.r1-fleet/v1".into(),
        driver: "rust-ort-r1-ccx-fleet-v1".into(),
        bundle_id,
        workers,
        streams_per_worker: cfg.streams,
        total_streams_offered: workers * cfg.streams,
        sustained_workers,
        all_sustained: cfg.paced.then_some(sustained_workers == workers),
        utterances: all_completed.len(),
        audio_seconds: (total_audio_s * 1000.0).round() / 1000.0,
        audio_processed_seconds: (total_processed_s * 1000.0).round() / 1000.0,
        audio_unfinished_seconds: (total_unfinished_s * 1000.0).round() / 1000.0,
        process_cpu_seconds: (proc_cpu_s * 1000.0).round() / 1000.0,
        observer_cpu_seconds: (obs_cpu_s * 1000.0).round() / 1000.0,
        service_cpu_seconds: (svc_cpu_s * 1000.0).round() / 1000.0,
        wall_seconds: (wall_s * 1000.0).round() / 1000.0,
        rtf_service_cpu: svc_cpu_s / audio_denom.max(1e-9),
        peak_rss_bytes: peak_rss_b.max(1),
        forced_advances: total_forced,
        observer_cpus: obs_cpus_vec,
        effective_observer_cpus: eff_obs_cpus,
        worker_summaries: summaries,
        series,
        receipt,
    })
}
