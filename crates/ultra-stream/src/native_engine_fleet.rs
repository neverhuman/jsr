//! CCX-pinned shared-weight multi-worker [`NativeEngineBackend`] fleet
//! (`run_sp_a_native_engine_shared_workers`) with disjoint-CCX `/proc` observer thread.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use ultra_core::receipt::{Receipt, RECEIPT_SCHEMA};
use ultra_eval::procfs::TreeSampler;
use ultra_frontend::npz::load_npz_audio_only;

use crate::api::StreamError;
use crate::contrast::{build_event_hashes_and_chunks, current_affinity, self_cpu_seconds};
use crate::fleet_inproc::{physical_first_cpus, pin_current_thread_verified, WorkerSummary};
use crate::grid_sim::{simulate_grid, sustains_lateness, SimOutcome, SimParams};
use crate::native_engine::NativeEngineBackend;
use crate::native_engine_bank::{engine_status_events, EngineSpAConfig, NativeEngineSlotBank};
use crate::ort_r1_fleet::R1FleetResult;
use crate::ort_r1_ws::thread_cpu_ns;
use crate::scheduler::RealClock;
use crate::sherpa_ffi::verify_cpu_only_process;

/// CCX-pinned shared-weight multi-worker [`NativeEngineBackend`] fleet with disjoint observer CPUs.
pub fn run_sp_a_native_engine_shared_workers(
    cfg: &EngineSpAConfig,
    workers: usize,
    observer_cpus: Option<&[usize]>,
) -> Result<R1FleetResult, StreamError> {
    if workers == 0 {
        return Err(StreamError::Protocol("workers must be >= 1".into()));
    }
    let audio_utts = load_npz_audio_only(&cfg.npz_path, cfg.offset, cfg.limit)?;
    if audio_utts.is_empty() {
        return Err(StreamError::Protocol("no audio utterances loaded".into()));
    }
    let backend = NativeEngineBackend::load(&cfg.weights_dir, cfg.precision, cfg.threads, true)?;
    let bundle_id = backend.bundle_id();
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
    let pid = std::process::id() as i32;
    let stop_obs = Arc::new(AtomicBool::new(false));
    let obs_cpus_vec: Vec<usize> = observer_cpus.map_or_else(Vec::new, <[usize]>::to_vec);
    let cpu0 = self_cpu_seconds()?;
    let wall_t0 = Instant::now();
    let barrier = Barrier::new(workers);

    let (outcomes, obs_res) = std::thread::scope(|scope| {
        let stop_ref = Arc::clone(&stop_obs);
        let obs_pin = obs_cpus_vec.clone();
        let obs_handle = scope.spawn(
            move || -> Result<(u64, u64, f64, Vec<usize>), StreamError> {
                let eff_obs = pin_current_thread_verified(&obs_pin)?;
                let t0 = thread_cpu_ns(true);
                let mut sampler = TreeSampler::new(pid);
                sampler.sample();
                while !stop_ref.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(25));
                    sampler.sample();
                }
                sampler.sample();
                let dt = (thread_cpu_ns(true).saturating_sub(t0)) as f64 * 1e-9;
                let sum = sampler.summary();
                Ok((sum.peak_rss_bytes, sum.max_threads, dt, eff_obs))
            },
        );
        let mut handles = Vec::with_capacity(workers);
        for (w, cpus) in worker_cpus.iter().enumerate() {
            let b_ref = Arc::clone(&backend);
            let bar = &barrier;
            let utts_ref = &audio_utts;
            let cpus_clone = cpus.clone();
            handles.push(scope.spawn(move || {
                let pin_res = pin_current_thread_verified(&cpus_clone);
                let mut bank = NativeEngineSlotBank::with_start_offset(
                    b_ref,
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
                Ok((sim, bank.forced_advances, eff_cpus))
            }));
        }
        let outs: Vec<Result<(SimOutcome, u64, Vec<usize>), StreamError>> = handles
            .into_iter()
            .map(|h| {
                h.join().map_err(|_| {
                    StreamError::Runtime("ultra-engine fleet worker panicked".into())
                })?
            })
            .collect();
        stop_obs.store(true, Ordering::Relaxed);
        let obs_res = obs_handle
            .join()
            .map_err(|_| StreamError::Runtime("ultra-engine observer thread panicked".into()))
            .and_then(|r| r);
        (outs, obs_res)
    });
    let (peak_rss_b, max_thr, obs_cpu_s, eff_obs_cpus) = obs_res?;

    let proc_cpu_s = (self_cpu_seconds()? - cpu0).max(1e-6);
    let svc_cpu_s = (proc_cpu_s - obs_cpu_s).max(1e-6);
    let wall_s = wall_t0.elapsed().as_secs_f64().max(1e-6);
    let cpu_only = verify_cpu_only_process();
    let mut all_completed = Vec::new();
    let mut all_ticks = Vec::new();
    let mut total_audio_s = 0.0f64;
    let mut total_wakeups = 0u64;
    let mut total_forced = 0u64;
    let mut summaries = Vec::with_capacity(workers);
    let mut sustained_workers = 0usize;

    for (w, res) in outcomes.into_iter().enumerate() {
        let (sim, fa, eff_cpus) = res?;
        if !sim.drained {
            return Err(StreamError::Runtime(format!(
                "ultra-engine worker {w} did not drain"
            )));
        }
        total_forced += fa;
        total_audio_s += sim.audio_accepted_s;
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
            wall_seconds: (sim.wall_seconds * 1000.0).round() / 1000.0,
            accepted_packets: sim.accepted_packets,
            drained: sim.drained,
            sustained: sust,
            lateness_ms: sim.lateness_ms,
        });
        all_completed.extend(sim.completed);
    }

    let (hashes, chunk_ms) =
        build_event_hashes_and_chunks(&bundle_id, &audio_utts, &all_completed, &mut all_ticks)?;
    let eff_workers: Vec<&[usize]> = summaries
        .iter()
        .map(|s| s.effective_cpus.as_slice())
        .collect();
    let mut status_events = engine_status_events(&backend);
    status_events.extend([
        format!("observer_cpu_seconds:{obs_cpu_s:.6}"),
        format!("process_cpu_seconds:{proc_cpu_s:.6}"),
        format!("effective_observer_cpus:{eff_obs_cpus:?}"),
        format!("effective_worker_cpus:{eff_workers:?}"),
    ]);
    let receipt = Receipt {
        schema: RECEIPT_SCHEMA.into(),
        bundle: bundle_id.clone(),
        source_commit: cfg.source_commit.clone(),
        hardware: cfg.hardware_id.clone(),
        target: std::env::consts::ARCH.into(),
        configured_threads: workers as u32,
        observed_threads: Some(max_thr.max(1) as u32),
        cpu_only_verified: cpu_only,
        audio_seconds: (total_audio_s * 1000.0).round() / 1000.0,
        process_tree_cpu_seconds: Some((svc_cpu_s * 1000.0).round() / 1000.0),
        peak_rss_bytes: Some(peak_rss_b.max(1)),
        wakeups: Some(total_wakeups),
        events: Some(all_completed.len() as u64),
        cap_hits: Some(total_forced),
        chunk_ms,
        status_events,
        gaps: Vec::new(),
        hashes,
        label: cfg.honesty_label,
        partition: cfg.partition,
        completion_status: cfg.completion_status.clone(),
    };
    Ok(R1FleetResult {
        schema: "ultra.r1-fleet/v1".into(),
        driver: "rust-ultra-engine-ccx-fleet-v1".into(),
        bundle_id,
        workers,
        streams_per_worker: cfg.streams,
        total_streams_offered: workers * cfg.streams,
        sustained_workers,
        all_sustained: cfg.paced.then_some(sustained_workers == workers),
        utterances: all_completed.len(),
        audio_seconds: (total_audio_s * 1000.0).round() / 1000.0,
        process_cpu_seconds: (proc_cpu_s * 1000.0).round() / 1000.0,
        observer_cpu_seconds: (obs_cpu_s * 1000.0).round() / 1000.0,
        service_cpu_seconds: (svc_cpu_s * 1000.0).round() / 1000.0,
        wall_seconds: (wall_s * 1000.0).round() / 1000.0,
        rtf_service_cpu: svc_cpu_s / total_audio_s.max(1e-9),
        peak_rss_bytes: peak_rss_b.max(1),
        forced_advances: total_forced,
        observer_cpus: obs_cpus_vec,
        effective_observer_cpus: eff_obs_cpus,
        worker_summaries: summaries,
        receipt,
    })
}
