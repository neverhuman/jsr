//! Single-process, shared-weight multi-worker SP-A runner (`--workers W`, spec v1.0-B W01/SP-A).
//!
//! Spawns `W` worker threads inside ONE OS process sharing ONE immutable `Arc<SherpaZipformerBackend>`
//! (~72.6 MB int8 weights in physical RAM) so multi-core density scaling avoids `W` private 70 MB
//! weight copies while keeping per-worker ORT `num_threads` small (e.g. `1` or CCX-sized `4`).

use std::sync::{Arc, Barrier};

use serde::{Deserialize, Serialize};
use ultra_core::receipt::{Receipt, RECEIPT_SCHEMA};
use ultra_eval::procfs::TreeSampler;
use ultra_eval::score::ScoreRecord;
use ultra_frontend::npz::{load_npz_audio_only, AudioUtterance};

use crate::api::StreamError;
use crate::contrast::{
    build_event_hashes_and_chunks, current_affinity, read_loadavg, score_completed,
    self_cpu_seconds, WerSummary,
};
use crate::grid_sim::{
    simulate_grid, sustains_lateness, DecodeMode, LatenessQuantiles, SimOutcome, SimParams,
    SimSlotBank,
};
use crate::scheduler::{RealClock, SpAConfig};
use crate::sherpa_ffi::{verify_cpu_only_process, SherpaLib};
use crate::sherpa_online::{RawOnlineStream, SherpaZipformerBackend};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerSummary {
    pub worker_id: usize,
    pub pinned_cpus: Vec<usize>,
    pub utterances: usize,
    pub audio_accepted_s: f64,
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
    pub cpu_seconds: f64,
    pub wall_seconds: f64,
    pub rtf_cpu: f64,
    pub total_streams_at_full_load: f64,
    pub streams_per_physical_core_equiv: f64,
    pub peak_rss_bytes: u64,
    pub wer: WerSummary,
    pub affinity: Vec<usize>,
    pub load_before: [f64; 3],
    pub load_after: [f64; 3],
    pub worker_summaries: Vec<WorkerSummary>,
    pub receipt: Receipt,
    #[serde(skip)]
    pub scored_rows: Vec<ScoreRecord>,
}

struct WorkerBank<'a> {
    audio_utts: &'a [AudioUtterance],
    rot: usize,
    engine: Arc<SherpaZipformerBackend>,
    slots: Vec<Option<(usize, RawOnlineStream, usize, bool)>>,
    pad_zeros: Vec<f32>,
    scratch: Vec<f32>,
}

impl SimSlotBank for WorkerBank<'_> {
    fn num_utterances(&self) -> usize {
        self.audio_utts.len()
    }
    fn utt_samples(&self, utt_pos: usize) -> usize {
        self.audio_utts[utt_pos].pcm.len()
    }
    fn open_slot(&mut self, slot: usize, utt_pos: usize) -> Result<(), StreamError> {
        let real = (utt_pos + self.rot) % self.audio_utts.len();
        self.slots[slot] = Some((real, self.engine.create_raw_stream()?, 0, false));
        Ok(())
    }
    fn has_slot(&self, slot: usize) -> bool {
        self.slots[slot].is_some()
    }
    fn slot_utt(&self, slot: usize) -> usize {
        self.slots[slot].as_ref().map_or(0, |(real, _, _, _)| *real)
    }
    fn slot_cursor(&self, slot: usize) -> usize {
        self.slots[slot].as_ref().map_or(0, |(_, _, cur, _)| *cur)
    }
    fn slot_finished_input(&self, slot: usize) -> bool {
        self.slots[slot].as_ref().is_some_and(|(_, _, _, fin)| *fin)
    }
    fn accept_packet(&mut self, slot: usize, max_samples: usize) -> usize {
        let Some((real, st, cur, fin)) = self.slots[slot].as_mut() else {
            return 0;
        };
        let pcm = &self.audio_utts[*real].pcm;
        if *cur >= pcm.len() {
            return 0;
        }
        let end = (*cur + max_samples).min(pcm.len());
        let n = end - *cur;
        st.accept_i16(&pcm[*cur..end], &mut self.scratch);
        *cur = end;
        if *cur >= pcm.len() {
            st.accept_f32(&self.pad_zeros);
            st.input_finished();
            *fin = true;
        }
        n
    }
    fn finish_slot_input(&mut self, slot: usize) {
        if let Some((_, st, _, fin)) = self.slots[slot].as_mut() {
            if !*fin {
                st.accept_f32(&self.pad_zeros);
                st.input_finished();
                *fin = true;
            }
        }
    }
    fn slot_is_ready(&self, slot: usize) -> bool {
        self.slots[slot]
            .as_ref()
            .is_some_and(|(_, st, _, _)| st.is_ready())
    }
    fn decode_ready_step(&mut self, decode: DecodeMode, max_batch: usize) -> (usize, u64) {
        let mut addrs = Vec::new();
        let mut indices = Vec::new();
        for (idx, sl) in self.slots.iter().enumerate() {
            if let Some((_, st, _, _)) = sl.as_ref().filter(|(_, st, _, _)| st.is_ready()) {
                addrs.push(st.addr());
                indices.push(idx);
            }
        }
        if addrs.is_empty() {
            return (0, 0);
        }
        let mut w = 0u64;
        match decode {
            DecodeMode::Batch => {
                for ch in addrs.chunks(max_batch) {
                    self.engine.decode_batch_addrs(ch);
                    w += 1;
                }
            }
            DecodeMode::Single => {
                for idx in indices {
                    if let Some((_, st, _, _)) = &self.slots[idx] {
                        self.engine.decode_one(st);
                        w += 1;
                    }
                }
            }
        }
        (addrs.len(), w)
    }
    fn take_completed_slot(&mut self, slot: usize) -> String {
        match self.slots[slot].take() {
            Some((_, st, _, _)) => st.text().trim().to_owned(),
            None => String::new(),
        }
    }
}

fn pin_current_thread(cpus: &[usize]) {
    if cpus.is_empty() {
        return;
    }
    let mut set = nix::sched::CpuSet::new();
    for &c in cpus {
        let _ = set.set(c);
    }
    let _ = nix::sched::sched_setaffinity(nix::unistd::Pid::from_raw(0), &set);
}

pub fn run_sp_a_shared_workers(
    cfg: &SpAConfig,
    workers: usize,
) -> Result<SharedFleetResult, StreamError> {
    if workers == 0 {
        return Err(StreamError::Protocol("workers must be >= 1".into()));
    }
    let audio_utts = load_npz_audio_only(&cfg.npz_path, cfg.offset, cfg.limit)?;
    if audio_utts.is_empty() {
        return Err(StreamError::Protocol("no audio utterances loaded".into()));
    }
    let lib = SherpaLib::load_default()?;
    let sherpa_ver = lib.version();
    let ort_ver = lib.onnxruntime_version();
    let engine = SherpaZipformerBackend::new(lib, cfg.zipformer.clone())?;

    let aff = current_affinity();
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

    let mut sampler = TreeSampler::new(std::process::id() as i32);
    sampler.sample();
    let load_before = read_loadavg();
    let cpu0 = self_cpu_seconds()?;
    let wall_t0 = std::time::Instant::now();
    let barrier = Barrier::new(workers);

    let outcomes: Vec<Result<SimOutcome, StreamError>> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for (w, cpus) in worker_cpus.iter().enumerate() {
            let eng = Arc::clone(&engine);
            let bar = &barrier;
            let utts_ref = &audio_utts;
            let cpus_clone = cpus.clone();
            handles.push(scope.spawn(move || {
                pin_current_thread(&cpus_clone);
                let mut bank = WorkerBank {
                    audio_utts: utts_ref,
                    rot: (w * 137) % utts_ref.len(),
                    engine: eng,
                    slots: (0..cfg.streams).map(|_| None).collect(),
                    pad_zeros: vec![0.0f32; (16_000.0 * cfg.zipformer.tail_pad_s) as usize],
                    scratch: Vec::with_capacity(1024),
                };
                bar.wait();
                let mut clk = RealClock::new()?;
                simulate_grid(
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
                )
            }));
        }
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .map_err(|_| StreamError::Runtime("worker panicked".into()))?
            })
            .collect()
    });

    let cpu_s = (self_cpu_seconds()? - cpu0).max(1e-6);
    let wall_s = wall_t0.elapsed().as_secs_f64().max(1e-6);
    sampler.sample();
    let res_sum = sampler.summary();
    let load_after = read_loadavg();
    let cpu_only = verify_cpu_only_process();
    drop(engine);

    let mut all_completed = Vec::new();
    let mut all_ticks = Vec::new();
    let mut total_audio_s = 0.0f64;
    let mut total_wakeups = 0u64;
    let mut eff_grid = cfg.grid_ms;
    let mut summaries = Vec::with_capacity(workers);
    let mut sustained_workers = 0usize;

    for (w, res) in outcomes.into_iter().enumerate() {
        let sim = res?;
        if !sim.drained {
            return Err(StreamError::Runtime(format!("worker {w} did not drain")));
        }
        eff_grid = sim.effective_grid_ms;
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
            utterances: sim.completed.len(),
            audio_accepted_s: (sim.audio_accepted_s * 1000.0).round() / 1000.0,
            wall_seconds: (sim.wall_seconds * 1000.0).round() / 1000.0,
            accepted_packets: sim.accepted_packets,
            drained: sim.drained,
            sustained: sust,
            lateness_ms: sim.lateness_ms,
        });
        // Note: in WorkerBank, `slot_utt` already returns the real index in `audio_utts`!
        all_completed.extend(sim.completed);
    }

    if total_audio_s <= 0.0 {
        return Err(StreamError::Protocol("zero audio accepted".into()));
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
    let cfg_threads = (workers as u32) * (cfg.zipformer.num_threads.max(1) as u32);
    let receipt = Receipt {
        schema: RECEIPT_SCHEMA.into(),
        bundle: model_id,
        source_commit: cfg.source_commit.clone(),
        hardware: cfg.hardware_id.clone(),
        target: std::env::consts::ARCH.into(),
        configured_threads: cfg_threads,
        observed_threads: Some(res_sum.max_threads.max(1) as u32),
        cpu_only_verified: cpu_only,
        audio_seconds: (total_audio_s * 1000.0).round() / 1000.0,
        process_tree_cpu_seconds: Some((cpu_s * 1000.0).round() / 1000.0),
        peak_rss_bytes: Some(res_sum.peak_rss_bytes.max(1)),
        events: Some(all_completed.len() as u64),
        wakeups: Some(total_wakeups),
        cap_hits: Some(0),
        chunk_ms,
        status_events: Vec::new(),
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
    let per_core_full_load = total_audio_s / cpu_s.max(1e-9);
    let wall_streams = total_audio_s / wall_s.max(1e-9);

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
        cpu_seconds: (cpu_s * 1000.0).round() / 1000.0,
        wall_seconds: (wall_s * 1000.0).round() / 1000.0,
        rtf_cpu: ((cpu_s / total_audio_s.max(1e-9)) * 100_000.0).round() / 100_000.0,
        total_streams_at_full_load: ((per_core_full_load * pinned_cores) * 100.0).round() / 100.0,
        streams_per_physical_core_equiv: ((wall_streams / pinned_cores).min(per_core_full_load)
            * 100.0)
            .round()
            / 100.0,
        peak_rss_bytes: res_sum.peak_rss_bytes.max(1),
        wer: wer_summary,
        affinity: aff,
        load_before,
        load_after,
        worker_summaries: summaries,
        receipt,
        scored_rows,
    })
}
