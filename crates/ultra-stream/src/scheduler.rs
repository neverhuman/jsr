//! Pure-Rust SP-A runner over `SherpaZipformerBackend` using [`crate::grid_sim::simulate_grid`].

use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use ultra_core::receipt::{HonestyLabel, Partition, Receipt, RECEIPT_SCHEMA};
use ultra_eval::procfs::TreeSampler;
use ultra_eval::score::ScoreRecord;
use ultra_frontend::npz::{load_npz_audio_only, AudioUtterance};

use crate::api::StreamError;
use crate::contrast::{
    build_event_hashes_and_chunks, current_affinity, read_loadavg, score_completed,
    self_cpu_seconds,
};
pub use crate::contrast::{default_npz_for_split, WerSummary};
pub use crate::grid_sim::{
    simulate_grid, sustains_lateness, DecodeMode, LatenessQuantiles, SimClock, SimOutcome,
    SimParams, SimSlotBank, DEFAULT_LATENESS_BOUND_MS,
};
use crate::sherpa_ffi::{verify_cpu_only_process, SherpaLib};
use crate::sherpa_online::{RawOnlineStream, SherpaZipformerBackend, SherpaZipformerConfig};

pub(crate) struct RealClock {
    t0: Instant,
    c0: f64,
}

impl RealClock {
    pub(crate) fn new() -> Result<Self, StreamError> {
        Ok(Self {
            t0: Instant::now(),
            c0: self_cpu_seconds()?,
        })
    }
}

impl SimClock for RealClock {
    fn now_s(&self) -> f64 {
        self.t0.elapsed().as_secs_f64()
    }
    fn sleep_s(&mut self, dt_s: f64) {
        if dt_s > 0.0 {
            std::thread::sleep(Duration::from_secs_f64(dt_s));
        }
    }
    fn cpu_s(&self) -> Result<f64, StreamError> {
        Ok((self_cpu_seconds()? - self.c0).max(0.0))
    }
}

#[derive(Debug, Clone)]
pub struct SpAConfig {
    pub bundle_id: Option<String>,
    pub npz_path: PathBuf,
    pub split_name: String,
    pub partition: Partition,
    pub honesty_label: HonestyLabel,
    pub offset: usize,
    pub limit: Option<usize>,
    pub duration_s: Option<f64>,
    pub streams: usize,
    pub decode: DecodeMode,
    pub max_batch: usize,
    pub packet_ms: u32,
    pub grid_ms: u32,
    pub lateness_bound_ms: f64,
    pub paced: bool,
    pub zipformer: SherpaZipformerConfig,
    pub source_commit: String,
    pub hardware_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpAResult {
    pub schema: String,
    pub driver: String,
    pub sherpa_version: String,
    pub onnxruntime_version: String,
    pub split: String,
    pub utterances: usize,
    pub audio_seconds: f64,
    pub audio_completed_seconds: f64,
    pub streams: usize,
    pub decode: String,
    pub max_batch: usize,
    pub precision: String,
    pub threads: i32,
    pub paced: bool,
    pub tail_pad_s: f64,
    pub packet_ms: u32,
    pub grid_ms: u32,
    pub effective_grid_ms: u32,
    pub accepted_packets: u64,
    pub offered_to_decode_packets: u64,
    pub drained: bool,
    pub lateness_bound_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sustained: Option<bool>,
    pub cpu_seconds: f64,
    pub wall_seconds: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drain_cpu_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drain_wall_seconds: Option<f64>,
    pub rtf_cpu: f64,
    pub streams_per_core_at_full_load: f64,
    pub wer: WerSummary,
    pub max_ready_backlog: usize,
    pub affinity: Vec<usize>,
    pub load_before: [f64; 3],
    pub load_after: [f64; 3],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lateness_ms: Option<LatenessQuantiles>,
    pub receipt: Receipt,
    #[serde(skip)]
    pub scored_rows: Vec<ScoreRecord>,
}

pub(crate) struct SherpaSlot {
    utt_pos: usize,
    stream: RawOnlineStream,
    cursor: usize,
    finished_input: bool,
}

pub(crate) struct SherpaBank<'a> {
    audio_utts: &'a [AudioUtterance],
    rot: usize,
    engine: std::sync::Arc<SherpaZipformerBackend>,
    slots: Vec<Option<SherpaSlot>>,
    pad_zeros: Vec<f32>,
    scratch: Vec<f32>,
    sampler: Option<&'a mut TreeSampler>,
}

impl<'a> SherpaBank<'a> {
    pub(crate) fn new(
        audio_utts: &'a [AudioUtterance],
        rot: usize,
        engine: std::sync::Arc<SherpaZipformerBackend>,
        streams: usize,
        tail_pad_s: f64,
        sampler: Option<&'a mut TreeSampler>,
    ) -> Self {
        Self {
            audio_utts,
            rot,
            engine,
            slots: (0..streams).map(|_| None).collect(),
            pad_zeros: vec![0.0f32; (16_000.0 * tail_pad_s) as usize],
            scratch: Vec::with_capacity(1024),
            sampler,
        }
    }
}

impl SimSlotBank for SherpaBank<'_> {
    fn num_utterances(&self) -> usize {
        self.audio_utts.len()
    }
    fn utt_samples(&self, utt_pos: usize) -> usize {
        self.audio_utts[utt_pos].pcm.len()
    }
    fn open_slot(&mut self, slot: usize, utt_pos: usize) -> Result<(), StreamError> {
        let real = (utt_pos + self.rot) % self.audio_utts.len();
        self.slots[slot] = Some(SherpaSlot {
            utt_pos: real,
            stream: self.engine.create_raw_stream()?,
            cursor: 0,
            finished_input: false,
        });
        Ok(())
    }
    fn has_slot(&self, slot: usize) -> bool {
        self.slots[slot].is_some()
    }
    fn slot_utt(&self, slot: usize) -> usize {
        self.slots[slot].as_ref().map_or(0, |s| s.utt_pos)
    }
    fn slot_cursor(&self, slot: usize) -> usize {
        self.slots[slot].as_ref().map_or(0, |s| s.cursor)
    }
    fn slot_finished_input(&self, slot: usize) -> bool {
        self.slots[slot].as_ref().is_some_and(|s| s.finished_input)
    }
    fn accept_packet(&mut self, slot: usize, max_samples: usize) -> usize {
        let Some(st) = self.slots[slot].as_mut() else {
            return 0;
        };
        let pcm = &self.audio_utts[st.utt_pos].pcm;
        if st.cursor >= pcm.len() {
            return 0;
        }
        let end = (st.cursor + max_samples).min(pcm.len());
        let n = end - st.cursor;
        st.stream
            .accept_i16(&pcm[st.cursor..end], &mut self.scratch);
        st.cursor = end;
        if st.cursor >= pcm.len() {
            st.stream.accept_f32(&self.pad_zeros);
            st.stream.input_finished();
            st.finished_input = true;
        }
        n
    }
    fn finish_slot_input(&mut self, slot: usize) {
        if let Some(st) = self.slots[slot].as_mut() {
            if !st.finished_input {
                st.stream.accept_f32(&self.pad_zeros);
                st.stream.input_finished();
                st.finished_input = true;
            }
        }
    }
    fn slot_is_ready(&self, slot: usize) -> bool {
        self.slots[slot]
            .as_ref()
            .is_some_and(|s| s.stream.is_ready())
    }
    fn decode_ready_step(&mut self, decode: DecodeMode, max_batch: usize) -> (usize, u64) {
        let mut addrs = Vec::new();
        let mut indices = Vec::new();
        for (idx, sl) in self.slots.iter().enumerate() {
            if let Some(s) = sl.as_ref().filter(|s| s.stream.is_ready()) {
                addrs.push(s.stream.addr());
                indices.push(idx);
            }
        }
        if addrs.is_empty() {
            return (0, 0);
        }
        let mut w = 0u64;
        match decode {
            DecodeMode::Batch => {
                for chunk in addrs.chunks(max_batch) {
                    self.engine.decode_batch_addrs(chunk);
                    w += 1;
                }
            }
            DecodeMode::Single => {
                for idx in indices {
                    if let Some(s) = &self.slots[idx] {
                        self.engine.decode_one(&s.stream);
                        w += 1;
                    }
                }
            }
        }
        (addrs.len(), w)
    }
    fn take_completed_slot(&mut self, slot: usize) -> String {
        match self.slots[slot].take() {
            Some(st) => st.stream.text().trim().to_owned(),
            None => String::new(),
        }
    }
    fn on_periodic_tick(&mut self) {
        if let Some(s) = self.sampler.as_mut() {
            s.sample();
        }
    }
}

pub(crate) struct SpAMeta {
    pub driver: String,
    pub sherpa_ver: String,
    pub ort_ver: String,
    pub model_id: String,
    pub precision: String,
    pub threads: i32,
    pub tail_pad_s: f64,
    pub npz_path: PathBuf,
    pub split_name: String,
    pub partition: Partition,
    pub honesty_label: HonestyLabel,
    pub offset: usize,
    pub limit: Option<usize>,
    pub duration_s: Option<f64>,
    pub streams: usize,
    pub decode: DecodeMode,
    pub max_batch: usize,
    pub packet_ms: u32,
    pub grid_ms: u32,
    pub lateness_bound_ms: f64,
    pub paced: bool,
    pub source_commit: String,
    pub hardware_id: String,
    pub cap_hits: Option<u64>,
    pub completion_status: String,
    pub validate_evidence: bool,
}

pub(crate) fn finalize_sp_a_result(
    meta: SpAMeta,
    audio_utts: &[AudioUtterance],
    mut sim: SimOutcome,
    max_threads: u64,
    peak_rss_bytes: u64,
    load_before: [f64; 3],
    load_after: [f64; 3],
) -> Result<SpAResult, StreamError> {
    let cpu_only = verify_cpu_only_process();
    if sim.audio_accepted_s <= 0.0 {
        return Err(StreamError::Protocol(
            "zero audio accepted before duration cutoff".into(),
        ));
    }
    let (hashes, chunk_ms) = build_event_hashes_and_chunks(
        &meta.model_id,
        audio_utts,
        &sim.completed,
        &mut sim.tick_compute_ms,
    )?;
    let cpu_s = sim.cpu_seconds.max(1e-6);
    let wall_s = sim.wall_seconds.max(1e-6);
    let receipt = Receipt {
        schema: RECEIPT_SCHEMA.into(),
        bundle: meta.model_id,
        source_commit: meta.source_commit,
        hardware: meta.hardware_id,
        target: std::env::consts::ARCH.into(),
        configured_threads: meta.threads.max(1) as u32,
        observed_threads: Some(max_threads.max(1) as u32),
        cpu_only_verified: cpu_only,
        audio_seconds: (sim.audio_accepted_s * 1000.0).round() / 1000.0,
        process_tree_cpu_seconds: Some((cpu_s * 1000.0).round() / 1000.0),
        peak_rss_bytes: Some(peak_rss_bytes.max(1)),
        events: Some(sim.completed.len() as u64),
        wakeups: Some(sim.wakeups),
        cap_hits: meta.cap_hits,
        chunk_ms,
        status_events: Vec::new(),
        gaps: Vec::new(),
        hashes,
        label: meta.honesty_label,
        partition: meta.partition,
        completion_status: meta.completion_status,
    };
    if meta.validate_evidence {
        let mut check = receipt.clone();
        check.completion_status = "complete".into();
        check.validate_as_evidence()?;
    }

    let (wer_summary, scored_rows) = score_completed(
        &meta.npz_path,
        &meta.split_name,
        meta.offset,
        meta.limit,
        meta.duration_s.is_some(),
        &sim.completed,
    )?;
    let sustained = sim
        .lateness_ms
        .as_ref()
        .map(|lat| sustains_lateness(lat, sim.drained, meta.lateness_bound_ms));

    Ok(SpAResult {
        schema: "ultra.sp-a/v3".into(),
        driver: meta.driver,
        sherpa_version: meta.sherpa_ver,
        onnxruntime_version: meta.ort_ver,
        split: meta.split_name,
        utterances: sim.completed.len(),
        audio_seconds: (sim.audio_accepted_s * 1000.0).round() / 1000.0,
        audio_completed_seconds: (sim.audio_completed_s * 1000.0).round() / 1000.0,
        streams: meta.streams,
        decode: meta.decode.as_str().into(),
        max_batch: meta.max_batch,
        precision: meta.precision,
        threads: meta.threads,
        paced: meta.paced,
        tail_pad_s: meta.tail_pad_s,
        packet_ms: meta.packet_ms,
        grid_ms: meta.grid_ms,
        effective_grid_ms: sim.effective_grid_ms,
        accepted_packets: sim.accepted_packets,
        offered_to_decode_packets: sim.offered_to_decode_packets,
        drained: sim.drained,
        lateness_bound_ms: meta.lateness_bound_ms,
        sustained,
        cpu_seconds: (cpu_s * 1000.0).round() / 1000.0,
        wall_seconds: (wall_s * 1000.0).round() / 1000.0,
        drain_cpu_seconds: sim.drain_cpu_seconds,
        drain_wall_seconds: sim.drain_wall_seconds,
        rtf_cpu: ((cpu_s / sim.audio_accepted_s.max(1e-9)) * 100_000.0).round() / 100_000.0,
        streams_per_core_at_full_load: ((sim.audio_accepted_s / cpu_s.max(1e-9)) * 100.0).round()
            / 100.0,
        wer: wer_summary,
        max_ready_backlog: sim.max_ready_batch,
        affinity: current_affinity(),
        load_before,
        load_after,
        lateness_ms: sim.lateness_ms,
        receipt,
        scored_rows,
    })
}

pub fn run_sp_a_zipformer(cfg: &SpAConfig) -> Result<SpAResult, StreamError> {
    let audio_utts = load_npz_audio_only(&cfg.npz_path, cfg.offset, cfg.limit)?;
    if audio_utts.is_empty() {
        return Err(StreamError::Protocol("no audio utterances loaded".into()));
    }
    let lib = SherpaLib::load_default()?;
    let sherpa_ver = lib.version();
    let ort_ver = lib.onnxruntime_version();
    let engine = SherpaZipformerBackend::new(lib, cfg.zipformer.clone())?;

    let mut sampler = TreeSampler::new(std::process::id() as i32);
    sampler.sample();
    let load_before = read_loadavg();

    let mut bank = SherpaBank::new(
        &audio_utts,
        0,
        engine,
        cfg.streams,
        cfg.zipformer.tail_pad_s,
        Some(&mut sampler),
    );
    let mut clock = RealClock::new()?;
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
        &mut clock,
    )?;
    drop(bank);
    sampler.sample();
    let res_sum = sampler.summary();
    let load_after = read_loadavg();

    let model_id = match &cfg.bundle_id {
        Some(b) => b.clone(),
        None => format!(
            "sherpa-onnx-streaming-zipformer-en-2023-06-26-{}",
            cfg.zipformer.precision
        ),
    };
    finalize_sp_a_result(
        SpAMeta {
            driver: "rust-sherpa-c-api".into(),
            sherpa_ver,
            ort_ver,
            model_id,
            precision: cfg.zipformer.precision.clone(),
            threads: cfg.zipformer.num_threads,
            tail_pad_s: cfg.zipformer.tail_pad_s,
            npz_path: cfg.npz_path.clone(),
            split_name: cfg.split_name.clone(),
            partition: cfg.partition,
            honesty_label: cfg.honesty_label,
            offset: cfg.offset,
            limit: cfg.limit,
            duration_s: cfg.duration_s,
            streams: cfg.streams,
            decode: cfg.decode,
            max_batch: cfg.max_batch,
            packet_ms: cfg.packet_ms,
            grid_ms: cfg.grid_ms,
            lateness_bound_ms: cfg.lateness_bound_ms,
            paced: cfg.paced,
            source_commit: cfg.source_commit.clone(),
            hardware_id: cfg.hardware_id.clone(),
            // Sherpa C-API does not expose transducer cap-hit count; leave None rather than invent 0.
            cap_hits: None,
            completion_status: "complete".into(),
            validate_evidence: false,
        },
        &audio_utts,
        sim,
        res_sum.max_threads,
        res_sum.peak_rss_bytes,
        load_before,
        load_after,
    )
}
