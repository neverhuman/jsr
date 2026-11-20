//! `SimSlotBank`, packet-streaming runner, and SP-A driver for [`OrtTransducerBackend`].

use std::path::PathBuf;
use std::sync::Arc;

use ultra_core::receipt::{HonestyLabel, Partition};
use ultra_eval::procfs::TreeSampler;
use ultra_frontend::npz::{load_npz_audio_only, AudioUtterance};
use ultra_gp::R1Precision;

use crate::api::StreamError;
use crate::contrast::read_loadavg;
use crate::grid_sim::{simulate_grid, DecodeMode, SimParams, SimSlotBank};
use crate::ort_r1::{BatchWorkspace, OrtTransducerBackend};
use crate::ort_r1_stream::OrtR1RawStream;
use crate::scheduler::{finalize_sp_a_result, RealClock, SpAMeta, SpAResult};

struct R1ActiveSlot {
    utt_pos: usize,
    cursor: usize,
    finished: bool,
    raw: OrtR1RawStream,
}

/// [`SimSlotBank`] implementation driving [`OrtTransducerBackend`] (`S` or `S_MR`).
pub struct OrtR1SlotBank<'a> {
    pub engine: Arc<OrtTransducerBackend>,
    pub utts: &'a [AudioUtterance],
    slots: Vec<Option<R1ActiveSlot>>,
    ws: BatchWorkspace,
    sampler: Option<&'a mut TreeSampler>,
    pub forced_advances: u64,
    pub last_error: Option<StreamError>,
}

impl<'a> OrtR1SlotBank<'a> {
    pub fn new(
        engine: Arc<OrtTransducerBackend>,
        utts: &'a [AudioUtterance],
        streams: usize,
        sampler: Option<&'a mut TreeSampler>,
    ) -> Self {
        Self {
            engine,
            utts,
            slots: (0..streams).map(|_| None).collect(),
            ws: BatchWorkspace::new(),
            sampler,
            forced_advances: 0,
            last_error: None,
        }
    }
}

impl SimSlotBank for OrtR1SlotBank<'_> {
    fn num_utterances(&self) -> usize {
        self.utts.len()
    }

    fn utt_samples(&self, utt_pos: usize) -> usize {
        self.utts[utt_pos].pcm.len()
    }

    fn open_slot(&mut self, slot: usize, utt_pos: usize) -> Result<(), StreamError> {
        self.slots[slot] = Some(R1ActiveSlot {
            utt_pos,
            cursor: 0,
            finished: false,
            raw: self.engine.create_raw_stream(),
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
        self.slots[slot].as_ref().is_some_and(|s| s.finished)
    }

    fn accept_packet(&mut self, slot: usize, max_samples: usize) -> usize {
        let Some(st) = self.slots[slot].as_mut() else {
            return 0;
        };
        let pcm = &self.utts[st.utt_pos].pcm;
        if st.cursor >= pcm.len() {
            return 0;
        }
        let end = (st.cursor + max_samples).min(pcm.len());
        let slice = &pcm[st.cursor..end];
        let n = slice.len();
        st.raw.accept_i16(slice);
        st.cursor = end;
        if st.cursor >= pcm.len() {
            st.raw.input_finished();
            st.finished = true;
        }
        n
    }

    fn finish_slot_input(&mut self, slot: usize) {
        if let Some(st) = self.slots[slot].as_mut() {
            if !st.finished {
                st.raw.input_finished();
                st.finished = true;
            }
        }
    }

    fn slot_is_ready(&self, slot: usize) -> bool {
        self.slots[slot]
            .as_ref()
            .is_some_and(|st| st.raw.is_ready())
    }

    fn decode_ready_step(&mut self, decode: DecodeMode, max_batch: usize) -> (usize, u64) {
        let ready_indices: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(idx, s)| s.as_ref().filter(|st| st.raw.is_ready()).map(|_| idx))
            .collect();
        if ready_indices.is_empty() {
            return (0, 0);
        }
        let max_ready = ready_indices.len();
        let step_cap = match decode {
            DecodeMode::Single => 1,
            DecodeMode::Batch => max_batch.max(1),
        };
        let mut wakeups = 0u64;
        for chunk in ready_indices.chunks(step_cap) {
            let mut refs: Vec<&mut OrtR1RawStream> = Vec::with_capacity(chunk.len());
            let mut iter_idx = 0usize;
            for (slot_i, opt) in self.slots.iter_mut().enumerate() {
                if iter_idx < chunk.len() && chunk[iter_idx] == slot_i {
                    if let Some(st) = opt.as_mut() {
                        refs.push(&mut st.raw);
                    }
                    iter_idx += 1;
                }
            }
            match self.engine.decode_streams(&mut refs, &mut self.ws) {
                Ok((_, w)) => wakeups += w,
                Err(e) => {
                    self.last_error = Some(e);
                    break;
                }
            }
        }
        (max_ready, wakeups)
    }

    fn take_completed_slot(&mut self, slot: usize) -> String {
        match self.slots[slot].take() {
            Some(st) => {
                self.forced_advances += st.raw.rnnt.forced_advances;
                st.raw.text()
            }
            None => String::new(),
        }
    }

    fn on_periodic_tick(&mut self) {
        if let Some(s) = self.sampler.as_mut() {
            s.sample();
        }
    }
}

/// Output of [`run_r1_packet_streaming_with_tokens`].
#[derive(Debug, Clone)]
pub struct R1PacketStreamingRun {
    pub hyps: Vec<String>,
    pub tokens: Vec<Vec<i64>>,
    pub forced_advances: u64,
}

/// Exact Rust mirror of `python/ultra/train/infer.py:run_streaming` (`packet_ms` packets across
/// `streams` concurrent slots) returning hypotheses, token-ID sequences, and forced advances.
pub fn run_r1_packet_streaming_with_tokens(
    engine: &Arc<OrtTransducerBackend>,
    utts: &[AudioUtterance],
    streams: usize,
    packet_ms: u32,
) -> Result<R1PacketStreamingRun, StreamError> {
    if streams == 0 || packet_ms == 0 {
        return Err(StreamError::Protocol(
            "streams and packet_ms must be > 0".into(),
        ));
    }
    let packet_samples = 16_000usize * (packet_ms as usize) / 1000;
    let mut next_utt = 0usize;
    let mut active: Vec<(usize, R1ActiveSlot)> = Vec::with_capacity(streams);
    let mut occupied = vec![false; streams];
    let mut hyps = vec![String::new(); utts.len()];
    let mut toks = vec![Vec::new(); utts.len()];
    let mut ws = BatchWorkspace::new();
    let mut forced_advances = 0u64;

    while next_utt < utts.len() || !active.is_empty() {
        for (slot_id, occ) in occupied.iter_mut().enumerate() {
            if !*occ && next_utt < utts.len() {
                *occ = true;
                active.push((
                    slot_id,
                    R1ActiveSlot {
                        utt_pos: next_utt,
                        cursor: 0,
                        finished: false,
                        raw: engine.create_raw_stream(),
                    },
                ));
                next_utt += 1;
            }
        }
        for (_, st) in &mut active {
            let pcm = &utts[st.utt_pos].pcm;
            if st.cursor < pcm.len() {
                let end = (st.cursor + packet_samples).min(pcm.len());
                st.raw.accept_i16(&pcm[st.cursor..end]);
                st.cursor = end;
                if st.cursor >= pcm.len() {
                    st.raw.input_finished();
                    st.finished = true;
                }
            }
        }
        loop {
            let mut ready: Vec<&mut OrtR1RawStream> = active
                .iter_mut()
                .filter_map(|(_, st)| st.raw.is_ready().then_some(&mut st.raw))
                .collect();
            if ready.is_empty() {
                break;
            }
            engine.decode_streams(&mut ready, &mut ws)?;
        }
        let mut remaining = Vec::with_capacity(active.len());
        for (slot_id, st) in active {
            if st.cursor >= utts[st.utt_pos].pcm.len() && !st.raw.is_ready() {
                occupied[slot_id] = false;
                forced_advances += st.raw.rnnt.forced_advances;
                hyps[st.utt_pos] = st.raw.text();
                toks[st.utt_pos] = st.raw.rnnt.tokens;
            } else {
                remaining.push((slot_id, st));
            }
        }
        active = remaining;
    }
    Ok(R1PacketStreamingRun {
        hyps,
        tokens: toks,
        forced_advances,
    })
}

/// Exact Rust mirror of `python/ultra/train/infer.py:run_streaming` returning `(hyps, forced_advances)`.
pub fn run_r1_packet_streaming(
    engine: &Arc<OrtTransducerBackend>,
    utts: &[AudioUtterance],
    streams: usize,
    packet_ms: u32,
) -> Result<(Vec<String>, u64), StreamError> {
    let run = run_r1_packet_streaming_with_tokens(engine, utts, streams, packet_ms)?;
    Ok((run.hyps, run.forced_advances))
}

#[derive(Debug, Clone)]
pub struct R1SpAConfig {
    pub bundle_dir: PathBuf,
    pub precision: R1Precision,
    pub threads: i32,
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
    pub completion_status: String,
}

/// Run the fixed-deadline SP-A loop over [`OrtTransducerBackend`] and emit a validated receipt.
pub fn run_sp_a_ort_r1(cfg: &R1SpAConfig) -> Result<SpAResult, StreamError> {
    let engine = OrtTransducerBackend::load(&cfg.bundle_dir, cfg.precision, cfg.threads, true)?;
    let utts = load_npz_audio_only(&cfg.npz_path, cfg.offset, cfg.limit)?;
    if utts.is_empty() {
        return Err(StreamError::Protocol("empty utterance slice".into()));
    }

    let mut sampler = TreeSampler::new(std::process::id() as i32);
    sampler.sample();
    let load_before = read_loadavg();

    let mut bank = OrtR1SlotBank::new(Arc::clone(&engine), &utts, cfg.streams, Some(&mut sampler));
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
    let forced = bank.forced_advances;
    if let Some(err) = bank.last_error.take() {
        return Err(err);
    }
    drop(bank);
    sampler.sample();
    let res_sum = sampler.summary();
    let load_after = read_loadavg();

    let model_id = format!(
        "ultra-r1-{}-{}-{}",
        engine.bundle.family.as_str(),
        engine.bundle.precision.as_str(),
        &engine.bundle.manifest_sha256[..12]
    );

    finalize_sp_a_result(
        SpAMeta {
            driver: "rust-ort-c-api-v1".into(),
            sherpa_ver: "n/a".into(),
            ort_ver: engine.ort.version.clone(),
            model_id,
            precision: cfg.precision.as_str().into(),
            threads: cfg.threads,
            tail_pad_s: 0.0,
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
            cap_hits: Some(forced),
            completion_status: cfg.completion_status.clone(),
            validate_evidence: true,
        },
        &utts,
        sim,
        res_sum.max_threads,
        res_sum.peak_rss_bytes,
        load_before,
        load_after,
    )
}
