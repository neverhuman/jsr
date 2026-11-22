//! [`SimSlotBank`], packet-streaming parity runner, and single/batched SP-A driver
//! (`run_sp_a_native_engine`) for [`NativeEngineBackend`].

use std::path::PathBuf;
use std::sync::Arc;

use ultra_core::receipt::{HonestyLabel, Partition};
use ultra_engine::{EngineStream, Precision, Workspace};
use ultra_eval::procfs::TreeSampler;
use ultra_frontend::npz::AudioUtterance;

use crate::api::StreamError;
use crate::contrast::{load_selected_audio, read_loadavg};
use crate::grid_sim::{simulate_grid, DecodeMode, SimParams, SimSlotBank};
use crate::native_engine::NativeEngineBackend;
use crate::scheduler::{finalize_sp_a_result, RealClock, SpAMeta, SpAResult};

struct EngineActiveSlot {
    utt_pos: usize,
    cursor: usize,
    finished: bool,
    raw: EngineStream,
}

/// [`SimSlotBank`] driving [`NativeEngineBackend`] inside the shared SP-A grid simulator.
pub struct NativeEngineSlotBank<'a> {
    pub backend: Arc<NativeEngineBackend>,
    pub utts: &'a [AudioUtterance],
    utt_offset: usize,
    slots: Vec<Option<EngineActiveSlot>>,
    ws: Workspace,
    sampler: Option<&'a mut TreeSampler>,
    pub forced_advances: u64,
    pub last_error: Option<StreamError>,
}

impl<'a> NativeEngineSlotBank<'a> {
    pub fn with_start_offset(
        backend: Arc<NativeEngineBackend>,
        utts: &'a [AudioUtterance],
        utt_offset: usize,
        streams: usize,
        sampler: Option<&'a mut TreeSampler>,
    ) -> Self {
        Self {
            backend,
            utts,
            utt_offset,
            slots: (0..streams).map(|_| None).collect(),
            ws: Workspace::default(),
            sampler,
            forced_advances: 0,
            last_error: None,
        }
    }
}

impl SimSlotBank for NativeEngineSlotBank<'_> {
    fn num_utterances(&self) -> usize {
        self.utts.len()
    }

    fn utt_samples(&self, utt_pos: usize) -> usize {
        self.utts[utt_pos % self.utts.len()].pcm.len()
    }

    fn open_slot(&mut self, slot: usize, utt_pos: usize) -> Result<(), StreamError> {
        let idx = (utt_pos + self.utt_offset) % self.utts.len();
        self.slots[slot] = Some(EngineActiveSlot {
            utt_pos: idx,
            cursor: 0,
            finished: false,
            raw: self.backend.create_raw_stream(),
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
        if let Err(e) = st.raw.accept_i16(slice) {
            self.last_error = Some(e.into());
            return 0;
        }
        st.cursor = end;
        if st.cursor >= pcm.len() {
            st.raw.input_finished();
            st.finished = true;
        }
        slice.len()
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
            .is_some_and(|st| self.backend.engine.is_ready(&st.raw))
    }

    fn decode_ready_step(&mut self, decode: DecodeMode, max_batch: usize) -> (usize, u64) {
        let ready_indices: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(idx, s)| {
                s.as_ref()
                    .filter(|st| self.backend.engine.is_ready(&st.raw))
                    .map(|_| idx)
            })
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
            let mut refs: Vec<&mut EngineStream> = Vec::with_capacity(chunk.len());
            let mut iter_idx = 0usize;
            for (slot_i, opt) in self.slots.iter_mut().enumerate() {
                if iter_idx < chunk.len() && chunk[iter_idx] == slot_i {
                    if let Some(st) = opt.as_mut() {
                        refs.push(&mut st.raw);
                    }
                    iter_idx += 1;
                }
            }
            match self.backend.engine.decode(&mut refs, &mut self.ws) {
                Ok(n_adv) => wakeups += u64::from(n_adv > 0),
                Err(e) => {
                    self.last_error = Some(e.into());
                    break;
                }
            }
        }
        (max_ready, wakeups)
    }

    fn take_completed_slot(&mut self, slot: usize) -> String {
        match self.slots[slot].take() {
            Some(st) => {
                self.forced_advances += st.raw.forced_advances();
                self.backend.engine.text(&st.raw)
            }
            None => String::new(),
        }
    }

    fn on_periodic_tick(&mut self) {
        if let Some(s) = self.sampler.as_mut() {
            s.sample();
        }
    }

    fn take_error(&mut self) -> Option<StreamError> {
        self.last_error.take()
    }
}

/// Output of [`run_engine_packet_streaming_with_tokens`].
#[derive(Debug, Clone)]
pub struct EnginePacketStreamingRun {
    pub hyps: Vec<String>,
    pub tokens: Vec<Vec<i64>>,
    pub forced_advances: u64,
    pub max_active_streams: usize,
    pub decode_batches: u64,
}

/// Packet-streaming runner (`packet_ms` packets across `streams` concurrent slots) over
/// [`NativeEngineBackend`].
pub fn run_engine_packet_streaming_with_tokens(
    backend: &Arc<NativeEngineBackend>,
    utts: &[AudioUtterance],
    streams: usize,
    packet_ms: u32,
) -> Result<EnginePacketStreamingRun, StreamError> {
    if streams == 0 || packet_ms == 0 {
        return Err(StreamError::Protocol(
            "streams and packet_ms must be > 0".into(),
        ));
    }
    let packet_samples = 16_000usize * (packet_ms as usize) / 1000;
    let mut next_utt = 0usize;
    let mut active: Vec<(usize, EngineActiveSlot)> = Vec::with_capacity(streams);
    let mut occupied = vec![false; streams];
    let mut hyps = vec![String::new(); utts.len()];
    let mut toks = vec![Vec::new(); utts.len()];
    let mut ws = Workspace::default();
    let mut forced_advances = 0u64;
    let mut max_active_streams = 0usize;
    let mut decode_batches = 0u64;

    while next_utt < utts.len() || !active.is_empty() {
        for (slot_id, occ) in occupied.iter_mut().enumerate() {
            if !*occ && next_utt < utts.len() {
                *occ = true;
                active.push((
                    slot_id,
                    EngineActiveSlot {
                        utt_pos: next_utt,
                        cursor: 0,
                        finished: false,
                        raw: backend.create_raw_stream(),
                    },
                ));
                next_utt += 1;
            }
        }
        max_active_streams = max_active_streams.max(active.len());
        for (_, st) in &mut active {
            let pcm = &utts[st.utt_pos].pcm;
            if st.cursor < pcm.len() {
                let end = (st.cursor + packet_samples).min(pcm.len());
                st.raw.accept_i16(&pcm[st.cursor..end])?;
                st.cursor = end;
                if st.cursor >= pcm.len() {
                    st.raw.input_finished();
                    st.finished = true;
                }
            }
        }
        loop {
            let mut ready: Vec<&mut EngineStream> = active
                .iter_mut()
                .filter_map(|(_, st)| backend.engine.is_ready(&st.raw).then_some(&mut st.raw))
                .collect();
            if ready.is_empty() {
                break;
            }
            let adv = backend.engine.decode(&mut ready, &mut ws)?;
            decode_batches += u64::from(adv > 0);
        }
        let mut remaining = Vec::with_capacity(active.len());
        for (slot_id, st) in active {
            if st.cursor >= utts[st.utt_pos].pcm.len() && !backend.engine.is_ready(&st.raw) {
                occupied[slot_id] = false;
                forced_advances += st.raw.forced_advances();
                hyps[st.utt_pos] = backend.engine.text(&st.raw);
                toks[st.utt_pos] = st.raw.tokens().iter().map(|&t| i64::from(t)).collect();
            } else {
                remaining.push((slot_id, st));
            }
        }
        active = remaining;
    }
    Ok(EnginePacketStreamingRun {
        hyps,
        tokens: toks,
        forced_advances,
        max_active_streams,
        decode_batches,
    })
}

#[derive(Debug, Clone)]
pub struct EngineSpAConfig {
    pub weights_dir: PathBuf,
    pub precision: Precision,
    pub threads: i32,
    pub npz_path: PathBuf,
    pub split_name: String,
    pub partition: Partition,
    pub honesty_label: HonestyLabel,
    pub offset: usize,
    pub limit: Option<usize>,
    pub selected_indices: Option<Vec<usize>>,
    pub duration_s: Option<f64>,
    pub streams: usize,
    pub decode: DecodeMode,
    pub max_batch: usize,
    pub max_chunks: usize,
    pub beam: usize,
    pub packet_ms: u32,
    pub grid_ms: u32,
    pub lateness_bound_ms: f64,
    pub paced: bool,
    pub source_commit: String,
    pub hardware_id: String,
    pub completion_status: String,
}

pub(crate) fn engine_status_events(b: &NativeEngineBackend) -> Vec<String> {
    let a = &b.engine.model.arch;
    let mut evs = vec![
        format!("isa:{}", b.engine.isa.name()),
        format!("int8_source:{}", b.int8_source),
        format!("weights_sha256:{}", a.weights_sha256),
        format!("checkpoint_sha256:{}", a.checkpoint_sha256),
        format!("max_chunks:{}", b.engine.max_chunks),
        format!("beam:{}", b.engine.beam),
    ];
    if let Some(prep) = &b.engine.model.int8_prepared {
        evs.push(format!("weights_int8_sha256:{}", prep.sha256));
    }
    evs
}

/// Run the fixed-deadline SP-A loop over [`NativeEngineBackend`] and emit a validated receipt.
pub fn run_sp_a_native_engine(cfg: &EngineSpAConfig) -> Result<SpAResult, StreamError> {
    let backend = NativeEngineBackend::load_with_options(
        &cfg.weights_dir,
        cfg.precision,
        cfg.threads,
        true,
        cfg.max_chunks.max(1),
        cfg.beam.max(1),
    )?;
    let utts = load_selected_audio(
        &cfg.npz_path,
        cfg.offset,
        cfg.limit,
        cfg.selected_indices.as_deref(),
    )?;
    if utts.is_empty() {
        return Err(StreamError::Protocol("empty utterance slice".into()));
    }
    let mut sampler = TreeSampler::new(std::process::id() as i32);
    sampler.sample();
    let load_before = read_loadavg();
    let mut bank = NativeEngineSlotBank::with_start_offset(
        Arc::clone(&backend),
        &utts,
        0,
        cfg.streams,
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
    let forced = bank.forced_advances;
    if let Some(err) = bank.last_error.take() {
        return Err(err);
    }
    drop(bank);
    sampler.sample();
    let res_sum = sampler.summary();
    let load_after = read_loadavg();
    let mut res = finalize_sp_a_result(
        SpAMeta {
            driver: "rust-ultra-engine-v1".into(),
            sherpa_ver: "n/a".into(),
            ort_ver: "none".into(),
            model_id: backend.bundle_id(),
            precision: cfg.precision.as_str().into(),
            threads: cfg.threads,
            tail_pad_s: 0.0,
            npz_path: cfg.npz_path.clone(),
            split_name: cfg.split_name.clone(),
            partition: cfg.partition,
            honesty_label: cfg.honesty_label,
            offset: cfg.offset,
            limit: cfg.limit,
            selected_indices: cfg.selected_indices.clone(),
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
    )?;
    res.receipt
        .status_events
        .extend(engine_status_events(&backend));
    Ok(res)
}
