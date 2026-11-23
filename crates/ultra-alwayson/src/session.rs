//! Explicit 5-state H47 sleep ladder (`Listening`, `Waking`, `Active`, `Draining`, `Recovery`),
//! sleep-depth `poll_hint_ms` (`20 / 80 / 160 ms`), one-pass batched `1.0 s` pre-roll replay,
//! exact capture-position recovery continuity, and idempotent EOF handling.

mod ops;

use ultra_gate::{Sentinel0, HOP, KS};

use crate::types::OpenSegment;
use crate::{
    AlwaysOnConfig, AlwaysOnError, AsrBackend, ContinuityGap, LadderState, PcmRing, SampleCounters,
    Segment, SessionEvent, SessionStats, SAMPLE_RATE_HZ,
};

pub(super) const SAMPLES_PER_MS: u32 = (SAMPLE_RATE_HZ / 1000) as u32;
pub(super) const HOPS_PER_SEC: u64 = 100;

/// H47 always-on session driving the 5-state sleep ladder over any [`AsrBackend`].
#[derive(Debug)]
pub struct AlwaysOnSession<B: AsrBackend> {
    pub(super) backend: B,
    pub(super) cfg: AlwaysOnConfig,
    pub(super) ring: PcmRing,
    pub(super) gate: Sentinel0,
    pub(super) state: LadderState,
    pub(super) finished: bool,
    pub(super) pending: [i16; HOP],
    pub(super) pending_len: usize,
    pub(super) replay_scratch: Vec<i16>,
    pub(super) hot_recent: [bool; KS],
    pub(super) hop_idx: u64,
    pub(super) sleep_hops_run: u64,
    pub(super) idle_frames: usize,
    pub(super) open_seg: OpenSegment,
    pub(super) ledger: Vec<Segment>,
    pub(super) gaps: Vec<ContinuityGap>,
    pub(super) stats: SessionStats,
}

impl<B: AsrBackend> AlwaysOnSession<B> {
    /// Construct a validated always-on session (`Listening { depth: 1 }` when gated, `Active` when ungated).
    pub fn new(mut backend: B, cfg: AlwaysOnConfig) -> Result<Self, AlwaysOnError> {
        cfg.validate()?;
        let ring = PcmRing::new((cfg.ring_ms * SAMPLES_PER_MS) as usize)?;
        let mut stats = SessionStats::default();
        let state = if cfg.gate_enabled {
            backend.drop_stream();
            LadderState::Listening { depth: 1 }
        } else {
            backend.reset_stream();
            stats.episode_resets = 1;
            LadderState::Active
        };
        Ok(Self {
            backend,
            cfg,
            ring,
            gate: Sentinel0::new(cfg.gate),
            state,
            finished: false,
            pending: [0i16; HOP],
            pending_len: 0,
            replay_scratch: Vec::with_capacity((cfg.replay_ms * SAMPLES_PER_MS) as usize + 4096),
            hot_recent: [false; KS],
            hop_idx: 0,
            sleep_hops_run: 0,
            idle_frames: 0,
            open_seg: OpenSegment::default(),
            ledger: Vec::new(),
            gaps: Vec::new(),
            stats,
        })
    }

    pub fn state(&self) -> LadderState {
        self.state
    }
    pub fn is_finished(&self) -> bool {
        self.finished
    }
    pub fn ledger(&self) -> &[Segment] {
        &self.ledger
    }
    pub fn continuity_gaps(&self) -> &[ContinuityGap] {
        &self.gaps
    }
    pub fn backend(&self) -> &B {
        &self.backend
    }
    pub fn ring_counters(&self) -> SampleCounters {
        self.ring.counters()
    }
    pub fn stats(&self) -> SessionStats {
        SessionStats {
            ring: self.ring.counters(),
            ..self.stats
        }
    }

    /// Capture poll interval hint in ms: `20 / 80 / 160` by sleep depth, `10` when awake.
    pub fn poll_hint_ms(&self) -> u32 {
        match self.state {
            LadderState::Listening { depth: 1 } => self.cfg.poll_ms[0],
            LadderState::Listening { depth: 2 } => self.cfg.poll_ms[1],
            LadderState::Listening { .. } => self.cfg.poll_ms[2],
            _ => 10,
        }
    }

    /// Recover explicitly from [`LadderState::Recovery`], resetting the backend cleanly at the
    /// exact post-gap absolute capture cursor (`ring.end()`).
    pub fn recover(&mut self, on_event: &mut dyn FnMut(SessionEvent<'_>)) {
        if self.finished || self.state != LadderState::Recovery {
            return;
        }
        self.idle_frames = 0;
        self.sleep_hops_run = 0;
        if self.cfg.gate_enabled {
            self.backend.drop_stream();
            self.transition(LadderState::Listening { depth: 1 }, on_event);
        } else {
            self.backend.reset_stream();
            self.stats.episode_resets += 1;
            self.open_seg = OpenSegment {
                s0: self.ring.end(),
                ..OpenSegment::default()
            };
            self.transition(LadderState::Active, on_event);
        }
    }

    /// Push an arbitrary-sized PCM16 chunk (`80`, `160`, `320`, `640`, or odd sizes).
    ///
    /// Returns [`AlwaysOnError::AlreadyFinished`] after `finish()`, [`AlwaysOnError::NeedsRecovery`]
    /// while in `Recovery`, or [`AlwaysOnError::BackendFault`] with exact `consumed_samples` and
    /// `capture_end` if a backend call fails mid-chunk (recording a [`ContinuityGap`] and advancing
    /// the ring timeline to `capture_end` so post-`recover()` segments never drift).
    pub fn push_pcm(
        &mut self,
        pcm: &[i16],
        on_event: &mut dyn FnMut(SessionEvent<'_>),
    ) -> Result<(), AlwaysOnError> {
        if self.finished {
            return Err(AlwaysOnError::AlreadyFinished);
        }
        if self.state == LadderState::Recovery {
            return Err(AlwaysOnError::NeedsRecovery);
        }
        let capture_end = self.stats.total_samples_pushed + pcm.len() as u64;
        self.stats.total_samples_pushed = capture_end;
        let mut cursor = 0usize;
        let mut consumed_ok = 0usize;
        while cursor < pcm.len() {
            let take = (HOP - self.pending_len).min(pcm.len() - cursor);
            self.pending[self.pending_len..self.pending_len + take]
                .copy_from_slice(&pcm[cursor..cursor + take]);
            self.pending_len += take;
            cursor += take;
            if self.pending_len == HOP {
                let hop = self.pending;
                self.pending_len = 0;
                if let Err(err) = self.push_hop(&hop, on_event) {
                    let gap_start = self.ring.end().saturating_sub(HOP as u64);
                    return Err(self.record_fault_gap(
                        gap_start,
                        capture_end,
                        consumed_ok,
                        pcm.len().saturating_sub(consumed_ok),
                        err,
                        on_event,
                    ));
                }
                consumed_ok = cursor;
            }
        }
        Ok(())
    }

    /// Idempotent EOF: flushes trailing partial-hop samples, handles any short audible startup
    /// onset buffered before Sentinel-0 exited warm-up, closes any open segment, and marks the
    /// session terminally finished. Repeated calls return `Ok(())` without duplicating segments.
    pub fn finish(
        &mut self,
        on_event: &mut dyn FnMut(SessionEvent<'_>),
    ) -> Result<(), AlwaysOnError> {
        if self.finished {
            return Ok(());
        }
        if self.state == LadderState::Recovery {
            return Err(AlwaysOnError::NeedsRecovery);
        }
        let capture_end = self.stats.total_samples_pushed;
        let trailing = self.pending_len;
        if trailing > 0 {
            let tail = self.pending;
            self.pending_len = 0;
            self.ring.push(&tail[..trailing]);
            if matches!(self.state, LadderState::Active | LadderState::Draining) {
                let step = match self.backend.accept_and_decode_all(&tail[..trailing]) {
                    Ok(s) => s,
                    Err(e) => {
                        let gap_start = self.ring.end().saturating_sub(trailing as u64);
                        return Err(self.record_fault_gap(
                            gap_start,
                            capture_end,
                            0,
                            trailing,
                            e,
                            on_event,
                        ));
                    }
                };
                self.stats.recogniser_audio_samples += trailing as u64;
                self.stats.engine_decode_calls += step.decode_calls;
            }
        }
        if matches!(self.state, LadderState::Listening { .. }) {
            if let Err(e) = self.flush_short_onset_if_needed(on_event) {
                return Err(self.record_fault_gap(
                    self.ring.base(),
                    capture_end,
                    0,
                    0,
                    e,
                    on_event,
                ));
            }
        }
        if matches!(self.state, LadderState::Active | LadderState::Draining) {
            if let Err(e) = self.close_segment(true, on_event) {
                return Err(self.record_fault_gap(
                    self.open_seg.s0,
                    capture_end,
                    0,
                    0,
                    e,
                    on_event,
                ));
            }
            if self.cfg.gate_enabled {
                self.backend.drop_stream();
                self.transition(LadderState::Listening { depth: 1 }, on_event);
            }
        }
        self.finished = true;
        Ok(())
    }
}
