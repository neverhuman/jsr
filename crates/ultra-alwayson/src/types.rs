//! Configuration, state, ledger, event, continuity-gap, and telemetry types for H47.

use ultra_gate::GateConfig;

use crate::{AlwaysOnError, SampleCounters};

/// Explicit 5-state H47 always-on sleep ladder (explicitly NO warm-pause / suppressed-emission).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LadderState {
    /// Asleep: only the PCM ring and pure-integer Sentinel-0 run (zero engine calls).
    /// Sleep depth escalates `1 -> 2 -> 3` (`poll_hint_ms`: `20 -> 80 -> 160 ms`).
    Listening { depth: u8 },
    /// Triggered by Sentinel-0 or hot-at-close: executing one-pass batched pre-roll replay
    /// into a freshly reset stream before entering `Active`.
    Waking,
    /// Speech active: decoding incoming hops normally with zero consecutive blank frames.
    Active,
    /// Post-speech blank tail: tokens continue to emit normally if speech resumes (no warm-pause
    /// or suppressed emission), closing once `idle_ms` of blank encoder frames pass.
    Draining,
    /// Fault isolation state entered when the recogniser backend returns an error; refuses audio
    /// until `recover()` resets state cleanly.
    Recovery,
}

/// Reason a segment was opened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentCause {
    /// Stream start in ungated mode (`gate_enabled = false`).
    Start,
    /// Woken from `Listening` by Sentinel-0.
    Wake,
    /// Previous segment closed after `idle_ms` of blanks while Sentinel-0 remained hot (or ungated).
    HotAtClose,
    /// Short audible onset buffered while Sentinel-0 was still in startup warm-up, flushed at EOF.
    EofShortOnset,
}

/// Explicit capture continuity gap recorded whenever a backend fault interrupts a `push_pcm` or
/// `finish` call. Post-`recover()` segments always resume at `end_sample` (exact capture time).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinuityGap {
    /// First absolute capture sample whose decoding was interrupted or lost by the fault.
    pub start_sample: u64,
    /// Exclusive absolute capture sample at which post-`recover()` audio will resume.
    pub end_sample: u64,
    /// Samples in the failing `push_pcm` call already forwarded before the fault.
    pub consumed_in_call: usize,
    /// Samples in the failing `push_pcm` call never forwarded due to the fault.
    pub unconsumed_in_call: usize,
    /// Fault description from the recogniser backend.
    pub reason: String,
}

/// Closed isolated segment recorded in the session ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    /// Inclusive absolute start sample index (in caller capture coordinates).
    pub s0: u64,
    /// Exclusive absolute end sample index (`s0 + 640 * frames - 240`, or EOF cursor on `finish()`).
    pub e: u64,
    /// Complete 25 Hz encoder frames processed in this segment.
    pub frames: u32,
    pub cause: SegmentCause,
    pub tokens: Vec<u32>,
    pub trigger_hop: u64,
    pub trigger_path: u8,
    /// Number of batched backend decode calls needed for the wake pre-roll (`1` for one-pass).
    pub wake_decode_calls: u64,
    /// True if closed via `finish()` (EOF partial-chunk flush).
    pub eof_flushed: bool,
}

/// Synchronous events emitted from [`crate::AlwaysOnSession::push_pcm`] / `finish`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionEvent<'a> {
    StateChanged {
        from: LadderState,
        to: LadderState,
    },
    Wake {
        s0: u64,
        trigger_hop: u64,
        replay_samples: usize,
        decode_calls: u64,
    },
    Tokens {
        segment: usize,
        tokens: &'a [u32],
    },
    Sleep {
        segment: usize,
    },
    ContinuityGap {
        start_sample: u64,
        end_sample: u64,
    },
}

/// Configuration for [`crate::AlwaysOnSession`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlwaysOnConfig {
    /// Pre-roll replayed from the ring at wake in ms (default `500` = `8_000` samples; `<= idle_ms - 300`).
    pub replay_ms: u32,
    /// Consecutive blank encoder-frame ms before closing a segment (default `1500`; Q pass 2 optimum
    /// avoiding mid-utterance restarts and hot-close word duplication under recogniser-blank close).
    pub idle_ms: u32,
    /// Optional onset phase-lock offset in ms (`0..=150`, multiple of `10`; default `0`).
    pub phase_ms: u32,
    /// Fixed circular PCM16 ring duration in ms (default `3000` = `48_000` samples = `96 KB`).
    pub ring_ms: u32,
    /// Sleep-depth `poll_hint_ms` table for depths 1 / 2 / 3 (default `[20, 80, 160]`).
    pub poll_ms: [u32; 3],
    /// Seconds asleep before escalating from depth 1->2 and 2->3 (default `[10, 120]`).
    pub depth_after_s: [u32; 2],
    /// Pure-integer Sentinel-0 thresholds.
    pub gate: GateConfig,
    /// When `false`, behaves as the ungated reset-with-replay control arm (never sleeps).
    pub gate_enabled: bool,
    /// When `true` (default), if `finish()` arrives while `Listening` before Sentinel-0 exits its
    /// `70 ms` startup warm-up and buffered audio exceeds `-72 dBFS`, flush it as an isolated
    /// `EofShortOnset` segment so short startups are never silently omitted.
    pub flush_short_onset_on_eof: bool,
    /// When `false` (default), mid-stream sleep transitions (`Draining -> Listening` / `HotAtClose`)
    /// close on complete chunks only without calling `input_finished()` on a trailing noise chunk.
    /// Final stream EOF `finish()` still flushes cleanly.
    pub flush_on_sleep_close: bool,
}

impl Default for AlwaysOnConfig {
    fn default() -> Self {
        Self {
            replay_ms: 500,
            idle_ms: 1500,
            phase_ms: 0,
            ring_ms: 3000,
            poll_ms: [20, 80, 160],
            depth_after_s: [10, 120],
            gate: GateConfig::default(),
            gate_enabled: true,
            flush_short_onset_on_eof: true,
            flush_on_sleep_close: false,
        }
    }
}

impl AlwaysOnConfig {
    /// Validate H47 temporal and ring-sizing constraints.
    pub fn validate(&self) -> Result<(), AlwaysOnError> {
        if !self.replay_ms.is_multiple_of(10)
            || !self.phase_ms.is_multiple_of(10)
            || self.phase_ms > 150
        {
            return Err(AlwaysOnError::InvalidConfig(
                "replay_ms and phase_ms must be multiples of 10 (phase_ms <= 150)".into(),
            ));
        }
        if self.idle_ms < 400 || !self.idle_ms.is_multiple_of(10) {
            return Err(AlwaysOnError::InvalidConfig(
                "idle_ms must be >= 400 and a multiple of 10".into(),
            ));
        }
        if self.replay_ms.saturating_add(300) > self.idle_ms {
            return Err(AlwaysOnError::InvalidConfig(
                "replay_ms must be <= idle_ms - 300 to prevent token double-decoding".into(),
            ));
        }
        let min_ring_ms = self.replay_ms + self.phase_ms + self.poll_ms[2] + 100;
        if self.ring_ms < min_ring_ms {
            return Err(AlwaysOnError::InvalidConfig(format!(
                "ring_ms ({}) must be >= {min_ring_ms}",
                self.ring_ms
            )));
        }
        if self.depth_after_s[0] == 0 || self.depth_after_s[1] < self.depth_after_s[0] {
            return Err(AlwaysOnError::InvalidConfig(
                "depth_after_s must satisfy 0 < d0 <= d1".into(),
            ));
        }
        Ok(())
    }
}

/// Session-level counters and duty-cycle telemetry.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SessionStats {
    pub total_samples_pushed: u64,
    pub recogniser_audio_samples: u64,
    pub engine_decode_calls: u64,
    pub wake_count: u64,
    pub hot_at_close_count: u64,
    pub short_onset_flushes: u64,
    pub unflushed_short_onset_samples: u64,
    pub episode_resets: u64,
    pub recovery_count: u64,
    pub ring_underflows: u64,
    pub listening_hops: u64,
    pub total_hops: u64,
    pub ring: SampleCounters,
}

/// Active open-segment cursor metadata tracked while `Active` / `Draining`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct OpenSegment {
    pub s0: u64,
    pub cause: SegmentCause,
    pub trigger_hop: u64,
    pub trigger_path: u8,
    pub wake_decode_calls: u64,
}

impl Default for OpenSegment {
    fn default() -> Self {
        Self {
            s0: 0,
            cause: SegmentCause::Start,
            trigger_hop: 0,
            trigger_path: 0,
            wake_decode_calls: 0,
        }
    }
}
