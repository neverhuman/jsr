//! Internal hop-stepping, wake-replay, short-onset EOF flushing, segment-closing, and
//! fault-recovery continuity helpers for [`super::AlwaysOnSession`].

use ultra_gate::{ilog2_q8, HOP, KS};

use super::{AlwaysOnSession, HOPS_PER_SEC, SAMPLES_PER_MS};
use crate::types::OpenSegment;
use crate::{
    AlwaysOnError, AsrBackend, ContinuityGap, LadderState, Segment, SegmentCause, SessionEvent,
};

/// Pure-integer check matching Sentinel-0's Band-A `-72 dBFS` absolute floor (`nabs_fast`) over
/// 2-hop (`320`-sample) blocks so digital silence / sub-`-72 dBFS` noise stay at 0 engine calls
/// while short audible onsets buffered during the `70 ms` warm-up are flushed at EOF.
fn has_above_floor_audio(pcm: &[i16], nabs_fast: i32) -> bool {
    if pcm.is_empty() {
        return false;
    }
    for block in pcm.chunks(2 * HOP) {
        let mut ea: u64 = 0;
        let mut prev = 0i32;
        for &s in block {
            let d = i64::from(i32::from(s) - prev);
            ea += (d * d) as u64;
            prev = i32::from(s);
        }
        if ilog2_q8(ea + 1) >= nabs_fast {
            return true;
        }
    }
    false
}

impl<B: AsrBackend> AlwaysOnSession<B> {
    pub(super) fn transition(
        &mut self,
        to: LadderState,
        on_event: &mut dyn FnMut(SessionEvent<'_>),
    ) {
        if self.state != to {
            let from = self.state;
            self.state = to;
            on_event(SessionEvent::StateChanged { from, to });
        }
    }

    pub(super) fn record_fault_gap(
        &mut self,
        start_sample: u64,
        capture_end: u64,
        consumed_in_call: usize,
        unconsumed_in_call: usize,
        err: AlwaysOnError,
        on_event: &mut dyn FnMut(SessionEvent<'_>),
    ) -> AlwaysOnError {
        let message = err.to_string();
        self.pending_len = 0;
        self.backend.drop_stream();
        self.ring.reset_at_cursor(capture_end);
        self.stats.recovery_count += 1;
        let gap = ContinuityGap {
            start_sample,
            end_sample: capture_end,
            consumed_in_call,
            unconsumed_in_call,
            reason: message.clone(),
        };
        self.gaps.push(gap);
        on_event(SessionEvent::ContinuityGap {
            start_sample,
            end_sample: capture_end,
        });
        self.transition(LadderState::Recovery, on_event);
        AlwaysOnError::BackendFault {
            consumed_samples: consumed_in_call,
            capture_end,
            message,
        }
    }

    pub(super) fn flush_short_onset_if_needed(
        &mut self,
        on_event: &mut dyn FnMut(SessionEvent<'_>),
    ) -> Result<(), AlwaysOnError> {
        // Only needed when EOF arrives before Sentinel-0 exited its startup warm-up (`KS - 1` hops).
        if self.gate.hops() >= KS as u64 || self.ring.is_empty() {
            return Ok(());
        }
        let s0 = self.ring.base();
        let now = self.ring.end();
        let mut inspect = Vec::new();
        self.ring.copy_range_readonly(s0, now, &mut inspect)?;
        if !has_above_floor_audio(&inspect, self.cfg.gate.nabs_fast) {
            return Ok(());
        }
        if !self.cfg.flush_short_onset_on_eof {
            self.stats.unflushed_short_onset_samples += inspect.len() as u64;
            return Ok(());
        }
        self.stats.short_onset_flushes += 1;
        self.open_segment(s0, SegmentCause::EofShortOnset, self.hop_idx, 0, on_event)
    }

    pub(super) fn push_hop(
        &mut self,
        hop: &[i16; HOP],
        on_event: &mut dyn FnMut(SessionEvent<'_>),
    ) -> Result<(), AlwaysOnError> {
        self.ring.push(hop);
        let g = self.gate.push_hop(hop);
        self.hot_recent[(self.hop_idx % KS as u64) as usize] = g.hot;
        let (now, cur_hop) = (self.ring.end(), self.hop_idx);
        self.hop_idx += 1;
        self.stats.total_hops += 1;

        match self.state {
            LadderState::Listening { .. } => {
                self.stats.listening_hops += 1;
                self.sleep_hops_run += 1;
                if g.hot {
                    let back = u64::from((self.cfg.replay_ms + self.cfg.phase_ms) * SAMPLES_PER_MS);
                    let start = now.saturating_sub(back);
                    self.open_segment(start, SegmentCause::Wake, cur_hop, g.path, on_event)?;
                } else {
                    let d = self.sleep_depth(self.sleep_hops_run);
                    self.transition(LadderState::Listening { depth: d }, on_event);
                }
            }
            LadderState::Active | LadderState::Draining => {
                self.step_active_hop(hop, now, cur_hop, g.path, on_event)?;
            }
            LadderState::Waking | LadderState::Recovery => {
                return Err(AlwaysOnError::NeedsRecovery)
            }
        }
        Ok(())
    }

    fn step_active_hop(
        &mut self,
        hop: &[i16; HOP],
        now: u64,
        cur_hop: u64,
        path: u8,
        on_event: &mut dyn FnMut(SessionEvent<'_>),
    ) -> Result<(), AlwaysOnError> {
        let step = self.backend.accept_and_decode_all(hop)?;
        self.stats.recogniser_audio_samples += HOP as u64;
        self.stats.engine_decode_calls += step.decode_calls;
        if step.new_tokens > 0 {
            self.idle_frames = 0;
            self.transition(LadderState::Active, on_event);
            on_event(SessionEvent::Tokens {
                segment: self.ledger.len(),
                tokens: self.backend.tokens(),
            });
        } else if step.frames_advanced > 0 {
            self.idle_frames += step.frames_advanced;
            if self.idle_frames * 40 >= self.cfg.idle_ms as usize {
                self.close_segment(false, on_event)?;
                if !self.cfg.gate_enabled || self.hot_recent.iter().any(|&h| h) {
                    let back = u64::from(self.cfg.replay_ms * SAMPLES_PER_MS);
                    let start = now.saturating_sub(back);
                    self.open_segment(start, SegmentCause::HotAtClose, cur_hop, path, on_event)?;
                } else {
                    self.backend.drop_stream();
                    self.sleep_hops_run = 0;
                    self.transition(LadderState::Listening { depth: 1 }, on_event);
                }
            } else {
                self.transition(LadderState::Draining, on_event);
            }
        }
        Ok(())
    }

    fn sleep_depth(&self, sleep_hops: u64) -> u8 {
        if sleep_hops >= u64::from(self.cfg.depth_after_s[1]) * HOPS_PER_SEC {
            3
        } else if sleep_hops >= u64::from(self.cfg.depth_after_s[0]) * HOPS_PER_SEC {
            2
        } else {
            1
        }
    }

    pub(super) fn open_segment(
        &mut self,
        desired: u64,
        cause: SegmentCause,
        trigger_hop: u64,
        trigger_path: u8,
        on_event: &mut dyn FnMut(SessionEvent<'_>),
    ) -> Result<(), AlwaysOnError> {
        self.transition(LadderState::Waking, on_event);
        let base = self.ring.base();
        let s0 = if desired < base {
            if self.ring.len() == self.ring.capacity() {
                self.stats.ring_underflows += 1;
            }
            base
        } else {
            desired
        };
        let replayed = self
            .ring
            .copy_range_into(s0, self.ring.end(), &mut self.replay_scratch)?;
        self.backend.reset_stream();
        self.stats.episode_resets += 1;
        match cause {
            SegmentCause::Wake => self.stats.wake_count += 1,
            SegmentCause::HotAtClose => self.stats.hot_at_close_count += 1,
            SegmentCause::Start | SegmentCause::EofShortOnset => {}
        }
        let step = self.backend.accept_and_decode_all(&self.replay_scratch)?;
        self.stats.recogniser_audio_samples += replayed as u64;
        self.stats.engine_decode_calls += step.decode_calls;
        self.open_seg = OpenSegment {
            s0,
            cause,
            trigger_hop,
            trigger_path,
            wake_decode_calls: step.decode_calls,
        };
        self.idle_frames = 0;
        self.sleep_hops_run = 0;
        on_event(SessionEvent::Wake {
            s0,
            trigger_hop,
            replay_samples: replayed,
            decode_calls: step.decode_calls,
        });
        if step.new_tokens > 0 {
            on_event(SessionEvent::Tokens {
                segment: self.ledger.len(),
                tokens: self.backend.tokens(),
            });
        }
        self.transition(LadderState::Active, on_event);
        Ok(())
    }

    pub(super) fn close_segment(
        &mut self,
        eof: bool,
        on_event: &mut dyn FnMut(SessionEvent<'_>),
    ) -> Result<(), AlwaysOnError> {
        let do_flush = eof || self.cfg.flush_on_sleep_close;
        if do_flush {
            let step = self.backend.finish_and_drain()?;
            self.stats.engine_decode_calls += step.decode_calls;
        }
        let frames = self.backend.frames_processed();
        let e = if do_flush {
            self.ring.end()
        } else if frames > 0 {
            self.open_seg.s0 + 640 * (frames as u64) - 240
        } else {
            self.open_seg.s0
        };
        let seg_idx = self.ledger.len();
        self.ledger.push(Segment {
            s0: self.open_seg.s0,
            e,
            frames: frames as u32,
            cause: self.open_seg.cause,
            tokens: self.backend.tokens().to_vec(),
            trigger_hop: self.open_seg.trigger_hop,
            trigger_path: self.open_seg.trigger_path,
            wake_decode_calls: self.open_seg.wake_decode_calls,
            eof_flushed: do_flush,
        });
        on_event(SessionEvent::Sleep { segment: seg_idx });
        Ok(())
    }
}
