//! Opt-in live-endpointer [`CommitPassSession`] adapter (`ultra stream --commit-pass`).
//! Streams `c=4` provisional previews via `accept_i16_live`, tracks causal blank-frame hangover
//! (`200/300/500/800 ms`) + real pre-roll ring (`--commit-preroll-ms`), runs single in-flight
//! `c=16` beam+LM commit with `40 s` cap fallback, and updates optional session lexical memory.

use std::collections::VecDeque;
use std::sync::Arc;

use serde_json::json;
use ultra_core::events::TranscriptEvent;
use ultra_engine::commit::{
    CommitConfig, CommitLmConfig, CommitPassSession, SessionLexicalConfig, SessionLexicalMemory,
    DEFAULT_MAX_PCM_SAMPLES, DEFAULT_SAMPLE_RATE, DEFAULT_TAIL_PAD_SAMPLES,
};

use crate::api::{
    AsrStream, PollResult, PushReport, PushStatus, StreamClock, StreamConfig, StreamError,
    StreamMetrics, TerminalOutcome,
};
use crate::native_engine::NativeEngineBackend;
use crate::ort_r1::R1_SAMPLES_PER_ENC_FRAME;

#[derive(Debug, Clone)]
pub struct LiveCommitConfig {
    pub commit: CommitConfig,
    pub hangover_ms: u32,
    pub preroll_ms: u32,
    pub commit_lm: Option<CommitLmConfig>,
    pub session_bias: SessionLexicalConfig,
}

impl Default for LiveCommitConfig {
    fn default() -> Self {
        Self {
            commit: CommitConfig {
                commit_chunk: 16,
                tail_pad_samples: DEFAULT_TAIL_PAD_SAMPLES,
                max_pcm_samples: DEFAULT_MAX_PCM_SAMPLES,
                ..CommitConfig::default()
            },
            hangover_ms: 300,
            preroll_ms: 300,
            commit_lm: None,
            session_bias: SessionLexicalConfig::default(),
        }
    }
}

impl LiveCommitConfig {
    pub fn validate(&self) -> Result<(), StreamError> {
        self.commit.validate()?;
        self.session_bias.validate()?;
        if !(1..=10_000).contains(&self.hangover_ms) {
            return Err(StreamError::Protocol(
                "commit hangover_ms must be in 1..=10000".into(),
            ));
        }
        if self.preroll_ms > 5_000 {
            return Err(StreamError::Protocol(
                "commit preroll_ms must be in 0..=5000".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Default)]
struct LiveCommitState {
    metrics: StreamMetrics,
    last_tokens_len: usize,
    last_enc_frames: usize,
    last_hyp: String,
    speech_armed: bool,
    trailing_blank_samples: u64,
    speech_end_ns: Option<u64>,
    commit_in_flight: bool,
    finished: bool,
    cancelled: bool,
    failed: Option<String>,
}

/// Live-endpointer [`AsrStream`] wrapping [`CommitPassSession`] with single in-flight commit pass.
pub struct LiveCommitAsrStream {
    sess: CommitPassSession,
    live_cfg: LiveCommitConfig,
    session_bias: SessionLexicalMemory,
    clock: StreamClock,
    st: LiveCommitState,
    events: VecDeque<TranscriptEvent>,
}

impl LiveCommitAsrStream {
    pub fn new(
        backend: &Arc<NativeEngineBackend>,
        cfg: &StreamConfig,
        live_cfg: &LiveCommitConfig,
        clock: StreamClock,
    ) -> Result<Self, StreamError> {
        live_cfg.validate()?;
        let bid = backend.bundle_id();
        let eid = if cfg.model_config_id == "default-model-v1" {
            bid.as_str()
        } else {
            cfg.model_config_id.as_str()
        };
        let sess = CommitPassSession::with_stream_id(
            backend.engine.clone(),
            live_cfg.commit,
            &cfg.stream_id,
            eid,
        )?
        .with_commit_lm(live_cfg.commit_lm.clone())?;
        Ok(Self {
            sess,
            live_cfg: live_cfg.clone(),
            session_bias: SessionLexicalMemory::new(live_cfg.session_bias)?,
            clock,
            st: LiveCommitState::default(),
            events: VecDeque::new(),
        })
    }

    pub fn session_bias(&self) -> &SessionLexicalMemory {
        &self.session_bias
    }
    fn hangover_samples(&self) -> u64 {
        (u64::from(self.live_cfg.hangover_ms) * DEFAULT_SAMPLE_RATE as u64).div_ceil(1000)
    }
    fn preroll_keep_samples(&self) -> usize {
        let p = (self.live_cfg.preroll_ms as usize * DEFAULT_SAMPLE_RATE).div_ceil(1000);
        (p + 8 * R1_SAMPLES_PER_ENC_FRAME as usize).min(self.live_cfg.commit.max_pcm_samples)
    }

    fn run_commit_endpoint(&mut self, trigger: &'static str) -> Result<(), StreamError> {
        if self.st.commit_in_flight {
            return Err(StreamError::Protocol(
                "concurrent commit pass already in flight for stream".into(),
            ));
        }
        self.st.commit_in_flight = true;
        let (seg_start, seg_end) = self.sess.segment_sample_range();
        let live_forced = self.sess.live_progress().2;
        let res = self.sess.commit_endpoint_event(self.clock.now_ns());
        self.st.commit_in_flight = false;
        let (cr, final_ev) = res.map_err(|e| {
            self.st.failed = Some(e.to_string());
            StreamError::from(e)
        })?;
        let done_ns = self.clock.now_ns();
        let e2f_ms = self.st.speech_end_ns.map_or(cr.commit_wall_ms, |t0| {
            (done_ns.saturating_sub(t0) as f64 / 1e6).max(cr.commit_wall_ms)
        });
        let m = &mut self.st.metrics;
        m.cap_hits += live_forced;
        m.refiner_attempts += 1;
        *if cr.revised {
            &mut m.refiner_accepts
        } else {
            &mut m.refiner_rejects
        } += 1;
        m.refiner_charged_cpu_s += cr.commit_wall_ms / 1000.0;
        self.session_bias
            .observe_committed_segment(&cr.committed, None, !cr.pcm_truncated);
        let detail = json!({
            "trigger": trigger, "start_sample": seg_start, "end_sample": seg_end,
            "hangover_ms": self.live_cfg.hangover_ms, "preroll_ms": self.live_cfg.preroll_ms,
            "commit_wall_ms": (cr.commit_wall_ms * 1000.0).round() / 1000.0,
            "endpoint_to_final_ms": (e2f_ms * 1000.0).round() / 1000.0,
            "revised": cr.revised, "pcm_truncated": cr.pcm_truncated,
            "session_bias_active_words": self.session_bias.active_words(),
        })
        .to_string();
        let status_ev = self.sess.status_event(done_ns, "commit_pass", &detail);
        self.events.extend([final_ev, status_ev]);
        self.st.metrics.events_emitted += 2;
        self.st.metrics.processed_samples = self.st.metrics.accepted_samples;
        self.st.reset_segment_tracking();
        Ok(())
    }

    fn retained_bytes(&self) -> u64 {
        let pcm = self.sess.buffered_samples() * std::mem::size_of::<i16>();
        let evs = self.events.capacity() * std::mem::size_of::<TranscriptEvent>();
        (pcm + evs + std::mem::size_of::<Self>()) as u64
    }
}

impl LiveCommitState {
    fn reset_segment_tracking(&mut self) {
        (
            self.last_tokens_len,
            self.last_enc_frames,
            self.speech_armed,
        ) = (0, 0, false);
        (self.trailing_blank_samples, self.speech_end_ns) = (0, None);
        self.last_hyp.clear();
    }
}

impl AsrStream for LiveCommitAsrStream {
    fn push_pcm(&mut self, start_sample: u64, pcm: &[i16]) -> Result<PushReport, StreamError> {
        if self.st.finished || self.st.cancelled || self.st.failed.is_some() {
            return Err(ultra_core::CoreError::Lifecycle("push after terminal state").into());
        }
        let steps = self.sess.accept_i16_live(pcm).map_err(|e| {
            self.st.failed = Some(e.to_string());
            StreamError::from(e)
        })?;
        self.st.metrics.accepted_samples += pcm.len() as u64;
        if steps > 0 {
            self.st.metrics.wakeups += steps as u64;
            let (cur_tok, cur_frames, _, seg_open) = self.sess.live_progress();
            let delta_frames = cur_frames.saturating_sub(self.st.last_enc_frames);
            self.st.last_enc_frames = cur_frames;
            let seg_start = self.sess.segment_sample_range().0;
            self.st.metrics.processed_samples = (seg_start
                + (cur_frames as u64) * R1_SAMPLES_PER_ENC_FRAME)
                .min(self.st.metrics.accepted_samples);
            if cur_tok > self.st.last_tokens_len {
                (self.st.last_tokens_len, self.st.speech_armed) = (cur_tok, true);
                (self.st.trailing_blank_samples, self.st.speech_end_ns) = (0, None);
                let hyp = self.sess.provisional_text();
                if hyp != self.st.last_hyp {
                    self.events
                        .push_back(self.sess.preview_event(self.clock.now_ns())?);
                    self.st.last_hyp = hyp;
                    self.st.metrics.events_emitted += 1;
                }
            } else if self.st.speech_armed && delta_frames > 0 {
                if self.st.trailing_blank_samples == 0 {
                    self.st.speech_end_ns = Some(self.clock.now_ns());
                }
                self.st.trailing_blank_samples += (delta_frames as u64) * R1_SAMPLES_PER_ENC_FRAME;
                if self.st.trailing_blank_samples >= self.hangover_samples() {
                    self.run_commit_endpoint("hangover")?;
                }
            } else if !self.st.speech_armed && !seg_open {
                let reset_live = cur_frames >= 100; // >= 4 s of pure silence
                self.sess
                    .trim_idle_pcm(self.preroll_keep_samples(), reset_live);
                if reset_live {
                    (self.st.last_tokens_len, self.st.last_enc_frames) = (0, 0);
                }
            }
        }
        Ok(PushReport {
            start_sample,
            accepted_samples: pcm.len(),
            status: PushStatus::AcceptedAll,
        })
    }

    fn poll_event(&mut self) -> Result<PollResult, StreamError> {
        if let Some(ev) = self.events.pop_front() {
            Ok(PollResult::Event(Box::new(ev)))
        } else if let Some(msg) = &self.st.failed {
            Ok(PollResult::Ended(TerminalOutcome::Failed(msg.clone())))
        } else if self.st.cancelled {
            Ok(PollResult::Ended(TerminalOutcome::Cancelled))
        } else if self.st.finished {
            Ok(PollResult::Ended(TerminalOutcome::Finished))
        } else {
            Ok(PollResult::Pending)
        }
    }

    fn finish_input(&mut self) -> Result<(), StreamError> {
        if self.st.cancelled || self.st.failed.is_some() {
            return Err(ultra_core::CoreError::Lifecycle("finish_input on terminal stream").into());
        }
        if !self.st.finished {
            let (s0, s1) = self.sess.segment_sample_range();
            if s1 > s0 || self.sess.live_progress().3 || self.st.metrics.events_emitted == 0 {
                self.run_commit_endpoint("eof")?;
            }
            self.st.finished = true;
            self.st.metrics.processed_samples = self.st.metrics.accepted_samples;
        }
        Ok(())
    }

    fn cancel(&mut self) -> Result<(), StreamError> {
        if !self.st.finished && self.st.failed.is_none() {
            self.st.cancelled = true;
            self.events.clear();
        }
        Ok(())
    }

    fn metrics(&self) -> StreamMetrics {
        let mut m = self.st.metrics;
        m.cap_hits += self.sess.live_progress().2;
        m.retained_bytes = self.retained_bytes();
        m
    }
}
