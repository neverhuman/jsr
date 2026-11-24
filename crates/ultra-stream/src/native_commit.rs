//! Opt-in live-endpointer [`CommitPassSession`] adapter (`ultra stream --commit-pass`).
//! Streams `c=4` provisional previews via `accept_i16_live`, preserves monotonic encoder sample
//! offsets across idle resets, and runs single in-flight `c=16` commit jobs with deadline fallback.

use std::collections::VecDeque;
use std::sync::Arc;

use ultra_core::events::{Emitter, TranscriptEvent};
use ultra_engine::commit::{CommitPassSession, CommitSnapshot, CommitTiming, SessionLexicalMemory};
use ultra_engine::EffectiveDecodeConfig;

use crate::api::{
    AsrStream, PollResult, PushReport, PushStatus, StreamClock, StreamConfig, StreamError,
    StreamMetrics, TerminalOutcome,
};
use crate::commit_config::{CommitWorker, PendingCommit};
use crate::commit_queue::{emit_preview_event, finalize_and_flush, FinalizeInput, LiveCommitState};
use crate::native_engine::NativeEngineBackend;
use crate::ort_r1::R1_SAMPLES_PER_ENC_FRAME;

pub use crate::commit_queue::LiveCommitConfig;

/// Live-endpointer [`AsrStream`] wrapping [`CommitPassSession`] + 1-in-flight async commit worker.
pub struct LiveCommitAsrStream {
    sess: CommitPassSession,
    emitter: Emitter,
    worker: CommitWorker,
    pending: Option<PendingCommit>,
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
        if live_cfg.commit.ctc_weight > 0.0 && backend.engine.model.ctc_head.is_none() {
            return Err(StreamError::Protocol(
                "live commit ctc_weight > 0 requires a model loaded with ctc_head weights (disable NativeEngineLoadConfig::live_only)".into(),
            ));
        }
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
        let worker = CommitWorker::spawn(sess.commit_engine().clone(), live_cfg.commit);
        Ok(Self {
            sess,
            emitter: Emitter::new(&cfg.stream_id, eid),
            worker,
            pending: None,
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

    /// Resolved [`EffectiveDecodeConfig`] for the live provisional pass (`F10`).
    pub fn provisional_decode_config(&self) -> EffectiveDecodeConfig {
        self.sess
            .provisional_engine()
            .effective_decode_config()
            .with_endpoint_policy(format!(
                "live_commit_hangover_{}ms_deadline_{}ms",
                self.live_cfg.hangover_ms, self.live_cfg.deadline_ms
            ))
            .with_session_bias_config(self.session_bias.config())
    }

    /// Resolved [`EffectiveDecodeConfig`] for the background commit pass (`F10`).
    pub fn commit_decode_config(&self) -> EffectiveDecodeConfig {
        self.sess
            .commit_engine()
            .effective_decode_config()
            .with_endpoint_policy(format!(
                "commit_pass_chunk{}_hangover_{}ms_deadline_{}ms",
                self.live_cfg.commit.commit_chunk,
                self.live_cfg.hangover_ms,
                self.live_cfg.deadline_ms
            ))
            .with_tail_pad_samples(self.live_cfg.commit.tail_pad_samples)
            .with_session_bias_config(self.session_bias.config())
    }

    /// Total CPU seconds burned by timed-out, stale, or cancelled background commit jobs (`F09`).
    pub fn stale_worker_cpu_s(&self) -> f64 {
        self.st.stale_worker_cpu_s + self.worker.unreaped_worker_cpu_s()
    }

    /// Count of timed-out, stale, or cancelled background commit jobs (`F09`).
    pub fn stale_worker_jobs(&self) -> u64 {
        self.st.stale_worker_jobs + self.worker.unreaped_worker_jobs()
    }

    fn settle_pending(&mut self, wait_until_deadline: bool) -> Result<(), StreamError> {
        let (cfg, emitter, bias, clock, events) = (
            &self.live_cfg,
            &mut self.emitter,
            &mut self.session_bias,
            &self.clock,
            &mut self.events,
        );
        self.worker.settle(
            &mut self.pending,
            cfg,
            clock,
            &mut self.st,
            wait_until_deadline,
            |st, inp| finalize_and_flush(cfg, emitter, st, bias, clock, events, inp),
        )
    }

    fn emit_fallback(
        &mut self,
        trigger: &'static str,
        snap: &CommitSnapshot,
        speech_end_ns: Option<u64>,
        reason: &'static str,
    ) -> Result<(), StreamError> {
        finalize_and_flush(
            &self.live_cfg,
            &mut self.emitter,
            &mut self.st,
            &mut self.session_bias,
            &self.clock,
            &mut self.events,
            FinalizeInput {
                trigger,
                s0: snap.start_sample,
                s1: snap.end_sample,
                provisional: &snap.provisional,
                raw_committed: &snap.provisional,
                commit_timing: CommitTiming::default(),
                speech_end_ns,
                pcm_truncated: snap.pcm_truncated,
                fallback_reason: reason,
            },
        )
    }

    fn dispatch_commit_endpoint(&mut self, trigger: &'static str) -> Result<(), StreamError> {
        if self.pending.is_some() {
            self.settle_pending(true)?;
        } else {
            self.worker.drain_stale(&mut self.st);
        }
        let mut snap = self.sess.take_commit_snapshot().map_err(|e| {
            self.st.failed = Some(e.to_string());
            StreamError::from(e)
        })?;
        self.st.metrics.cap_hits += snap.forced_advances;
        self.st.live_base_sample = self.st.metrics.accepted_samples;
        self.st.metrics.processed_samples = self.st.metrics.accepted_samples;
        let speech_end_ns = self.st.speech_end_ns;
        self.st.reset_segment_tracking();

        let now_ns = self.clock.now_ns();
        let deadline_ns = self.live_cfg.deadline_ns(speech_end_ns, now_ns);
        let fallback = if snap.pcm_truncated {
            Some("pcm_cap")
        } else if now_ns >= deadline_ns {
            Some("deadline")
        } else if self.worker.is_busy() {
            Some("worker_busy")
        } else {
            None
        };
        if let Some(reason) = fallback {
            return self.emit_fallback(trigger, &snap, speech_end_ns, reason);
        }
        let job_id = self.st.next_job_id;
        self.st.next_job_id += 1;
        if !self
            .worker
            .try_submit(job_id, std::mem::take(&mut snap.pcm))
        {
            return self.emit_fallback(trigger, &snap, speech_end_ns, "worker_busy");
        }
        self.pending = Some(PendingCommit {
            id: job_id,
            trigger,
            seg_start: snap.start_sample,
            seg_end: snap.end_sample,
            provisional: snap.provisional,
            speech_end_ns,
            deadline_ns,
        });
        Ok(())
    }
}

impl AsrStream for LiveCommitAsrStream {
    fn push_pcm(&mut self, start_sample: u64, pcm: &[i16]) -> Result<PushReport, StreamError> {
        if self.st.finished || self.st.cancelled || self.st.failed.is_some() {
            return Err(ultra_core::CoreError::Lifecycle("push after terminal state").into());
        }
        self.settle_pending(false)?;
        let steps = self.sess.accept_i16_live(pcm).map_err(|e| {
            self.st.failed = Some(e.to_string());
            StreamError::from(e)
        })?;
        self.st.metrics.accepted_samples += pcm.len() as u64;
        if steps > 0 {
            self.st.metrics.wakeups += steps as u64;
            let (cur_tok, cur_frames, _, _) = self.sess.live_progress();
            let delta_frames = cur_frames.saturating_sub(self.st.last_enc_frames);
            self.st.last_enc_frames = cur_frames;
            let proc_now = (self.st.live_base_sample
                + (cur_frames as u64) * R1_SAMPLES_PER_ENC_FRAME)
                .min(self.st.metrics.accepted_samples);
            self.st.metrics.processed_samples = self.st.metrics.processed_samples.max(proc_now);
            let prog = self.sess.last_decode_progress();
            if prog.has_activity() || cur_tok != self.st.last_tokens_len {
                (self.st.last_tokens_len, self.st.speech_armed) = (cur_tok, true);
                (self.st.trailing_blank_samples, self.st.speech_end_ns) = (0, None);
                let hyp = self.sess.provisional_text();
                if hyp != self.st.last_hyp {
                    self.st.last_hyp = hyp.clone();
                    let end_samp = self.sess.total_samples();
                    if self.pending.is_some() {
                        self.st.deferred_preview = Some((hyp, end_samp));
                    } else {
                        emit_preview_event(
                            &mut self.emitter,
                            &mut self.st,
                            self.clock.now_ns(),
                            &mut self.events,
                            &hyp,
                            end_samp,
                        )?;
                    }
                }
            } else if self.st.speech_armed && delta_frames > 0 {
                if self.st.trailing_blank_samples == 0 {
                    self.st.speech_end_ns = Some(self.clock.now_ns());
                }
                self.st.trailing_blank_samples += (delta_frames as u64) * R1_SAMPLES_PER_ENC_FRAME;
                if self.st.trailing_blank_samples >= self.live_cfg.hangover_samples() {
                    self.dispatch_commit_endpoint("hangover")?;
                }
            } else if !self.st.speech_armed {
                let reset_live = cur_frames >= 100; // >= 4 s of pure silence
                self.sess
                    .trim_idle_pcm(self.live_cfg.preroll_keep_samples(), reset_live);
                if reset_live {
                    self.st.live_base_sample = self.st.metrics.accepted_samples;
                    self.st.metrics.processed_samples = self.st.metrics.accepted_samples;
                    (self.st.last_tokens_len, self.st.last_enc_frames) = (0, 0);
                }
            }
        }
        self.settle_pending(false)?;
        Ok(PushReport {
            start_sample,
            accepted_samples: pcm.len(),
            status: PushStatus::AcceptedAll,
        })
    }

    fn poll_event(&mut self) -> Result<PollResult, StreamError> {
        let _ = self.settle_pending(false);
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
            self.settle_pending(true)?;
            let (s0, s1) = self.sess.segment_sample_range();
            if s1 > s0 || self.st.seg_open || self.st.metrics.events_emitted == 0 {
                self.dispatch_commit_endpoint("eof")?;
                self.settle_pending(true)?;
            }
            self.st.finished = true;
            self.st.metrics.processed_samples = self.st.metrics.accepted_samples;
        }
        Ok(())
    }

    fn cancel(&mut self) -> Result<(), StreamError> {
        if !self.st.finished && self.st.failed.is_none() {
            self.st.cancelled = true;
            self.pending = None;
            self.worker.drain_stale(&mut self.st);
            self.events.clear();
        }
        Ok(())
    }

    fn metrics(&self) -> StreamMetrics {
        let mut m = self.st.metrics;
        m.refiner_charged_cpu_s += self.worker.unreaped_worker_cpu_s();
        m.cap_hits += self.sess.live_progress().2;
        let pcm = self.sess.buffered_samples() * std::mem::size_of::<i16>();
        let evs = self.events.capacity() * std::mem::size_of::<TranscriptEvent>();
        m.retained_bytes = (pcm + evs + std::mem::size_of::<Self>()) as u64;
        m
    }
}
