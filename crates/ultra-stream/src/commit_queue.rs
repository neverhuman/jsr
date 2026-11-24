//! Configuration, state, and status formatting for [`crate::LiveCommitAsrStream`].

use serde_json::json;
use ultra_core::events::{Emitter, ProcessingEvidence, TranscriptEvent};
use ultra_engine::commit::{
    reconcile_immutable_suffix, CommitConfig, CommitLmConfig, CommitTiming, SessionLexicalConfig,
    SessionLexicalMemory, DEFAULT_MAX_PCM_SAMPLES, DEFAULT_SAMPLE_RATE, DEFAULT_TAIL_PAD_SAMPLES,
};

use crate::api::{StreamError, StreamMetrics};
use crate::ort_r1::R1_SAMPLES_PER_ENC_FRAME;

#[derive(Debug, Clone)]
pub struct LiveCommitConfig {
    pub commit: CommitConfig,
    pub hangover_ms: u32,
    pub preroll_ms: u32,
    pub deadline_ms: u32,
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
            deadline_ms: 800,
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
        if self.deadline_ms > 60_000 {
            return Err(StreamError::Protocol(
                "commit deadline_ms must be in 0..=60000".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn hangover_samples(&self) -> u64 {
        (u64::from(self.hangover_ms) * DEFAULT_SAMPLE_RATE as u64).div_ceil(1000)
    }

    pub(crate) fn preroll_keep_samples(&self) -> usize {
        let p = (self.preroll_ms as usize * DEFAULT_SAMPLE_RATE).div_ceil(1000);
        (p + 8 * R1_SAMPLES_PER_ENC_FRAME as usize).min(self.commit.max_pcm_samples)
    }

    pub(crate) fn deadline_ns(&self, speech_end_ns: Option<u64>, now_ns: u64) -> u64 {
        if self.deadline_ms == 0 {
            return u64::MAX;
        }
        let guard_ms = if self.deadline_ms > 60 {
            self.deadline_ms - 40
        } else {
            self.deadline_ms
        };
        speech_end_ns
            .unwrap_or(now_ns)
            .saturating_add(u64::from(guard_ms) * 1_000_000)
    }
}

#[derive(Default)]
pub(crate) struct LiveCommitState {
    pub metrics: StreamMetrics,
    pub live_base_sample: u64,
    pub last_tokens_len: usize,
    pub last_enc_frames: usize,
    pub last_hyp: String,
    pub speech_armed: bool,
    pub trailing_blank_samples: u64,
    pub speech_end_ns: Option<u64>,
    pub seg_open: bool,
    pub next_job_id: u64,
    pub stale_worker_cpu_s: f64,
    pub stale_worker_jobs: u64,
    pub deferred_preview: Option<(String, u64)>,
    pub finished: bool,
    pub cancelled: bool,
    pub failed: Option<String>,
}

impl LiveCommitState {
    pub fn reset_segment_tracking(&mut self) {
        (
            self.last_tokens_len,
            self.last_enc_frames,
            self.speech_armed,
        ) = (0, 0, false);
        (self.trailing_blank_samples, self.speech_end_ns) = (0, None);
        self.last_hyp.clear();
    }
}

pub(crate) struct FinalizeInput<'a> {
    pub trigger: &'static str,
    pub s0: u64,
    pub s1: u64,
    pub provisional: &'a str,
    pub raw_committed: &'a str,
    pub commit_timing: CommitTiming,
    pub speech_end_ns: Option<u64>,
    pub pcm_truncated: bool,
    pub fallback_reason: &'static str,
}

pub(crate) fn emit_preview_event(
    emitter: &mut Emitter,
    st: &mut LiveCommitState,
    now_ns: u64,
    events: &mut std::collections::VecDeque<TranscriptEvent>,
    hyp: &str,
    owned_to: u64,
) -> Result<(), StreamError> {
    let imm = if st.seg_open { emitter.immutable() } else { "" };
    let suffix = reconcile_immutable_suffix(imm, hyp);
    let ev = emitter.preview(now_ns, &suffix, owned_to)?;
    st.seg_open = true;
    events.push_back(ev);
    st.metrics.events_emitted += 1;
    Ok(())
}

pub(crate) fn finalize_and_flush(
    cfg: &LiveCommitConfig,
    emitter: &mut Emitter,
    st: &mut LiveCommitState,
    bias: &mut SessionLexicalMemory,
    clock: &crate::api::StreamClock,
    events: &mut std::collections::VecDeque<TranscriptEvent>,
    inp: FinalizeInput<'_>,
) -> Result<(), StreamError> {
    let s1 = inp.s1;
    let pair = build_final_pair(cfg, emitter, st, bias, clock.now_ns(), inp)?;
    events.extend(pair);
    st.metrics.events_emitted += 2;
    if let Some((hyp, end_samp)) = st.deferred_preview.take() {
        emit_preview_event(emitter, st, clock.now_ns(), events, &hyp, end_samp.max(s1))?;
    }
    Ok(())
}

pub(crate) fn build_final_pair(
    cfg: &LiveCommitConfig,
    emitter: &mut Emitter,
    st: &mut LiveCommitState,
    bias: &mut SessionLexicalMemory,
    done_ns: u64,
    inp: FinalizeInput<'_>,
) -> Result<[TranscriptEvent; 2], StreamError> {
    let active_imm = if st.seg_open {
        emitter.immutable().to_owned()
    } else {
        String::new()
    };
    let suffix = reconcile_immutable_suffix(&active_imm, inp.raw_committed);
    let committed = format!("{active_imm}{suffix}");
    let proc = ProcessingEvidence {
        accepted_through_sample: st.metrics.accepted_samples.max(inp.s1),
        required_processed_through_sample: inp.s1,
        flushed: true,
    };
    let final_ev = emitter.finalize(done_ns, &suffix, inp.s1, proc)?;
    st.seg_open = false;
    let revised = committed.trim() != inp.provisional.trim();
    let commit_wall_ms = inp.commit_timing.wall_ms();
    let commit_cpu_ms = inp.commit_timing.cpu_ms();
    let e2f_ms = inp.speech_end_ns.map_or(commit_wall_ms, |t0| {
        (done_ns.saturating_sub(t0) as f64 / 1e6).max(commit_wall_ms)
    });
    st.metrics.refiner_attempts += 1;
    *if revised {
        &mut st.metrics.refiner_accepts
    } else {
        &mut st.metrics.refiner_rejects
    } += 1;
    st.metrics.refiner_charged_cpu_s += inp.commit_timing.cpu_s;
    bias.observe_committed_segment(&committed, None, inp.fallback_reason == "none");
    let detail = json!({
        "trigger": inp.trigger, "start_sample": inp.s0, "end_sample": inp.s1,
        "hangover_ms": cfg.hangover_ms, "preroll_ms": cfg.preroll_ms,
        "deadline_ms": cfg.deadline_ms, "fallback_reason": inp.fallback_reason,
        "commit_wall_ms": (commit_wall_ms * 1000.0).round() / 1000.0,
        "commit_wall_s": (inp.commit_timing.wall_s * 1e6).round() / 1e6,
        "commit_cpu_ms": (commit_cpu_ms * 1000.0).round() / 1000.0,
        "commit_cpu_s": (inp.commit_timing.cpu_s * 1e6).round() / 1e6,
        "stale_worker_cpu_s": (st.stale_worker_cpu_s * 1e6).round() / 1e6,
        "stale_worker_jobs": st.stale_worker_jobs,
        "endpoint_to_final_ms": (e2f_ms * 1000.0).round() / 1000.0,
        "revised": revised, "pcm_truncated": inp.pcm_truncated,
        "session_bias_active_words": bias.active_words(),
    })
    .to_string();
    let status_ev = emitter.status(done_ns, "commit_pass", &detail);
    Ok([final_ev, status_ev])
}
