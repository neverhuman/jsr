//! Deterministic `FakeBackend` and Spec §38.1 / §38.3 failure-probe fixtures (W00).

use std::collections::VecDeque;

use ultra_core::events::{Emitter, ProcessingEvidence, TranscriptEvent};

use crate::api::{
    AsrStream, Backend, BackendCaps, OutputPolicy, PollResult, PushReport, PushStatus,
    StreamConfig, StreamError, StreamMetrics, TerminalOutcome,
};

/// Scheduled word emission on the 16 kHz audio sample clock.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FakeWord {
    pub emit_at_sample: u64,
    pub word: String,
    pub difficult: bool,
}

#[derive(Debug, Clone)]
pub struct FakeBackendConfig {
    /// Fixed logical inference stride in samples (`1_600` = 100 ms @ 16 kHz).
    pub chunk_samples: usize,
    pub words: Vec<FakeWord>,
    /// §38.3 probe 1: deliberately omit `difficult` words to verify `CQ(D)` and `WER` catch it.
    pub omit_difficult_words: bool,
    /// Optional causal acoustic probe: when `Some(thresh)`, each `chunk_samples` window whose
    /// mean `|pcm|` exceeds `thresh` emits `"tone_{chunk_idx}"` derived solely from past/current
    /// accepted samples.
    pub energy_word_threshold: Option<i16>,
}

impl Default for FakeBackendConfig {
    fn default() -> Self {
        Self {
            chunk_samples: 1_600,
            words: Vec::new(),
            omit_difficult_words: false,
            energy_word_threshold: None,
        }
    }
}

/// Deterministic reference-free test backend whose logical schedule depends only on total
/// accepted samples (`chunk_samples` multiples), never on transport packet boundaries.
#[derive(Debug, Clone)]
pub struct FakeBackend {
    cfg: FakeBackendConfig,
}

impl FakeBackend {
    pub fn new(cfg: FakeBackendConfig) -> Self {
        Self { cfg }
    }
}

impl Backend for FakeBackend {
    fn capabilities(&self) -> BackendCaps {
        BackendCaps {
            name: "deterministic-fake-v1".into(),
            supports_batch_step: true,
            supports_partial_push: true,
            supports_commit_policy: true,
            chunk_samples: self.cfg.chunk_samples,
            right_context_samples: 0,
            cpu_only_verified: true,
        }
    }

    fn create_stream(&self, cfg: &StreamConfig) -> Result<Box<dyn AsrStream>, StreamError> {
        Ok(Box::new(FakeStream {
            emitter: Emitter::new(&cfg.stream_id, &cfg.model_config_id),
            stream_cfg: cfg.clone(),
            backend_cfg: self.cfg.clone(),
            pcm_tail: VecDeque::new(),
            accepted_samples: 0,
            processed_samples: 0,
            chunk_idx: 0,
            next_word_idx: 0,
            committed_words: 0,
            emitted_words: Vec::new(),
            events_emitted: 0,
            wakeups: 0,
            finished: false,
            cancelled: false,
            events: VecDeque::new(),
        }))
    }
}

struct FakeStream {
    emitter: Emitter,
    stream_cfg: StreamConfig,
    backend_cfg: FakeBackendConfig,
    pcm_tail: VecDeque<i16>,
    accepted_samples: u64,
    processed_samples: u64,
    chunk_idx: u64,
    next_word_idx: usize,
    committed_words: usize,
    emitted_words: Vec<String>,
    events_emitted: u64,
    wakeups: u64,
    finished: bool,
    cancelled: bool,
    events: VecDeque<TranscriptEvent>,
}

impl FakeStream {
    fn now_ns(&self) -> u64 {
        (self.processed_samples * 1_000_000_000) / 16_000
    }

    fn record_ev(&mut self, ev: TranscriptEvent) {
        self.events_emitted += 1;
        self.events.push_back(ev);
    }

    fn step_to(&mut self, limit_sample: u64, flush_tail: bool) -> Result<(), StreamError> {
        let stride = self.backend_cfg.chunk_samples.max(1) as u64;
        while self.processed_samples + stride <= limit_sample
            || (flush_tail && self.processed_samples < limit_sample)
        {
            let next_proc = if self.processed_samples + stride <= limit_sample {
                self.processed_samples + stride
            } else {
                limit_sample
            };
            let take_n = (next_proc - self.processed_samples) as usize;
            let mut abs_sum = 0u64;
            let mut popped = 0usize;
            while popped < take_n {
                if let Some(s) = self.pcm_tail.pop_front() {
                    abs_sum += u64::from(s.unsigned_abs());
                }
                popped += 1;
            }
            self.processed_samples = next_proc;
            self.wakeups += 1;
            let mut changed = false;

            if let Some(thresh) = self.backend_cfg.energy_word_threshold {
                let mean_abs = if take_n > 0 {
                    abs_sum / (take_n as u64)
                } else {
                    0
                };
                if mean_abs >= u64::from(thresh.unsigned_abs()) {
                    self.emitted_words.push(format!("tone{}", self.chunk_idx));
                    changed = true;
                }
            }
            self.chunk_idx += 1;

            while let Some(w) = self.backend_cfg.words.get(self.next_word_idx) {
                if w.emit_at_sample <= self.processed_samples {
                    self.next_word_idx += 1;
                    if !(self.backend_cfg.omit_difficult_words && w.difficult) {
                        self.emitted_words.push(w.word.clone());
                        changed = true;
                    }
                } else {
                    break;
                }
            }
            if changed {
                let ts = self.now_ns();
                match self.stream_cfg.policy {
                    OutputPolicy::EditableDictation => {
                        let suffix = self.emitted_words[self.committed_words..].join(" ");
                        let ev = self.emitter.preview(ts, &suffix, self.processed_samples)?;
                        self.record_ev(ev);
                    }
                    OutputPolicy::Commit => {
                        while self.committed_words < self.emitted_words.len() {
                            let prefix_space = if self.emitter.immutable().is_empty() {
                                ""
                            } else {
                                " "
                            };
                            let piece = format!(
                                "{prefix_space}{}",
                                self.emitted_words[self.committed_words]
                            );
                            self.committed_words += 1;
                            let ev = self.emitter.commit(
                                ts,
                                &piece,
                                piece.len(),
                                self.processed_samples,
                            )?;
                            self.record_ev(ev);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl AsrStream for FakeStream {
    fn push_pcm(&mut self, start_sample: u64, pcm: &[i16]) -> Result<PushReport, StreamError> {
        if self.finished || self.cancelled {
            return Err(ultra_core::CoreError::Lifecycle(
                "push after finish_input or terminal state",
            )
            .into());
        }
        if self.backend_cfg.energy_word_threshold.is_some() {
            self.pcm_tail.extend(pcm.iter().copied());
        }
        self.accepted_samples += pcm.len() as u64;
        let target = self.accepted_samples;
        self.step_to(target, false)?;
        Ok(PushReport {
            start_sample,
            accepted_samples: pcm.len(),
            status: PushStatus::AcceptedAll,
        })
    }

    fn poll_event(&mut self) -> Result<PollResult, StreamError> {
        if let Some(ev) = self.events.pop_front() {
            return Ok(PollResult::Event(Box::new(ev)));
        }
        if self.cancelled {
            return Ok(PollResult::Ended(TerminalOutcome::Cancelled));
        }
        if self.finished {
            return Ok(PollResult::Ended(TerminalOutcome::Finished));
        }
        Ok(PollResult::Pending)
    }

    fn finish_input(&mut self) -> Result<(), StreamError> {
        if self.cancelled {
            return Err(ultra_core::CoreError::Lifecycle(
                "finish_input on a cancelled or failed stream",
            )
            .into());
        }
        if !self.finished {
            self.finished = true;
            let target = self.accepted_samples;
            self.step_to(target, true)?;
            let remaining = if self.committed_words < self.emitted_words.len() {
                let prefix_space = if self.emitter.immutable().is_empty() {
                    ""
                } else {
                    " "
                };
                let tail = self.emitted_words[self.committed_words..].join(" ");
                self.committed_words = self.emitted_words.len();
                format!("{prefix_space}{tail}")
            } else {
                String::new()
            };
            let ts = self.now_ns();
            let ev = self.emitter.finalize(
                ts,
                &remaining,
                self.processed_samples,
                ProcessingEvidence {
                    accepted_through_sample: self.accepted_samples,
                    required_processed_through_sample: self.processed_samples,
                    flushed: true,
                },
            )?;
            self.record_ev(ev);
        }
        Ok(())
    }

    fn cancel(&mut self) -> Result<(), StreamError> {
        if !self.finished {
            self.cancelled = true;
            self.events.clear();
            self.pcm_tail.clear();
        }
        Ok(())
    }

    fn metrics(&self) -> StreamMetrics {
        let retained_samples = self.accepted_samples.saturating_sub(self.processed_samples);
        StreamMetrics {
            accepted_samples: self.accepted_samples,
            processed_samples: self.processed_samples,
            wakeups: self.wakeups,
            events_emitted: self.events_emitted,
            cap_hits: 0,
            refiner_attempts: 0,
            refiner_accepts: 0,
            refiner_rejects: 0,
            refiner_charged_cpu_s: 0.0,
            retained_bytes: retained_samples * 2,
        }
    }
}

/// §38.3 probe 6: Simulates cancelling an uninterruptible native C++ inference call.
/// Memory reservation and CPU accounting stay held until the in-flight worker actually exits,
/// and spawning unlimited replacement workers while one is still draining is refused!
#[derive(Debug)]
pub struct UninterruptibleWorkerSim {
    max_workers: usize,
    in_flight_cancelled: usize,
    active_workers: usize,
    reserved_bytes_per_worker: u64,
    cpu_charged_s: f64,
}

impl UninterruptibleWorkerSim {
    pub fn new(max_workers: usize, reserved_bytes_per_worker: u64) -> Self {
        Self {
            max_workers: max_workers.max(1),
            in_flight_cancelled: 0,
            active_workers: 0,
            reserved_bytes_per_worker,
            cpu_charged_s: 0.0,
        }
    }

    pub fn spawn_worker(&mut self) -> Result<(), StreamError> {
        if self.active_workers + self.in_flight_cancelled >= self.max_workers {
            return Err(StreamError::ResourceLimit(
                "cannot spawn replacement worker while cancelled native call still holds slot"
                    .into(),
            ));
        }
        self.active_workers += 1;
        Ok(())
    }

    /// Cancels the logical stream immediately, moving the worker to `in_flight_cancelled`
    /// while keeping its memory reservation active until `reap_finished_cancelled_call`.
    pub fn cancel_active(&mut self) {
        if self.active_workers > 0 {
            self.active_workers -= 1;
            self.in_flight_cancelled += 1;
        }
    }

    pub fn reserved_bytes(&self) -> u64 {
        ((self.active_workers + self.in_flight_cancelled) as u64) * self.reserved_bytes_per_worker
    }

    pub fn cpu_charged_s(&self) -> f64 {
        self.cpu_charged_s
    }

    /// Called only when the underlying native thread/process finally exits.
    pub fn reap_finished_cancelled_call(&mut self, tail_cpu_s: f64) {
        if self.in_flight_cancelled > 0 {
            self.in_flight_cancelled -= 1;
            self.cpu_charged_s += tail_cpu_s.max(0.0);
        }
    }
}
