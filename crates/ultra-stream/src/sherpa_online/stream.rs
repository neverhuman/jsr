//! Causal `AsrStream` adapter for Online Streaming Zipformer with monotonic `StreamClock`
//! observation timestamps and decoded-chunk-gated `processed_samples` accounting (WP4).

use std::collections::VecDeque;
use std::sync::Arc;

use ultra_core::events::{Emitter, ProcessingEvidence, TranscriptEvent};

use crate::api::{
    AsrStream, OutputPolicy, PollResult, PushReport, PushStatus, StreamClock, StreamConfig,
    StreamError, StreamMetrics, TerminalOutcome,
};
use crate::sherpa_online::{RawOnlineStream, SherpaZipformerBackend};

/// Zipformer online streaming stride (`320 ms @ 16 kHz = 5,120` samples).
pub(crate) const ZIPFORMER_CHUNK_SAMPLES: u64 = 5_120;

/// Minimal operations trait so WP4 clock/watermark/error invariants are unit-tested without FFI.
pub(crate) trait OnlineStreamDriver: Send {
    fn accept_i16(&mut self, pcm: &[i16]);
    fn finish_with_tail_pad(&mut self, tail_pad_samples: usize);
    fn is_ready(&self) -> bool;
    fn decode_one(&mut self);
    fn try_text(&self) -> Result<String, StreamError>;
}

pub(crate) struct RealZipformerDriver {
    pub(crate) engine: Arc<SherpaZipformerBackend>,
    pub(crate) raw: RawOnlineStream,
    pub(crate) scratch: Vec<f32>,
}

impl OnlineStreamDriver for RealZipformerDriver {
    fn accept_i16(&mut self, pcm: &[i16]) {
        self.raw.accept_i16(pcm, &mut self.scratch);
    }

    fn finish_with_tail_pad(&mut self, tail_pad_samples: usize) {
        let zeros = vec![0.0f32; tail_pad_samples];
        self.raw.accept_f32(&zeros);
        self.raw.input_finished();
    }

    fn is_ready(&self) -> bool {
        self.raw.is_ready()
    }

    fn decode_one(&mut self) {
        self.engine.decode_one(&self.raw);
    }

    fn try_text(&self) -> Result<String, StreamError> {
        self.raw.try_text()
    }
}

pub(crate) struct ZipformerAsrStream<D: OnlineStreamDriver> {
    driver: D,
    clock: StreamClock,
    tail_pad_samples: usize,
    emitter: Emitter,
    cfg: StreamConfig,
    accepted_samples: u64,
    processed_samples: u64,
    wakeups: u64,
    events_emitted: u64,
    last_hyp: String,
    finished: bool,
    cancelled: bool,
    events: VecDeque<TranscriptEvent>,
}

impl<D: OnlineStreamDriver> ZipformerAsrStream<D> {
    pub(crate) fn new(
        driver: D,
        clock: StreamClock,
        tail_pad_samples: usize,
        model_id: &str,
        cfg: StreamConfig,
    ) -> Self {
        Self {
            driver,
            clock,
            tail_pad_samples,
            emitter: Emitter::new(&cfg.stream_id, model_id),
            cfg,
            accepted_samples: 0,
            processed_samples: 0,
            wakeups: 0,
            events_emitted: 0,
            last_hyp: String::new(),
            finished: false,
            cancelled: false,
            events: VecDeque::new(),
        }
    }

    fn record_ev(&mut self, ev: TranscriptEvent) {
        self.events_emitted += 1;
        self.events.push_back(ev);
    }

    fn decode_ready(&mut self) -> Result<(), StreamError> {
        let mut decoded_chunks = 0u64;
        while self.driver.is_ready() {
            self.driver.decode_one();
            self.wakeups += 1;
            decoded_chunks += 1;
        }
        if decoded_chunks == 0 {
            return Ok(());
        }
        self.processed_samples = (self.processed_samples
            + decoded_chunks * ZIPFORMER_CHUNK_SAMPLES)
            .min(self.accepted_samples);

        let hyp = self.driver.try_text()?.trim().to_owned();
        if hyp != self.last_hyp {
            self.last_hyp.clone_from(&hyp);
            let ts = self.clock.now_ns();
            let owned_to = self.processed_samples;
            match self.cfg.policy {
                OutputPolicy::EditableDictation => {
                    let ev = self.emitter.preview(ts, &hyp, owned_to)?;
                    self.record_ev(ev);
                }
                OutputPolicy::Commit => {
                    let imm_len = self.emitter.immutable().len();
                    if hyp.starts_with(self.emitter.immutable()) && hyp.len() > imm_len {
                        let tail = &hyp[imm_len..];
                        let ev = match tail.rfind(' ').filter(|&sp| sp > 0) {
                            Some(sp) => self.emitter.commit(ts, tail, sp, owned_to)?,
                            None => self.emitter.preview(ts, tail, owned_to)?,
                        };
                        self.record_ev(ev);
                    }
                }
            }
        }
        Ok(())
    }
}

impl<D: OnlineStreamDriver> AsrStream for ZipformerAsrStream<D> {
    fn push_pcm(&mut self, start_sample: u64, pcm: &[i16]) -> Result<PushReport, StreamError> {
        if self.finished || self.cancelled {
            return Err(ultra_core::CoreError::Lifecycle(
                "push after finish_input or terminal state",
            )
            .into());
        }
        self.driver.accept_i16(pcm);
        self.accepted_samples += pcm.len() as u64;
        self.decode_ready()?;
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
            self.driver.finish_with_tail_pad(self.tail_pad_samples);
            while self.driver.is_ready() {
                self.driver.decode_one();
                self.wakeups += 1;
            }
            let hyp = self.driver.try_text()?.trim().to_owned();
            let imm_len = self.emitter.immutable().len();
            let suffix = if imm_len == 0 {
                hyp.as_str()
            } else if hyp.len() > imm_len && hyp.is_char_boundary(imm_len) {
                &hyp[imm_len..]
            } else {
                ""
            };
            let ts = self.clock.now_ns();
            let ev = self.emitter.finalize(
                ts,
                suffix,
                self.accepted_samples,
                ProcessingEvidence {
                    accepted_through_sample: self.accepted_samples,
                    required_processed_through_sample: self.accepted_samples,
                    flushed: true,
                },
            )?;
            self.finished = true;
            self.processed_samples = self.accepted_samples;
            self.record_ev(ev);
        }
        Ok(())
    }

    fn cancel(&mut self) -> Result<(), StreamError> {
        if !self.finished {
            self.cancelled = true;
            self.events.clear();
        }
        Ok(())
    }

    fn metrics(&self) -> StreamMetrics {
        let buffered = self.accepted_samples.saturating_sub(self.processed_samples);
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
            retained_bytes: buffered * 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    use super::*;

    struct MockOnline {
        pending: u64,
        flushed: bool,
        text: String,
        fail_text: Arc<AtomicBool>,
        now_ns: Arc<AtomicU64>,
    }

    impl OnlineStreamDriver for MockOnline {
        fn accept_i16(&mut self, pcm: &[i16]) {
            self.pending += pcm.len() as u64;
        }
        fn finish_with_tail_pad(&mut self, _tail: usize) {
            self.flushed = true;
        }
        fn is_ready(&self) -> bool {
            self.pending >= ZIPFORMER_CHUNK_SAMPLES || (self.flushed && self.pending > 0)
        }
        fn decode_one(&mut self) {
            self.pending = self.pending.saturating_sub(ZIPFORMER_CHUNK_SAMPLES);
            self.now_ns.fetch_add(45_000_000, Ordering::SeqCst);
            self.text = "alpha beta".into();
        }
        fn try_text(&self) -> Result<String, StreamError> {
            if self.fail_text.swap(false, Ordering::SeqCst) {
                return Err(StreamError::Runtime("null native result".into()));
            }
            Ok(self.text.clone())
        }
    }

    #[test]
    fn wp4_subchunk_leaves_processed_zero_and_stamps_monotonic_post_decode_time() {
        let now_ns = Arc::new(AtomicU64::new(320_000_000));
        let fail_text = Arc::new(AtomicBool::new(false));
        let clk_ref = Arc::clone(&now_ns);
        let clock = StreamClock::custom(move || clk_ref.load(Ordering::SeqCst));
        let mut s = ZipformerAsrStream::new(
            MockOnline {
                pending: 0,
                flushed: false,
                text: String::new(),
                fail_text: Arc::clone(&fail_text),
                now_ns: Arc::clone(&now_ns),
            },
            clock,
            0,
            "m0",
            StreamConfig::default(),
        );

        // 20 ms push (320 samples < 5,120 chunk): no decode ran -> processed_samples stays 0!
        s.push_pcm(0, &[1i16; 320]).expect("subchunk");
        assert_eq!(s.metrics().accepted_samples, 320);
        assert_eq!(s.metrics().processed_samples, 0);
        assert_eq!(s.metrics().wakeups, 0);

        // Push 5,000 more samples (total 5,320 -> 1 chunk of 5,120 decoded, 200 remain un-decoded).
        s.push_pcm(320, &[1i16; 5_000]).expect("chunk");
        assert_eq!(s.metrics().accepted_samples, 5_320);
        assert_eq!(s.metrics().processed_samples, 5_120);
        assert_eq!(s.metrics().wakeups, 1);
        let PollResult::Event(ev) = s.poll_event().expect("preview") else {
            panic!("expected Preview");
        };
        assert_eq!(ev.observed_monotonic_ns, 365_000_000);

        // Error in try_text during finish_input does NOT set finished = true.
        fail_text.store(true, Ordering::SeqCst);
        assert!(s.finish_input().is_err());
        assert_eq!(s.poll_event().expect("still open"), PollResult::Pending);
    }
}
