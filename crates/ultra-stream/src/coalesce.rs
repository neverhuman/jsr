//! Chunk-coalesced wakeup wrapper over any [`AsrStream`] (H47 live power optimization).
//!
//! Buffers short incoming transport packets (e.g., 10/20/37 ms or arbitrary odd sizes) until a
//! full model chunk (`chunk_samples`, e.g., `4 * 640 = 2_560` samples = 160 ms) is accumulated
//! before waking the wrapped stream and publishing events. Preserves exact original audio-clock
//! `start_sample` timestamps and flushes any partial trailing samples cleanly on gaps, explicit
//! endpoint flushes, and EOF (`finish_input`).

use crate::api::{
    AsrStream, PollResult, PushReport, PushStatus, StreamError, StreamMetrics, TerminalOutcome,
};

/// Wakeup and flush counters recorded by [`ChunkWakeCoalescer`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CoalesceStats {
    pub chunk_samples: usize,
    pub packets_in: u64,
    pub inner_pushes: u64,
    pub gap_flushes: u64,
    pub eof_flushes: u64,
    pub manual_flushes: u64,
    pub buffered_samples: usize,
}

/// [`AsrStream`] adapter that coalesces short packets into `chunk_samples`-aligned wakes.
pub struct ChunkWakeCoalescer {
    inner: Box<dyn AsrStream>,
    chunk_samples: usize,
    buf: Vec<i16>,
    buf_start_sample: Option<u64>,
    expected_sample: u64,
    total_accepted_samples: u64,
    packets_in: u64,
    inner_pushes: u64,
    gap_flushes: u64,
    eof_flushes: u64,
    manual_flushes: u64,
    inner_dirty: bool,
    finished: bool,
    cancelled: bool,
    terminal: Option<TerminalOutcome>,
}

impl ChunkWakeCoalescer {
    /// Wrap `inner`, coalescing contiguous pushes until `chunk_samples` samples accumulate.
    /// `chunk_samples == 0` is normalized to `1` (pass-through).
    pub fn new(inner: Box<dyn AsrStream>, chunk_samples: usize) -> Self {
        let chunk = chunk_samples.max(1);
        Self {
            inner,
            chunk_samples: chunk,
            buf: Vec::with_capacity(chunk),
            buf_start_sample: None,
            expected_sample: 0,
            total_accepted_samples: 0,
            packets_in: 0,
            inner_pushes: 0,
            gap_flushes: 0,
            eof_flushes: 0,
            manual_flushes: 0,
            inner_dirty: false,
            finished: false,
            cancelled: false,
            terminal: None,
        }
    }

    /// Snapshot wakeup reduction and buffering statistics.
    pub fn stats(&self) -> CoalesceStats {
        CoalesceStats {
            chunk_samples: self.chunk_samples,
            packets_in: self.packets_in,
            inner_pushes: self.inner_pushes,
            gap_flushes: self.gap_flushes,
            eof_flushes: self.eof_flushes,
            manual_flushes: self.manual_flushes,
            buffered_samples: self.buf.len(),
        }
    }

    pub fn inner(&self) -> &dyn AsrStream {
        self.inner.as_ref()
    }

    pub fn into_inner(self) -> Box<dyn AsrStream> {
        self.inner
    }

    /// Flush all currently buffered samples to `inner` (e.g., at an endpoint or before a gap).
    /// Returns how many samples were accepted by `inner`.
    pub fn flush_pending(&mut self) -> Result<usize, StreamError> {
        let n = self.flush_prefix(self.buf.len())?;
        if n > 0 {
            self.manual_flushes += 1;
        }
        Ok(n)
    }

    fn flush_prefix(&mut self, len: usize) -> Result<usize, StreamError> {
        let want = len.min(self.buf.len());
        if want == 0 {
            return Ok(0);
        }
        let start = self
            .buf_start_sample
            .unwrap_or_else(|| self.expected_sample.saturating_sub(self.buf.len() as u64));
        let rep = self.inner.push_pcm(start, &self.buf[..want])?;
        self.inner_pushes += 1;
        self.inner_dirty = true;
        let acc = rep.accepted_samples.min(self.buf.len());
        if acc == self.buf.len() {
            self.buf.clear();
            self.buf_start_sample = None;
        } else if acc > 0 {
            self.buf.drain(..acc);
            self.buf_start_sample = Some(start.saturating_add(acc as u64));
        }
        Ok(acc)
    }

    fn flush_all_before_boundary(&mut self) -> Result<bool, StreamError> {
        while !self.buf.is_empty() {
            let acc = self.flush_prefix(self.buf.len())?;
            if acc == 0 {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

impl AsrStream for ChunkWakeCoalescer {
    fn push_pcm(&mut self, start_sample: u64, pcm: &[i16]) -> Result<PushReport, StreamError> {
        if self.finished || self.cancelled || self.terminal.is_some() {
            return Err(ultra_core::CoreError::Lifecycle(
                "push after finish_input or terminal state",
            )
            .into());
        }
        if start_sample < self.expected_sample {
            return Err(StreamError::Protocol(format!(
                "overlapping/duplicate start_sample {start_sample} < expected {}",
                self.expected_sample
            )));
        }

        // Input discontinuity: flush any buffered pre-gap audio first with its original
        // audio-clock start_sample, then deliver the post-gap packet immediately so gap
        // events fire on time without mixing pre- and post-gap audio.
        if start_sample > self.expected_sample {
            let had_buffered = !self.buf.is_empty();
            if !self.flush_all_before_boundary()? {
                return Ok(PushReport {
                    start_sample,
                    accepted_samples: 0,
                    status: PushStatus::Backpressure {
                        accepted_samples: 0,
                    },
                });
            }
            if had_buffered {
                self.gap_flushes += 1;
            }
            self.expected_sample = start_sample;
            let rep = self.inner.push_pcm(start_sample, pcm)?;
            self.packets_in += 1;
            self.inner_pushes += 1;
            self.inner_dirty = true;
            self.expected_sample = start_sample.saturating_add(rep.accepted_samples as u64);
            self.total_accepted_samples = self
                .total_accepted_samples
                .saturating_add(rep.accepted_samples as u64);
            return Ok(rep);
        }

        if pcm.is_empty() {
            return Ok(PushReport {
                start_sample,
                accepted_samples: 0,
                status: PushStatus::AcceptedAll,
            });
        }

        if self.buf.is_empty() {
            self.buf_start_sample = Some(start_sample);
        }
        self.buf.extend_from_slice(pcm);
        self.expected_sample = start_sample.saturating_add(pcm.len() as u64);
        self.total_accepted_samples = self.total_accepted_samples.saturating_add(pcm.len() as u64);
        self.packets_in += 1;

        if self.buf.len() >= self.chunk_samples {
            let flush_len = (self.buf.len() / self.chunk_samples) * self.chunk_samples;
            let _ = self.flush_prefix(flush_len)?;
        }

        Ok(PushReport {
            start_sample,
            accepted_samples: pcm.len(),
            status: PushStatus::AcceptedAll,
        })
    }

    fn poll_event(&mut self) -> Result<PollResult, StreamError> {
        if !self.inner_dirty {
            if let Some(term) = &self.terminal {
                return Ok(PollResult::Ended(term.clone()));
            }
            return Ok(PollResult::Pending);
        }
        let res = self.inner.poll_event()?;
        match &res {
            PollResult::Pending => self.inner_dirty = false,
            PollResult::Ended(term) => {
                self.terminal = Some(term.clone());
                self.inner_dirty = false;
            }
            PollResult::Event(_) => {}
        }
        Ok(res)
    }

    fn finish_input(&mut self) -> Result<(), StreamError> {
        if self.cancelled {
            return Err(ultra_core::CoreError::Lifecycle(
                "finish_input on a cancelled or failed stream",
            )
            .into());
        }
        if !self.finished {
            let had_buffered = !self.buf.is_empty();
            if !self.flush_all_before_boundary()? {
                return Err(StreamError::ResourceLimit(
                    "inner stream backpressured while flushing trailing EOF samples".into(),
                ));
            }
            if had_buffered {
                self.eof_flushes += 1;
            }
            self.inner.finish_input()?;
            self.inner_dirty = true;
            self.finished = true;
        }
        Ok(())
    }

    fn cancel(&mut self) -> Result<(), StreamError> {
        if !self.finished && !self.cancelled {
            self.buf.clear();
            self.buf_start_sample = None;
            self.cancelled = true;
            self.inner_dirty = true;
            self.inner.cancel()?;
        }
        Ok(())
    }

    fn metrics(&self) -> StreamMetrics {
        let mut m = self.inner.metrics();
        m.accepted_samples = self.total_accepted_samples.max(m.accepted_samples);
        m.retained_bytes = m
            .retained_bytes
            .saturating_add((self.buf.capacity() * std::mem::size_of::<i16>()) as u64);
        m
    }
}
