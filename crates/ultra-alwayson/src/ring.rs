//! Fixed-capacity circular PCM16 ring buffer (`3.0 s` = `48_000` samples = `96 KB` default)
//! with exact sample conservation counters (`samples_in`, `samples_replayed`, `samples_dropped`).

use crate::AlwaysOnError;

/// Canonical capture sample rate (`16 kHz` mono PCM16).
pub const SAMPLE_RATE_HZ: usize = 16_000;

/// Default circular ring capacity: `3.0 s` (`48_000` `i16` samples = `96_000` bytes).
pub const DEFAULT_RING_SAMPLES: usize = 3 * SAMPLE_RATE_HZ;

/// Exact sample accounting counters maintained by [`PcmRing`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SampleCounters {
    /// Total PCM16 samples ever pushed into the ring.
    pub samples_in: u64,
    /// Total PCM16 samples copied out of the ring for wake/segment replay.
    pub samples_replayed: u64,
    /// Total PCM16 samples evicted by circular wrap-around.
    pub samples_dropped: u64,
}

/// Fixed-capacity circular PCM16 ring buffer.
///
/// Conservation invariant (holds after every operation):
/// - `counters.samples_in == counters.samples_dropped + len() as u64`
/// - `base() == counters.samples_dropped`
/// - `end() == counters.samples_in`
#[derive(Debug, Clone)]
pub struct PcmRing {
    buf: Vec<i16>,
    head: usize,
    len: usize,
    counters: SampleCounters,
}

impl PcmRing {
    /// Allocate a fixed-capacity ring holding `capacity_samples` `i16` samples.
    pub fn new(capacity_samples: usize) -> Result<Self, AlwaysOnError> {
        if capacity_samples == 0 {
            return Err(AlwaysOnError::InvalidConfig(
                "ring capacity must be > 0 samples".into(),
            ));
        }
        Ok(Self {
            buf: vec![0i16; capacity_samples],
            head: 0,
            len: 0,
            counters: SampleCounters::default(),
        })
    }

    /// Allocate the default `3.0 s` (`48_000` samples = `96 KB`) circular PCM16 ring.
    pub fn default_3s() -> Self {
        Self {
            buf: vec![0i16; DEFAULT_RING_SAMPLES],
            head: 0,
            len: 0,
            counters: SampleCounters::default(),
        }
    }

    /// Fixed capacity in samples.
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Number of live samples currently retained in the ring (`<= capacity()`).
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if no samples have been pushed yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Absolute sample index of the oldest retained sample (`== counters.samples_dropped`).
    pub fn base(&self) -> u64 {
        self.counters.samples_in - self.len as u64
    }

    /// Exclusive absolute sample index of the newest retained sample (`== counters.samples_in`).
    pub fn end(&self) -> u64 {
        self.counters.samples_in
    }

    /// Snapshot of exact sample conservation counters.
    pub fn counters(&self) -> SampleCounters {
        self.counters
    }

    /// Verify exact sample conservation (`samples_in == samples_dropped + len`).
    pub fn check_conservation(&self) -> bool {
        self.counters.samples_in == self.counters.samples_dropped + self.len as u64
            && self.base() == self.counters.samples_dropped
            && self.end() == self.counters.samples_in
    }

    /// Push an arbitrary-length slice of PCM16 samples with zero heap allocations.
    pub fn push(&mut self, samples: &[i16]) {
        if samples.is_empty() {
            return;
        }
        let cap = self.buf.len();
        let incoming = samples.len();
        if incoming >= cap {
            let dropped = self.len as u64 + (incoming - cap) as u64;
            let tail = &samples[incoming - cap..];
            self.buf.copy_from_slice(tail);
            self.head = 0;
            self.len = cap;
            self.counters.samples_in += incoming as u64;
            self.counters.samples_dropped += dropped;
            return;
        }

        let new_len = self.len + incoming;
        if new_len > cap {
            self.counters.samples_dropped += (new_len - cap) as u64;
        }
        let first = (cap - self.head).min(incoming);
        self.buf[self.head..self.head + first].copy_from_slice(&samples[..first]);
        let rem = incoming - first;
        if rem > 0 {
            self.buf[..rem].copy_from_slice(&samples[first..]);
        }
        self.head = (self.head + incoming) % cap;
        self.len = new_len.min(cap);
        self.counters.samples_in += incoming as u64;
    }

    /// Drop all buffered samples and advance the ring's absolute timeline to `new_cursor`
    /// (`>= end()`) after a backend fault so post-recovery segments keep exact capture positions
    /// and never replay across a continuity gap. Preserves `samples_in == samples_dropped + len`.
    pub fn reset_at_cursor(&mut self, new_cursor: u64) {
        let advance = new_cursor.saturating_sub(self.counters.samples_in);
        self.counters.samples_in += advance;
        self.counters.samples_dropped += self.len as u64 + advance;
        self.len = 0;
        self.head = 0;
    }

    /// Copy live samples `[start..end)` into `out` (clearing `out` first) and charge them to
    /// `counters.samples_replayed`. Refuses overwritten (`start < base`) or future (`end > end()`)
    /// indices.
    pub fn copy_range_into(
        &mut self,
        start: u64,
        end: u64,
        out: &mut Vec<i16>,
    ) -> Result<usize, AlwaysOnError> {
        let count = self.copy_range_readonly(start, end, out)?;
        self.counters.samples_replayed += count as u64;
        Ok(count)
    }

    /// Read-only copy of live samples `[start..end)` into `out` without mutating replay counters.
    pub fn copy_range_readonly(
        &self,
        start: u64,
        end: u64,
        out: &mut Vec<i16>,
    ) -> Result<usize, AlwaysOnError> {
        let base = self.base();
        let ring_end = self.end();
        if start > end || start < base || end > ring_end {
            return Err(AlwaysOnError::RingOutOfRange {
                start,
                end,
                base,
                ring_end,
            });
        }
        out.clear();
        let count = (end - start) as usize;
        if count == 0 {
            return Ok(0);
        }
        let cap = self.buf.len();
        let oldest = (self.head + cap - self.len) % cap;
        let offset = (start - base) as usize;
        let phys_start = (oldest + offset) % cap;
        let first = (cap - phys_start).min(count);
        out.reserve(count);
        out.extend_from_slice(&self.buf[phys_start..phys_start + first]);
        let rem = count - first;
        if rem > 0 {
            out.extend_from_slice(&self.buf[..rem]);
        }
        Ok(count)
    }
}
