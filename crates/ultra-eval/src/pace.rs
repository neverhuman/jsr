//! Paced PCM replay (v1.0-B §12.2 paced workloads, §38.1 transport 10/20/37 ms).
//!
//! The harness owns pacing. A packet covering samples `[s, s + n)` of a 16 kHz int16
//! stream becomes available when its last sample would have been captured, i.e. at
//! `origin + (s + n) / 16000 * pace` seconds. `pace = 1.0` is real time, `0.0` is
//! unpaced (every packet immediately), `0.5` replays twice as fast. Release times are
//! computed from the fixed origin and slept *until* (never "sleep for a packet
//! duration"), so per-packet scheduling jitter never accumulates into drift. The last
//! packet may be short.

use std::time::{Duration, Instant};

use crate::EvalError;

/// Sample rate of the audio contract.
pub const SAMPLE_RATE_HZ: u64 = 16_000;

/// Samples per packet of `packet_ms` milliseconds at 16 kHz.
pub fn packet_samples(packet_ms: u32) -> usize {
    (u64::from(packet_ms) * SAMPLE_RATE_HZ / 1000) as usize
}

/// Offset from the origin at which audio up to `end_sample` is available.
pub fn due_offset(end_sample: u64, pace: f64) -> Duration {
    Duration::from_secs_f64(end_sample as f64 / SAMPLE_RATE_HZ as f64 * pace)
}

/// One released packet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Packet<'a> {
    pub start_sample: u64,
    pub pcm: &'a [i16],
    /// When the packet was due (origin + capture end).
    pub due: Instant,
    /// When the iterator actually released it (>= `due` when paced).
    pub released_at: Instant,
}

/// Iterator over packets of a PCM buffer, paced against a fixed origin.
#[derive(Debug, Clone)]
pub struct PacedReplay<'a> {
    pcm: &'a [i16],
    packet: usize,
    /// Optional model chunk stride in samples (`0` = wake per packet, `> packet` = coalesce
    /// incoming transport packets until the next chunk boundary is complete before waking).
    chunk_samples: usize,
    pace: f64,
    origin: Instant,
    pos: usize,
}

impl<'a> PacedReplay<'a> {
    /// `packet_ms > 0`; `pace` finite and >= 0. The origin is "now".
    pub fn new(pcm: &'a [i16], packet_ms: u32, pace: f64) -> Result<Self, EvalError> {
        Self::with_origin(pcm, packet_ms, pace, Instant::now())
    }

    /// As [`PacedReplay::new`] with an explicit origin (stream start on the monotonic
    /// clock), so several streams can be staggered against one clock.
    pub fn with_origin(
        pcm: &'a [i16],
        packet_ms: u32,
        pace: f64,
        origin: Instant,
    ) -> Result<Self, EvalError> {
        Self::with_origin_and_chunk_coalescing(pcm, packet_ms, 0, pace, origin)
    }

    /// Paced replay that coalesces incoming `packet_ms` transport packets up to the earliest
    /// transport boundary reaching each `chunk_samples` model chunk boundary, waking once per
    /// chunk instead of once per short packet while preserving exact audio-clock offsets and
    /// trailing EOF flush.
    pub fn with_chunk_coalescing(
        pcm: &'a [i16],
        packet_ms: u32,
        chunk_samples: usize,
        pace: f64,
    ) -> Result<Self, EvalError> {
        Self::with_origin_and_chunk_coalescing(pcm, packet_ms, chunk_samples, pace, Instant::now())
    }

    /// As [`PacedReplay::with_chunk_coalescing`] with an explicit monotonic origin.
    pub fn with_origin_and_chunk_coalescing(
        pcm: &'a [i16],
        packet_ms: u32,
        chunk_samples: usize,
        pace: f64,
        origin: Instant,
    ) -> Result<Self, EvalError> {
        if packet_ms == 0 {
            return Err(EvalError::Input("packet_ms must be > 0".into()));
        }
        if !(pace.is_finite() && pace >= 0.0) {
            return Err(EvalError::Input(format!(
                "pace {pace} must be finite and >= 0"
            )));
        }
        Ok(PacedReplay {
            pcm,
            packet: packet_samples(packet_ms),
            chunk_samples,
            pace,
            origin,
            pos: 0,
        })
    }

    pub fn origin(&self) -> Instant {
        self.origin
    }
}

fn sleep_until(deadline: Instant) {
    loop {
        let now = Instant::now();
        if now >= deadline {
            return;
        }
        std::thread::sleep(deadline - now);
    }
}

impl<'a> Iterator for PacedReplay<'a> {
    type Item = Packet<'a>;

    fn next(&mut self) -> Option<Packet<'a>> {
        if self.pos >= self.pcm.len() {
            return None;
        }
        let start = self.pos;
        let next_end = if self.chunk_samples > self.packet {
            let chunk_target = ((start / self.chunk_samples) + 1) * self.chunk_samples;
            chunk_target.div_ceil(self.packet) * self.packet
        } else {
            start + self.packet
        };
        let end = next_end.min(self.pcm.len());
        self.pos = end;
        let due = self.origin + due_offset(end as u64, self.pace);
        if self.pace > 0.0 {
            sleep_until(due);
        }
        Some(Packet {
            start_sample: start as u64,
            pcm: &self.pcm[start..end],
            due,
            released_at: Instant::now(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize) -> Vec<i16> {
        (0..n).map(|i| (i % 65_536) as u16 as i16).collect()
    }

    #[test]
    fn packet_sizes_give_identical_timelines() {
        let pcm = ramp(16_000 + 123); // short final packet for every size
        let mut timelines = Vec::new();
        for ms in [10, 20, 37] {
            let mut cat = Vec::new();
            let mut expect_start = 0u64;
            let mut n = 0;
            for p in PacedReplay::new(&pcm, ms, 0.0).unwrap() {
                assert_eq!(p.start_sample, expect_start);
                assert!(p.pcm.len() <= packet_samples(ms));
                expect_start += p.pcm.len() as u64;
                cat.extend_from_slice(p.pcm);
                n += 1;
            }
            assert_eq!(n, pcm.len().div_ceil(packet_samples(ms)));
            timelines.push(cat);
        }
        assert_eq!(timelines[0], pcm);
        assert_eq!(timelines[0], timelines[1]);
        assert_eq!(timelines[1], timelines[2]);
        assert_eq!(packet_samples(37), 592);
    }

    #[test]
    fn empty_and_invalid() {
        assert_eq!(PacedReplay::new(&[], 10, 1.0).unwrap().count(), 0);
        assert!(PacedReplay::new(&[0], 0, 1.0).is_err());
        assert!(PacedReplay::new(&[0], 10, -1.0).is_err());
        assert!(PacedReplay::new(&[0], 10, f64::NAN).is_err());
    }

    #[test]
    fn due_times_follow_capture_end() {
        assert_eq!(due_offset(16_000, 1.0), Duration::from_secs(1));
        assert_eq!(due_offset(160, 1.0), Duration::from_millis(10));
        assert_eq!(due_offset(16_000, 0.0), Duration::ZERO);
    }

    #[test]
    fn pacing_is_faithful_without_drift() {
        // 2 s of audio at pace 0.05 -> 100 ms, in 200 packets of 10 ms (0.5 ms apart).
        let pcm = ramp(32_000);
        let origin = Instant::now();
        let mut worst_late = Duration::ZERO;
        let mut last = None;
        for p in PacedReplay::with_origin(&pcm, 10, 0.05, origin).unwrap() {
            assert!(p.released_at >= p.due, "released before due");
            let expected = origin + due_offset(p.start_sample + p.pcm.len() as u64, 0.05);
            assert_eq!(p.due, expected);
            worst_late = worst_late.max(p.released_at - p.due);
            last = Some(p);
        }
        let last = last.unwrap();
        let total = last.released_at - origin;
        assert!(total >= Duration::from_millis(100), "{total:?}");
        // Sleep-until: the final packet is late only by its own jitter, not by the
        // sum of 200 per-packet oversleeps.
        assert!(
            last.released_at - last.due < Duration::from_millis(20),
            "{total:?}"
        );
        assert!(worst_late < Duration::from_millis(50), "{worst_late:?}");
    }

    #[test]
    fn chunk_coalescing_reduces_wakeups_and_preserves_audio_clock() {
        // 10 s + 137 trailing samples at 16 kHz, 20 ms packets, chunk 4 = 2560 samples (160 ms).
        let pcm = ramp(160_000 + 137);
        let per_pkt: Vec<_> = PacedReplay::new(&pcm, 20, 0.0).unwrap().collect();
        let per_chunk: Vec<_> = PacedReplay::with_chunk_coalescing(&pcm, 20, 2_560, 0.0)
            .unwrap()
            .collect();

        // 501 per-packet wakes -> 63 per-chunk wakes (>= 7.9x reduction, 50/s -> 6.25/s).
        assert_eq!(per_pkt.len(), 501);
        assert_eq!(per_chunk.len(), 63);
        assert!(per_pkt.len() >= per_chunk.len() * 7);

        let mut reassembled = Vec::with_capacity(pcm.len());
        let mut expected_start = 0u64;
        for p in &per_chunk {
            assert_eq!(p.start_sample, expected_start);
            expected_start += p.pcm.len() as u64;
            reassembled.extend_from_slice(p.pcm);
        }
        assert_eq!(reassembled, pcm);
        assert_eq!(per_chunk.last().unwrap().pcm.len(), (160_000 % 2_560) + 137);

        // Odd 37 ms packet size: exact timeline preserved without drift.
        let origin = Instant::now();
        let odd: Vec<_> =
            PacedReplay::with_origin_and_chunk_coalescing(&pcm, 37, 2_560, 0.0, origin)
                .unwrap()
                .collect();
        let mut odd_cat = Vec::with_capacity(pcm.len());
        let mut odd_start = 0u64;
        for p in &odd {
            assert_eq!(p.start_sample, odd_start);
            odd_start += p.pcm.len() as u64;
            odd_cat.extend_from_slice(p.pcm);
        }
        assert_eq!(odd_cat, pcm);
        assert!(odd.len() <= 64);
    }
}
