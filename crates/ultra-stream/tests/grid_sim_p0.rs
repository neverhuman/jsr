//! Regression tests for `grid_sim` P0 fixes:
//! 1. Persistent decode errors terminate `simulate_grid` immediately instead of spinning forever.
//! 2. Rotated slot banks (`utt_offset > 0`) with heterogeneous utterance lengths do not double-apply
//!    `utt_offset` in `utt_samples(slot_utt(slot))`.

use ultra_stream::grid_sim::{simulate_grid, DecodeMode, SimClock, SimParams, SimSlotBank};
use ultra_stream::StreamError;

struct FakeClock(f64);

impl SimClock for FakeClock {
    fn now_s(&self) -> f64 {
        self.0
    }
    fn sleep_s(&mut self, dt_s: f64) {
        self.0 += dt_s;
    }
    fn cpu_s(&self) -> Result<f64, StreamError> {
        Ok(0.01)
    }
}

struct FailingDecodeBank {
    slot_open: bool,
    cursor: usize,
    err: Option<StreamError>,
    decode_calls: usize,
}

impl SimSlotBank for FailingDecodeBank {
    fn num_utterances(&self) -> usize {
        1
    }
    fn utt_samples(&self, _utt_pos: usize) -> usize {
        3200
    }
    fn open_slot(&mut self, _slot: usize, _utt_pos: usize) -> Result<(), StreamError> {
        self.slot_open = true;
        self.cursor = 0;
        Ok(())
    }
    fn has_slot(&self, _slot: usize) -> bool {
        self.slot_open
    }
    fn slot_utt(&self, _slot: usize) -> usize {
        0
    }
    fn slot_cursor(&self, _slot: usize) -> usize {
        self.cursor
    }
    fn slot_finished_input(&self, _slot: usize) -> bool {
        self.cursor >= 3200
    }
    fn accept_packet(&mut self, _slot: usize, max_samples: usize) -> usize {
        let take = max_samples.min(3200 - self.cursor);
        self.cursor += take;
        take
    }
    fn finish_slot_input(&mut self, _slot: usize) {
        self.cursor = 3200;
    }
    fn slot_is_ready(&self, _slot: usize) -> bool {
        true
    }
    fn decode_ready_step(&mut self, _decode: DecodeMode, _max_batch: usize) -> (usize, u64) {
        self.decode_calls += 1;
        // Record fatal error while still reporting ready_n = 1 (previously caused infinite loop).
        self.err = Some(StreamError::Runtime("injected decode failure".into()));
        (1, 0)
    }
    fn take_completed_slot(&mut self, _slot: usize) -> String {
        self.slot_open = false;
        String::new()
    }
    fn take_error(&mut self) -> Option<StreamError> {
        self.err.take()
    }
}

#[test]
fn simulate_grid_aborts_immediately_on_decode_error() {
    let mut bank = FailingDecodeBank {
        slot_open: false,
        cursor: 0,
        err: None,
        decode_calls: 0,
    };
    let mut clock = FakeClock(0.0);
    let err = simulate_grid(
        &mut bank,
        &SimParams {
            streams: 1,
            decode: DecodeMode::Batch,
            max_batch: 4,
            packet_ms: 20,
            grid_ms: 160,
            coalesce_ms: 0,
            paced: false,
            duration_s: None,
        },
        &mut clock,
    )
    .expect_err("must propagate injected decode failure");
    assert!(err.to_string().contains("injected decode failure"));
    assert_eq!(bank.decode_calls, 1, "must stop on first failing step");
}

struct RotatedHeterogeneousBank {
    lengths: Vec<usize>,
    utt_offset: usize,
    slot: Option<(usize, usize, bool)>,
}

impl SimSlotBank for RotatedHeterogeneousBank {
    fn num_utterances(&self) -> usize {
        self.lengths.len()
    }
    fn utt_samples(&self, utt_pos: usize) -> usize {
        // Must index directly by the stored dataset id `utt_pos`, NOT add `utt_offset` again.
        self.lengths[utt_pos % self.lengths.len()]
    }
    fn open_slot(&mut self, _slot: usize, logical_pos: usize) -> Result<(), StreamError> {
        let idx = (logical_pos + self.utt_offset) % self.lengths.len();
        self.slot = Some((idx, 0, false));
        Ok(())
    }
    fn has_slot(&self, _slot: usize) -> bool {
        self.slot.is_some()
    }
    fn slot_utt(&self, _slot: usize) -> usize {
        self.slot.map_or(0, |s| s.0)
    }
    fn slot_cursor(&self, _slot: usize) -> usize {
        self.slot.map_or(0, |s| s.1)
    }
    fn slot_finished_input(&self, _slot: usize) -> bool {
        self.slot.is_some_and(|s| s.2)
    }
    fn accept_packet(&mut self, _slot: usize, max_samples: usize) -> usize {
        let Some((idx, cur, fin)) = self.slot.as_mut() else {
            return 0;
        };
        let len = self.lengths[*idx];
        let take = max_samples.min(len.saturating_sub(*cur));
        *cur += take;
        if *cur >= len {
            *fin = true;
        }
        take
    }
    fn finish_slot_input(&mut self, _slot: usize) {
        if let Some((_, _, fin)) = self.slot.as_mut() {
            *fin = true;
        }
    }
    fn slot_is_ready(&self, _slot: usize) -> bool {
        false
    }
    fn decode_ready_step(&mut self, _decode: DecodeMode, _max_batch: usize) -> (usize, u64) {
        (0, 0)
    }
    fn take_completed_slot(&mut self, _slot: usize) -> String {
        let id = self.slot.take().map_or(0, |s| s.0);
        format!("utt-{id}")
    }
}

#[test]
fn rotated_heterogeneous_bank_completes_all_utterances_without_double_offset() {
    let mut bank = RotatedHeterogeneousBank {
        lengths: vec![100, 200, 300],
        utt_offset: 1,
        slot: None,
    };
    let mut clock = FakeClock(0.0);
    let out = simulate_grid(
        &mut bank,
        &SimParams {
            streams: 1,
            decode: DecodeMode::Batch,
            max_batch: 4,
            packet_ms: 20,
            grid_ms: 160,
            coalesce_ms: 0,
            paced: false,
            duration_s: None,
        },
        &mut clock,
    )
    .expect("rotated unpaced simulation succeeds");
    let ids: Vec<usize> = out.completed.iter().map(|(id, _)| *id).collect();
    assert_eq!(ids, vec![1, 2, 0]);
    assert_eq!(out.accepted_samples, 600);
    assert_eq!(out.processed_samples, 600);
    assert!((out.audio_completed_s - 600.0 / 16_000.0).abs() < 1e-9);
}
