//! Deterministic fake-clock / fake-recogniser probes for the Rust SP-A grid scheduler
//! (verifying M1 packet-end deadlines, M2 cutoff catch-up + drain, and M4 oldest-unprocessed
//! lateness & no-growth sustainability rules).

use std::cell::Cell;
use std::rc::Rc;

use ultra_stream::{
    simulate_grid, sustains_lateness, DecodeMode, LatenessQuantiles, SimClock, SimParams,
    SimSlotBank, StreamError, DEFAULT_LATENESS_BOUND_MS,
};

#[derive(Clone)]
struct SharedTime {
    t_s: Rc<Cell<f64>>,
    cpu_s: Rc<Cell<f64>>,
    sleep_calls: Rc<Cell<usize>>,
}

impl SharedTime {
    fn new() -> Self {
        Self {
            t_s: Rc::new(Cell::new(0.0)),
            cpu_s: Rc::new(Cell::new(0.0)),
            sleep_calls: Rc::new(Cell::new(0)),
        }
    }
}

impl SimClock for SharedTime {
    fn now_s(&self) -> f64 {
        self.t_s.get()
    }
    fn sleep_s(&mut self, dt_s: f64) {
        let calls = self.sleep_calls.get() + 1;
        assert!(calls <= 100_000, "unbounded sleep spin detected");
        self.sleep_calls.set(calls);
        if dt_s > 0.0 {
            self.t_s.set(self.t_s.get() + dt_s);
        }
    }
    fn cpu_s(&self) -> Result<f64, StreamError> {
        Ok(self.cpu_s.get())
    }
}

struct FakeSlot {
    utt: usize,
    cursor: usize,
    ready_chunks: usize,
    pending_samples: usize,
    finished: bool,
}

struct FakeRecBank {
    utt_lens: Vec<usize>,
    slots: Vec<Option<FakeSlot>>,
    chunk_samples: usize,
    decode_cost_s: f64,
    first_accept_t: Option<f64>,
    clock: SharedTime,
}

impl FakeRecBank {
    fn new(
        utt_secs: &[f64],
        streams: usize,
        chunk_ms: u32,
        decode_cost_s: f64,
        clock: SharedTime,
    ) -> Self {
        Self {
            utt_lens: utt_secs
                .iter()
                .map(|s| (s * 16_000.0).round() as usize)
                .collect(),
            slots: (0..streams).map(|_| None).collect(),
            chunk_samples: (u64::from(chunk_ms) * 16_000 / 1000).max(1) as usize,
            decode_cost_s,
            first_accept_t: None,
            clock,
        }
    }
}

impl SimSlotBank for FakeRecBank {
    fn num_utterances(&self) -> usize {
        self.utt_lens.len()
    }
    fn utt_samples(&self, utt_pos: usize) -> usize {
        self.utt_lens[utt_pos]
    }
    fn open_slot(&mut self, slot: usize, utt_pos: usize) -> Result<(), StreamError> {
        self.slots[slot] = Some(FakeSlot {
            utt: utt_pos,
            cursor: 0,
            ready_chunks: 0,
            pending_samples: 0,
            finished: false,
        });
        Ok(())
    }
    fn has_slot(&self, slot: usize) -> bool {
        self.slots[slot].is_some()
    }
    fn slot_utt(&self, slot: usize) -> usize {
        self.slots[slot].as_ref().map_or(0, |s| s.utt)
    }
    fn slot_cursor(&self, slot: usize) -> usize {
        self.slots[slot].as_ref().map_or(0, |s| s.cursor)
    }
    fn slot_finished_input(&self, slot: usize) -> bool {
        self.slots[slot].as_ref().is_some_and(|s| s.finished)
    }
    fn accept_packet(&mut self, slot: usize, max_samples: usize) -> usize {
        if self.first_accept_t.is_none() {
            self.first_accept_t = Some(self.clock.now_s());
        }
        let len = self.utt_lens[self.slot_utt(slot)];
        let Some(st) = self.slots[slot].as_mut() else {
            return 0;
        };
        if st.cursor >= len {
            return 0;
        }
        let end = (st.cursor + max_samples).min(len);
        let n = end - st.cursor;
        st.cursor = end;
        st.pending_samples += n;
        while st.pending_samples >= self.chunk_samples {
            st.pending_samples -= self.chunk_samples;
            st.ready_chunks += 1;
        }
        if st.cursor >= len {
            st.finished = true;
            if st.pending_samples > 0 {
                st.pending_samples = 0;
                st.ready_chunks += 1;
            }
        }
        n
    }
    fn finish_slot_input(&mut self, slot: usize) {
        if let Some(st) = self.slots[slot].as_mut() {
            if !st.finished {
                st.finished = true;
                if st.pending_samples > 0 {
                    st.pending_samples = 0;
                    st.ready_chunks += 1;
                }
            }
        }
    }
    fn slot_is_ready(&self, slot: usize) -> bool {
        self.slots[slot]
            .as_ref()
            .is_some_and(|s| s.ready_chunks > 0)
    }
    fn decode_ready_step(&mut self, _decode: DecodeMode, _max_batch: usize) -> (usize, u64) {
        let mut ready = 0usize;
        for s in self.slots.iter_mut().flatten() {
            if s.ready_chunks > 0 {
                s.ready_chunks -= 1;
                ready += 1;
            }
        }
        if ready == 0 {
            return (0, 0);
        }
        self.clock
            .t_s
            .set(self.clock.t_s.get() + self.decode_cost_s);
        self.clock
            .cpu_s
            .set(self.clock.cpu_s.get() + self.decode_cost_s);
        (ready, 1)
    }
    fn take_completed_slot(&mut self, slot: usize) -> String {
        let u = self.slots[slot].take().map_or(0, |s| s.utt);
        format!("hyp-{u}")
    }
}

#[test]
fn m1_first_packet_waits_for_capture_deadline_and_grid_records_effective_ms() {
    let mut clk = SharedTime::new();
    let mut bank = FakeRecBank::new(&[1.0], 1, 320, 0.010, clk.clone());
    let out = simulate_grid(
        &mut bank,
        &SimParams {
            streams: 1,
            decode: DecodeMode::Batch,
            max_batch: 64,
            packet_ms: 20,
            grid_ms: 320,
            coalesce_ms: 0,
            paced: true,
            duration_s: None,
        },
        &mut clk,
    )
    .expect("sim");
    assert!(
        (bank.first_accept_t.expect("accepted") - 0.020).abs() < 1e-9,
        "first 20 ms packet must be accepted at 20 ms, never 0 ms"
    );
    assert_eq!(out.effective_grid_ms, 320);
    assert!(out.drained);
    assert!((out.audio_accepted_s - 1.0).abs() < 1e-9);
    assert!(sustains_lateness(
        out.lateness_ms.as_ref().expect("lat"),
        out.drained,
        DEFAULT_LATENESS_BOUND_MS
    ));

    // 37 ms packetization with 320 ms requested grid -> records 333 ms effective grid:
    let mut clk37 = SharedTime::new();
    let mut bank37 = FakeRecBank::new(&[1.0], 1, 320, 0.005, clk37.clone());
    let out37 = simulate_grid(
        &mut bank37,
        &SimParams {
            streams: 1,
            decode: DecodeMode::Batch,
            max_batch: 64,
            packet_ms: 37,
            grid_ms: 320,
            coalesce_ms: 0,
            paced: true,
            duration_s: None,
        },
        &mut clk37,
    )
    .expect("sim 37ms");
    assert_eq!(out37.effective_grid_ms, 333);
}

#[test]
fn m2_and_m4_cutoff_short_windows_and_overloaded_catchup_preserve_deadlines() {
    // Reproduce codex-review's exact three cutoff fixtures:
    // (0.005s, 0.1s) -> 0.000s accepted; (0.030s, 0.1s) -> 0.020s; (1.0s, 0.5s) -> 1.000s!
    for (dur, cost, expected_acc_s) in [(0.005, 0.1, 0.000), (0.030, 0.1, 0.020), (1.0, 0.5, 1.000)]
    {
        let mut clk = SharedTime::new();
        let mut bank = FakeRecBank::new(&[30.0], 1, 20, cost, clk.clone());
        let out = simulate_grid(
            &mut bank,
            &SimParams {
                streams: 1,
                decode: DecodeMode::Batch,
                max_batch: 64,
                packet_ms: 20,
                grid_ms: 20,
                coalesce_ms: 0,
                paced: true,
                duration_s: Some(dur),
            },
            &mut clk,
        )
        .expect("sim cutoff");
        assert!(
            (out.audio_accepted_s - expected_acc_s).abs() < 1e-9,
            "duration={dur} expected {expected_acc_s}s accepted, got {}",
            out.audio_accepted_s
        );
        assert!(out.drained);
    }

    // Overload fixture (100ms decode per 20ms packet over 1s): oldest-unprocessed lateness grows
    // far above bound and MUST NOT report sustained!
    let mut clk_ov = SharedTime::new();
    let mut bank_ov = FakeRecBank::new(&[2.0], 1, 20, 0.100, clk_ov.clone());
    let ov = simulate_grid(
        &mut bank_ov,
        &SimParams {
            streams: 1,
            decode: DecodeMode::Batch,
            max_batch: 64,
            packet_ms: 20,
            grid_ms: 20,
            coalesce_ms: 0,
            paced: true,
            duration_s: Some(1.0),
        },
        &mut clk_ov,
    )
    .expect("sim overload");
    let lat = ov.lateness_ms.expect("lateness recorded");
    assert!(
        lat.p99 > 800.0,
        "overload lateness must not be hidden: {lat:?}"
    );
    assert!(!sustains_lateness(
        &lat,
        ov.drained,
        DEFAULT_LATENESS_BOUND_MS
    ));

    // Monotonically growing backlog below bound (10 ms -> 290 ms for 320 ms bound) is rejected:
    let growing = LatenessQuantiles {
        p50: 150.0,
        p95: 295.0,
        p99: 300.0,
        max: 300.0,
        last: 300.0,
        first_third_mean: 10.0,
        last_third_mean: 290.0,
    };
    assert!(!sustains_lateness(&growing, true, 320.0));

    // Non-finite / negative lateness is rejected:
    let bad = LatenessQuantiles {
        p50: f64::NEG_INFINITY,
        ..growing
    };
    assert!(!sustains_lateness(&bad, true, 800.0));
}
