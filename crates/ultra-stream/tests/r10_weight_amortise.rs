//! R10 (`p7-weight-amortise`) verification tests:
//! 1. Deterministic fake-clock deadline-bound enforcement, clean drain, separate
//!    `engine_decode_calls` vs `paced_wakeups` accounting, and `~N×` decode-call reduction on
//!    staggered arrivals (`N = 1, 4, 8, 16`).
//! 2. 0-diff bit-exact transcript parity across `N = 1` (`Single`), `N = 4`, `N = 8`, and `N = 16`
//!    (`Batch` with and without deadline-aware coalescing) over `NativeEngineSlotBank` (INT8).

use std::cell::Cell;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

use ultra_engine::model::{Arch, Block, Linear, Model, Norm};
use ultra_engine::{Engine, Isa, Precision};
use ultra_frontend::npz::AudioUtterance;
use ultra_frontend::{GlobalCmvn, TrainerMelConfig};
use ultra_gp::TokenTable;
use ultra_stream::{
    simulate_grid, sustains_lateness, DecodeMode, NativeEngineBackend, NativeEngineSlotBank,
    SimClock, SimParams, SimSlotBank, StreamError,
};

#[derive(Clone)]
struct FakeTime {
    t_s: Rc<Cell<f64>>,
    cpu_s: Rc<Cell<f64>>,
}

impl FakeTime {
    fn new() -> Self {
        Self {
            t_s: Rc::new(Cell::new(0.0)),
            cpu_s: Rc::new(Cell::new(0.0)),
        }
    }
}

impl SimClock for FakeTime {
    fn now_s(&self) -> f64 {
        self.t_s.get()
    }
    fn sleep_s(&mut self, dt_s: f64) {
        if dt_s > 0.0 {
            self.t_s.set(self.t_s.get() + dt_s);
        }
    }
    fn cpu_s(&self) -> Result<f64, StreamError> {
        Ok(self.cpu_s.get())
    }
}

struct StaggeredSlot {
    utt: usize,
    cursor: usize,
    pending_samples: usize,
    ready_chunks: usize,
    finished: bool,
}

/// Deterministic bank whose slots start with staggered chunk phases across the 160 ms chunk
/// period (mirroring real multi-stream arrivals on a 20 ms packet grid).
struct StaggeredRecBank {
    utt_lens: Vec<usize>,
    slots: Vec<Option<StaggeredSlot>>,
    chunk_samples: usize,
    decode_cost_s: f64,
    clock: FakeTime,
}

impl StaggeredRecBank {
    fn new(
        utt_secs: &[f64],
        streams: usize,
        chunk_ms: u32,
        decode_cost_s: f64,
        clock: FakeTime,
    ) -> Self {
        Self {
            utt_lens: utt_secs
                .iter()
                .map(|s| (s * 16_000.0).round() as usize)
                .collect(),
            slots: (0..streams).map(|_| None).collect(),
            chunk_samples: (u64::from(chunk_ms) * 16_000 / 1000).max(1) as usize,
            decode_cost_s,
            clock,
        }
    }
}

impl SimSlotBank for StaggeredRecBank {
    fn num_utterances(&self) -> usize {
        self.utt_lens.len()
    }
    fn utt_samples(&self, utt_pos: usize) -> usize {
        self.utt_lens[utt_pos]
    }
    fn open_slot(&mut self, slot: usize, utt_pos: usize) -> Result<(), StreamError> {
        let streams = self.slots.len().max(1);
        // Stagger initial phase across the chunk period so slots become ready on distinct ticks.
        let phase_samples = (slot * self.chunk_samples) / streams;
        self.slots[slot] = Some(StaggeredSlot {
            utt: utt_pos,
            cursor: 0,
            pending_samples: phase_samples,
            ready_chunks: 0,
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
    fn decode_ready_step(&mut self, decode: DecodeMode, max_batch: usize) -> (usize, u64) {
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
        let calls = match decode {
            DecodeMode::Single => ready as u64,
            DecodeMode::Batch => ready.div_ceil(max_batch.max(1)) as u64,
        };
        let dt = self.decode_cost_s * (calls as f64);
        self.clock.t_s.set(self.clock.t_s.get() + dt);
        self.clock.cpu_s.set(self.clock.cpu_s.get() + dt);
        (ready, calls)
    }
    fn take_completed_slot(&mut self, slot: usize) -> String {
        let u = self.slots[slot].take().map_or(0, |s| s.utt);
        format!("hyp-{u}")
    }
}

#[test]
fn r10_deadline_coalescer_bounds_wait_reduces_decode_calls_and_drains() {
    let utts: Vec<f64> = (0..16).map(|i| 1.6 + (i % 4) as f64 * 0.16).collect();
    let decode_cost_s = 0.005;

    // 1. N = 1: coalescer fires immediately with 0 ms added wait; wakeups & decode calls stay distinct.
    let mut clk0 = FakeTime::new();
    let mut bank0 = StaggeredRecBank::new(&utts[..1], 1, 160, decode_cost_s, clk0.clone());
    let base1 = simulate_grid(
        &mut bank0,
        &SimParams {
            streams: 1,
            decode: DecodeMode::Batch,
            max_batch: 64,
            packet_ms: 20,
            grid_ms: 20,
            coalesce_ms: 0,
            paced: true,
            duration_s: None,
        },
        &mut clk0,
    )
    .expect("N=1 base");

    let mut clk1 = FakeTime::new();
    let mut bank1 = StaggeredRecBank::new(&utts[..1], 1, 160, decode_cost_s, clk1.clone());
    let coal1 = simulate_grid(
        &mut bank1,
        &SimParams {
            streams: 1,
            decode: DecodeMode::Batch,
            max_batch: 64,
            packet_ms: 20,
            grid_ms: 20,
            coalesce_ms: 100,
            paced: true,
            duration_s: None,
        },
        &mut clk1,
    )
    .expect("N=1 coalesced");

    assert!(coal1.drained);
    assert_eq!(coal1.engine_decode_calls, base1.engine_decode_calls);
    assert_eq!(coal1.wakeups, coal1.engine_decode_calls);
    assert!(
        coal1.paced_wakeups > coal1.engine_decode_calls,
        "paced_wakeups ({}) must be tracked separately from engine_decode_calls ({})",
        coal1.paced_wakeups,
        coal1.engine_decode_calls
    );
    let lat1 = coal1.lateness_ms.as_ref().expect("N=1 lateness");
    assert!(
        (lat1.max - 5.0).abs() < 1e-6,
        "N=1 must not wait for other streams (got max={} ms)",
        lat1.max
    );

    // 2. N = 4, 8, 16 staggered arrivals: coalescer enforces strict <= coalesce_ms hold bound,
    //    drains cleanly, sustains lateness << 160 ms, and slashes engine_decode_calls by ~N×.
    for &(streams, coalesce_ms, min_reduction) in
        &[(4usize, 120u32, 3.0f64), (8, 120, 4.0), (16, 120, 4.0)]
    {
        let mut cb = FakeTime::new();
        let mut bb = StaggeredRecBank::new(&utts, streams, 160, decode_cost_s, cb.clone());
        let uncoal = simulate_grid(
            &mut bb,
            &SimParams {
                streams,
                decode: DecodeMode::Batch,
                max_batch: 64,
                packet_ms: 20,
                grid_ms: 20,
                coalesce_ms: 0,
                paced: true,
                duration_s: Some(3.2),
            },
            &mut cb,
        )
        .expect("uncoalesced");

        let mut cc = FakeTime::new();
        let mut bc = StaggeredRecBank::new(&utts, streams, 160, decode_cost_s, cc.clone());
        let coal = simulate_grid(
            &mut bc,
            &SimParams {
                streams,
                decode: DecodeMode::Batch,
                max_batch: 64,
                packet_ms: 20,
                grid_ms: 20,
                coalesce_ms,
                paced: true,
                duration_s: Some(3.2),
            },
            &mut cc,
        )
        .expect("coalesced");

        assert!(coal.drained, "streams={streams} must drain cleanly");
        let lat = coal.lateness_ms.as_ref().expect("lateness recorded");
        // Strictly bounded: steady-state hold <= coalesce_ms + 1 decode; even EOF 2-chunk tail
        // drain never exceeds coalesce_ms + 2 decodes (and sustains_lateness verifies p99 <= 160 ms).
        assert!(
            lat.p50 <= f64::from(coalesce_ms) + decode_cost_s * 1000.0 + 1e-3,
            "streams={streams} p50 exceeded coalesce bound: p50={} ms (coalesce={coalesce_ms} ms)",
            lat.p50
        );
        assert!(
            lat.max <= f64::from(coalesce_ms) + 2.0 * decode_cost_s * 1000.0 + 1e-3,
            "streams={streams} max exceeded coalesce+drain bound: max={} ms (coalesce={coalesce_ms} ms)",
            lat.max
        );
        assert!(
            sustains_lateness(lat, coal.drained, 160.0),
            "streams={streams} must sustain < 160 ms bound: {lat:?}"
        );
        let ratio = (uncoal.engine_decode_calls as f64) / (coal.engine_decode_calls.max(1) as f64);
        assert!(
            ratio >= min_reduction,
            "streams={streams} expected >= {min_reduction}x decode_call reduction, got {ratio:.2}x ({} -> {})",
            uncoal.engine_decode_calls,
            coal.engine_decode_calls
        );
        assert!(
            coal.paced_wakeups > coal.engine_decode_calls,
            "streams={streams} paced_wakeups ({}) must stay distinct from engine_decode_calls ({})",
            coal.paced_wakeups,
            coal.engine_decode_calls
        );
    }
}

fn next_f32(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let u = ((*seed >> 40) as f32) / ((1u64 << 24) as f32);
    (u - 0.5) * 0.4
}

fn mk_lin(seed: &mut u64, n: usize, m: usize) -> Linear {
    let w: Vec<f32> = (0..n * m).map(|_| next_f32(seed)).collect();
    let b: Vec<f32> = (0..n).map(|_| next_f32(seed) * 0.1).collect();
    Linear::from_fp32(&w, n, m, Some(&b))
}

fn synthetic_int8_model() -> Arc<Model> {
    let (d, h, dh, ffn, k, sc, f1, f2) = (32, 4, 8, 64, 5, 8, 39, 19);
    let (vocab, pe, jn) = (4, 8, 16);
    let mut seed = 0x5EED_CAFEu64;
    let nm = |n| Norm {
        g: vec![1.0; n],
        b: vec![0.0; n],
    };
    let blk = Block {
        ff1_norm: nm(d),
        ff1_w1: mk_lin(&mut seed, ffn, d),
        ff1_w2: mk_lin(&mut seed, d, ffn),
        att_norm: nm(d),
        qkv: mk_lin(&mut seed, 3 * d, d),
        att_out: mk_lin(&mut seed, d, d),
        conv_norm: nm(d),
        pw1: mk_lin(&mut seed, 2 * d, d),
        dw_w: (0..k * d).map(|_| next_f32(&mut seed)).collect(),
        dw_b: vec![0.0; d],
        dnorm: nm(d),
        pw2: mk_lin(&mut seed, d, d),
        ff2_norm: nm(d),
        ff2_w1: mk_lin(&mut seed, ffn, d),
        ff2_w2: mk_lin(&mut seed, d, ffn),
        post_norm: None,
    };
    let tok = TokenTable::from_str_checked(
        "<blk> 0\n\u{2581}one 1\n\u{2581}two 2\n\u{2581}three 3\n",
        vocab,
    )
    .expect("tok");
    let arch = Arch {
        n_mels: 80,
        d,
        n_blocks: 1,
        lower: 1,
        mid: 0,
        fin: 0,
        heads: h,
        dh,
        ffn,
        kernel: k,
        stem_ch: sc,
        f1,
        f2,
        chunk: 4,
        left: 16,
        post_block_norm: false,
        vocab,
        pred_emb: pe,
        joiner: jn,
        blank: 0,
        max_symbols: 2,
        ctx_init: vec![0, 0],
        log_floor: 1e-6,
        checkpoint_sha256: "a".repeat(64),
        weights_sha256: "b".repeat(64),
        qat_stem_int8: false,
    };
    let mut m = Model {
        arch,
        root: PathBuf::new(),
        c1_w: (0..9 * sc).map(|_| next_f32(&mut seed)).collect(),
        c1_b: vec![0.0; sc],
        c2: mk_lin(&mut seed, sc, 9 * sc),
        stem_out: mk_lin(&mut seed, d, f2 * sc),
        blocks: vec![blk],
        final_norm: Some(nm(d)),
        pair: None,
        restore: None,
        enc_proj: mk_lin(&mut seed, jn, d),
        emb: (0..vocab * pe).map(|_| next_f32(&mut seed)).collect(),
        pred_proj: mk_lin(&mut seed, jn, 2 * pe),
        pred_cache: None,
        joiner_out: mk_lin(&mut seed, vocab, jn),
        blank_cert_f32: None,
        blank_cert_i8: None,
        rope_inv: (0..dh / 2)
            .map(|i| 10_000.0f32.powf(-2.0 * (i as f32) / (dh as f32)))
            .collect(),
        mel_cfg: Arc::new(TrainerMelConfig::default_htk(GlobalCmvn::identity()).expect("mel")),
        tokens: Arc::new(tok),
        ctc_head: None,
        assets: Default::default(),
        int8_prepared: None,
    };
    m.quantize_int8_where(|_, _| true);
    Arc::new(m)
}

#[test]
fn r10_native_engine_0diff_transcript_parity_across_n1_n4_n8_n16_and_coalescing() {
    let model = synthetic_int8_model();
    let engine = Engine::new(model, Isa::detect(), Precision::Int8).expect("int8 engine");
    let backend = Arc::new(NativeEngineBackend::from_engine(
        engine,
        "synth-r10-int8".into(),
        "test",
    ));

    // 16 distinct heterogeneous-length utterances (0.35 s to 1.25 s) so slots finish & refill out
    // of phase across N = 4, 8, 16, thoroughly exercising partial & full batches (M = 4..64).
    let utts: Vec<AudioUtterance> = (0..16)
        .map(|i| {
            let n_samples = 5_600 + i * 960;
            let pcm: Vec<i16> = (0..n_samples)
                .map(|t| {
                    let freq = 220.0 + (i as f32) * 35.0;
                    let phase = 2.0 * std::f32::consts::PI * freq * (t as f32) / 16_000.0;
                    (phase.sin() * 12_000.0) as i16
                })
                .collect();
            AudioUtterance { index: i, pcm }
        })
        .collect();

    let run_mode = |streams: usize, decode: DecodeMode, coalesce_ms: u32| {
        let mut bank =
            NativeEngineSlotBank::with_start_offset(Arc::clone(&backend), &utts, 0, streams, None);
        let mut clk = FakeTime::new();
        let sim = simulate_grid(
            &mut bank,
            &SimParams {
                streams,
                decode,
                max_batch: 64,
                packet_ms: 20,
                grid_ms: 20,
                coalesce_ms,
                paced: false,
                duration_s: None,
            },
            &mut clk,
        )
        .expect("simulate_grid");
        assert!(
            sim.drained,
            "streams={streams} coalesce={coalesce_ms} must drain"
        );
        let hyps: BTreeMap<usize, String> = sim.completed.into_iter().collect();
        assert_eq!(hyps.len(), utts.len());
        (hyps, sim.engine_decode_calls)
    };

    let (ref_hyps, ref_calls) = run_mode(1, DecodeMode::Single, 0);
    for &(streams, coalesce_ms) in &[
        (4usize, 0u32),
        (4, 80),
        (8, 0),
        (8, 100),
        (16, 0),
        (16, 120),
    ] {
        let (got_hyps, got_calls) = run_mode(streams, DecodeMode::Batch, coalesce_ms);
        assert_eq!(
            got_hyps, ref_hyps,
            "0-diff transcript parity failed at N={streams}, coalesce_ms={coalesce_ms}"
        );
        if coalesce_ms > 0 {
            assert!(
                got_calls < ref_calls / 2,
                "expected coalescing at N={streams} to slash decode calls vs N=1 ({got_calls} vs {ref_calls})"
            );
        }
    }
}
