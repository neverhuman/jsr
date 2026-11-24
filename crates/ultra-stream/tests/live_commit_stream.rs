//! Integration tests for Wave 1D opt-in live-endpointer `CommitPassSession` (`--commit-pass`)
//! and session lexical memory (`P8`).

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use ultra_core::events::{Body, TranscriptEvent};
use ultra_engine::commit::{
    CommitConfig, SessionLexicalConfig, SessionLexicalMemory, DEFAULT_SAMPLE_RATE,
};
use ultra_engine::model::{Arch, Linear, Model};
use ultra_engine::{Engine, Isa, Precision};
use ultra_frontend::{GlobalCmvn, TrainerMelConfig};
use ultra_gp::TokenTable;
use ultra_stream::{
    AsrStream, LiveCommitConfig, NativeEngineBackend, OutputPolicy, PollResult, StreamClock,
    StreamConfig, StreamCoordinator,
};

fn wave(n: usize, s: f32, p: f32) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * p).sin() * s).collect()
}
fn lin(n: usize, k: usize, s: f32, p: f32, b: Option<&[f32]>) -> Linear {
    Linear::from_fp32(&wave(n * k, s, p), n, k, b)
}

/// Synthetic model: emits `<blk>` on zero-PCM silence and word tokens on 440 Hz tone.
fn speech_or_silence_model() -> Arc<Model> {
    let mut fb = vec![0u8; 257 * 80 * 4];
    for m in 0..80 {
        for bin in 1..40 {
            let o = (bin * 80 + m) * 4;
            fb[o..o + 4].copy_from_slice(&0.25f32.to_le_bytes());
        }
    }
    let cmvn = GlobalCmvn::new(&[-13.815_511; 80], &[1.0; 80]).expect("cmvn");
    let mel_cfg = Arc::new(TrainerMelConfig::from_fbank_le_bytes(&fb, 1e-6, cmvn).expect("mel"));
    let toks = "<blk> 0\n<unk> 1\n\u{2581}Reykjavik 2\n\u{2581}world 3\n\u{2581}yes 4\n\u{2581}no 5\n\u{2581}go 6\n\u{2581}ok 7\n";
    let tokens = Arc::new(TokenTable::from_str_checked(toks, 8).expect("toks"));
    let mut jw = wave(32, 2.0, 0.71);
    jw[..4].fill(0.0);
    let jb = [0.8, -2.0, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2];
    let c2 = Linear::from_fp32(&[0.15; 2 * 18], 2, 18, None);
    let stem_out = Linear::from_fp32(&[0.15; 4 * 38], 4, 38, None);
    Arc::new(Model {
        arch: Arch {
            n_mels: 80,
            d: 4,
            n_blocks: 0,
            lower: 0,
            mid: 0,
            fin: 0,
            heads: 1,
            dh: 4,
            ffn: 16,
            kernel: 3,
            stem_ch: 2,
            f1: 39,
            f2: 19,
            chunk: 4,
            left: 16,
            post_block_norm: false,
            vocab: 8,
            pred_emb: 4,
            joiner: 4,
            blank: 0,
            max_symbols: 1,
            ctx_init: vec![0, 0],
            log_floor: 1e-6,
            checkpoint_sha256: String::new(),
            weights_sha256: String::new(),
            qat_stem_int8: false,
        },
        root: PathBuf::new(),
        c1_w: vec![0.15; 18],
        c1_b: vec![0.0; 2],
        c2,
        stem_out,
        blocks: Vec::new(),
        final_norm: None,
        pair: None,
        restore: None,
        enc_proj: lin(4, 4, 0.5, 0.43, None),
        ctc_head: None,
        emb: wave(32, 0.05, 0.37),
        pred_proj: lin(4, 8, 0.05, 0.53, None),
        pred_cache: None,
        joiner_out: Linear::from_fp32(&jw, 8, 4, Some(&jb)),
        blank_cert_f32: None,
        blank_cert_i8: None,
        rope_inv: vec![1.0, 0.1],
        mel_cfg,
        tokens,
        assets: Default::default(),
        int8_prepared: None,
    })
}

fn backend() -> Arc<NativeEngineBackend> {
    let engine = Engine::new(speech_or_silence_model(), Isa::Scalar, Precision::F32).expect("eng");
    Arc::new(NativeEngineBackend {
        engine,
        arch_sha256: "000000000000deadbeef".into(),
        int8_source: "n/a",
    })
}

fn tone(n: usize, hz: f32) -> Vec<i16> {
    let k = std::f32::consts::TAU * hz / 16e3;
    (0..n)
        .map(|i| ((k * i as f32).sin() * 12e3) as i16)
        .collect()
}

fn drain(coord: &mut StreamCoordinator) -> Vec<TranscriptEvent> {
    let mut out = Vec::new();
    while let PollResult::Event(ev) = coord.poll_event().expect("poll") {
        out.push(*ev);
    }
    out
}

fn run_stream(
    b: &Arc<NativeEngineBackend>,
    pcm: &[i16],
    live_cfg: &LiveCommitConfig,
) -> (Vec<TranscriptEvent>, String, ultra_stream::StreamMetrics) {
    let tick = Arc::new(AtomicU64::new(0));
    let t = Arc::clone(&tick);
    let clk = StreamClock::custom(move || t.fetch_add(1_000_000, Ordering::SeqCst) + 1);
    let cfg = StreamConfig {
        stream_id: "live-commit-test".into(),
        policy: OutputPolicy::EditableDictation,
        segment_hard_cap_samples: (live_cfg.commit.max_pcm_samples as u64).max(16_000 * 40),
        endpoint_hangover_ms: u64::from(live_cfg.hangover_ms),
        ..StreamConfig::default()
    };
    let inner = b
        .create_live_commit_stream_with_clock(&cfg, live_cfg, clk.clone())
        .expect("create live commit stream");
    let mut coord = StreamCoordinator::with_clock(cfg, inner, clk);
    let mut evs = Vec::new();
    for (k, pkt) in pcm.chunks(320).enumerate() {
        coord.push_pcm((k * 320) as u64, pkt).expect("push");
        evs.extend(drain(&mut coord));
    }
    coord.finish_input().expect("finish");
    evs.extend(drain(&mut coord));
    let txt = coord.ledger().committed_text();
    let m = coord.metrics();
    (evs, txt, m)
}

#[test]
fn live_commit_hangover_triggers_at_200_300_500_800_ms_with_span_continuity() {
    let b = backend();
    let mut pcm = Vec::new();
    pcm.extend(vec![0i16; 16_000]); // 1.0 s leading silence (tests preroll ring retention)
    pcm.extend(tone(4_800, 440.0)); // 300 ms speech
    pcm.extend(vec![0i16; 20_800]); // 1.3 s silence (triggers hangover up to 800 ms)
    pcm.extend(tone(4_800, 770.0)); // 300 ms speech 2
    pcm.extend(vec![0i16; 20_800]); // 1.3 s silence (triggers hangover 2)

    let mut first_end_samples = Vec::new();
    for hangover_ms in [200u32, 300, 500, 800] {
        let cfg = LiveCommitConfig {
            hangover_ms,
            preroll_ms: 300,
            session_bias: SessionLexicalConfig {
                enabled: true,
                bias_bonus: 0.35,
                ..SessionLexicalConfig::default()
            },
            ..LiveCommitConfig::default()
        };
        let (evs, text, metrics) = run_stream(&b, &pcm, &cfg);
        assert!(!text.is_empty(), "hangover={hangover_ms} must commit text");
        let finals: Vec<(u64, u64, u64)> = evs
            .iter()
            .filter_map(|e| match &e.body {
                Body::Final { update, .. } => Some((
                    update.segment_id,
                    update.ownership_start_sample,
                    update.ownership_end_sample,
                )),
                _ => None,
            })
            .collect();
        assert!(finals.len() >= 2, "expected >= 2 segments, got {finals:?}");
        // Verify strict contiguous sample ownership with zero overlaps or gaps.
        assert_eq!(finals[0].1, 0);
        for w in finals.windows(2) {
            assert_eq!(w[1].1, w[0].2, "segment ownership must be contiguous");
        }
        assert_eq!(finals.last().expect("last").2, pcm.len() as u64);
        first_end_samples.push(finals[0].2);

        let statuses: Vec<serde_json::Value> = evs
            .iter()
            .filter_map(|e| match &e.body {
                Body::Status { code, detail } if code == "commit_pass" => {
                    serde_json::from_str(detail).ok()
                }
                _ => None,
            })
            .collect();
        assert_eq!(statuses.len(), finals.len());
        assert!(statuses.iter().any(|s| s["trigger"] == "hangover"));
        assert!(statuses.iter().all(|s| s["preroll_ms"] == 300));
        assert!(statuses
            .iter()
            .all(|s| s["endpoint_to_final_ms"].as_f64().is_some_and(|v| v >= 0.0)));
        assert!(metrics.refiner_charged_cpu_s >= 0.0);
        assert!(metrics.events_emitted >= finals.len() as u64 * 2);
    }
    assert!(
        first_end_samples.windows(2).all(|w| w[0] <= w[1])
            && first_end_samples[0] < first_end_samples[3],
        "hangover 200<=300<=500<=800 must trigger monotonically: {first_end_samples:?}"
    );
}

#[test]
fn live_commit_cap_fallback_keeps_all_audio_and_default_off_is_unchanged() {
    let b = backend();
    // 1. Cap fallback: continuous speech exceeding max_pcm_samples (1.0 s cap vs 1.5 s audio).
    let pcm_long = tone(24_000, 440.0);
    let cap_cfg = LiveCommitConfig {
        commit: CommitConfig {
            max_pcm_samples: DEFAULT_SAMPLE_RATE, // 1.0 s cap for fast unit test
            ..CommitConfig::default()
        },
        hangover_ms: 300,
        ..LiveCommitConfig::default()
    };
    let (evs, text, _) = run_stream(&b, &pcm_long, &cap_cfg);
    assert!(!text.is_empty());
    let final_end = evs
        .iter()
        .filter_map(|e| match &e.body {
            Body::Final { update, .. } => Some(update.ownership_end_sample),
            _ => None,
        })
        .next_back()
        .expect("final");
    assert_eq!(
        final_end,
        pcm_long.len() as u64,
        "zero audio dropped on cap"
    );
    assert!(evs.iter().any(|e| match &e.body {
        Body::Status { code, detail } if code == "commit_pass" => {
            serde_json::from_str::<serde_json::Value>(detail)
                .ok()
                .and_then(|v| v["pcm_truncated"].as_bool())
                == Some(true)
        }
        _ => false,
    }));

    // 2. Default-off NativeEngineBackend::create_stream_with_clock remains bit-identical.
    let cfg = StreamConfig::default();
    let run_plain = || {
        let tick = Arc::new(AtomicU64::new(0));
        let t = Arc::clone(&tick);
        let clk = StreamClock::custom(move || t.fetch_add(1_000_000, Ordering::SeqCst) + 1);
        let inner = b.create_stream_with_clock(&cfg, clk.clone());
        let mut c = StreamCoordinator::with_clock(cfg.clone(), inner, clk);
        for (k, pkt) in pcm_long.chunks(320).enumerate() {
            c.push_pcm((k * 320) as u64, pkt).expect("push");
            let _ = drain(&mut c);
        }
        c.finish_input().expect("fin");
        let _ = drain(&mut c);
        c.ledger().committed_text()
    };
    assert_eq!(run_plain(), run_plain());
    assert_eq!(
        text,
        run_plain(),
        "cap fallback must match live provisional"
    );

    // 3. SessionLexicalMemory default-off sanity.
    let mem = SessionLexicalMemory::new(SessionLexicalConfig::default()).expect("off");
    assert!(mem.bias_map().is_empty());
}
