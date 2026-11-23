//! Regression test for idle-reset sample-offset preservation, zero missing-input gaps across
//! >= 32 s streams with multiple > 4 s silence resets, and complete EOF tail span coverage.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use ultra_core::events::{Body, TranscriptEvent};
use ultra_engine::commit::CommitConfig;
use ultra_engine::model::{Arch, Linear, Model};
use ultra_engine::{Engine, Isa, Precision};
use ultra_frontend::{GlobalCmvn, TrainerMelConfig};
use ultra_gp::TokenTable;
use ultra_stream::{
    AsrStream, LiveCommitConfig, NativeEngineBackend, PollResult, StreamClock, StreamConfig,
    StreamCoordinator,
};

fn wave(n: usize, s: f32, p: f32) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * p).sin() * s).collect()
}
fn lin(n: usize, k: usize, s: f32, p: f32, b: Option<&[f32]>) -> Linear {
    Linear::from_fp32(&wave(n * k, s, p), n, k, b)
}

fn tiny_backend() -> Arc<NativeEngineBackend> {
    let mut fb = vec![0u8; 257 * 80 * 4];
    for m in 0..80 {
        for bin in 1..40 {
            let o = (bin * 80 + m) * 4;
            fb[o..o + 4].copy_from_slice(&0.25f32.to_le_bytes());
        }
    }
    let cmvn = GlobalCmvn::new(&[-13.815_511; 80], &[1.0; 80]).expect("cmvn");
    let mel_cfg = Arc::new(TrainerMelConfig::from_fbank_le_bytes(&fb, 1e-6, cmvn).expect("mel"));
    let toks = "<blk> 0\n<unk> 1\n\u{2581}hello 2\n\u{2581}world 3\n\u{2581}yes 4\n\u{2581}no 5\n\u{2581}go 6\n\u{2581}ok 7\n";
    let tokens = Arc::new(TokenTable::from_str_checked(toks, 8).expect("toks"));
    let mut jw = wave(32, 2.0, 0.71);
    jw[..4].fill(0.0);
    let jb = [0.8, -2.0, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2];
    let model = Arc::new(Model {
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
        c2: Linear::from_fp32(&[0.15; 2 * 18], 2, 18, None),
        stem_out: Linear::from_fp32(&[0.15; 4 * 38], 4, 38, None),
        blocks: Vec::new(),
        final_norm: None,
        pair: None,
        restore: None,
        enc_proj: lin(4, 4, 0.5, 0.43, None),
        emb: wave(32, 0.05, 0.37),
        pred_proj: lin(4, 8, 0.05, 0.53, None),
        pred_cache: None,
        joiner_out: Linear::from_fp32(&jw, 8, 4, Some(&jb)),
        blank_cert_f32: None,
        blank_cert_i8: None,
        rope_inv: vec![1.0, 0.1],
        mel_cfg,
        tokens,
        int8_prepared: None,
        ctc_head: None,
    });
    let engine = Engine::new(model, Isa::Scalar, Precision::F32).expect("eng");
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

fn drain(c: &mut StreamCoordinator) -> Vec<TranscriptEvent> {
    let mut out = Vec::new();
    while let PollResult::Event(ev) = c.poll_event().expect("poll") {
        out.push(*ev);
    }
    out
}

#[test]
fn live_commit_survives_32s_multi_idle_resets_with_zero_gaps_and_full_eof_span() {
    let b = tiny_backend();
    // 32.5 s stream with 3 speech bursts separated by 13 s and 14 s silence gaps (> 10 s
    // coordinator backpressure budget & multiple 4 s idle resets) + 320-sample EOF tail.
    let mut pcm = Vec::new();
    pcm.extend(tone(8_000, 440.0)); // 0.5 s speech 1
    pcm.extend(vec![0i16; 16_000 * 13]); // 13 s silence (3+ idle resets!)
    pcm.extend(tone(8_000, 660.0)); // 0.5 s speech 2
    pcm.extend(vec![0i16; 16_000 * 14]); // 14 s silence (3+ idle resets!)
    pcm.extend(tone(8_000, 550.0)); // 0.5 s speech 3
    pcm.extend(vec![0i16; 16_000 * 4 + 320]); // 4.02 s tail silence

    let live_cfg = LiveCommitConfig {
        commit: CommitConfig {
            commit_chunk: 16,
            tail_pad_samples: 5_120,
            max_pcm_samples: 16_000 * 30,
            ..CommitConfig::default()
        },
        hangover_ms: 200,
        preroll_ms: 1_000,
        deadline_ms: 800,
        ..LiveCommitConfig::default()
    };
    let scfg = StreamConfig {
        segment_hard_cap_samples: u64::MAX,
        endpoint_hangover_ms: 200,
        ..StreamConfig::default()
    };
    let tick = Arc::new(AtomicU64::new(0));
    let t = Arc::clone(&tick);
    let clk = StreamClock::custom(move || t.fetch_add(1_000_000, Ordering::SeqCst) + 1);
    let inner = b
        .create_live_commit_stream_with_clock(&scfg, &live_cfg, clk.clone())
        .expect("live commit stream");
    let mut coord = StreamCoordinator::with_clock(scfg, inner, clk);

    let mut evs = Vec::new();
    let mut accepted_total = 0u64;
    for (k, pkt) in pcm.chunks(320).enumerate() {
        let rep = coord.push_pcm((k * 320) as u64, pkt).expect("push");
        assert_eq!(
            rep.accepted_samples,
            pkt.len(),
            "zero backpressure stalls after idle resets at packet {k}"
        );
        accepted_total += rep.accepted_samples as u64;
        evs.extend(drain(&mut coord));
    }
    coord.finish_input().expect("finish");
    evs.extend(drain(&mut coord));

    assert_eq!(accepted_total, pcm.len() as u64);
    let gaps = evs
        .iter()
        .filter(|e| matches!(e.body, Body::Gap { .. }))
        .count();
    assert_eq!(gaps, 0, "must emit zero missing-input gaps");

    let finals: Vec<(u64, u64)> = evs
        .iter()
        .filter_map(|e| match &e.body {
            Body::Final { update, .. } => {
                Some((update.ownership_start_sample, update.ownership_end_sample))
            }
            _ => None,
        })
        .collect();
    assert!(
        finals.len() >= 3,
        "expected >= 3 finalized segments, got {finals:?}"
    );
    assert_eq!(finals[0].0, 0);
    for w in finals.windows(2) {
        assert_eq!(w[1].0, w[0].1, "segment ownership must stay contiguous");
    }
    assert_eq!(
        finals.last().expect("last").1,
        pcm.len() as u64,
        "EOF tail must be spanned completely"
    );
    // Verify every Final is immediately followed by a commit_pass Status with matching span.
    let mut matched = 0usize;
    for w in evs.windows(2) {
        if let Body::Final { update, .. } = &w[0].body {
            match &w[1].body {
                Body::Status { code, detail } if code == "commit_pass" => {
                    let v: serde_json::Value = serde_json::from_str(detail).expect("json");
                    assert_eq!(
                        v["start_sample"].as_u64(),
                        Some(update.ownership_start_sample)
                    );
                    assert_eq!(v["end_sample"].as_u64(), Some(update.ownership_end_sample));
                    matched += 1;
                }
                other => panic!("expected commit_pass status right after Final, got {other:?}"),
            }
        }
    }
    assert_eq!(matched, finals.len());
}

#[test]
fn live_commit_late_queued_result_after_deadline_expiry_never_corrupts_ledger() {
    let b = tiny_backend();
    let mut pcm = Vec::new();
    pcm.extend(tone(8_000, 440.0)); // 0.5 s speech 1
    pcm.extend(vec![0i16; 16_000]); // 1.0 s silence (hangover 1)
    pcm.extend(tone(8_000, 660.0)); // 0.5 s speech 2
    pcm.extend(vec![0i16; 16_000]); // 1.0 s silence (hangover 2)

    let live_cfg = LiveCommitConfig {
        hangover_ms: 200,
        preroll_ms: 300,
        deadline_ms: 800,
        ..LiveCommitConfig::default()
    };
    let scfg = StreamConfig {
        segment_hard_cap_samples: u64::MAX,
        endpoint_hangover_ms: 200,
        ..StreamConfig::default()
    };
    let now_ns = Arc::new(AtomicU64::new(1_000_000_000));
    let clk_ref = Arc::clone(&now_ns);
    let clk = StreamClock::custom(move || clk_ref.fetch_add(1_000_000, Ordering::SeqCst));
    let inner = b
        .create_live_commit_stream_with_clock(&scfg, &live_cfg, clk.clone())
        .expect("stream");
    let mut coord = StreamCoordinator::with_clock(scfg, inner, clk);

    let mut evs = Vec::new();
    let mut jumped = false;
    for (k, pkt) in pcm.chunks(320).enumerate() {
        coord.push_pcm((k * 320) as u64, pkt).expect("push");
        evs.extend(drain(&mut coord));
        // Right after speech 1 + 240 ms silence (hangover 1 dispatched), jump clock +2 s past
        // the 800 ms deadline so segment 0 expires into greedy fallback while job 0 queues late.
        if !jumped && k * 320 >= 12_800 {
            now_ns.fetch_add(2_000_000_000, Ordering::SeqCst);
            jumped = true;
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }
    coord.finish_input().expect("finish");
    evs.extend(drain(&mut coord));

    let finals: Vec<(u64, u64)> = evs
        .iter()
        .filter_map(|e| match &e.body {
            Body::Final { update, .. } => {
                Some((update.ownership_start_sample, update.ownership_end_sample))
            }
            _ => None,
        })
        .collect();
    assert!(finals.len() >= 2);
    assert_eq!(finals[0].0, 0);
    assert!(finals.windows(2).all(|w| w[1].0 == w[0].1));
    assert_eq!(finals.last().expect("last").1, pcm.len() as u64);
    assert!(evs.iter().all(|e| !matches!(e.body, Body::Gap { .. })));
    // Confirm every Final has an immediate commit_pass Status and no duplicate Final arrived.
    let status_count = evs
        .iter()
        .filter(|e| matches!(&e.body, Body::Status { code, .. } if code == "commit_pass"))
        .count();
    assert_eq!(status_count, finals.len());
}
