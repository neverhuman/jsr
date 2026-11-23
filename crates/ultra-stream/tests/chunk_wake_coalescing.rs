//! Witness tests for H47 chunk-coalesced wakeups and event publishing:
//! 1. Byte-identical transcript & token-stream parity across 20 ms per-packet, chunk-coalesced,
//!    10 ms, 37 ms, and arbitrary 137-sample odd packets + partial trailing EOF flush.
//! 2. Wakeup reduction >= 7x (50/s -> 6.25/s at chunk 4 = 2_560 samples) on 10 s of 20 ms audio.
//! 3. Gap, EOF, and mid-stream flush preservation: zero lost/duplicated samples, monotonic
//!    audio-clock timestamps, immediate `Body::Gap` delivery with pre-gap flush first.
//! 4. Design-clock emission timestamp parity: 0 ms shift at 20 ms packets and <= 20 ms across
//!    short packetizations (`10 ms`, `137` samples).

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use ultra_core::events::{Body, Ledger, TranscriptEvent};
use ultra_engine::Precision;
use ultra_eval::pace::PacedReplay;
use ultra_frontend::npz::load_npz_audio_only;
use ultra_stream::{
    AsrStream, Backend, ChunkWakeCoalescer, FakeBackend, FakeBackendConfig, FakeWord,
    NativeEngineBackend, OutputPolicy, PollResult, StreamClock, StreamConfig, StreamCoordinator,
    TerminalOutcome,
};

fn drain_events(s: &mut dyn AsrStream) -> Vec<TranscriptEvent> {
    let mut out = Vec::new();
    while let PollResult::Event(ev) = s.poll_event().expect("poll") {
        out.push(*ev);
    }
    out
}

fn feed_and_collect(
    stream: &mut dyn AsrStream,
    pcm: &[i16],
    pkt_samples: usize,
) -> (Vec<TranscriptEvent>, Ledger) {
    let (mut events, mut ledger, mut pos) = (Vec::new(), Ledger::new(), 0usize);
    while pos < pcm.len() {
        let end = (pos + pkt_samples).min(pcm.len());
        stream.push_pcm(pos as u64, &pcm[pos..end]).expect("push");
        for ev in drain_events(stream) {
            ledger.apply(&ev).expect("ledger apply");
            events.push(ev);
        }
        pos = end;
    }
    stream.finish_input().expect("finish_input");
    for ev in drain_events(stream) {
        ledger.apply(&ev).expect("ledger apply eof");
        events.push(ev);
    }
    assert_eq!(
        stream.poll_event().expect("ended"),
        PollResult::Ended(TerminalOutcome::Finished)
    );
    (events, ledger)
}

#[test]
fn chunk_coalesced_byte_identical_across_packet_sizes_and_reduces_wakeups_8x() {
    let chunk = 2_560usize; // 160 ms @ 16 kHz (chunk 4)
    let word = |emit_at_sample, w: &str| FakeWord {
        emit_at_sample,
        word: w.into(),
        difficult: false,
    };
    let backend = FakeBackend::new(FakeBackendConfig {
        chunk_samples: chunk,
        words: vec![
            word(2_560, "wake"),
            word(25_600, "once"),
            word(81_920, "per"),
            word(160_000, "chunk"),
        ],
        omit_difficult_words: false,
        energy_word_threshold: Some(500),
    });
    // 10 s + 137 trailing samples (tests partial EOF tail).
    let mut pcm = vec![0i16; 160_000 + 137];
    for (i, s) in pcm.iter_mut().enumerate() {
        *s = if (i / chunk).is_multiple_of(5) {
            2_000
        } else {
            0
        };
    }

    for policy in [OutputPolicy::EditableDictation, OutputPolicy::Commit] {
        let cfg = StreamConfig {
            stream_id: format!("w-{}", policy.as_str()),
            policy,
            ..StreamConfig::default()
        };
        let mut base = StreamCoordinator::new(cfg.clone(), backend.create_stream(&cfg).unwrap());
        let (base_evs, base_ledger) = feed_and_collect(&mut base, &pcm, 320);

        for pkt_samples in [160usize, 320, 592, 137, chunk] {
            let inner = StreamCoordinator::new(cfg.clone(), backend.create_stream(&cfg).unwrap());
            let mut coal = ChunkWakeCoalescer::new(Box::new(inner), chunk);
            let (evs, ledger) = feed_and_collect(&mut coal, &pcm, pkt_samples);
            assert_eq!(ledger.committed_text(), base_ledger.committed_text());
            assert_eq!(evs, base_evs, "packet size {pkt_samples} diverged");

            let st = coal.stats();
            assert_eq!(st.eof_flushes, 1, "partial 137-sample tail must flush once");
            if pkt_samples == 320 {
                // 10 s @ 20 ms = 501 packets -> 62 chunks + 1 EOF tail = 63 pushes (7.95x).
                assert_eq!(st.packets_in, 501);
                assert_eq!(st.inner_pushes, 63);
                assert!(st.packets_in >= st.inner_pushes * 7);
            }
        }
    }
}

#[test]
fn gap_and_eof_flush_preserve_all_samples_and_monotonic_audio_clock() {
    let chunk = 2_560usize;
    let word = |emit_at_sample, w: &str| FakeWord {
        emit_at_sample,
        word: w.into(),
        difficult: false,
    };
    let backend = FakeBackend::new(FakeBackendConfig {
        chunk_samples: chunk,
        words: vec![word(2_560, "before"), word(5_120, "after")],
        omit_difficult_words: false,
        energy_word_threshold: None,
    });
    let cfg = StreamConfig {
        stream_id: "gap-coalesce".into(),
        policy: OutputPolicy::Commit,
        ..StreamConfig::default()
    };
    let clock_ns = Arc::new(AtomicU64::new(0));
    let clk_ref = Arc::clone(&clock_ns);
    let clk = StreamClock::custom(move || clk_ref.load(Ordering::SeqCst));
    let inner =
        StreamCoordinator::with_clock(cfg.clone(), backend.create_stream(&cfg).unwrap(), clk);
    let mut coal = ChunkWakeCoalescer::new(Box::new(inner), chunk);
    let (mut ledger, mut events) = (Ledger::new(), Vec::new());

    // Push 3_200 samples in 20 ms packets (1 full chunk + 640 buffered samples).
    for k in 0..10u64 {
        clock_ns.store(((k + 1) * 320 * 1_000_000_000) / 16_000, Ordering::SeqCst);
        coal.push_pcm(k * 320, &[100i16; 320]).unwrap();
        for ev in drain_events(&mut coal) {
            ledger.apply(&ev).unwrap();
            events.push(ev);
        }
    }
    assert_eq!(coal.stats().buffered_samples, 640);
    assert_eq!(coal.stats().inner_pushes, 1);

    // Gap: jump from 3_200 -> 6_400; pre-gap 640 samples MUST flush BEFORE Gap(3200..6400).
    clock_ns.store((3_520 * 1_000_000_000) / 16_000, Ordering::SeqCst);
    coal.push_pcm(6_400, &[200i16; 320]).unwrap();
    for ev in drain_events(&mut coal) {
        ledger.apply(&ev).unwrap();
        events.push(ev);
    }
    assert_eq!(coal.stats().gap_flushes, 1);
    assert!(events.iter().any(|e| matches!(
        e.body,
        Body::Gap {
            start_sample: Some(3_200),
            end_sample: Some(6_400),
            ..
        }
    )));

    for k in 1..10u64 {
        clock_ns.store(
            ((3_520 + k * 320) * 1_000_000_000) / 16_000,
            Ordering::SeqCst,
        );
        coal.push_pcm(6_400 + k * 320, &[200i16; 320]).unwrap();
        for ev in drain_events(&mut coal) {
            ledger.apply(&ev).unwrap();
            events.push(ev);
        }
    }
    assert!(coal.flush_pending().unwrap() > 0);
    assert_eq!(coal.stats().manual_flushes, 1);

    clock_ns.store((6_511 * 1_000_000_000) / 16_000, Ordering::SeqCst);
    coal.push_pcm(9_600, &[200i16; 111]).unwrap();
    coal.finish_input().unwrap();
    for ev in drain_events(&mut coal) {
        ledger.apply(&ev).unwrap();
        events.push(ev);
    }
    assert_eq!(coal.stats().eof_flushes, 1);
    assert_eq!(coal.metrics().accepted_samples, 3_200 + 3_200 + 111);
    assert!(ledger.committed_text().contains("before"));
    assert!(ledger.committed_text().contains("after"));
    for win in events.windows(2) {
        assert!(win[1].observed_monotonic_ns >= win[0].observed_monotonic_ns);
    }
}

#[test]
fn native_engine_chunk_coalescing_byte_identical_and_design_clock_shift_within_20ms() {
    let root = std::env::var("JSR_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/mnt/xbabe3-external/jsr"));
    let weights = root.join("bundles/r1-d2-s5000-fused-int8-research-20261121-v2");
    let npz = root.join("cache/librispeech_pcm/dev-clean.npz");
    if !weights.join("weights.int8.safetensors").exists() || !npz.exists() {
        eprintln!(
            "SKIPPED native_engine chunk coalescing witness: weights or dev-clean.npz absent"
        );
        return;
    }

    let backend = NativeEngineBackend::load(&weights, Precision::Int8, 1, false).unwrap();
    let chunk_samples = backend.capabilities().chunk_samples;
    assert_eq!(chunk_samples, 2_560);
    let utts = load_npz_audio_only(&npz, 0, Some(3)).unwrap();

    let run_dc = |pcm: &[i16], pkt_samples: usize, coalesce: bool| {
        let clock_ns = Arc::new(AtomicU64::new(0));
        let clk_ref = Arc::clone(&clock_ns);
        let clk = StreamClock::custom(move || clk_ref.load(Ordering::SeqCst));
        let cfg = StreamConfig {
            stream_id: "native-dc".into(),
            policy: OutputPolicy::Commit,
            ..StreamConfig::default()
        };
        let inner = StreamCoordinator::with_clock(
            cfg.clone(),
            backend.create_stream_with_clock(&cfg, clk.clone()),
            clk,
        );
        let mut stream: Box<dyn AsrStream> = if coalesce {
            Box::new(ChunkWakeCoalescer::new(Box::new(inner), chunk_samples))
        } else {
            Box::new(inner)
        };
        let (mut ledger, mut emit_ns, mut pos) = (Ledger::new(), Vec::new(), 0usize);
        while pos < pcm.len() {
            let end = (pos + pkt_samples).min(pcm.len());
            clock_ns.store(((end as u64) * 1_000_000_000) / 16_000, Ordering::SeqCst);
            stream.push_pcm(pos as u64, &pcm[pos..end]).unwrap();
            for ev in drain_events(stream.as_mut()) {
                if matches!(ev.body, Body::Preview(_) | Body::Commit(_)) {
                    emit_ns.push(ev.observed_monotonic_ns);
                }
                ledger.apply(&ev).unwrap();
            }
            pos = end;
        }
        stream.finish_input().unwrap();
        for ev in drain_events(stream.as_mut()) {
            ledger.apply(&ev).unwrap();
        }
        (ledger.committed_text(), emit_ns)
    };

    for u in &utts {
        let (base_txt, base_ts) = run_dc(&u.pcm, 320, false);
        let (coal20_txt, coal20_ts) = run_dc(&u.pcm, 320, true);
        let (coal10_txt, coal10_ts) = run_dc(&u.pcm, 160, true);
        let (coal137_txt, coal137_ts) = run_dc(&u.pcm, 137, true);
        let (base10_txt, base10_ts) = run_dc(&u.pcm, 160, false);
        let (base137_txt, base137_ts) = run_dc(&u.pcm, 137, false);

        assert_eq!(coal20_txt, base_txt);
        assert_eq!(coal10_txt, base_txt);
        assert_eq!(coal137_txt, base_txt);
        assert_eq!(base10_txt, base_txt);
        assert_eq!(base137_txt, base_txt);

        // 20 ms packets: EXACT 0 ms design-clock shift!
        assert_eq!(coal20_ts, base_ts);
        for (c, b) in coal10_ts.iter().zip(&base10_ts) {
            assert!(*c >= *b && c - b <= 20_000_000, "10ms shift {}ns", c - b);
        }
        for (c, b) in coal137_ts.iter().zip(&base137_ts) {
            assert!(
                *c >= *b && c - b <= 20_000_000,
                "137-sample shift {}ns",
                c - b
            );
        }

        let cfg = StreamConfig::default();
        let mut coord = StreamCoordinator::new(cfg.clone(), backend.create_stream(&cfg).unwrap());
        for pkt in PacedReplay::with_chunk_coalescing(&u.pcm, 20, chunk_samples, 0.0).unwrap() {
            coord.push_pcm(pkt.start_sample, pkt.pcm).unwrap();
            let _ = drain_events(&mut coord);
        }
        coord.finish_input().unwrap();
        let _ = drain_events(&mut coord);
        assert_eq!(coord.ledger().committed_text(), base_txt);
    }
}
