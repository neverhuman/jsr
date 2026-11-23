//! Unit and integration tests for `ultra-alwayson`:
//! (a) 0% speech stays in `Listening` with zero engine calls and escalates `poll_hint_ms` 20/80/160,
//! (b) speech burst wakes, replays `1.0 s` pre-roll in one pass, and matches isolated decode bit-for-bit,
//! (c) exact sample conservation holds across arbitrary chunk sizes (`80 / 160 / 320 / 640 / 73`),
//! (d) `Recovery` fault isolation and ring-overwrite refusal guards.

use ultra_alwayson::{
    AlwaysOnConfig, AlwaysOnError, AlwaysOnSession, AsrBackend, LadderState, MockBackend, PcmRing,
    SegmentCause, SessionEvent, DEFAULT_RING_SAMPLES,
};

fn make_speech_burst_stream() -> Vec<i16> {
    // 14 s @ 16 kHz: 2.5 s quiet -> 1.5 s burst 1 -> 3.5 s quiet -> 1.5 s burst 2 -> 5.0 s quiet.
    let total = 14 * 16_000;
    let mut s: u32 = 42_424;
    (0..total)
        .map(|i| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let mut v = i32::from((s >> 16) as u16 as i16) >> 10;
            let sq = if (i / 20) % 2 == 0 { 1 } else { -1 };
            if (40_000..64_000).contains(&i) || (120_000..144_000).contains(&i) {
                v += 3_200 * sq;
            }
            v as i16
        })
        .collect()
}

#[test]
fn ring_conservation_and_bounds_guard() {
    let mut ring = PcmRing::default_3s();
    assert_eq!(ring.capacity(), DEFAULT_RING_SAMPLES);
    assert_eq!(ring.capacity(), 48_000);
    assert!(ring.check_conservation());

    let data: Vec<i16> = (0..120_000).map(|i| (i % 32_000) as i16).collect();
    for chunk in data.chunks(777) {
        ring.push(chunk);
        assert!(ring.check_conservation());
    }
    let c = ring.counters();
    assert_eq!(c.samples_in, 120_000);
    assert_eq!(c.samples_dropped, 72_000);
    assert_eq!(ring.base(), 72_000);
    assert_eq!(ring.end(), 120_000);

    // Reading valid live range succeeds and accounts for samples_replayed.
    let mut out = Vec::new();
    let n = ring
        .copy_range_into(104_000, 120_000, &mut out)
        .expect("live range");
    assert_eq!(n, 16_000);
    assert_eq!(out, &data[104_000..120_000]);
    assert_eq!(ring.counters().samples_replayed, 16_000);

    // Overwritten and future reads must fail loudly.
    assert!(matches!(
        ring.copy_range_into(0, 16_000, &mut out),
        Err(AlwaysOnError::RingOutOfRange { .. })
    ));
    assert!(matches!(
        ring.copy_range_into(110_000, 120_001, &mut out),
        Err(AlwaysOnError::RingOutOfRange { .. })
    ));
}

#[test]
fn zero_percent_speech_stays_listening_with_zero_engine_calls_and_depth_escalation() {
    let mut session =
        AlwaysOnSession::new(MockBackend::default(), AlwaysOnConfig::default()).expect("session");
    assert_eq!(session.state(), LadderState::Listening { depth: 1 });
    assert_eq!(session.poll_hint_ms(), 20);

    let silence_1s = vec![0i16; 16_000];
    let mut noop = |_: SessionEvent<'_>| {};

    // Push 15 s -> depth 2 (80 ms).
    for _ in 0..15 {
        session.push_pcm(&silence_1s, &mut noop).expect("push");
    }
    assert_eq!(session.state(), LadderState::Listening { depth: 2 });
    assert_eq!(session.poll_hint_ms(), 80);

    // Push another 120 s (total 135 s) -> depth 3 (160 ms).
    for _ in 0..120 {
        session.push_pcm(&silence_1s, &mut noop).expect("push");
    }
    assert_eq!(session.state(), LadderState::Listening { depth: 3 });
    assert_eq!(session.poll_hint_ms(), 160);
    session.finish(&mut noop).expect("finish");

    let stats = session.stats();
    assert_eq!(
        stats.engine_decode_calls, 0,
        "0% speech must make 0 engine calls"
    );
    assert_eq!(stats.recogniser_audio_samples, 0);
    assert_eq!(stats.wake_count, 0);
    assert!(session.ledger().is_empty());
    assert!(!session.backend().has_active_stream());
    assert_eq!(stats.ring.samples_in, 135 * 16_000);
    assert_eq!(stats.ring.samples_replayed, 0);
    assert_eq!(stats.ring.samples_dropped, 132 * 16_000);
}

#[test]
fn speech_burst_wakes_replays_in_one_pass_and_matches_isolated_decode_bit_for_bit() {
    let pcm = make_speech_burst_stream();
    let cfg = AlwaysOnConfig::default();
    assert_eq!(cfg.replay_ms, 500, "default pre-roll is 500 ms");
    assert_eq!(cfg.idle_ms, 1500, "default idle close is 1500 ms");
    assert!(
        !cfg.flush_on_sleep_close,
        "mid-stream sleep close defaults to complete chunks only (no trailing noise flush)"
    );
    let mut session = AlwaysOnSession::new(MockBackend::default(), cfg).expect("session");

    let mut saw_waking = false;
    let mut saw_active = false;
    let mut saw_draining = false;
    let mut saw_sleep = false;

    let mut on_ev = |ev: SessionEvent<'_>| match ev {
        SessionEvent::StateChanged { to, .. } => match to {
            LadderState::Waking => saw_waking = true,
            LadderState::Active => saw_active = true,
            LadderState::Draining => saw_draining = true,
            LadderState::Listening { .. } => saw_sleep = true,
            LadderState::Recovery => {}
        },
        SessionEvent::Wake {
            decode_calls,
            replay_samples,
            ..
        } => {
            assert_eq!(decode_calls, 1, "wake replay must run in one batched pass");
            assert_eq!(replay_samples, 8_000, "500 ms pre-roll = 8_000 samples");
        }
        SessionEvent::Tokens { .. }
        | SessionEvent::Sleep { .. }
        | SessionEvent::ContinuityGap { .. } => {}
    };

    for chunk in pcm.chunks(320) {
        session.push_pcm(chunk, &mut on_ev).expect("push");
    }
    session.finish(&mut on_ev).expect("finish");

    assert!(saw_waking && saw_active && saw_draining && saw_sleep);
    assert_eq!(
        session.ledger().len(),
        2,
        "expected 2 isolated speech segments"
    );

    for (idx, seg) in session.ledger().iter().enumerate() {
        assert_eq!(seg.cause, SegmentCause::Wake);
        assert_eq!(seg.wake_decode_calls, 1, "segment {idx} one-pass replay");
        assert!(
            !seg.eof_flushed,
            "mid-stream sleep close must not flush trailing noise chunk"
        );
        assert!(!seg.tokens.is_empty(), "segment {idx} should emit tokens");
        let isolated = session
            .backend()
            .decode_isolated(&pcm[seg.s0 as usize..seg.e as usize])
            .expect("isolated decode");
        assert_eq!(
            seg.tokens, isolated,
            "segment {idx} must match standalone isolated decode bit-for-bit"
        );
    }

    let stats = session.stats();
    assert_eq!(stats.wake_count, 2);
    assert_eq!(stats.ring.samples_in, pcm.len() as u64);
    assert_eq!(stats.ring.samples_replayed, 16_000);
    assert!(
        stats.recogniser_audio_samples < pcm.len() as u64,
        "sleep gate must reduce recogniser work below 100% duty"
    );
}

#[test]
fn sample_conservation_and_bit_exact_invariance_across_chunk_sizes() {
    let pcm = make_speech_burst_stream();
    let run = |packet: usize| {
        let mut s = AlwaysOnSession::new(MockBackend::default(), AlwaysOnConfig::default())
            .expect("session");
        let mut noop = |_: SessionEvent<'_>| {};
        for pk in pcm.chunks(packet) {
            s.push_pcm(pk, &mut noop).expect("push");
        }
        s.finish(&mut noop).expect("finish");
        (s.ledger().to_vec(), s.stats())
    };

    let (ref_ledger, ref_stats) = run(160);
    assert_eq!(ref_stats.ring.samples_in, pcm.len() as u64);
    assert_eq!(
        ref_stats.ring.samples_in,
        ref_stats.ring.samples_dropped + DEFAULT_RING_SAMPLES as u64
    );

    for packet in [73, 80, 320, 592, 640, 1024] {
        let (ledger, stats) = run(packet);
        assert_eq!(ledger, ref_ledger, "ledger diverged for packet {packet}");
        assert_eq!(
            stats.ring, ref_stats.ring,
            "ring counters diverged for packet {packet}"
        );
        assert_eq!(
            stats.recogniser_audio_samples, ref_stats.recogniser_audio_samples,
            "recogniser audio diverged for packet {packet}"
        );
    }
}

#[test]
fn recovery_state_isolates_backend_fault_and_recovers_cleanly() {
    let pcm = make_speech_burst_stream();
    let backend = MockBackend::default().with_fault_on_call(1);
    let mut session = AlwaysOnSession::new(backend, AlwaysOnConfig::default()).expect("session");
    let mut noop = |_: SessionEvent<'_>| {};

    let mut hit_err = false;
    for chunk in pcm.chunks(320) {
        if let Err(err) = session.push_pcm(chunk, &mut noop) {
            assert!(matches!(err, AlwaysOnError::BackendFault { .. }));
            hit_err = true;
            break;
        }
    }
    assert!(hit_err, "faulty backend call must surface error");
    assert_eq!(session.state(), LadderState::Recovery);
    assert_eq!(session.stats().recovery_count, 1);

    // Further pushes fail with NeedsRecovery until recover() is called.
    assert_eq!(
        session.push_pcm(&[0i16; 160], &mut noop),
        Err(AlwaysOnError::NeedsRecovery)
    );
    session.recover(&mut noop);
    assert_eq!(session.state(), LadderState::Listening { depth: 1 });
    session
        .push_pcm(&[0i16; 320], &mut noop)
        .expect("recovered push");
}

#[test]
fn ungated_mode_replays_at_close_and_matches_isolated_segments() {
    let pcm = make_speech_burst_stream();
    let cfg = AlwaysOnConfig {
        gate_enabled: false,
        ..AlwaysOnConfig::default()
    };
    let mut session = AlwaysOnSession::new(MockBackend::default(), cfg).expect("session");
    assert_eq!(session.state(), LadderState::Active);
    let mut noop = |_: SessionEvent<'_>| {};
    for chunk in pcm.chunks(320) {
        session.push_pcm(chunk, &mut noop).expect("push");
    }
    session.finish(&mut noop).expect("finish");

    assert!(session.stats().hot_at_close_count >= 1);
    assert_eq!(session.stats().listening_hops, 0);
    for (idx, seg) in session.ledger().iter().enumerate() {
        let isolated = session
            .backend()
            .decode_isolated(&pcm[seg.s0 as usize..seg.e as usize])
            .expect("isolated");
        assert_eq!(seg.tokens, isolated, "ungated segment {idx} mismatch");
    }
}
