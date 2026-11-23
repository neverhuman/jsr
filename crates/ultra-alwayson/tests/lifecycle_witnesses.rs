//! Regression tests for the 3 `@codex-lead` (06:41:24Z) exact-source lifecycle witnesses:
//! 1. Idempotent `finish()` + `AlreadyFinished` rejection on push-after-EOF (gated & ungated).
//! 2. Mid-call `push_pcm` backend fault continuity (500-sample call failing on call 2 -> exact
//!    `ContinuityGap` + post-`recover()` segment metadata at exact capture `[500, 660)`).
//! 3. Short 60 ms (`960`-sample) audible onset finishing before Sentinel-0 warm-up completes.

use ultra_alwayson::{
    AlwaysOnConfig, AlwaysOnError, AlwaysOnSession, AsrBackend, LadderState, MockBackend,
    SegmentCause, SessionEvent,
};

fn loud_square_pcm(len: usize) -> Vec<i16> {
    (0..len)
        .map(|i| if (i / 20) % 2 == 0 { 3_200 } else { -3_200 })
        .collect()
}

#[test]
fn witness1_idempotent_finish_and_push_after_eof_rejected_ungated_and_gated() {
    let pcm = loud_square_pcm(640);
    let mut noop = |_: SessionEvent<'_>| {};

    // Ungated: repeated finish() must emit exactly 1 segment over [0, 640), never duplicate.
    let ungated_cfg = AlwaysOnConfig {
        gate_enabled: false,
        ..AlwaysOnConfig::default()
    };
    let mut ungated = AlwaysOnSession::new(MockBackend::default(), ungated_cfg).expect("session");
    ungated.push_pcm(&pcm, &mut noop).expect("push");
    ungated.finish(&mut noop).expect("finish 1");
    ungated.finish(&mut noop).expect("finish 2 idempotent");
    assert!(ungated.is_finished());
    assert_eq!(
        ungated.ledger().len(),
        1,
        "no duplicate segment on 2nd finish()"
    );
    assert_eq!((ungated.ledger()[0].s0, ungated.ledger()[0].e), (0, 640));
    assert_eq!(
        ungated.push_pcm(&[0i16; 160], &mut noop),
        Err(AlwaysOnError::AlreadyFinished)
    );

    // Gated: push after successful EOF must also return AlreadyFinished.
    let mut gated =
        AlwaysOnSession::new(MockBackend::default(), AlwaysOnConfig::default()).expect("session");
    gated.push_pcm(&[0i16; 320], &mut noop).expect("push");
    gated.finish(&mut noop).expect("finish 1");
    gated.finish(&mut noop).expect("finish 2 idempotent");
    assert!(gated.is_finished());
    assert_eq!(
        gated.push_pcm(&[0i16; 160], &mut noop),
        Err(AlwaysOnError::AlreadyFinished)
    );
}

#[test]
fn witness2_mid_call_fault_preserves_exact_capture_coordinates_and_gap() {
    // Exact Codex witness: ungated mode, push 500 samples, fail backend call 2 (hop 1: [160..320)).
    let cfg = AlwaysOnConfig {
        gate_enabled: false,
        ..AlwaysOnConfig::default()
    };
    let backend = MockBackend::default().with_fault_on_call(2);
    let mut session = AlwaysOnSession::new(backend, cfg).expect("session");
    let mut gap_events = Vec::new();
    let first_500 = loud_square_pcm(500);
    let err = {
        let mut on_ev = |ev: SessionEvent<'_>| {
            if let SessionEvent::ContinuityGap {
                start_sample,
                end_sample,
            } = ev
            {
                gap_events.push((start_sample, end_sample));
            }
        };
        session
            .push_pcm(&first_500, &mut on_ev)
            .expect_err("call 2 must fail")
    };
    assert!(
        matches!(
            err,
            AlwaysOnError::BackendFault {
                consumed_samples: 160,
                capture_end: 500,
                ..
            }
        ),
        "unexpected error: {err:?}"
    );
    assert_eq!(session.state(), LadderState::Recovery);
    assert_eq!(gap_events, vec![(160, 500)]);
    assert_eq!(session.continuity_gaps().len(), 1);
    let gap = &session.continuity_gaps()[0];
    assert_eq!((gap.start_sample, gap.end_sample), (160, 500));
    assert_eq!((gap.consumed_in_call, gap.unconsumed_in_call), (160, 340));
    assert_eq!(session.ring_counters().samples_in, 500);
    assert_eq!(session.ring_counters().samples_dropped, 500);

    // Recover and supply the subsequent 160 samples at capture [500, 660).
    let mut noop = |_: SessionEvent<'_>| {};
    session.recover(&mut noop);
    assert_eq!(session.state(), LadderState::Active);
    let next_160 = loud_square_pcm(160);
    session
        .push_pcm(&next_160, &mut noop)
        .expect("post-recover push");
    session.finish(&mut noop).expect("finish");

    assert_eq!(session.ledger().len(), 1);
    let seg = &session.ledger()[0];
    assert_eq!(
        (seg.s0, seg.e),
        (500, 660),
        "post-recovery segment must carry exact capture coordinates [500, 660)"
    );
    assert_eq!(session.ring_counters().samples_in, 660);
}

#[test]
fn witness3_short_60ms_startup_onset_flushes_at_eof_and_matches_isolated() {
    // 60 ms (960 samples = 6 hops < KS-1=7 warm-up hops) audible speech onset.
    let pcm = loud_square_pcm(960);
    let isolated = MockBackend::default()
        .decode_isolated(&pcm)
        .expect("isolated");
    assert!(
        !isolated.is_empty(),
        "60 ms nonzero input must produce at least one token in isolated mock"
    );

    // Default config (flush_short_onset_on_eof = true): never silently omits the short startup.
    let mut session =
        AlwaysOnSession::new(MockBackend::default(), AlwaysOnConfig::default()).expect("session");
    let mut noop = |_: SessionEvent<'_>| {};
    session.push_pcm(&pcm, &mut noop).expect("push 60 ms");
    session.finish(&mut noop).expect("finish");

    assert_eq!(
        session.ledger().len(),
        1,
        "short 60 ms onset must emit 1 segment"
    );
    let seg = &session.ledger()[0];
    assert_eq!(seg.cause, SegmentCause::EofShortOnset);
    assert_eq!((seg.s0, seg.e), (0, 960));
    assert_eq!(
        seg.tokens, isolated,
        "must match isolated decode bit-for-bit"
    );
    assert_eq!(session.stats().short_onset_flushes, 1);
    assert_eq!(session.stats().recogniser_audio_samples, 960);

    // 60 ms of digital silence must STILL make zero engine calls and emit zero segments.
    let mut silent =
        AlwaysOnSession::new(MockBackend::default(), AlwaysOnConfig::default()).expect("session");
    silent
        .push_pcm(&vec![0i16; 960], &mut noop)
        .expect("push silence");
    silent.finish(&mut noop).expect("finish");
    assert!(silent.ledger().is_empty());
    assert_eq!(silent.stats().engine_decode_calls, 0);
    assert_eq!(silent.stats().recogniser_audio_samples, 0);

    // Explicit opt-out (flush_short_onset_on_eof = false) accounts for unflushed samples explicitly.
    let opt_out_cfg = AlwaysOnConfig {
        flush_short_onset_on_eof: false,
        ..AlwaysOnConfig::default()
    };
    let mut opt_out = AlwaysOnSession::new(MockBackend::default(), opt_out_cfg).expect("session");
    opt_out.push_pcm(&pcm, &mut noop).expect("push");
    opt_out.finish(&mut noop).expect("finish");
    assert!(opt_out.ledger().is_empty());
    assert_eq!(opt_out.stats().unflushed_short_onset_samples, 960);
}

#[test]
fn config_enforces_300ms_no_double_decode_margin() {
    let def = AlwaysOnConfig::default();
    assert_eq!((def.replay_ms, def.idle_ms), (500, 1500));
    assert!(!def.flush_on_sleep_close);
    assert!(def.validate().is_ok());
    assert!(AlwaysOnSession::new(MockBackend::default(), def).is_ok());

    let boundary = AlwaysOnConfig {
        replay_ms: 500,
        idle_ms: 800,
        ..AlwaysOnConfig::default()
    };
    assert!(boundary.validate().is_ok());

    let too_tight = AlwaysOnConfig {
        replay_ms: 500,
        idle_ms: 700,
        ..AlwaysOnConfig::default()
    };
    assert!(matches!(
        too_tight.validate(),
        Err(AlwaysOnError::InvalidConfig(_))
    ));
    assert!(matches!(
        AlwaysOnSession::new(MockBackend::default(), too_tight),
        Err(AlwaysOnError::InvalidConfig(_))
    ));
}

#[test]
fn witness4_hot_close_replay_produces_zero_duplicate_tokens() {
    // Burst 1: [0..1.0 s), non-speech blank gap: [1.0..2.8 s) (> 1.5 s idle triggers HotAtClose
    // with 500 ms replay over [2.0..2.5 s) inside the blank gap), Burst 2: [2.8..3.8 s).
    let total = 4 * 16_000;
    let mut pcm = vec![0i16; total];
    for (i, s) in pcm.iter_mut().enumerate() {
        let sq = if (i / 20) % 2 == 0 { 1 } else { -1 };
        if i < 16_000 {
            *s = 3_200 * sq;
        } else if (44_800..60_800).contains(&i) {
            *s = 3_400 * sq;
        }
    }
    let cfg = AlwaysOnConfig {
        gate_enabled: false,
        ..AlwaysOnConfig::default()
    };
    assert_eq!((cfg.replay_ms, cfg.idle_ms), (500, 1500));
    let mut session = AlwaysOnSession::new(MockBackend::default(), cfg).expect("session");
    let mut noop = |_: SessionEvent<'_>| {};
    for pk in pcm.chunks(320) {
        session.push_pcm(pk, &mut noop).expect("push");
    }
    session.finish(&mut noop).expect("finish");

    assert!(
        session.stats().hot_at_close_count >= 1,
        "must exercise at least one HotAtClose replay boundary"
    );
    assert!(session.ledger().len() >= 2);
    let hot_seg = session
        .ledger()
        .iter()
        .find(|seg| seg.cause == SegmentCause::HotAtClose)
        .expect("HotAtClose segment");
    assert!(
        hot_seg.s0 >= 16_000,
        "500 ms replay window must start strictly after burst 1 (s0={})",
        hot_seg.s0
    );

    // Total emitted tokens across all segments must equal the single-pass isolated decode of
    // the full stream with zero duplicated boundary tokens.
    let combined: Vec<u32> = session
        .ledger()
        .iter()
        .flat_map(|seg| seg.tokens.iter().copied())
        .collect();
    let full_isolated = session
        .backend()
        .decode_isolated(&pcm)
        .expect("isolated full");
    assert_eq!(
        combined, full_isolated,
        "hot-close replay must emit zero duplicate tokens across segment boundary"
    );
}
