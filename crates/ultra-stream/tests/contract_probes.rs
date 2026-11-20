//! Integration tests for W00 instrument verification, Spec §38.1 transport/session/causality
//! contracts, and Spec §38.3 failure probes.

use ultra_core::events::{Body, Emitter, ProcessingEvidence, SegmentStatus, TranscriptEvent};
use ultra_eval::delay::{delay_report, normalise_timed, word_delays, CQ_DEADLINES_MS};
use ultra_eval::normalise::normalise_words;
use ultra_eval::pace::PacedReplay;
use ultra_eval::stream::{replay, TimedEvent};
use ultra_eval::wer::align;
use ultra_frontend::mel::{MelConfig, StreamingLogMel};
use ultra_stream::{
    AsrStream, Backend, DeliveryFilter, FakeBackend, FakeBackendConfig, FakeWord, OutputPolicy,
    PollResult, StreamConfig, StreamCoordinator, TerminalOutcome, UninterruptibleWorkerSim,
};

fn collect_all_events(coord: &mut StreamCoordinator) -> Vec<TranscriptEvent> {
    let mut out = Vec::new();
    while let PollResult::Event(ev) = coord.poll_event().expect("poll") {
        out.push(*ev);
    }
    out
}

#[test]
fn transport_packetization_10_20_37ms_produces_identical_events() {
    let words = vec![
        FakeWord {
            emit_at_sample: 3_200,
            word: "go".into(),
            difficult: false,
        },
        FakeWord {
            emit_at_sample: 6_400,
            word: "go".into(),
            difficult: false,
        },
        FakeWord {
            emit_at_sample: 14_400,
            word: "ultra".into(),
            difficult: true,
        },
    ];
    let backend = FakeBackend::new(FakeBackendConfig {
        chunk_samples: 1_600,
        words,
        omit_difficult_words: false,
        energy_word_threshold: None,
    });
    let pcm = vec![42i16; 16_000 + 333];

    for policy in [OutputPolicy::EditableDictation, OutputPolicy::Commit] {
        let mut by_pkt = Vec::new();
        for ms in [10u32, 20, 37] {
            let cfg = StreamConfig {
                stream_id: "s0".into(),
                policy,
                ..StreamConfig::default()
            };
            let mut coord =
                StreamCoordinator::new(cfg.clone(), backend.create_stream(&cfg).expect("stream"));
            let mut events = Vec::new();
            for pkt in PacedReplay::new(&pcm, ms, 0.0).expect("replay") {
                coord.push_pcm(pkt.start_sample, pkt.pcm).expect("push");
                events.extend(collect_all_events(&mut coord));
            }
            coord.finish_input().expect("finish");
            events.extend(collect_all_events(&mut coord));
            assert_eq!(
                coord.poll_event().expect("ended"),
                PollResult::Ended(TerminalOutcome::Finished)
            );
            assert_eq!(coord.ledger().committed_text(), "go go ultra");
            by_pkt.push(events);
        }
        assert_eq!(by_pkt[0], by_pkt[1]);
        assert_eq!(by_pkt[1], by_pkt[2]);
    }
}

#[test]
fn probe_acoustic_causality_future_audio_never_alters_earlier_events() {
    // Spec §38.1 Acoustic Causality: mutating future audio strictly after horizon H (16_000 samples
    // = 1.0 s) must leave all log-mel frames and all emitted TranscriptEvents up to H bit-identical!
    let horizon = 16_000usize;
    let mut pcm_a = vec![0i16; 32_000];
    for (i, s) in pcm_a[..horizon].iter_mut().enumerate() {
        *s = if (i / 1_600) % 2 == 0 { 8_000 } else { 100 };
    }
    let mut pcm_b = pcm_a.clone();
    // Mutate everything strictly after `horizon` in arm B:
    for s in &mut pcm_b[horizon..] {
        *s = 24_000;
    }

    // 1. Frontend log-mel causality:
    let mut mel_a = StreamingLogMel::new(MelConfig::default());
    let mut mel_b = StreamingLogMel::new(MelConfig::default());
    let frames_a = mel_a.push(&pcm_a[..horizon]);
    let frames_b = mel_b.push(&pcm_b[..horizon]);
    assert!(!frames_a.is_empty());
    assert_eq!(frames_a, frames_b);

    // 2. Coordinator + causal energy backend causality before vs after horizon:
    let backend = FakeBackend::new(FakeBackendConfig {
        chunk_samples: 1_600,
        words: Vec::new(),
        omit_difficult_words: false,
        energy_word_threshold: Some(2_000),
    });
    let run_prefix_and_tail = |pcm: &[i16]| {
        let cfg = StreamConfig {
            stream_id: "causal-0".into(),
            policy: OutputPolicy::Commit,
            ..StreamConfig::default()
        };
        let mut c = StreamCoordinator::new(cfg.clone(), backend.create_stream(&cfg).expect("st"));
        let mut pre = Vec::new();
        let mut post = Vec::new();
        for pkt in PacedReplay::new(pcm, 20, 0.0).expect("replay") {
            let end_sample = (pkt.start_sample as usize) + pkt.pcm.len();
            c.push_pcm(pkt.start_sample, pkt.pcm).expect("push");
            let evs = collect_all_events(&mut c);
            if end_sample <= horizon {
                pre.extend(evs);
            } else {
                post.extend(evs);
            }
        }
        c.finish_input().expect("finish");
        post.extend(collect_all_events(&mut c));
        (pre, post)
    };
    let (pre_a, post_a) = run_prefix_and_tail(&pcm_a);
    let (pre_b, post_b) = run_prefix_and_tail(&pcm_b);
    assert!(!pre_a.is_empty());
    assert_eq!(pre_a, pre_b, "future audio leaked into pre-horizon events!");
    assert_ne!(
        post_a, post_b,
        "post-horizon events must reflect tail mutation"
    );
}

#[test]
fn probe_omitting_difficult_words_improves_conditional_delay_but_worsens_cq_and_wer() {
    let words = vec![
        FakeWord {
            emit_at_sample: 4_000,
            word: "easy".into(),
            difficult: false,
        },
        FakeWord {
            emit_at_sample: 19_200,
            word: "electroencephalogram".into(),
            difficult: true,
        },
    ];
    let pcm = vec![0i16; 24_000];
    let ref_timed = normalise_timed(&[("easy", 0.20), ("electroencephalogram", 0.60)]);
    let ref_words = normalise_words("easy electroencephalogram");

    let run_arm = |omit: bool| {
        let b = FakeBackend::new(FakeBackendConfig {
            chunk_samples: 800,
            words: words.clone(),
            omit_difficult_words: omit,
            energy_word_threshold: None,
        });
        let cfg = StreamConfig::default();
        let mut c = StreamCoordinator::new(cfg.clone(), b.create_stream(&cfg).expect("stream"));
        let mut timed = Vec::new();
        for pkt in PacedReplay::new(&pcm, 20, 0.0).expect("replay") {
            let end_s = (pkt.start_sample + pkt.pcm.len() as u64) as f64 / 16_000.0;
            c.push_pcm(pkt.start_sample, pkt.pcm).expect("push");
            for ev in collect_all_events(&mut c) {
                timed.push(TimedEvent {
                    t_s: end_s,
                    event: ev,
                });
            }
        }
        c.finish_input().expect("finish");
        for ev in collect_all_events(&mut c) {
            timed.push(TimedEvent {
                t_s: 1.50,
                event: ev,
            });
        }
        let ss = replay(&timed).expect("replay score");
        let wd = word_delays(&ref_timed, &ss.words, &ss.stable_t_s);
        let dr = delay_report(&wd, &CQ_DEADLINES_MS);
        let w = align(&ref_words, &ss.words).counts;
        (dr, w)
    };

    let (honest_dr, honest_wer) = run_arm(false);
    let (cheating_dr, cheating_wer) = run_arm(true);
    assert!(cheating_dr.p95_s.expect("p95") < honest_dr.p95_s.expect("p95"));
    let cq750 = |dr: &ultra_eval::delay::DelayReport| {
        dr.cq
            .iter()
            .find(|p| p.deadline_ms == 750)
            .expect("750")
            .cq
            .expect("cq")
    };
    assert!((cq750(&honest_dr) - 1.0).abs() < 1e-9);
    assert!((cq750(&cheating_dr) - 0.5).abs() < 1e-9);
    assert_eq!(honest_wer.errors(), 0);
    assert_eq!(cheating_wer.deletions, 1);
}

#[test]
fn probe_shifting_scorer_reference_times_never_changes_runtime_events() {
    let b = FakeBackend::new(FakeBackendConfig {
        chunk_samples: 1_600,
        words: vec![FakeWord {
            emit_at_sample: 8_000,
            word: "hello".into(),
            difficult: false,
        }],
        omit_difficult_words: false,
        energy_word_threshold: None,
    });
    let cfg = StreamConfig::default();
    let mut c = StreamCoordinator::new(cfg.clone(), b.create_stream(&cfg).expect("stream"));
    let pcm = vec![1i16; 16_000];
    let mut timed = Vec::new();
    for pkt in PacedReplay::new(&pcm, 20, 0.0).expect("replay") {
        let t_s = (pkt.start_sample + pkt.pcm.len() as u64) as f64 / 16_000.0;
        c.push_pcm(pkt.start_sample, pkt.pcm).expect("push");
        for ev in collect_all_events(&mut c) {
            timed.push(TimedEvent { t_s, event: ev });
        }
    }
    c.finish_input().expect("finish");
    for ev in collect_all_events(&mut c) {
        timed.push(TimedEvent {
            t_s: 1.0,
            event: ev,
        });
    }

    let ss = replay(&timed).expect("stream replay");
    let ref_early = normalise_timed(&[("hello", 0.20)]);
    let ref_late = normalise_timed(&[("hello", 0.40)]);
    let d1 = delay_report(
        &word_delays(&ref_early, &ss.words, &ss.stable_t_s),
        &CQ_DEADLINES_MS,
    );
    let d2 = delay_report(
        &word_delays(&ref_late, &ss.words, &ss.stable_t_s),
        &CQ_DEADLINES_MS,
    );
    assert_eq!(ss.final_text, "hello");
    assert!((d1.p50_s.expect("d1") - 0.30).abs() < 1e-6);
    assert!((d2.p50_s.expect("d2") - 0.10).abs() < 1e-6);
}

#[test]
fn probe_endpoint_hangover_exhausts_causal_budget_and_late_refiner_is_rejected_yet_charged() {
    let b = FakeBackend::new(FakeBackendConfig {
        chunk_samples: 1_600,
        words: vec![FakeWord {
            emit_at_sample: 1_600,
            word: "draft".into(),
            difficult: false,
        }],
        omit_difficult_words: false,
        energy_word_threshold: None,
    });
    let cfg = StreamConfig {
        final_budget_ms: 400,
        endpoint_hangover_ms: 400,
        ..StreamConfig::default()
    };
    let mut c = StreamCoordinator::new(cfg.clone(), b.create_stream(&cfg).expect("stream"));
    c.push_pcm(0, &[0i16; 1_600]).expect("push");
    let _ = collect_all_events(&mut c);

    c.mark_observable_pause_start(100);
    assert_eq!(c.remaining_refinement_budget_ms(500), 0);
    assert!(c.try_begin_refiner(500, 50).is_none());

    let ticket = c.try_begin_refiner(200, 100).expect("ticket admitted");
    c.push_pcm(1_600, &[1i16; 1_600]).expect("new speech");
    let accepted = c
        .apply_refiner_result(&ticket, 280, "refined text".into(), 0.125, 8 * 1024 * 1024)
        .expect("compare-and-apply");
    assert!(!accepted);
    let m = c.metrics();
    assert_eq!(m.refiner_attempts, 1);
    assert_eq!(m.refiner_rejects, 1);
    assert!((m.refiner_charged_cpu_s - 0.125).abs() < 1e-9);
    assert!(m.retained_bytes >= 8 * 1024 * 1024);
}

#[test]
fn probe_replay_dedup_multibyte_utf8_gap_and_uninterruptible_cancel_and_1h_session() {
    let mut filter = DeliveryFilter::new();
    let mut em = Emitter::new("utf8-s", "model-v1");
    let ev_commit = em
        .commit(
            100_000_000,
            "café naïve 東京",
            "café naïve 東京".len(),
            1_600,
        )
        .expect("emit utf8 commit");
    assert!(filter.check_and_record(&ev_commit).expect("first apply"));
    assert!(!filter
        .check_and_record(&ev_commit)
        .expect("identical replay is no-op"));

    let mut em2 = Emitter::new("utf8-s", "model-v1");
    let tampered = em2
        .finalize(
            100_000_000,
            "café altered",
            1_600,
            ProcessingEvidence {
                accepted_through_sample: 1_600,
                required_processed_through_sample: 1_600,
                flushed: true,
            },
        )
        .expect("tampered same seq");
    assert!(filter.check_and_record(&tampered).is_err());

    // Gap preserves committed prefix and closes segment Incomplete; cancel never emits false Final.
    let b = FakeBackend::new(FakeBackendConfig {
        chunk_samples: 1_600,
        words: vec![
            FakeWord {
                emit_at_sample: 1_600,
                word: "kept".into(),
                difficult: false,
            },
            FakeWord {
                emit_at_sample: 4_800,
                word: "later".into(),
                difficult: false,
            },
        ],
        omit_difficult_words: false,
        energy_word_threshold: None,
    });
    let cfg = StreamConfig {
        policy: OutputPolicy::Commit,
        ..StreamConfig::default()
    };
    let mut c = StreamCoordinator::new(cfg.clone(), b.create_stream(&cfg).expect("stream"));
    c.push_pcm(0, &[0i16; 1_600]).expect("push");
    let _ = collect_all_events(&mut c);
    assert_eq!(c.ledger().committed_text(), "kept");

    c.push_pcm(3_200, &[0i16; 1_600]).expect("gap push");
    let gap_evs = collect_all_events(&mut c);
    assert!(gap_evs.iter().any(|e| matches!(
        e.body,
        Body::Gap {
            start_sample: Some(1_600),
            end_sample: Some(3_200),
            ..
        }
    )));
    assert_eq!(
        c.ledger().segment(0).expect("seg0").status,
        SegmentStatus::Incomplete
    );
    assert_eq!(c.ledger().committed_text(), "kept");

    // Push more audio so segment 1 opens, then cancel -> SegmentIncomplete, never Final!
    c.push_pcm(4_800, &[0i16; 1_600]).expect("post-gap push");
    let _ = collect_all_events(&mut c);
    c.cancel().expect("cancel");
    let cancel_evs = collect_all_events(&mut c);
    assert!(!cancel_evs
        .iter()
        .any(|e| matches!(e.body, Body::Final { .. })));
    assert!(cancel_evs
        .iter()
        .any(|e| matches!(e.body, Body::SegmentIncomplete { .. })));

    let mut pool = UninterruptibleWorkerSim::new(1, 512 * 1024 * 1024);
    pool.spawn_worker().expect("spawn 1");
    pool.cancel_active();
    assert_eq!(pool.reserved_bytes(), 512 * 1024 * 1024);
    assert!(pool.spawn_worker().is_err());
    pool.reap_finished_cancelled_call(0.42);
    assert_eq!(pool.reserved_bytes(), 0);
    assert!((pool.cpu_charged_s() - 0.42).abs() < 1e-9);
    pool.spawn_worker().expect("slot freed after reap");

    let b_long = FakeBackend::new(FakeBackendConfig {
        chunk_samples: 16_000,
        words: Vec::new(),
        omit_difficult_words: false,
        energy_word_threshold: None,
    });
    let cfg_long = StreamConfig::default();
    let mut c_long = StreamCoordinator::new(
        cfg_long.clone(),
        b_long.create_stream(&cfg_long).expect("stream"),
    );
    let one_sec = [0i16; 16_000];
    for sec in 0..3_600u64 {
        c_long.push_pcm(sec * 16_000, &one_sec).expect("1h push");
        let _ = collect_all_events(&mut c_long);
    }
    let m_long = c_long.metrics();
    assert_eq!(m_long.accepted_samples, 3_600 * 16_000);
    assert_eq!(m_long.cap_hits, 120);
    assert!(m_long.retained_bytes <= 32_000);
}
