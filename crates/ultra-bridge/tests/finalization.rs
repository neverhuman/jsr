//! Independent FC diagnostics for the trained CPU benchmark. Synthetic clocks/audio only.
//! Run with `cargo test -p ultra-bridge --test finalization`.

use ultra_eval::finalize::{episodes_from_finals, finalization_report, FinalObs, RefEpisode};
use ultra_eval::stream::{replay, ScriptedStream};

fn reference(start: u64, end: u64) -> RefEpisode {
    RefEpisode {
        id: "episode".into(),
        speech_start_sample: start,
        speech_end_sample: end,
        ref_end_s: end as f64 / 16000.0,
    }
}

fn closure(id: u64, start: u64, end: u64, time: f64) -> FinalObs {
    FinalObs {
        segment_id: id,
        owned_start_sample: start,
        owned_end_sample: end,
        t_s: time,
    }
}

#[test]
fn forced_segments_jointly_close_episode_and_duplicate_delivery_adds_no_coverage() {
    let mut stream = ScriptedStream::new("review", 16000);
    let first = stream.finalize(1.2, "one");
    let mut retry = first.clone();
    retry.t_s = 1.4;
    let second = stream.finalize(2.4, "two");
    let scored = replay(&[first, retry, second]).expect("valid event stream");
    assert_eq!(scored.finals.len(), 2);
    let eps = episodes_from_finals(&[reference(0, 32000)], &scored.finals).expect("valid closures");
    assert_eq!(eps[0].final_t_s, Some(2.4));
    let report = finalization_report(&eps, &[1500]);
    assert_eq!(report.fc[0].successes, 1);
}

#[test]
fn gap_closed_interval_and_unmatched_episode_stay_in_denominator() {
    let mut stream = ScriptedStream::new("review", 16000);
    let preview = stream.preview(0.5, "lost");
    let gap = stream.gap(1.0, Some((0, 16000)));
    let complete = stream.finalize(2.4, "next");
    let scored = replay(&[preview, gap, complete]).expect("valid discontinuous stream");
    let refs = [
        reference(0, 32000),
        reference(16000, 32000),
        reference(40000, 48000),
    ];
    let eps = episodes_from_finals(&refs, &scored.finals).expect("valid reference join");
    assert!(eps[0].final_t_s.is_none());
    assert_eq!(eps[1].final_t_s, Some(2.4));
    assert!(eps[2].final_t_s.is_none());
    let report = finalization_report(&eps, &[1500]);
    assert_eq!(report.episodes, 3);
    assert_eq!(report.fc[0].fc, Some(1.0 / 3.0));
}

#[test]
fn later_valid_final_cannot_hide_premature_ownership() {
    let eps = episodes_from_finals(
        &[reference(0, 32000)],
        &[closure(0, 0, 16000, 0.1), closure(1, 16000, 32000, 2.5)],
    )
    .expect("structurally valid but premature closures");
    assert!(eps[0].premature);
    assert!(eps[0].final_t_s.is_none());
    assert_eq!(finalization_report(&eps, &[1500]).fc[0].successes, 0);
}

#[test]
fn malformed_and_conflicting_raw_closures_are_instrument_errors() {
    let refs = [reference(0, 32000)];
    for finals in [
        vec![
            closure(0, 0, 16000, f64::NAN),
            closure(1, 16000, 32000, 2.5),
        ],
        vec![closure(0, 0, 16000, 1.2), closure(0, 16000, 32000, 2.5)],
        vec![closure(0, 0, 20000, 1.5), closure(1, 16000, 32000, 2.5)],
    ] {
        assert!(episodes_from_finals(&refs, &finals).is_err());
    }
}
