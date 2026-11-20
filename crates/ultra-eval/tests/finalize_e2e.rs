//! Event stream -> replay -> ownership-mapped episodes -> FC(D), end to end.

use ultra_eval::finalize::{episodes_from_finals, finalization_report, RefEpisode};
use ultra_eval::stream::{replay, ScriptedStream};

#[test]
fn two_utterances_one_finalized_in_time_one_lost_to_a_gap() {
    // 1600 samples (0.1 s) of ownership per scripted call.
    let mut sc = ScriptedStream::new("s", 1600);
    let evs = vec![
        sc.preview(0.2, "hello"),        // owns 0..1600
        sc.finalize(0.9, "hello world"), // owns 0..3200, Final at 0.9 s
        sc.preview(1.2, "second"),       // next segment owns 3200..4800
        sc.gap(1.5, Some((4800, 9600))), // segment closed incomplete
    ];
    let s = replay(&evs).unwrap();
    assert_eq!(s.finals.len(), 1);
    let refs = [
        RefEpisode {
            id: "u1".into(),
            speech_start_sample: 100,
            speech_end_sample: 3000,
            ref_end_s: 0.1875,
        },
        RefEpisode {
            id: "u2".into(),
            speech_start_sample: 3300,
            speech_end_sample: 4700,
            ref_end_s: 0.29375,
        },
    ];
    let eps = episodes_from_finals(&refs, &s.finals).unwrap();
    assert_eq!(eps[0].final_t_s, Some(0.9));
    assert_eq!(
        eps[1].final_t_s, None,
        "speech owned only by an incomplete segment is a miss"
    );
    let r = finalization_report(&eps, &[800, 1500]);
    assert_eq!((r.episodes, r.finalized), (2, 1));
    assert_eq!(r.fc[0].fc, Some(0.5)); // u1 delay 0.7125 s <= 0.8 s
    assert_eq!(r.fc[1].fc, Some(0.5));
}
