//! Event stream -> stable times -> CQ(D), end to end (v1.0-B §12.6 scorer checks:
//! revisions, insertions, repetitions).

use ultra_core::events::{EventKind, TranscriptEvent};
use ultra_eval::delay::{delay_report, normalise_timed, word_delays};
use ultra_eval::stream::{replay, TimedEvent};
use ultra_eval::wer::align_text;

fn ev(t: f64, seq: u64, seg: u64, kind: EventKind) -> TimedEvent {
    TimedEvent {
        t_s: t,
        event: TranscriptEvent {
            stream_id: "s".into(),
            segment_id: seg,
            event_sequence: seq,
            audio_start_sample: 0,
            audio_end_sample: 1,
            kind,
        },
    }
}
fn pv(rev: u64, s: &str) -> EventKind {
    EventKind::Preview {
        revision_id: rev,
        replacement_suffix: s.into(),
    }
}

#[test]
fn editable_and_commit_cq_differ_and_errors_stay_in_denominator() {
    // Reference: "go go home now" with word ends 0.5, 1.0, 1.5, 2.0 s.
    let r = normalise_timed(&[("go", 0.5), ("go", 1.0), ("home", 1.5), ("now", 2.0)]);
    let evs = vec![
        ev(0.6, 1, 0, pv(1, "go")),
        ev(1.1, 2, 0, pv(2, "go go")),
        ev(1.7, 3, 0, pv(3, "go go hum")), // wrong, revised later
        ev(
            1.9,
            4,
            0,
            EventKind::Commit {
                text: "go go ".into(),
            },
        ),
        ev(2.2, 5, 0, pv(4, "home uh")), // insertion "uh" survives; "now" never output
        ev(
            2.6,
            6,
            0,
            EventKind::Final {
                text: "home uh".into(),
            },
        ),
    ];
    let s = replay(&evs).unwrap();
    assert_eq!(s.commit_text, "go go home uh");
    assert_eq!(s.words, vec!["go", "go", "home", "uh"]);
    // The 0.6 s "go" could be either final "go": both get the later 1.1 s.
    assert_eq!(s.stable_t_s, vec![1.1, 1.1, 2.2, 2.2]);
    assert_eq!(s.commit_t_s, vec![1.9, 1.9, 2.6, 2.6]);

    let a = align_text("go go home now", &s.final_text);
    assert_eq!(
        (
            a.counts.substitutions,
            a.counts.deletions,
            a.counts.insertions
        ),
        (1, 0, 0)
    );

    let editable = word_delays(&r, &s.words, &s.stable_t_s);
    let commit = word_delays(&r, &s.words, &s.commit_t_s);
    let e = delay_report(&editable, &[250, 750]);
    let c = delay_report(&commit, &[250, 750, 1500]);
    // "now" is substituted by "uh": 3 of 4 reference words matched in both views.
    assert_eq!(e.matched_words, 3);
    assert_eq!(e.ref_words, 4);
    // Editable delays: 0.6, 0.1, 0.7 -> CQ(250) = 1/4, CQ(750) = 3/4.
    assert_eq!(e.cq[0].cq, Some(0.25));
    assert_eq!(e.cq[1].cq, Some(0.75));
    // Commit delays: 1.4, 0.9, 1.1 -> CQ(250) = 0, CQ(750) = 0, CQ(1500) = 3/4.
    assert_eq!(c.cq[0].cq, Some(0.0));
    assert_eq!(c.cq[1].cq, Some(0.0));
    assert_eq!(c.cq[2].cq, Some(0.75));
}
