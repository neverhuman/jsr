//! Actual DEV reference join, separate from inference and human alignment verification.
//! Run explicitly: cargo test -p ultra-bridge --test timing_join -- --ignored --nocapture
use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::Deserialize;
use ultra_bridge::records::hash_file;
use ultra_eval::delay::normalise_timed;
use ultra_eval::normalise::normalise_words;

#[derive(Deserialize)]
struct Timing {
    id: String,
    words: Vec<Word>,
}
#[derive(Deserialize)]
struct Word {
    w: String,
    start_s: f64,
    end_s: f64,
}

#[test]
#[ignore = "requires the pinned M1 DEV400 score records and acquired MFA DEV references"]
fn pinned_dev_panel_preserves_every_rust_normalised_reference_word() {
    let root = PathBuf::from(
        std::env::var("JSR_DATA").unwrap_or_else(|_| "/mnt/xbabe3-external/jsr".into()),
    );
    let scores = root.join("runs/bridge/m1-corrected-r0-dev400/scores.jsonl");
    let timing = root.join("corpora/librispeech/word_timing/dev-clean.jsonl");
    assert_eq!(
        hash_file(&scores).expect("frozen Rust DEV records"),
        "88caf9c1cc0a977e9a903c8769f9cc63c1c5d4c46a059da813f9a4c710beb41e"
    );
    assert_eq!(
        hash_file(&timing).expect("pinned MFA DEV timings"),
        "af8eb43d2c1aeeb46aab64b73bbb8a6bcbb808f586488ddb9f24067c8a29dd1a"
    );
    let records =
        ultra_eval::score::parse_jsonl(&std::fs::read_to_string(scores).expect("records"))
            .expect("valid score records");
    let mut timings = BTreeMap::new();
    for line in std::fs::read_to_string(timing).expect("timings").lines() {
        let row: Timing = serde_json::from_str(line).expect("MFA timing row");
        assert!(
            timings.insert(row.id, row.words).is_none(),
            "duplicate timing ID"
        );
    }
    let (mut raw_words, mut normalised_words) = (0, 0);
    for record in &records {
        assert_eq!(record.dataset, "dev-clean");
        let words = timings
            .get(&record.id)
            .expect("every expected DEV ID timed");
        let duration = record.duration_s.expect("source audio duration");
        let mut previous_end = 0.0;
        for w in words {
            assert!(w.start_s.is_finite() && w.end_s.is_finite());
            assert!(w.start_s >= previous_end && w.end_s >= w.start_s);
            assert!(w.end_s <= duration + 0.001);
            previous_end = w.end_s;
        }
        let timed = normalise_timed(&words.iter().map(|w| (&w.w, w.end_s)).collect::<Vec<_>>());
        let reference = normalise_words(&record.reference);
        assert_eq!(
            timed.iter().map(|w| &w.word).collect::<Vec<_>>(),
            reference.iter().collect::<Vec<_>>(),
            "timing/reference word join differs for {}",
            record.id
        );
        raw_words += words.len();
        normalised_words += timed.len();
    }
    assert_eq!((records.len(), normalised_words), (400, 7715));
    println!("DEV timing join: {} inputs; {raw_words} raw timed words -> {normalised_words} Rust reference words; no dropped input", records.len());
}
