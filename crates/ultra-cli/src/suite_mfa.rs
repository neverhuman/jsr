//! MFA `ultra.word-timing/v1` post-inference loader and strict validator (`duplicate ID`,
//! `non-finite/reversed timestamps`, `audio-end bound`, and `normalise_words` identity).

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;
use ultra_core::receipt::sha256_hex;
use ultra_eval::delay::{normalise_timed, RefWord};
use ultra_eval::normalise::normalise_words;
use ultra_frontend::npz::RefUtterance;

use crate::{jsr_data_root, CliError};

#[derive(Deserialize)]
struct MfaRow {
    id: String,
    words: Vec<MfaWord>,
}

#[derive(Deserialize)]
struct MfaWord {
    w: String,
    start_s: f64,
    end_s: f64,
}

pub(crate) struct MfaUtterance {
    pub timed_words: Vec<RefWord>,
    pub start_s: f64,
    pub end_s: f64,
}

pub(crate) fn parse_and_validate_mfa(
    jsonl_txt: &str,
    expected_refs: &BTreeMap<String, (&RefUtterance, f64)>,
    src_label: &str,
) -> Result<BTreeMap<String, MfaUtterance>, CliError> {
    let mut seen_ids = BTreeSet::new();
    let mut map = BTreeMap::new();
    for (line_no, line) in jsonl_txt.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: MfaRow = serde_json::from_str(line).map_err(|e| {
            CliError::Config(format!(
                "malformed MFA timing JSON at {src_label}:{}: {e}",
                line_no + 1
            ))
        })?;
        if !seen_ids.insert(row.id.clone()) {
            return Err(CliError::Config(format!(
                "duplicate MFA utterance id `{}` in {src_label}",
                row.id
            )));
        }
        if row.words.is_empty() {
            continue;
        }
        let mut prev_start = 0.0f64;
        for w in &row.words {
            if !w.start_s.is_finite()
                || !w.end_s.is_finite()
                || w.start_s < 0.0
                || w.end_s < w.start_s
                || w.start_s + 1e-9 < prev_start
            {
                return Err(CliError::Config(format!(
                    "non-finite, reversed, or out-of-order MFA timestamp in {src_label} for `{}`",
                    row.id
                )));
            }
            prev_start = w.start_s;
        }
        let start_s = row.words.first().map_or(0.0, |w| w.start_s);
        let end_s = row.words.last().map_or(0.0, |w| w.end_s);
        let pairs: Vec<(String, f64)> = row.words.into_iter().map(|w| (w.w, w.end_s)).collect();
        let timed_words = normalise_timed(&pairs);

        if let Some(&(r, dur_s)) = expected_refs.get(&row.id) {
            if end_s > dur_s + 0.05 {
                return Err(CliError::Config(format!(
                    "MFA end_s {end_s:.3} exceeds audio duration {dur_s:.3}s for `{}`",
                    row.id
                )));
            }
            let ref_norm = normalise_words(&r.text);
            let mfa_norm: Vec<&str> = timed_words.iter().map(|tw| tw.word.as_str()).collect();
            let ref_strs: Vec<&str> = ref_norm.iter().map(String::as_str).collect();
            if mfa_norm != ref_strs {
                return Err(CliError::Config(format!(
                    "MFA normalized words mismatch reference for `{}` (mfa={} vs ref={})",
                    row.id,
                    mfa_norm.len(),
                    ref_strs.len()
                )));
            }
        }
        if !timed_words.is_empty() {
            map.insert(
                row.id,
                MfaUtterance {
                    timed_words,
                    start_s,
                    end_s,
                },
            );
        }
    }
    Ok(map)
}

pub(crate) fn load_mfa_post_inference(
    split_name: &str,
    refs: &[RefUtterance],
    durs_s: &[f64],
) -> Result<(Option<String>, BTreeMap<String, MfaUtterance>), CliError> {
    let path = std::path::PathBuf::from(jsr_data_root())
        .join("corpora/librispeech/word_timing")
        .join(format!("{split_name}.jsonl"));
    if !path.is_file() {
        return Ok((None, BTreeMap::new()));
    }
    let bytes = std::fs::read(&path)?;
    let sha = sha256_hex(&bytes);
    let txt = std::str::from_utf8(&bytes)
        .map_err(|_| CliError::Config(format!("non-UTF8 MFA file {}", path.display())))?;
    let expected: BTreeMap<String, (&RefUtterance, f64)> = refs
        .iter()
        .zip(durs_s)
        .map(|(r, &d)| (r.uid.clone(), (r, d)))
        .collect();
    let map = parse_and_validate_mfa(txt, &expected, &path.display().to_string())?;
    Ok((Some(sha), map))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mfa_validator_rejects_duplicates_out_of_bounds_and_word_mismatches() {
        let r = RefUtterance {
            index: 0,
            uid: "1-2-0000".into(),
            speaker: 1,
            text: "HELLO WORLD".into(),
            duration_s: 2.0,
        };
        let mut expected = BTreeMap::new();
        expected.insert("1-2-0000".to_owned(), (&r, 2.0));

        let ok = r#"{"id":"1-2-0000","words":[{"w":"hello","start_s":0.1,"end_s":0.5},{"w":"world","start_s":0.6,"end_s":1.2}]}"#;
        let parsed = parse_and_validate_mfa(ok, &expected, "test").expect("valid mfa");
        assert_eq!(parsed["1-2-0000"].timed_words.len(), 2);

        let dup = format!("{ok}\n{ok}\n");
        assert!(parse_and_validate_mfa(&dup, &expected, "test").is_err());

        let oob = r#"{"id":"1-2-0000","words":[{"w":"hello","start_s":0.1,"end_s":0.5},{"w":"world","start_s":0.6,"end_s":2.5}]}"#;
        assert!(parse_and_validate_mfa(oob, &expected, "test").is_err());

        let mismatch = r#"{"id":"1-2-0000","words":[{"w":"hello","start_s":0.1,"end_s":0.5}]}"#;
        assert!(parse_and_validate_mfa(mismatch, &expected, "test").is_err());
    }
}
