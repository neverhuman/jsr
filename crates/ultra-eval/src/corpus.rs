//! Corpus pooling (v1.0-B §30.1): per-dataset pooled WER, a frozen-weight macro
//! average, and silence (zero-reference-word) utterances scored as insertions per hour,
//! never as a WER.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::wer::EditCounts;
use crate::EvalError;

/// One scored utterance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UtteranceResult {
    pub id: String,
    pub dataset: String,
    pub speaker: String,
    pub counts: EditCounts,
    /// Audio duration; needed for silence insertions per hour.
    pub duration_s: Option<f64>,
}

/// Zero-reference-word utterances of one dataset. WER is undefined for them.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SilenceScore {
    pub utterances: u64,
    pub inserted_words: u64,
    /// Utterances with at least one inserted word (false-transcript episodes).
    pub false_transcript_utterances: u64,
    /// Total duration; `None` if any silence utterance lacks a duration.
    pub seconds: Option<f64>,
    pub insertions_per_hour: Option<f64>,
    pub false_transcripts_per_hour: Option<f64>,
}

/// Pooled score of one dataset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetScore {
    /// Pooled counts over utterances with at least one reference word.
    pub counts: EditCounts,
    pub wer: Option<f64>,
    pub utterances: u64,
    pub speakers: u64,
    pub silence: SilenceScore,
}

/// Pool utterances per dataset: WER = sum(errors) / sum(reference words) over the
/// utterances that have reference words. Zero-reference utterances go to `silence`
/// (inserted words per hour) and never enter a WER denominator or numerator.
pub fn pool(utts: &[UtteranceResult]) -> BTreeMap<String, DatasetScore> {
    let mut acc: BTreeMap<String, (EditCounts, u64, BTreeSet<String>, SilenceAcc)> =
        BTreeMap::new();
    for u in utts {
        let e = acc.entry(u.dataset.clone()).or_default();
        e.1 += 1;
        e.2.insert(u.speaker.clone());
        if u.counts.ref_words == 0 {
            e.3.utterances += 1;
            e.3.inserted += u.counts.insertions;
            e.3.false_tr += u64::from(u.counts.insertions > 0);
            e.3.seconds = match (e.3.seconds, u.duration_s) {
                (Some(a), Some(b)) if b.is_finite() && b >= 0.0 => Some(a + b),
                _ => None,
            };
        } else {
            e.0.add(&u.counts);
        }
    }
    acc.into_iter()
        .map(|(name, (counts, n, spk, s))| {
            let per_hour = |x: u64| match s.seconds {
                Some(sec) if sec > 0.0 => Some(x as f64 * 3600.0 / sec),
                _ => None,
            };
            let silence = SilenceScore {
                utterances: s.utterances,
                inserted_words: s.inserted,
                false_transcript_utterances: s.false_tr,
                seconds: (s.utterances > 0).then_some(s.seconds).flatten(),
                insertions_per_hour: per_hour(s.inserted),
                false_transcripts_per_hour: per_hour(s.false_tr),
            };
            let score = DatasetScore {
                counts,
                wer: counts.wer(),
                utterances: n,
                speakers: spk.len() as u64,
                silence,
            };
            (name, score)
        })
        .collect()
}

#[derive(Debug)]
struct SilenceAcc {
    utterances: u64,
    inserted: u64,
    false_tr: u64,
    seconds: Option<f64>,
}

impl Default for SilenceAcc {
    fn default() -> Self {
        SilenceAcc {
            utterances: 0,
            inserted: 0,
            false_tr: 0,
            seconds: Some(0.0),
        }
    }
}

/// Pooled WER over all datasets (sum of errors / sum of reference words).
pub fn pooled_wer(scores: &BTreeMap<String, DatasetScore>) -> Option<f64> {
    let mut c = EditCounts::default();
    for s in scores.values() {
        c.add(&s.counts);
    }
    c.wer()
}

/// Frozen dataset weights for the macro average (v1.0-B §30.1, §30.4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MacroWeights {
    weights: BTreeMap<String, f64>,
}

impl MacroWeights {
    /// Validate: non-empty, every weight finite and > 0, sum within 1e-9 of 1.
    pub fn new(weights: BTreeMap<String, f64>) -> Result<Self, EvalError> {
        if weights.is_empty() {
            return Err(EvalError::Weights("no datasets".into()));
        }
        if let Some((k, w)) = weights.iter().find(|(_, w)| !(w.is_finite() && **w > 0.0)) {
            return Err(EvalError::Weights(format!("{k} has weight {w}")));
        }
        let sum: f64 = weights.values().sum();
        if (sum - 1.0).abs() > 1e-9 {
            return Err(EvalError::Weights(format!("weights sum to {sum}, not 1")));
        }
        Ok(MacroWeights { weights })
    }

    /// Equal weights over the named datasets.
    pub fn equal<I: IntoIterator<Item = String>>(names: I) -> Result<Self, EvalError> {
        let set: BTreeSet<String> = names.into_iter().collect();
        let w = 1.0 / set.len().max(1) as f64;
        let mut map: BTreeMap<String, f64> = set.into_iter().map(|n| (n, w)).collect();
        // Absorb rounding into the last entry so the sum is exactly representable near 1.
        let sum: f64 = map.values().sum();
        if let Some((_, last)) = map.iter_mut().next_back() {
            *last += 1.0 - sum;
        }
        Self::new(map)
    }

    pub fn weights(&self) -> &BTreeMap<String, f64> {
        &self.weights
    }

    /// SHA-256 of the canonical `name=weight` listing (weights printed with `{:?}`).
    pub fn sha256(&self) -> String {
        let s: String = self
            .weights
            .iter()
            .map(|(k, v)| format!("{k}={v:?}\n"))
            .collect();
        ultra_core::receipt::sha256_hex(s.as_bytes())
    }

    /// Weighted average of per-dataset values. The value map must cover exactly the
    /// weighted datasets.
    pub fn apply(&self, values: &BTreeMap<String, f64>) -> Result<f64, EvalError> {
        if values.len() != self.weights.len() || values.keys().ne(self.weights.keys()) {
            return Err(EvalError::Weights(format!(
                "datasets {:?} do not match frozen weights {:?}",
                values.keys().collect::<Vec<_>>(),
                self.weights.keys().collect::<Vec<_>>()
            )));
        }
        Ok(self.weights.iter().map(|(k, w)| w * values[k]).sum())
    }
}

/// Frozen-weight macro WER. Every weighted dataset must be present with a defined WER
/// and no unweighted dataset may be present.
pub fn macro_wer(
    scores: &BTreeMap<String, DatasetScore>,
    weights: &MacroWeights,
) -> Result<f64, EvalError> {
    let mut vals = BTreeMap::new();
    for (k, s) in scores {
        let w = s.wer.ok_or_else(|| EvalError::UndefinedWer(k.clone()))?;
        vals.insert(k.clone(), w);
    }
    weights.apply(&vals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wer::align;

    fn w(s: &str) -> Vec<&str> {
        s.split_whitespace().collect()
    }

    fn utt(id: &str, ds: &str, spk: &str, r: &str, h: &str, dur: Option<f64>) -> UtteranceResult {
        UtteranceResult {
            id: id.into(),
            dataset: ds.into(),
            speaker: spk.into(),
            counts: align(&w(r), &w(h)).counts,
            duration_s: dur,
        }
    }

    #[test]
    fn pooling_is_sum_errors_over_sum_words_not_mean_of_ratios() {
        let u = vec![
            utt("1", "d", "s1", "a", "b", None), // 1/1
            utt(
                "2",
                "d",
                "s2",
                "a b c d e f g h i",
                "a b c d e f g h i",
                None,
            ), // 0/9
        ];
        let p = pool(&u);
        assert_eq!(p["d"].wer, Some(0.1));
        assert_eq!(p["d"].speakers, 2);
    }

    #[test]
    fn zero_reference_utterances_are_insertions_per_hour() {
        let u = vec![
            utt("1", "d", "s", "a b", "a b", Some(2.0)),
            utt("2", "d", "s", "", "uh huh", Some(1800.0)),
            utt("3", "d", "s", "", "", Some(1800.0)),
            utt("4", "sil", "s", "", "x", Some(3600.0)),
        ];
        let p = pool(&u);
        assert_eq!(p["d"].wer, Some(0.0));
        assert_eq!(p["d"].counts.insertions, 0);
        assert_eq!(p["d"].silence.utterances, 2);
        assert_eq!(p["d"].silence.inserted_words, 2);
        assert_eq!(p["d"].silence.insertions_per_hour, Some(2.0));
        assert_eq!(p["d"].silence.false_transcripts_per_hour, Some(1.0));
        assert_eq!(p["sil"].wer, None);
        assert_eq!(p["sil"].silence.insertions_per_hour, Some(1.0));
        let wts = MacroWeights::equal(["d".to_string(), "sil".to_string()]).unwrap();
        assert_eq!(
            macro_wer(&p, &wts),
            Err(EvalError::UndefinedWer("sil".into()))
        );
        // Missing duration: per-hour undefined, never divided by zero.
        let p = pool(&[utt("5", "x", "s", "", "y", None)]);
        assert_eq!(p["x"].silence.insertions_per_hour, None);
    }

    #[test]
    fn macro_uses_frozen_weights() {
        let u = vec![
            utt("1", "a", "s", "w w w w", "w w w x", None), // 0.25
            utt("2", "b", "s", "w w", "x x", None),         // 1.0
        ];
        let p = pool(&u);
        let eq = MacroWeights::equal(["a".to_string(), "b".to_string()]).unwrap();
        assert!((macro_wer(&p, &eq).unwrap() - 0.625).abs() < 1e-12);
        let fw = MacroWeights::new([("a".into(), 0.75), ("b".into(), 0.25)].into()).unwrap();
        assert!((macro_wer(&p, &fw).unwrap() - 0.4375).abs() < 1e-12);
        assert!((pooled_wer(&p).unwrap() - 3.0 / 6.0).abs() < 1e-12);
        let wrong = MacroWeights::new([("a".into(), 1.0)].into()).unwrap();
        assert!(macro_wer(&p, &wrong).is_err());
        assert!(MacroWeights::new([("a".into(), 0.5)].into()).is_err());
        assert!(MacroWeights::new([("a".into(), f64::NAN)].into()).is_err());
        assert_ne!(eq.sha256(), fw.sha256());
    }

    #[test]
    fn label_poison_canary_raises_wer() {
        let refs = [
            "the quick brown fox jumps",
            "over the lazy dog today",
            "speech recognition on cpus",
            "streaming with low delay",
        ];
        let hyps = refs; // a perfect system
        let clean: Vec<_> = refs
            .iter()
            .zip(hyps)
            .enumerate()
            .map(|(i, (r, h))| utt(&i.to_string(), "d", "s", r, h, None))
            .collect();
        // Poison: rotate references across ids (scorer-only labels corrupted).
        let poisoned: Vec<_> = (0..refs.len())
            .map(|i| {
                utt(
                    &i.to_string(),
                    "d",
                    "s",
                    refs[(i + 1) % refs.len()],
                    hyps[i],
                    None,
                )
            })
            .collect();
        let c = pool(&clean)["d"].wer.unwrap();
        let p = pool(&poisoned)["d"].wer.unwrap();
        assert_eq!(c, 0.0);
        assert!(p > 0.5, "poisoned labels must raise WER, got {p}");
    }
}
