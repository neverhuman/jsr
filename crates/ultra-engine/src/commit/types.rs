//! Configuration, event/result records, tail-silence padding, bounded PCM buffer, and
//! immutable-prefix reconciliation helpers for the H45 endpoint commit pass.

use serde::{Deserialize, Serialize};

use crate::EngineError;

pub const DEFAULT_SAMPLE_RATE: usize = 16_000;
pub const DEFAULT_TAIL_PAD_SAMPLES: usize = 5_120;
/// Bounded per-segment PCM retention cap (`40 s` at `16 kHz`) preventing unbounded memory growth
/// while covering all canonical LibriSpeech dev utterances (`max 35.16 s`).
pub const DEFAULT_MAX_PCM_SAMPLES: usize = 40 * DEFAULT_SAMPLE_RATE;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CommitConfig {
    pub commit_chunk: usize,
    pub tail_pad_samples: usize,
    pub max_pcm_samples: usize,
    pub beam_size: usize,
    pub ctc_weight: f32,
    pub mbr_enabled: bool,
}

impl Default for CommitConfig {
    fn default() -> Self {
        Self {
            commit_chunk: 16,
            tail_pad_samples: DEFAULT_TAIL_PAD_SAMPLES,
            max_pcm_samples: DEFAULT_MAX_PCM_SAMPLES,
            beam_size: 1,
            ctc_weight: 0.0,
            mbr_enabled: false,
        }
    }
}

impl CommitConfig {
    pub fn validate(&self) -> Result<(), EngineError> {
        if !(1..=64).contains(&self.commit_chunk) {
            return Err(EngineError::Shape(format!(
                "commit_chunk {} not in 1..=64",
                self.commit_chunk
            )));
        }
        if self.max_pcm_samples == 0 {
            return Err(EngineError::Shape(
                "max_pcm_samples must be positive".into(),
            ));
        }
        if !(1..=16).contains(&self.beam_size) {
            return Err(EngineError::Shape(format!(
                "beam_size {} not in 1..=16",
                self.beam_size
            )));
        }
        if !self.ctc_weight.is_finite() || self.ctc_weight < 0.0 {
            return Err(EngineError::Format(format!(
                "invalid ctc_weight {}",
                self.ctc_weight
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CommitEvent {
    Provisional {
        prefix: String,
        text: String,
        audio_ms: f64,
    },
    Committed {
        prefix: String,
        provisional: String,
        text: String,
        revised: bool,
        commit_ms: f64,
        audio_ms: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommitResult {
    pub provisional: String,
    pub committed: String,
    pub committed_prefix: String,
    pub revised: bool,
    pub pcm_truncated: bool,
    pub commit_wall_ms: f64,
    pub audio_ms: f64,
    pub selected_hyp: usize,
}

impl CommitResult {
    pub fn into_event(self) -> CommitEvent {
        CommitEvent::Committed {
            prefix: self.committed_prefix,
            provisional: self.provisional,
            text: self.committed,
            revised: self.revised,
            commit_ms: self.commit_wall_ms,
            audio_ms: self.audio_ms,
        }
    }
}

/// Append `tail_pad_samples` zero samples of trailing silence to `pcm`.
pub fn pad_tail_silence(pcm: &[i16], tail_pad_samples: usize) -> Vec<i16> {
    let mut out = Vec::with_capacity(pcm.len() + tail_pad_samples);
    out.extend_from_slice(pcm);
    out.resize(pcm.len() + tail_pad_samples, 0);
    out
}

/// Push `pcm` into `buf` without exceeding `cap`, returning the count of dropped oldest samples.
pub(crate) fn push_bounded_pcm(buf: &mut Vec<i16>, pcm: &[i16], cap: usize) -> u64 {
    if pcm.len() >= cap {
        let dropped = (buf.len() + pcm.len() - cap) as u64;
        buf.clear();
        buf.extend_from_slice(&pcm[pcm.len() - cap..]);
        dropped
    } else if buf.len() + pcm.len() > cap {
        let drop_n = buf.len() + pcm.len() - cap;
        buf.drain(..drop_n);
        buf.extend_from_slice(pcm);
        drop_n as u64
    } else {
        buf.extend_from_slice(pcm);
        0
    }
}

/// Compute the uncommitted suffix to append after an already-committed `immutable` prefix of the
/// active segment without ever mutating or duplicating `immutable`.
pub fn reconcile_immutable_suffix(immutable: &str, candidate: &str) -> String {
    let cand = candidate.trim();
    if immutable.is_empty() {
        return cand.to_owned();
    }
    if let Some(rest) = cand.strip_prefix(immutable) {
        if rest.is_empty()
            || immutable.ends_with(char::is_whitespace)
            || rest.starts_with(char::is_whitespace)
        {
            return rest.to_owned();
        }
    }
    let imm_trim = immutable.trim_end();
    if let Some(rest) = (!imm_trim.is_empty())
        .then(|| cand.strip_prefix(imm_trim))
        .flatten()
    {
        if rest.is_empty() {
            return String::new();
        }
        if rest.starts_with(char::is_whitespace) {
            let tail = rest.trim_start();
            return if tail.is_empty() {
                String::new()
            } else if immutable.ends_with(char::is_whitespace) {
                tail.to_owned()
            } else {
                format!(" {tail}")
            };
        }
    }
    let imm_words: Vec<&str> = immutable.split_whitespace().collect();
    let cand_words: Vec<&str> = cand.split_whitespace().collect();
    if cand_words.len() <= imm_words.len() {
        return String::new();
    }
    let mut skip = imm_words.len();
    if let Some(&last_imm) = imm_words.last() {
        let lo = imm_words.len().saturating_sub(2);
        let hi = (imm_words.len() + 1).min(cand_words.len());
        if let Some(pos) = (lo..hi).find(|&i| cand_words[i] == last_imm) {
            skip = pos + 1;
        }
    }
    if skip >= cand_words.len() {
        return String::new();
    }
    let tail = cand_words[skip..].join(" ");
    if immutable.ends_with(char::is_whitespace) {
        tail
    } else {
        format!(" {tail}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commit_reconcile_immutable_suffix_preserves_precommitted_words() {
        assert_eq!(reconcile_immutable_suffix("", "the cat sat"), "the cat sat");
        assert_eq!(reconcile_immutable_suffix("the", "the cat sat"), " cat sat");
        assert_eq!(reconcile_immutable_suffix("the ", "the cat sat"), "cat sat");
        assert_eq!(reconcile_immutable_suffix("the", "a cat sat"), " cat sat");
        assert_eq!(
            reconcile_immutable_suffix("the red", "a red cat sat"),
            " cat sat"
        );
        assert_eq!(reconcile_immutable_suffix("the cat", "the"), "");
    }
}
