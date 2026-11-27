//! Frozen linear commit rescoring with native Transformer token probabilities.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::Path;

use serde::Deserialize;

use super::score_ngram_seq;
use crate::lm::{NgramLm, TfDirection, TfRescorer, TfWorkspace};
use crate::{EngineError, Isa};

#[derive(Debug, Deserialize)]
struct Weights {
    base: f64,
    l2r: f64,
    m5: f64,
    nw: f64,
    l2r_content: f64,
    nw_content: f64,
    logfreq_content: f64,
    oov_content: f64,
}

#[derive(Debug, Deserialize)]
struct TrainingFit {
    weights: Weights,
    skip: f32,
}

#[derive(Debug, Deserialize)]
struct Lexicon {
    pieces: Vec<String>,
    function_words: HashSet<String>,
    counts: HashMap<String, u64>,
}

/// Frozen content-word features attached to an opt-in L2R commit rescorer.
#[derive(Debug)]
pub struct CommitContentReranker {
    weights: Weights,
    skip: f32,
    pieces: Vec<String>,
    function_words: HashSet<String>,
    log_freq: HashMap<String, f64>,
    oov_log_freq: f64,
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, EngineError> {
    serde_json::from_reader(std::io::BufReader::new(File::open(path)?))
        .map_err(|e| EngineError::Format(format!("{}: {e}", path.display())))
}

impl CommitContentReranker {
    pub fn from_files(fit_path: &Path, lexicon_path: &Path) -> Result<Self, EngineError> {
        let TrainingFit { weights, skip } = read_json(fit_path)?;
        let Lexicon {
            pieces,
            function_words,
            counts,
        } = read_json(lexicon_path)?;
        let total: f64 = counts.values().map(|&n| n as f64).sum();
        if pieces.is_empty()
            || total == 0.0
            || !skip.is_finite()
            || skip < 0.0
            || ![
                weights.base,
                weights.l2r,
                weights.m5,
                weights.nw,
                weights.l2r_content,
                weights.nw_content,
                weights.logfreq_content,
                weights.oov_content,
            ]
            .iter()
            .all(|w| w.is_finite())
        {
            return Err(EngineError::Format(
                "invalid content reranker weights or lexicon".into(),
            ));
        }
        let pieces = pieces.into_iter().map(|s| s.to_uppercase()).collect();
        let log_freq = counts
            .into_iter()
            .map(|(word, n)| (word, ((n as f64 + 1.0) / total).ln()))
            .collect();
        Ok(Self {
            weights,
            skip,
            pieces,
            function_words,
            log_freq,
            oov_log_freq: (1.0 / total).ln(),
        })
    }

    pub(super) fn validate_model(&self, l2r: &TfRescorer) -> Result<(), EngineError> {
        if l2r.cfg.direction != TfDirection::L2r || l2r.cfg.vocab != self.pieces.len() {
            return Err(EngineError::Shape(
                "content reranker needs an L2R model with the lexicon's vocabulary".into(),
            ));
        }
        Ok(())
    }

    /// Apply the frozen routing threshold and compute scores for one native beam.
    /// The same objective is used by the commit path and offline n-best replay.
    pub fn score_batch(
        &self,
        l2r: &TfRescorer,
        isa: Isa,
        lm: &NgramLm,
        candidates: &[(&[u32], f32)],
        workspace: &mut TfWorkspace,
    ) -> Result<Vec<f32>, EngineError> {
        let (mut top, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for &(_, score) in candidates {
            if score > top {
                second = top;
                top = score;
            } else if score > second {
                second = score;
            }
        }
        if candidates.len() < 2 || top - second >= self.skip {
            return Ok(candidates.iter().map(|c| c.1).collect());
        }
        self.validate_model(l2r)?;
        let sequences: Vec<&[u32]> = candidates.iter().map(|c| c.0).collect();
        let mut token_logp = Vec::new();
        let scores =
            l2r.score_batch_with_token_logp(isa, &sequences, workspace, &mut token_logp)?;
        let mut offset = 0;
        let mut rescored = Vec::with_capacity(candidates.len());
        for (&(tokens, base), score) in candidates.iter().zip(scores) {
            let end = offset + score.n_tok;
            let lexical = self.lexical_score(tokens, &token_logp[offset..end])?;
            offset = end;
            let w = &self.weights;
            let fused = w.base * f64::from(base)
                + w.l2r * f64::from(score.total_logp)
                + w.m5 * f64::from(score_ngram_seq(lm, tokens))
                + lexical;
            rescored.push(fused as f32);
        }
        Ok(rescored)
    }

    fn lexical_score(&self, tokens: &[u32], logp: &[f32]) -> Result<f64, EngineError> {
        let mut score = 0.0;
        let mut word = String::new();
        let mut word_start = 0;
        for (position, &id) in tokens.iter().enumerate() {
            let piece = self.pieces.get(id as usize).ok_or_else(|| {
                EngineError::Shape(format!("content reranker token out of range: {id}"))
            })?;
            if position != 0 && piece.starts_with('▁') {
                score += self.word_score(&word, logp, word_start, position);
                word.clear();
                word_start = position;
            }
            word.push_str(if position == word_start {
                piece.trim_start_matches('▁')
            } else {
                piece
            });
        }
        if !tokens.is_empty() {
            score += self.word_score(&word, logp, word_start, tokens.len());
        }
        Ok(score)
    }

    fn word_score(&self, word: &str, logp: &[f32], start: usize, end: usize) -> f64 {
        let w = &self.weights;
        if self.function_words.contains(word) {
            return w.nw;
        }
        let content_logp: f64 = logp[start.min(logp.len())..end.min(logp.len())]
            .iter()
            .map(|&x| f64::from(x))
            .sum();
        let (log_freq, oov) = self
            .log_freq
            .get(word)
            .map_or((self.oov_log_freq, 1.0), |&f| (f, 0.0));
        w.nw + w.nw_content
            + w.l2r_content * content_logp
            + w.logfreq_content * log_freq
            + w.oov_content * oov
    }
}
