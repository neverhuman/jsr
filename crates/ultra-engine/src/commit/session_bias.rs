//! Opt-in session-scoped lexical bias memory (`P8`, H55 Wave 1D; default OFF).
//! Tracks rare/capitalized/multi-BPE words across `>= min_occurrences` high-confidence segments.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::EngineError;

const STOP_WORDS: &[&str] = &[
    "about", "above", "after", "again", "against", "always", "among", "another", "around",
    "because", "before", "being", "below", "between", "could", "doing", "during", "every", "first",
    "found", "great", "having", "house", "large", "little", "might", "never", "night", "often",
    "other", "people", "place", "point", "right", "shall", "should", "since", "small", "sound",
    "still", "state", "their", "there", "these", "thing", "think", "those", "three", "through",
    "today", "under", "until", "water", "where", "which", "while", "world", "would", "write",
    "years", "young",
];

/// Configuration for opt-in session lexical memory (`P8`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SessionLexicalConfig {
    pub enabled: bool,
    pub min_occurrences: u32,
    pub max_entries: usize,
    pub bias_bonus: f32,
    pub min_word_chars: usize,
    pub min_bpe_pieces: usize,
}

impl Default for SessionLexicalConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_occurrences: 2,
            max_entries: 64,
            bias_bonus: 0.0,
            min_word_chars: 5,
            min_bpe_pieces: 2,
        }
    }
}

impl SessionLexicalConfig {
    pub fn validate(&self) -> Result<(), EngineError> {
        if self.min_occurrences == 0 {
            return Err(EngineError::Format(
                "session lexical min_occurrences must be >= 1".into(),
            ));
        }
        if !(1..=1024).contains(&self.max_entries) {
            return Err(EngineError::Shape(
                "session lexical max_entries must be in 1..=1024".into(),
            ));
        }
        if !self.bias_bonus.is_finite() || self.bias_bonus < 0.0 {
            return Err(EngineError::Format(
                "session lexical bias_bonus must be finite >= 0".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Entry {
    count: u32,
    last_seq: u64,
}

/// Bounded session lexical memory tracking recurring high-confidence rare/named words.
#[derive(Debug, Clone)]
pub struct SessionLexicalMemory {
    cfg: SessionLexicalConfig,
    entries: BTreeMap<String, Entry>,
    seq: u64,
}

impl SessionLexicalMemory {
    pub fn new(cfg: SessionLexicalConfig) -> Result<Self, EngineError> {
        cfg.validate()?;
        Ok(Self {
            cfg,
            entries: BTreeMap::new(),
            seq: 0,
        })
    }
    pub fn config(&self) -> SessionLexicalConfig {
        self.cfg
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn reset_all(&mut self) {
        self.entries.clear();
        self.seq = 0;
    }

    /// Observe words from one committed segment (deduplicated per segment).
    pub fn observe_committed_segment(
        &mut self,
        text: &str,
        bpe_piece_count: Option<&dyn Fn(&str) -> usize>,
        high_confidence: bool,
    ) {
        if !self.cfg.enabled || !high_confidence {
            return;
        }
        self.seq = self.seq.saturating_add(1);
        let mut seen = BTreeSet::new();
        for (idx, raw) in text.split_whitespace().enumerate() {
            let trimmed = raw.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'');
            if trimmed.is_empty() || trimmed.chars().all(|c| c.is_ascii_digit()) {
                continue;
            }
            let key = trimmed.to_ascii_lowercase();
            if !seen.insert(key.clone()) {
                continue;
            }
            let cap = is_capitalized_word(trimmed, idx);
            let multi_bpe = bpe_piece_count.is_some_and(|f| f(trimmed) >= self.cfg.min_bpe_pieces);
            if !self.is_eligible(&key, cap, multi_bpe) {
                continue;
            }
            if let Some(ent) = self.entries.get_mut(&key) {
                ent.count = ent.count.saturating_add(1);
                ent.last_seq = self.seq;
                continue;
            }
            if self.entries.len() >= self.cfg.max_entries {
                self.evict_one();
            }
            self.entries.insert(
                key,
                Entry {
                    count: 1,
                    last_seq: self.seq,
                },
            );
        }
    }

    fn is_eligible(&self, lower: &str, capitalized: bool, multi_bpe: bool) -> bool {
        if STOP_WORDS.binary_search(&lower).is_ok() {
            return false;
        }
        capitalized || multi_bpe || lower.chars().count() >= self.cfg.min_word_chars
    }

    fn evict_one(&mut self) {
        let victim = self
            .entries
            .iter()
            .min_by(|(ka, ea), (kb, eb)| {
                (ea.count, ea.last_seq, *ka).cmp(&(eb.count, eb.last_seq, *kb))
            })
            .map(|(k, _)| k.clone());
        if let Some(k) = victim {
            self.entries.remove(&k);
        }
    }

    pub fn bias_map(&self) -> BTreeMap<String, f32> {
        if !self.cfg.enabled || self.cfg.bias_bonus <= 0.0 {
            return BTreeMap::new();
        }
        self.entries
            .iter()
            .filter(|(_, e)| e.count >= self.cfg.min_occurrences)
            .map(|(k, _)| (k.clone(), self.cfg.bias_bonus))
            .collect()
    }
    pub fn word_bonus(&self, word: &str) -> f32 {
        if !self.cfg.enabled || self.cfg.bias_bonus <= 0.0 {
            return 0.0;
        }
        let k = word
            .trim_matches(|c: char| !c.is_alphanumeric() && c != '\'')
            .to_ascii_lowercase();
        self.entries
            .get(&k)
            .filter(|e| e.count >= self.cfg.min_occurrences)
            .map_or(0.0, |_| self.cfg.bias_bonus)
    }
    pub fn active_words(&self) -> usize {
        if !self.cfg.enabled {
            return 0;
        }
        self.entries
            .values()
            .filter(|e| e.count >= self.cfg.min_occurrences)
            .count()
    }
}

impl super::CommitPassSession {
    pub fn live_progress(&self) -> (usize, usize, u64, bool) {
        let s = &self.stream;
        (
            s.tokens().len(),
            s.enc.t,
            s.forced_advances(),
            self.seg.segment_open,
        )
    }
    /// Aggregated [`crate::DecodeProgress`] from the most recent `accept_i16_live` call.
    pub fn last_decode_progress(&self) -> crate::DecodeProgress {
        self.last_live_progress
    }
    pub fn status_event(
        &mut self,
        now_ns: u64,
        code: &str,
        detail: &str,
    ) -> ultra_core::events::TranscriptEvent {
        self.emitter.status(now_ns, code, detail)
    }
    pub fn trim_idle_pcm(&mut self, keep_samples: usize, reset_live_stream: bool) {
        if self.seg.pcm_buf.len() > keep_samples {
            self.seg
                .pcm_buf
                .drain(..(self.seg.pcm_buf.len() - keep_samples));
        }
        if reset_live_stream {
            self.stream = self.provisional_engine.create_stream();
            self.last_live_progress = crate::DecodeProgress::default();
        }
    }
    pub fn accept_i16_live(&mut self, pcm: &[i16]) -> Result<usize, EngineError> {
        self.stream.accept_i16(pcm)?;
        let dropped =
            super::types::push_bounded_pcm(&mut self.seg.pcm_buf, pcm, self.cfg.max_pcm_samples);
        if dropped > 0 {
            self.seg.dropped_pcm_samples += dropped;
            self.seg.pcm_truncated = true;
        }
        self.seg.segment_samples += pcm.len() as u64;
        self.seg.total_samples += pcm.len() as u64;
        let mut steps = 0usize;
        let mut agg = crate::DecodeProgress::default();
        while self
            .provisional_engine
            .decode(&mut [&mut self.stream], &mut self.ws)?
            > 0
        {
            agg.merge(self.stream.last_progress());
            steps += 1;
        }
        self.last_live_progress = agg;
        Ok(steps)
    }
}

fn is_capitalized_word(word: &str, pos: usize) -> bool {
    let mut cs = word.chars();
    let Some(first) = cs.next() else { return false };
    let has_upper_rest = cs.clone().any(char::is_uppercase);
    has_upper_rest || (first.is_uppercase() && pos > 0 && cs.any(char::is_lowercase))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_bias_defaults_off_threshold_capacity_and_reset() {
        let mut off = SessionLexicalMemory::new(SessionLexicalConfig::default()).expect("default");
        off.observe_committed_segment("Hyperion Hyperion", None, true);
        off.observe_committed_segment("Hyperion", None, true);
        assert!(off.is_empty() && off.bias_map().is_empty() && off.word_bonus("Hyperion") == 0.0);

        let mut mem = SessionLexicalMemory::new(SessionLexicalConfig {
            enabled: true,
            bias_bonus: 0.35,
            ..Default::default()
        })
        .expect("enabled");
        mem.observe_committed_segment("we visited Reykjavik and Reykjavik", None, true);
        assert_eq!((mem.active_words(), mem.word_bonus("Reykjavik")), (0, 0.0));
        mem.observe_committed_segment("back in Reykjavik", None, false);
        assert_eq!(mem.active_words(), 0);
        mem.observe_committed_segment(
            "back in Reykjavik with SQL",
            Some(&|w| if w == "SQL" { 3 } else { 1 }),
            true,
        );
        assert_eq!(mem.active_words(), 1);
        assert!((mem.word_bonus("reykjavik") - 0.35).abs() < 1e-6);
        assert_eq!(mem.bias_map().get("reykjavik"), Some(&0.35));
        mem.reset_all();
        assert!(mem.is_empty() && mem.active_words() == 0);

        let mut cap = SessionLexicalMemory::new(SessionLexicalConfig {
            enabled: true,
            max_entries: 64,
            bias_bonus: 0.25,
            ..Default::default()
        })
        .expect("cap");
        cap.observe_committed_segment("keepword", None, true);
        cap.observe_committed_segment("keepword", None, true);
        for i in 0..90 {
            cap.observe_committed_segment(&format!("rareterm{i:03}"), None, true);
            assert!(cap.len() <= 64);
        }
        assert_eq!(cap.len(), 64);
        assert!((cap.word_bonus("keepword") - 0.25).abs() < 1e-6);
        assert_eq!(cap.word_bonus("rareterm000"), 0.0);
    }
}
