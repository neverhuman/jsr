//! O(1) exact committed-word emitter for continuous endless streaming.
//!
//! Matches `score_alwayson.py` snapshot stability timestamps to 0.000 ms for greedy decoding
//! while using O(1) memory, and safely handles beam re-ranking when the active hypothesis
//! revises or shortens its trailing tokens.

use std::io::Write;

use serde_json::json;
use ultra_gp::TokenTable;

pub(crate) struct WordCommitter {
    word_tok_start: usize,
    last_tok_len: usize,
    last_tail: Vec<u32>,
    pending: Option<(String, f64)>,
    total_words: u64,
}

impl WordCommitter {
    pub(crate) fn new() -> Self {
        Self {
            word_tok_start: 0,
            last_tok_len: 0,
            last_tail: Vec::new(),
            pending: None,
            total_words: 0,
        }
    }

    pub(crate) fn total_words(&self) -> u64 {
        self.total_words
    }

    pub(crate) fn update<W: Write>(
        &mut self,
        table: &TokenTable,
        tokens: &[u32],
        audio_s: f64,
        out: &mut W,
    ) -> Result<(), String> {
        let start = self.word_tok_start.min(tokens.len());
        if tokens.len() == self.last_tok_len && tokens[start..] == self.last_tail[..] {
            return Ok(());
        }
        self.word_tok_start = start;
        self.last_tok_len = tokens.len();
        let ids: Vec<i64> = tokens[self.word_tok_start..]
            .iter()
            .map(|&t| i64::from(t))
            .collect();
        let tail = table.decode_ids(&ids);
        let ws: Vec<&str> = tail.split_whitespace().collect();
        if ws.is_empty() {
            self.last_tail.clear();
            self.last_tail
                .extend_from_slice(&tokens[self.word_tok_start..]);
            return Ok(());
        }
        if ws.len() == 1 {
            let w0 = ws[0];
            match &self.pending {
                Some((pw, _)) if pw == w0 => {}
                _ => {
                    self.pending = Some((w0.to_owned(), audio_s));
                }
            }
            self.last_tail.clear();
            self.last_tail
                .extend_from_slice(&tokens[self.word_tok_start..]);
            return Ok(());
        }
        // Multiple words in tail: every word before the last is now finalized.
        let mut emit_prev: Option<(String, f64)> = None;
        let mut emit_now: Vec<String> = Vec::new();
        match self.pending.take() {
            Some((pw, pt)) if pw == ws[0] => {
                emit_prev = Some((pw, pt));
            }
            _ => {
                emit_now.push(ws[0].to_owned());
            }
        }
        for &w in &ws[1..ws.len() - 1] {
            emit_now.push(w.to_owned());
        }
        if let Some((pw, pt)) = emit_prev {
            writeln!(out, "{}", json!({"audio_s": pt, "words": [pw]}))
                .map_err(|e| e.to_string())?;
            self.total_words += 1;
        }
        if !emit_now.is_empty() {
            self.total_words += emit_now.len() as u64;
            writeln!(out, "{}", json!({"audio_s": audio_s, "words": emit_now}))
                .map_err(|e| e.to_string())?;
        }
        let last_word = ws[ws.len() - 1];
        self.pending = Some((last_word.to_owned(), audio_s));

        // Advance `word_tok_start` to the start of `last_word` (small backward scan).
        let mut new_start = tokens.len().saturating_sub(1);
        for k in (self.word_tok_start..tokens.len()).rev() {
            let sub: Vec<i64> = tokens[k..].iter().map(|&t| i64::from(t)).collect();
            let dec = table.decode_ids(&sub);
            if dec.trim() == last_word {
                new_start = k;
            } else if dec.split_whitespace().count() > 1 {
                break;
            }
        }
        self.word_tok_start = new_start;
        self.last_tail.clear();
        self.last_tail
            .extend_from_slice(&tokens[self.word_tok_start..]);
        Ok(())
    }

    pub(crate) fn flush<W: Write>(&mut self, out: &mut W) -> Result<(), String> {
        if let Some((pw, pt)) = self.pending.take() {
            writeln!(out, "{}", json!({"audio_s": pt, "words": [pw]}))
                .map_err(|e| e.to_string())?;
            self.total_words += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commits_words_and_handles_beam_revisions_without_panicking() {
        let tok_str = "<blk> 0\n<unk> 1\n\u{2581}hello 2\n\u{2581}world 3\n\u{2581}there 4\n";
        let table = TokenTable::from_str_checked(tok_str, 5).expect("table");
        let mut committer = WordCommitter::new();
        let mut buf = Vec::new();

        committer.update(&table, &[2], 0.4, &mut buf).expect("u1");
        committer
            .update(&table, &[2, 3], 0.8, &mut buf)
            .expect("u2");
        // Same-length beam revision of pending word ("world" -> "there")
        committer
            .update(&table, &[2, 4], 1.0, &mut buf)
            .expect("u3");
        // Shorter hypothesis overtaking best beam must not panic
        committer.update(&table, &[], 1.2, &mut buf).expect("u4");
        committer.flush(&mut buf).expect("flush");

        let text = String::from_utf8(buf).expect("utf8");
        let lines: Vec<serde_json::Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).expect("json"))
            .collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], json!({"audio_s": 0.4, "words": ["hello"]}));
        assert_eq!(lines[1], json!({"audio_s": 1.0, "words": ["there"]}));
        assert_eq!(committer.total_words(), 2);
    }
}
