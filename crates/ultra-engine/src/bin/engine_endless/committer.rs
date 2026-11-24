//! O(1) exact committed-word emitter for continuous endless streaming.
//!
//! - In greedy mode (`beam <= 1`), `stream.tokens()` is append-only and this emitter matches
//!   `score_alwayson.py` snapshot stability timestamps to 0.000 ms in O(1) memory.
//! - In beam mode (`beam > 1`), callers feed [`beam_lcp`] during streaming chunks so only the
//!   monotonically non-decreasing longest common prefix across all active beam hypotheses can
//!   finalize words; interim `best()` revisions never leak into emitted words, and [`finish`]
//!   flushes the final winning hypothesis at EOF so the joined trace equals `engine.text(&stream)`.

use std::io::Write;
use std::time::Instant;

use serde_json::json;
use ultra_engine::beam::Hyp;
use ultra_gp::TokenTable;

/// Longest common token prefix shared by all active beam hypotheses.
///
/// In `beam.rs`, every frame extends surviving hypotheses by appending zero or more tokens (and
/// `reset_context` collapses `hyps` to `[best_tokens]`), so this prefix is monotonically
/// non-decreasing and is always a prefix of the final winning hypothesis.
pub(crate) fn beam_lcp(hyps: &[Hyp]) -> &[u32] {
    let Some(first) = hyps.first() else {
        return &[];
    };
    let mut len = first.tokens.len();
    for h in &hyps[1..] {
        len = len.min(h.tokens.len());
        let mut k = 0;
        while k < len && first.tokens[k] == h.tokens[k] {
            k += 1;
        }
        len = k;
        if len == 0 {
            break;
        }
    }
    &first.tokens[..len]
}

pub(crate) struct WordCommitter {
    word_tok_start: usize,
    last_tok_len: usize,
    last_tail: Vec<u32>,
    /// `(word, first_observed_audio_s, first_observed_wall_ms)`
    pending: Option<(String, f64, f64)>,
    started_at: Instant,
    total_words: u64,
}

impl WordCommitter {
    pub(crate) fn new() -> Self {
        Self {
            word_tok_start: 0,
            last_tok_len: 0,
            last_tail: Vec::new(),
            pending: None,
            started_at: Instant::now(),
            total_words: 0,
        }
    }

    pub(crate) fn total_words(&self) -> u64 {
        self.total_words
    }

    #[inline]
    fn wall_ms(&self) -> f64 {
        self.started_at.elapsed().as_secs_f64() * 1000.0
    }

    fn write_row<W: Write>(
        out: &mut W,
        first_obs_audio_s: f64,
        emit_audio_s: f64,
        first_obs_wall_ms: f64,
        emitted_wall_ms: f64,
        words: &[String],
    ) -> Result<(), String> {
        let emit_audio = emit_audio_s.max(first_obs_audio_s);
        let emit_wall = emitted_wall_ms.max(first_obs_wall_ms);
        writeln!(
            out,
            "{}",
            json!({
                "audio_s": first_obs_audio_s,
                "emit_audio_s": emit_audio,
                "first_observed_ms": first_obs_audio_s * 1000.0,
                "emit_time_ms": emit_audio * 1000.0,
                "first_observed_wall_ms": first_obs_wall_ms,
                "emitted_at_ms": emit_wall,
                "words": words,
            })
        )
        .map_err(|e| e.to_string())
    }

    pub(crate) fn update<W: Write>(
        &mut self,
        table: &TokenTable,
        tokens: &[u32],
        audio_s: f64,
        out: &mut W,
    ) -> Result<(), String> {
        let wall_ms = self.wall_ms();
        self.update_with_wall_ms(table, tokens, audio_s, wall_ms, out)
    }

    pub(crate) fn update_with_wall_ms<W: Write>(
        &mut self,
        table: &TokenTable,
        tokens: &[u32],
        audio_s: f64,
        wall_ms: f64,
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
            // Clear any stale pending word if the active token tail became empty.
            self.pending = None;
            self.last_tail.clear();
            self.last_tail
                .extend_from_slice(&tokens[self.word_tok_start..]);
            return Ok(());
        }
        if ws.len() == 1 {
            let w0 = ws[0];
            match &self.pending {
                Some((pw, _, _)) if pw == w0 => {}
                _ => {
                    self.pending = Some((w0.to_owned(), audio_s, wall_ms));
                }
            }
            self.last_tail.clear();
            self.last_tail
                .extend_from_slice(&tokens[self.word_tok_start..]);
            return Ok(());
        }
        // Multiple words in tail: every word before the last is now finalized.
        let mut emit_prev: Option<(String, f64, f64)> = None;
        let mut emit_now: Vec<String> = Vec::new();
        match self.pending.take() {
            Some((pw, pt, pw_ms)) if pw == ws[0] => {
                emit_prev = Some((pw, pt, pw_ms));
            }
            _ => {
                emit_now.push(ws[0].to_owned());
            }
        }
        for &w in &ws[1..ws.len() - 1] {
            emit_now.push(w.to_owned());
        }
        if let Some((pw, pt, pw_ms)) = emit_prev {
            Self::write_row(out, pt, audio_s, pw_ms, wall_ms, &[pw])?;
            self.total_words += 1;
        }
        if !emit_now.is_empty() {
            self.total_words += emit_now.len() as u64;
            Self::write_row(out, audio_s, audio_s, wall_ms, wall_ms, &emit_now)?;
        }
        let last_word = ws[ws.len() - 1];
        self.pending = Some((last_word.to_owned(), audio_s, wall_ms));

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

    /// Finalize the trace at EOF against the winning token sequence (`stream.tokens()`).
    pub(crate) fn finish<W: Write>(
        &mut self,
        table: &TokenTable,
        final_tokens: &[u32],
        audio_s: f64,
        out: &mut W,
    ) -> Result<(), String> {
        let wall_ms = self.wall_ms();
        self.finish_with_wall_ms(table, final_tokens, audio_s, wall_ms, out)
    }

    pub(crate) fn finish_with_wall_ms<W: Write>(
        &mut self,
        table: &TokenTable,
        final_tokens: &[u32],
        audio_s: f64,
        wall_ms: f64,
        out: &mut W,
    ) -> Result<(), String> {
        self.update_with_wall_ms(table, final_tokens, audio_s, wall_ms, out)?;
        if let Some((pw, pt, pw_ms)) = self.pending.take() {
            Self::write_row(out, pt, audio_s, pw_ms, wall_ms, &[pw])?;
            self.total_words += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_table() -> TokenTable {
        let tok_str = "\
<blk> 0
<unk> 1
\u{2581}THE 2
\u{2581}A 3
\u{2581}CRANK 4
ED 5
\u{2581}SIBLE 6
\u{2581}INCRED 7
IBLE 8
\u{2581}AN 9
D 10
\u{2581}END 11
";
        TokenTable::from_str_checked(tok_str, 12).expect("table")
    }

    fn joined_words(buf: &[u8]) -> Vec<String> {
        let text = String::from_utf8(buf.to_vec()).expect("utf8");
        text.lines()
            .flat_map(|line| {
                let row: serde_json::Value = serde_json::from_str(line).expect("json");
                row["words"]
                    .as_array()
                    .expect("words array")
                    .iter()
                    .map(|w| w.as_str().expect("word str").to_owned())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn lcp_slices<'a>(seqs: &[&'a [u32]]) -> &'a [u32] {
        let Some(&first) = seqs.first() else {
            return &[];
        };
        let mut len = first.len();
        for &s in &seqs[1..] {
            len = len.min(s.len());
            let mut k = 0;
            while k < len && first[k] == s[k] {
                k += 1;
            }
            len = k;
        }
        &first[..len]
    }

    #[test]
    fn beam_lcp_trace_matches_final_text_across_mid_utterance_swaps() {
        let table = test_table();
        let mut committer = WordCommitter::new();
        let mut buf = Vec::new();

        // Step 1: beam disagrees on A vs THE and CRANKED SIBLE vs INCREDIBLE.
        // Interim best is [3, 4, 5, 6] ("A CRANKED SIBLE"), rival is [2, 7, 8] ("THE INCREDIBLE").
        let lcp1 = lcp_slices(&[&[3, 4, 5, 6], &[2, 7, 8]]);
        assert!(lcp1.is_empty());
        committer.update(&table, lcp1, 0.4, &mut buf).expect("u1");

        // Step 2: rival wins and both surviving beams share [2, 7, 8] ("THE INCREDIBLE"),
        // disagreeing on AN vs AND.
        let lcp2 = lcp_slices(&[&[2, 7, 8, 9], &[2, 7, 8, 9, 10]]);
        assert_eq!(lcp2, &[2, 7, 8, 9]);
        committer.update(&table, lcp2, 0.8, &mut buf).expect("u2");
        // "THE" and "INCREDIBLE" are finalized; "AN" stays pending because D may still merge.
        assert_eq!(joined_words(&buf), vec!["THE", "INCREDIBLE"]);

        // Step 3: final winning beam at EOF is [2, 7, 8, 9, 10, 11] ("THE INCREDIBLE AND END").
        let final_toks = &[2u32, 7, 8, 9, 10, 11];
        committer
            .finish(&table, final_toks, 1.2, &mut buf)
            .expect("finish");

        let final_ids: Vec<i64> = final_toks.iter().map(|&t| i64::from(t)).collect();
        let expected: Vec<String> = table
            .decode_ids(&final_ids)
            .split_whitespace()
            .map(str::to_owned)
            .collect();
        assert_eq!(joined_words(&buf), expected);
        assert_eq!(committer.total_words(), expected.len() as u64);
    }

    #[test]
    fn empty_final_beam_emits_no_stale_pending_word() {
        let table = test_table();
        let mut committer = WordCommitter::new();
        let mut buf = Vec::new();

        committer.update(&table, &[2], 0.4, &mut buf).expect("u1");
        committer
            .finish(&table, &[], 0.8, &mut buf)
            .expect("finish");

        assert!(buf.is_empty(), "empty final beam must emit nothing");
        assert_eq!(committer.total_words(), 0);
    }

    #[test]
    fn commit_timestamps_preserve_first_observed_and_record_honest_emission_time() {
        let table = test_table();
        let mut committer = WordCommitter::new();
        let mut buf = Vec::new();

        // Word "THE" first appears at audio_s=0.4 (400 ms), wall_ms=12.5, stays pending.
        committer
            .update_with_wall_ms(&table, &[2], 0.4, 12.5, &mut buf)
            .expect("u1");
        assert!(buf.is_empty(), "single pending word not yet emitted");

        // Second word arrives at audio_s=0.9 (900 ms), wall_ms=37.5 -> emits "THE".
        committer
            .update_with_wall_ms(&table, &[2, 11], 0.9, 37.5, &mut buf)
            .expect("u2");

        // EOF at audio_s=1.4 (1400 ms), wall_ms=55.0 -> emits "END".
        committer
            .finish_with_wall_ms(&table, &[2, 11], 1.4, 55.0, &mut buf)
            .expect("finish");

        let txt = String::from_utf8(buf).expect("utf8");
        let rows: Vec<serde_json::Value> = txt
            .lines()
            .map(|l| serde_json::from_str(l).expect("json"))
            .collect();
        assert_eq!(rows.len(), 2);

        // Row 0 ("THE"): first observed at 400 ms / 12.5 ms, honestly emitted at 900 ms / 37.5 ms.
        assert_eq!(rows[0]["words"], json!(["THE"]));
        assert!((rows[0]["first_observed_ms"].as_f64().unwrap() - 400.0).abs() < 1e-6);
        assert!((rows[0]["emit_time_ms"].as_f64().unwrap() - 900.0).abs() < 1e-6);
        assert!(
            rows[0]["emit_time_ms"].as_f64().unwrap()
                > rows[0]["first_observed_ms"].as_f64().unwrap()
        );
        assert!((rows[0]["first_observed_wall_ms"].as_f64().unwrap() - 12.5).abs() < 1e-6);
        assert!((rows[0]["emitted_at_ms"].as_f64().unwrap() - 37.5).abs() < 1e-6);
        assert!(
            rows[0]["emitted_at_ms"].as_f64().unwrap()
                >= rows[0]["first_observed_wall_ms"].as_f64().unwrap()
        );

        // Row 1 ("END"): first observed at 900 ms / 37.5 ms, honestly emitted at EOF 1400 ms / 55.0 ms.
        assert_eq!(rows[1]["words"], json!(["END"]));
        assert!((rows[1]["first_observed_ms"].as_f64().unwrap() - 900.0).abs() < 1e-6);
        assert!((rows[1]["emit_time_ms"].as_f64().unwrap() - 1400.0).abs() < 1e-6);
        assert!((rows[1]["emitted_at_ms"].as_f64().unwrap() - 55.0).abs() < 1e-6);
        assert!(
            rows[1]["emit_time_ms"].as_f64().unwrap()
                >= rows[1]["first_observed_ms"].as_f64().unwrap()
        );
        assert!(
            rows[1]["emitted_at_ms"].as_f64().unwrap()
                >= rows[1]["first_observed_wall_ms"].as_f64().unwrap()
        );
    }
}
