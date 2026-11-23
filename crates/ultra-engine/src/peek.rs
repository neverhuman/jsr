//! Non-mutating speculative onset peek (`H48d`, `onset_peek_design.md`).
//!
//! [`speculative_preview`] and [`PeekSession::peek_immutable`] borrow committed state immutably
//! (zero interior mutability) and decode greedily into call-owned scratch (`<= 3` frames).

use ultra_gp::chunk_bounds;

use crate::{Engine, EngineError, EngineStream, Workspace};

const SAMPLES_PER_MS: f64 = 16.0;
pub const MAX_PEEK_FRAMES: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeekConfig {
    pub max_peek_frames: usize,
    pub onset_only: bool,
    pub min_preview_samples: usize,
    pub max_onset_chunks: usize,
}
impl Default for PeekConfig {
    fn default() -> Self {
        Self {
            max_peek_frames: 2,
            onset_only: true,
            min_preview_samples: 1280,
            max_onset_chunks: 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PeekEvent {
    Preview {
        seq: u64,
        base_commit_seq: u64,
        through_sample: usize,
        text: String,
        audio_ms: f64,
    },
    Committed {
        seq: u64,
        through_sample: usize,
        text: String,
        audio_ms: f64,
    },
}

/// Runtime proxy telemetry for live diagnostics (single-commit arrival-clock proxies).
///
/// Formal registered `H48d` kill-rule scoring (2-step confirmation window, end-of-word clocks,
/// and non-speech audit) belongs to `eval-lane`'s offline trace scorer, not these runtime proxies.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PeekStats {
    pub preview_events: u64,
    pub preview_superseded: u64,
    pub preview_matched_next_commit: u64,
    pub preview_words_checked: u64,
    pub preview_words_mismatched: u64,
    pub preview_first_lead_ms_sum: f64,
}
impl PeekStats {
    /// Fraction of single-commit checked preview words mismatched on the next chunk (proxy).
    pub fn word_mismatch_proxy(&self) -> f64 {
        if self.preview_words_checked == 0 {
            return 0.0;
        }
        self.preview_words_mismatched as f64 / self.preview_words_checked as f64
    }
    /// Mean arrival-clock lead time (ms) across next-commit matched preview windows (proxy).
    pub fn mean_first_lead_ms_proxy(&self) -> f64 {
        if self.preview_matched_next_commit == 0 {
            return 0.0;
        }
        self.preview_first_lead_ms_sum / self.preview_matched_next_commit as f64
    }
}

/// Count `(preview_words, mismatched_words)` beyond `before` vs `after` (single-step proxy).
pub fn reconcile_preview_words(before: &str, preview: &str, after: &str) -> (u64, u64) {
    let b: Vec<&str> = before.split_whitespace().collect();
    let p: Vec<&str> = preview.split_whitespace().collect();
    let a: Vec<&str> = after.split_whitespace().collect();
    let c = b.iter().zip(&p).take_while(|(x, y)| x == y).count();
    let mis = (c..p.len()).filter(|&i| a.get(i) != Some(&p[i])).count();
    ((p.len() - c) as u64, mis as u64)
}

fn sample_at(t: usize, n: usize) -> usize {
    (640 * t).saturating_sub(240).min(n)
}

/// Non-mutating speculative preview over `peek_frames` (`1..=3`, `< engine.chunk`) partial frames.
pub fn speculative_preview(
    engine: &Engine,
    committed: &EngineStream,
    peek_frames: usize,
    peek_ws: &mut Workspace,
) -> Result<Option<String>, EngineError> {
    let f = peek_frames.min(MAX_PEEK_FRAMES) & if engine.model.arch.mid > 0 { !1 } else { !0 };
    if f == 0 || f >= engine.chunk || committed.is_finished() {
        return Ok(None);
    }
    let mut pe = engine.clone();
    (pe.chunk, pe.max_chunks, pe.beam, pe.endpoint_reset_ms) = (f, 1, 1, 0);
    if !pe.is_ready(committed) {
        return Ok(None);
    }
    let mut sc = committed.clone();
    if let Some(beam) = sc.dec.beam.take() {
        let best = beam.best_hyp();
        sc.dec.ctx =
            crate::beam::context(&best.tokens[best.ctx_start..], &engine.model.arch.ctx_init)
                .collect();
        sc.dec.dec_ok = false;
    } else if engine.beam > 1 {
        sc.dec.ctx = crate::beam::context(&sc.dec.tokens, &engine.model.arch.ctx_init).collect();
        sc.dec.dec_ok = false;
    }
    Ok((pe.decode(&mut [&mut sc], peek_ws)? > 0).then(|| pe.text(&sc)))
}

#[derive(Default)]
struct State {
    ws: [Workspace; 2],
    seqs: [u64; 2],
    samples: usize,
    last_text: String,
    last_peek_f: usize,
    onset_left: usize,
    armed_token_count: usize,
    pending: Option<(String, f64)>,
    stats: PeekStats,
}

/// Streaming session pairing committed [`EngineStream`] truth with non-mutating onset previews.
pub struct PeekSession {
    engine: Engine,
    pub config: PeekConfig,
    committed: EngineStream,
    s: State,
}
impl PeekSession {
    pub fn new(engine: Engine, config: PeekConfig) -> Self {
        let mut s = Self {
            committed: engine.create_stream(),
            engine,
            config,
            s: State::default(),
        };
        s.arm_onset();
        s
    }
    pub fn stats(&self) -> PeekStats {
        self.s.stats
    }
    pub fn committed_tokens(&self) -> &[u32] {
        self.committed.tokens()
    }
    pub fn committed_text(&self) -> String {
        self.engine.text(&self.committed)
    }
    /// Arm (or re-arm) the onset window from the current committed token count.
    pub fn arm_onset(&mut self) {
        let m = self.config.max_onset_chunks;
        self.s.onset_left = if m == 0 { usize::MAX } else { m };
        self.s.armed_token_count = self.committed.tokens().len();
    }
    pub fn reset(&mut self) {
        self.committed = self.engine.create_stream();
        (
            self.s.samples,
            self.s.seqs[1],
            self.s.last_peek_f,
            self.s.pending,
        ) = (0, 0, 0, None);
        self.s.last_text.clear();
        self.arm_onset();
    }
    /// Pure `&self` non-mutating peek using caller-owned scratch [`Workspace`].
    pub fn peek_immutable(&self, ws: &mut Workspace) -> Result<Option<String>, EngineError> {
        speculative_preview(&self.engine, &self.committed, self.peek_frames(), ws)
    }
    pub fn accept_i16(&mut self, pcm: &[i16]) -> Result<Vec<PeekEvent>, EngineError> {
        self.committed.accept_i16(pcm)?;
        self.s.samples += pcm.len();
        self.run(false)
    }
    pub fn finish(&mut self) -> Result<Vec<PeekEvent>, EngineError> {
        self.committed.input_finished();
        self.run(true)
    }
    fn peek_frames(&self) -> usize {
        let c = self
            .config
            .max_peek_frames
            .min(MAX_PEEK_FRAMES)
            .min(self.engine.chunk.saturating_sub(1));
        let (t0, nf, m) = (
            self.committed.enc.t,
            self.committed.n_feat(),
            self.engine.model.arch.mid,
        );
        (1..=c)
            .rfind(|&f| (m == 0 || f.is_multiple_of(2)) && chunk_bounds(t0, nf, f, false).is_some())
            .unwrap_or(0)
    }
    fn run(&mut self, finish: bool) -> Result<Vec<PeekEvent>, EngineError> {
        let (mut evs, prev_t, mut adv) = (Vec::new(), self.committed.enc.t, false);
        let audio_ms = self.s.samples as f64 / SAMPLES_PER_MS;
        let (e, st) = (&self.engine, &mut self.committed);
        while e.decode(&mut [st], &mut self.s.ws[0])? > 0 {
            adv = true;
        }
        if adv || finish {
            if adv && self.s.onset_left > 0 {
                let dc = (self.committed.enc.t - prev_t).div_ceil(self.engine.chunk.max(1));
                let no_new_tok = self.committed.tokens().len() <= self.s.armed_token_count;
                self.s.onset_left = self.s.onset_left.saturating_sub(dc) * usize::from(no_new_tok);
            }
            self.s.last_peek_f = 0;
            let text = self.engine.text(&self.committed);
            let had = self.s.pending.take().is_some_and(|(p, first_ms)| {
                let (pw, mis) = reconcile_preview_words(&self.s.last_text, &p, &text);
                self.s.stats.preview_superseded += 1;
                self.s.stats.preview_words_checked += pw;
                self.s.stats.preview_words_mismatched += mis;
                if pw > 0 && mis == 0 {
                    self.s.stats.preview_matched_next_commit += 1;
                    self.s.stats.preview_first_lead_ms_sum += (audio_ms - first_ms).max(0.0);
                }
                true
            });
            if had || text != self.s.last_text {
                self.s.seqs[0] += 1;
                (self.s.seqs[1], self.s.last_text) = (self.s.seqs[0], text.clone());
                let (seq, through_sample) = (
                    self.s.seqs[0],
                    sample_at(self.committed.enc.t, self.s.samples),
                );
                evs.push(PeekEvent::Committed {
                    seq,
                    through_sample,
                    text,
                    audio_ms,
                });
            }
        }
        let (t, n) = (self.committed.enc.t, self.s.samples);
        let stop = finish || (self.config.onset_only && self.s.onset_left == 0);
        if stop || n.saturating_sub(sample_at(t, n)) < self.config.min_preview_samples {
            return Ok(evs);
        }
        let f = self.peek_frames();
        if f <= self.s.last_peek_f {
            return Ok(evs);
        }
        self.s.last_peek_f = f;
        let Some(text) = speculative_preview(&self.engine, &self.committed, f, &mut self.s.ws[1])?
            .filter(|x| !x.is_empty() && *x != self.s.last_text)
        else {
            return Ok(evs);
        };
        let first_ms = match self.s.pending.take() {
            Some((p, old)) if p == text => {
                self.s.pending = Some((p, old));
                return Ok(evs);
            }
            Some((_, old)) => old,
            None => audio_ms,
        };
        self.s.pending = Some((text.clone(), first_ms));
        self.s.seqs[0] += 1;
        self.s.stats.preview_events += 1;
        let (seq, base_commit_seq, through_sample) =
            (self.s.seqs[0], self.s.seqs[1], sample_at(t + f, n));
        evs.push(PeekEvent::Preview {
            seq,
            base_commit_seq,
            through_sample,
            text,
            audio_ms,
        });
        Ok(evs)
    }
}
