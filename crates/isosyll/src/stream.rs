//! `isosyll stream`: synchronous live-endpointer + commit-pass segmenter, a port of
//! `LiveCommitAsrStream::push_pcm` (`ultra-stream/src/native_commit.rs`) with no worker thread
//! and no deadline fallback.

use ultra_engine::commit::{run_commit_pcm, CommitConfig, CommitPassSession, DEFAULT_SAMPLE_RATE};
use ultra_engine::Workspace;

use crate::decode::{commit_lm, e2s, live_engine, load_model, stream_commit_config};
use crate::model::ModelDir;

/// Encoder frame hop (`ultra_stream::ort_r1::R1_SAMPLES_PER_ENC_FRAME`).
const SAMPLES_PER_ENC_FRAME: u64 = 640;
// Endpointer for `isosyll stream`: commit after 600 ms of trailing blank. Longer than the 300 ms of
// `ultra stream --commit-pass` so segments end at sentence pauses rather than mid-phrase (a 109 s
// check: 300 ms split "THE TOPPER" into "THE TOR"; 1000 ms ran segments into the 40 s window).
// Display policy only; the paper's measured modes are `transcribe` final and live.
const HANGOVER_MS: usize = 600;
const PREROLL_MS: usize = 300;

/// Output of [`Segmenter`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SegEvent {
    /// Updated provisional text of the open segment.
    Partial(String),
    /// A committed (final) segment; never empty.
    Segment(String),
}

/// Synchronous live-endpointer + commit-pass segmenter (port of `LiveCommitAsrStream::push_pcm`).
pub struct Segmenter {
    sess: CommitPassSession,
    cfg: CommitConfig,
    ws: Workspace,
    hangover_samples: u64,
    preroll_keep: usize,
    last_tokens_len: usize,
    last_enc_frames: usize,
    last_hyp: String,
    speech_armed: bool,
    trailing_blank: u64,
}

impl Segmenter {
    pub fn new(dir: &ModelDir, threads: usize, rms_norm: bool) -> Result<Self, String> {
        let model = load_model(dir, false)?;
        let cfg = stream_commit_config(rms_norm);
        let live = live_engine(&model, &dir.live_lm(), threads, false)?;
        let lm = commit_lm(&dir.final_lm())?;
        let sess = CommitPassSession::with_stream_id(live, cfg, "isosyll", "isosyll-v4")
            .and_then(|s| s.with_commit_lm(Some(lm)))
            .map_err(e2s)?;
        let preroll = (PREROLL_MS * DEFAULT_SAMPLE_RATE).div_ceil(1000);
        Ok(Self {
            sess,
            cfg,
            ws: Workspace::default(),
            hangover_samples: ((HANGOVER_MS * DEFAULT_SAMPLE_RATE) as u64).div_ceil(1000),
            preroll_keep: (preroll + 8 * SAMPLES_PER_ENC_FRAME as usize).min(cfg.max_pcm_samples),
            last_tokens_len: 0,
            last_enc_frames: 0,
            last_hyp: String::new(),
            speech_armed: false,
            trailing_blank: 0,
        })
    }

    fn reset_tracking(&mut self) {
        self.last_tokens_len = 0;
        self.last_enc_frames = 0;
        self.speech_armed = false;
        self.trailing_blank = 0;
        self.last_hyp.clear();
    }

    fn commit(&mut self, out: &mut Vec<SegEvent>) -> Result<(), String> {
        let snap = self.sess.take_commit_snapshot().map_err(e2s)?;
        self.reset_tracking();
        let text = if snap.pcm_truncated {
            snap.provisional
        } else {
            run_commit_pcm(self.sess.commit_engine(), self.cfg, &snap.pcm, &mut self.ws)
                .map_err(e2s)?
                .0
        };
        let text = text.trim();
        if !text.is_empty() {
            out.push(SegEvent::Segment(text.to_owned()));
        }
        Ok(())
    }

    pub fn push(&mut self, pcm: &[i16]) -> Result<Vec<SegEvent>, String> {
        let mut out = Vec::new();
        let steps = self.sess.accept_i16_live(pcm).map_err(e2s)?;
        if steps == 0 {
            return Ok(out);
        }
        let (cur_tok, cur_frames, _, _) = self.sess.live_progress();
        let delta_frames = cur_frames.saturating_sub(self.last_enc_frames);
        self.last_enc_frames = cur_frames;
        let prog = self.sess.last_decode_progress();
        if prog.has_activity() || cur_tok != self.last_tokens_len {
            self.last_tokens_len = cur_tok;
            self.speech_armed = true;
            self.trailing_blank = 0;
            let hyp = self.sess.provisional_text();
            if hyp != self.last_hyp {
                self.last_hyp.clone_from(&hyp);
                out.push(SegEvent::Partial(hyp));
            }
        } else if self.speech_armed && delta_frames > 0 {
            self.trailing_blank += delta_frames as u64 * SAMPLES_PER_ENC_FRAME;
            if self.trailing_blank >= self.hangover_samples {
                self.commit(&mut out)?;
            }
        } else if !self.speech_armed {
            let reset_live = cur_frames >= 100; // >= 4 s of pure silence
            self.sess.trim_idle_pcm(self.preroll_keep, reset_live);
            if reset_live {
                self.last_tokens_len = 0;
                self.last_enc_frames = 0;
            }
        }
        Ok(out)
    }

    /// Flush the open segment at end of input.
    pub fn finish(&mut self) -> Result<Vec<SegEvent>, String> {
        let mut out = Vec::new();
        let (s0, s1) = self.sess.segment_sample_range();
        if s1 > s0 {
            self.commit(&mut out)?;
        }
        Ok(out)
    }
}
