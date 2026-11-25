//! Release decoders. Each reproduces a shipped binary's call sequence with its release flags.
//!
//! * [`FinalDecoder`]: `ultra-engine-commit` (`engine_commit_eval/main.rs`): `Model::load(.., true)`,
//!   `Engine::new(.., Int8).with_gemm_threads(t)`, `CommitPassSession::new` + `with_commit_lm`,
//!   320-sample `accept_i16` packets, `commit_endpoint`, `reset_all`.
//! * [`LiveDecoder`]: `ultra-cold-wav` backend `r1-engine-int8`
//!   (`ultra-bridge/src/bin/ultra-cold-wav.rs` `load_native` + `decode`, over
//!   `ultra-stream/src/native_engine.rs` `load_inner_cfg` / `NativeEngineAsrStream`).
//! * [`Segmenter`]: the live-endpointer + commit composition of `LiveCommitAsrStream`
//!   (`ultra-stream/src/native_commit.rs`), run synchronously (no worker thread, no deadline
//!   fallback), used by `stream`.
//!
//! Final mode on audio longer than the commit window is cut at pauses by [`plan_cuts`] and each
//! piece goes through the same single commit pass.

use std::path::Path;
use std::sync::Arc;

use ultra_engine::commit::{
    run_commit_pcm, CommitConfig, CommitLmConfig, CommitPassSession, DEFAULT_MAX_PCM_SAMPLES,
    DEFAULT_SAMPLE_RATE,
};
use ultra_engine::{Engine, EngineError, Isa, LmConfig, Model, NgramLm, Precision, Workspace};

use crate::model::ModelDir;
use crate::segment::plan_cuts;

/// Transport packet size used by both reference harnesses (20 ms at 16 kHz).
pub const PACKET_SAMPLES: usize = 320;
/// Encoder frame hop (`ultra_stream::ort_r1::R1_SAMPLES_PER_ENC_FRAME`).
const SAMPLES_PER_ENC_FRAME: u64 = 640;

// Final (commit) release flags.
const COMMIT_CHUNK: usize = 16;
const COMMIT_BEAM: usize = 16;
const MAX_SYM_PER_FRAME: usize = 1;
const TAIL_PAD_MS: usize = 320;
const LM_SCALE: f32 = 0.80;
const LODR_SCALE: f32 = 0.60;
const TOKEN_BONUS: f32 = 0.0;
// Live release flags.
const LIVE_CHUNK: usize = 4;
const LIVE_BEAM: usize = 4;
const LIVE_LM_WEIGHT: f32 = 0.70;
const LIVE_LODR_WEIGHT: f32 = 0.50;
// Live endpointer defaults of `ultra stream --commit-pass`.
const HANGOVER_MS: usize = 300;
const PREROLL_MS: usize = 300;

fn e2s(e: EngineError) -> String {
    e.to_string()
}

fn load_model(dir: &ModelDir, live_only: bool) -> Result<Arc<Model>, String> {
    let model = if live_only {
        Model::load_live(&dir.root, true)
    } else {
        Model::load(&dir.root, true)
    }
    .map_err(|e| format!("{}: {e}", dir.root.display()))?;
    if model.int8_prepared.is_none() {
        return Err(format!(
            "{}: bundle has no weights.int8.safetensors (INT8 bundle required)",
            dir.root.display()
        ));
    }
    Ok(Arc::new(model))
}

fn live_engine(
    model: &Arc<Model>,
    lm_path: &Path,
    threads: usize,
    live_only: bool,
) -> Result<Engine, String> {
    let lm = Arc::new(NgramLm::load(lm_path).map_err(|e| format!("{}: {e}", lm_path.display()))?);
    let cfg = LmConfig {
        lm_weight: LIVE_LM_WEIGHT,
        lodr_weight: LIVE_LODR_WEIGHT,
        blank_penalty: 0.0,
    };
    Engine::new(Arc::clone(model), Isa::detect(), Precision::Int8)
        .map_err(e2s)?
        .with_gemm_threads(threads)
        .with_max_chunks(1)
        .with_beam(LIVE_BEAM)
        .with_beam_max_symbols(1)
        .with_beam_margin(0.0)
        .with_kv_int8(false)
        .with_endpoint_reset(0, 0)
        .with_live_only(live_only)
        .with_tail_pad_samples(0)
        .with_chunk(LIVE_CHUNK)
        .map_err(e2s)?
        .with_lm(lm, cfg)
        .map_err(e2s)
}

/// `ultra-cold-wav` `r1-engine-int8` release decoder.
pub struct LiveDecoder {
    engine: Engine,
    ws: Workspace,
}

impl LiveDecoder {
    pub fn new(dir: &ModelDir, threads: usize) -> Result<Self, String> {
        let model = load_model(dir, true)?;
        Ok(Self {
            engine: live_engine(&model, &dir.live_lm(), threads, true)?,
            ws: Workspace::default(),
        })
    }

    pub fn decode(&mut self, pcm: &[i16]) -> Result<String, String> {
        let mut raw = self.engine.create_stream();
        for packet in pcm.chunks(PACKET_SAMPLES) {
            raw.accept_i16(packet).map_err(e2s)?;
            while self.engine.is_ready(&raw) {
                let n = self
                    .engine
                    .decode(&mut [&mut raw], &mut self.ws)
                    .map_err(e2s)?;
                if n == 0 {
                    break;
                }
            }
        }
        raw.input_finished();
        while self.engine.is_ready(&raw) {
            self.engine
                .decode(&mut [&mut raw], &mut self.ws)
                .map_err(e2s)?;
        }
        Ok(self.engine.text(&raw))
    }
}

fn commit_config(rms_norm: bool) -> CommitConfig {
    CommitConfig {
        commit_chunk: COMMIT_CHUNK,
        tail_pad_samples: TAIL_PAD_MS * DEFAULT_SAMPLE_RATE / 1000,
        max_pcm_samples: DEFAULT_MAX_PCM_SAMPLES,
        beam_size: COMMIT_BEAM,
        ctc_weight: 0.0,
        mbr_enabled: false,
        mbr_temp: 1.0,
        rms_norm,
    }
}

fn commit_lm(path: &Path) -> Result<CommitLmConfig, String> {
    CommitLmConfig::from_file(
        path,
        COMMIT_BEAM,
        LM_SCALE,
        LODR_SCALE,
        TOKEN_BONUS,
        MAX_SYM_PER_FRAME,
    )
    .map_err(|e| format!("{}: {e}", path.display()))
}

/// `ultra-engine-commit` release decoder, with pause-based cutting for audio beyond the commit window.
pub struct FinalDecoder {
    sess: CommitPassSession,
}

impl FinalDecoder {
    pub fn new(dir: &ModelDir, threads: usize, rms_norm: bool) -> Result<Self, String> {
        let model = load_model(dir, false)?;
        let eng = Engine::new(model, Isa::detect(), Precision::Int8)
            .map_err(e2s)?
            .with_gemm_threads(threads);
        let lm = commit_lm(&dir.final_lm())?;
        let sess = CommitPassSession::new(eng, commit_config(rms_norm))
            .and_then(|s| s.with_commit_lm(Some(lm)))
            .map_err(e2s)?;
        Ok(Self { sess })
    }

    /// Single commit pass (byte-identical to `ultra-engine-commit` for audio within the limit).
    fn commit_one(&mut self, pcm: &[i16]) -> Result<String, String> {
        for pkt in pcm.chunks(PACKET_SAMPLES) {
            self.sess.accept_i16(pkt).map_err(e2s)?;
        }
        let r = self.sess.commit_endpoint().map_err(e2s)?;
        self.sess.reset_all();
        Ok(r.committed)
    }

    pub fn decode(&mut self, pcm: &[i16]) -> Result<String, String> {
        if pcm.is_empty() {
            return Ok(String::new());
        }
        if pcm.len() <= DEFAULT_MAX_PCM_SAMPLES {
            return self.commit_one(pcm);
        }
        let mut parts: Vec<String> = Vec::new();
        let mut start = 0;
        for end in plan_cuts(pcm, DEFAULT_MAX_PCM_SAMPLES) {
            let t = self.commit_one(&pcm[start..end])?;
            let t = t.trim();
            if !t.is_empty() {
                parts.push(t.to_owned());
            }
            start = end;
        }
        Ok(parts.join(" "))
    }
}

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
        let cfg = commit_config(rms_norm);
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
