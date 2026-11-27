//! Release decoders. Each reproduces a shipped binary's call sequence with its release flags.
//!
//! * [`FinalDecoder`]: `ultra-engine-commit` (`engine_commit_eval/main.rs`): `Model::load(.., true)`,
//!   `Engine::new(.., Int8).with_gemm_threads(t)`, `CommitPassSession::new` + `with_commit_lm`,
//!   320-sample `accept_i16` packets, `commit_endpoint`, `reset_all`.
//! * [`LiveDecoder`]: `ultra-cold-wav` backend `r1-engine-int8`
//!   (`ultra-bridge/src/bin/ultra-cold-wav.rs` `load_native` + `decode`, over
//!   `ultra-stream/src/native_engine.rs` `load_inner_cfg` / `NativeEngineAsrStream`).
//! * `stream::Segmenter` (`stream.rs`) composes the two for `isosyll stream`.
//!
//! Final mode on audio longer than the commit window is cut at pauses by [`plan_cuts`] and each
//! piece goes through the same single commit pass.

use std::path::Path;
use std::sync::Arc;

use ultra_engine::commit::{
    CommitConfig, CommitLmConfig, CommitPassSession, DEFAULT_MAX_PCM_SAMPLES, DEFAULT_SAMPLE_RATE,
};
use ultra_engine::{Engine, EngineError, Isa, LmConfig, Model, NgramLm, Precision, Workspace};

use crate::model::ModelDir;
use crate::segment::plan_cuts;

/// Transport packet size used by both reference harnesses (20 ms at 16 kHz).
pub const PACKET_SAMPLES: usize = 320;

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
// v4.1.0: beam 12 with a 4.0 log-prob margin, blank penalty 0.24 and the compact M5 live LM
// (research H252: H247 step 450, DEV S-half).
const LIVE_BEAM: usize = 12;
const LIVE_BEAM_MARGIN: f32 = 4.0;
const LIVE_BLANK_PENALTY: f32 = 0.24;
const LIVE_LM_WEIGHT: f32 = 0.70;
const LIVE_LODR_WEIGHT: f32 = 0.50;

pub(crate) fn e2s(e: EngineError) -> String {
    e.to_string()
}

pub(crate) fn load_model(dir: &ModelDir, live_only: bool) -> Result<Arc<Model>, String> {
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

pub(crate) fn live_engine(
    model: &Arc<Model>,
    lm_path: &Path,
    threads: usize,
    live_only: bool,
) -> Result<Engine, String> {
    let lm = Arc::new(NgramLm::load(lm_path).map_err(|e| format!("{}: {e}", lm_path.display()))?);
    let cfg = LmConfig {
        lm_weight: LIVE_LM_WEIGHT,
        lodr_weight: LIVE_LODR_WEIGHT,
        blank_penalty: LIVE_BLANK_PENALTY,
    };
    Engine::new(Arc::clone(model), Isa::detect(), Precision::Int8)
        .map_err(e2s)?
        .with_gemm_threads(threads)
        .with_max_chunks(1)
        .with_beam(LIVE_BEAM)
        .with_beam_max_symbols(1)
        .with_beam_margin(LIVE_BEAM_MARGIN)
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

pub(crate) fn commit_config(rms_norm: bool) -> CommitConfig {
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

pub(crate) fn commit_lm(path: &Path) -> Result<CommitLmConfig, String> {
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
