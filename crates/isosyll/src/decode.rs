//! Release decoders. Each reproduces a shipped binary's call sequence with its release flags.
//!
//! * [`FinalDecoder`]: `ultra-engine-commit` (`engine_commit_eval/main.rs`) on the separate final
//!   acoustic model (`<model>/final/`): `Model::load(.., true)`, `Engine::new(.., Int8)
//!   .with_gemm_threads(t)`, `CommitPassSession::new` + `with_commit_lm` +
//!   `with_commit_tf_rescorer` (L2R Transformer with the frozen content-word reranker) +
//!   `with_wide_gate`, 320-sample `accept_i16` packets, `commit_endpoint`, `reset_all`
//!   (v4.2.0, research H294 `b48-train30`).
//! * [`LiveDecoder`]: `ultra-cold-wav` backend `r1-engine-int8`
//!   (`ultra-bridge/src/bin/ultra-cold-wav.rs` `load_native` + `decode`, over
//!   `ultra-stream/src/native_engine.rs` `load_inner_cfg` / `NativeEngineAsrStream`).
//! * `stream::Segmenter` (`stream.rs`) drives `isosyll stream` with the live model for both passes
//!   (its commit keeps the v4.1.0 c16 B16 M5 settings: [`stream_commit_config`]).
//!
//! Final mode on audio longer than the commit window is cut at pauses by [`plan_cuts`] and each
//! piece goes through the same single commit pass.

use std::path::Path;
use std::sync::Arc;

use ultra_engine::commit::{
    CommitConfig, CommitContentReranker, CommitLmConfig, CommitPassSession, CommitTfRescoreConfig,
    CommitWideGate, DEFAULT_MAX_PCM_SAMPLES, DEFAULT_SAMPLE_RATE,
};
use ultra_engine::{Engine, EngineError, Isa, LmConfig, Model, NgramLm, Precision, Workspace};

use crate::model::ModelDir;
use crate::segment::plan_cuts;

/// Transport packet size used by both reference harnesses (20 ms at 16 kHz).
pub const PACKET_SAMPLES: usize = 320;

// Final (commit) release flags. v4.2.0 (research H294 `b48-train30`): the F3 acoustic soup at
// c32 B16, M5 0.80 + LODR 0.60, a 320 ms zero tail; a B16 top-1 minus top-2 fused-score gap below
// the frozen TRAIN-30% cutoff re-searches the cached encoder projection at B48; the L2R t2
// Transformer with the frozen H261 content-word weights then reranks (skip margin 3.75, in the
// fit file). `rescore_weight` / `m5_sub` are required by the config but unused with the reranker.
const COMMIT_CHUNK: usize = 32;
const COMMIT_BEAM: usize = 16;
const WIDE_BEAM: usize = 48;
/// TRAIN-30% quantile of the B16 margin on the 9,196-utterance H286 F3 TRAIN lists
/// (`H294/thresholds.json`: 0.6467037200927734, exactly this f32).
const WIDE_MARGIN: f32 = 0.646_703_7;
const RESCORE_WEIGHT: f32 = 0.80;
const RESCORE_M5_SUB: f32 = 0.80;
const RESCORE_SKIP_MARGIN: f32 = 3.75;
// `isosyll stream` commit (live model): unchanged from v4.1.0.
const STREAM_COMMIT_CHUNK: usize = 16;
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
    load_bundle(&dir.root, live_only)
}

fn load_bundle(root: &Path, live_only: bool) -> Result<Arc<Model>, String> {
    let model = if live_only {
        Model::load_live(root, true)
    } else {
        Model::load(root, true)
    }
    .map_err(|e| format!("{}: {e}", root.display()))?;
    if model.int8_prepared.is_none() {
        return Err(format!(
            "{}: bundle has no weights.int8.safetensors (INT8 bundle required)",
            root.display()
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
        ..stream_commit_config(rms_norm)
    }
}

/// Commit settings of `isosyll stream` (live model, c16).
pub(crate) fn stream_commit_config(rms_norm: bool) -> CommitConfig {
    CommitConfig {
        commit_chunk: STREAM_COMMIT_CHUNK,
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

/// The final pass's L2R Transformer rescorer with the frozen content-word reranker.
pub(crate) fn commit_rescorer(dir: &ModelDir) -> Result<CommitTfRescoreConfig, String> {
    let (t2, fit, lexicon) = (
        dir.final_rescorer(),
        dir.content_fit(),
        dir.content_lexicon(),
    );
    let content = CommitContentReranker::from_files(&fit, &lexicon)
        .map_err(|e| format!("{} / {}: {e}", fit.display(), lexicon.display()))?;
    CommitTfRescoreConfig::from_files(Some(&t2), None, RESCORE_WEIGHT, RESCORE_M5_SUB)
        .and_then(|c| {
            c.ok_or_else(|| EngineError::Format("no rescorer".into()))?
                .with_skip_margin(RESCORE_SKIP_MARGIN)
                .with_act7(false)
                .with_content_reranker(content)
        })
        .map_err(|e| format!("{}: {e}", t2.display()))
}

pub(crate) fn wide_gate() -> CommitWideGate {
    CommitWideGate {
        width: WIDE_BEAM,
        margin: WIDE_MARGIN,
    }
}

/// `ultra-engine-commit` release decoder, with pause-based cutting for audio beyond the commit window.
pub struct FinalDecoder {
    sess: CommitPassSession,
}

impl FinalDecoder {
    pub fn new(dir: &ModelDir, threads: usize, rms_norm: bool) -> Result<Self, String> {
        let model = load_bundle(&dir.final_acoustic(), false)?;
        let eng = Engine::new(model, Isa::detect(), Precision::Int8)
            .map_err(e2s)?
            .with_gemm_threads(threads);
        let lm = commit_lm(&dir.final_lm())?;
        let tf = commit_rescorer(dir)?;
        let sess = CommitPassSession::new(eng, commit_config(rms_norm))
            .and_then(|s| s.with_commit_lm(Some(lm)))
            .and_then(|s| s.with_commit_tf_rescorer(Some(tf)))
            .map_err(e2s)?
            .with_wide_gate(Some(wide_gate()));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_pass_is_the_locked_h294_b48_train30_policy() {
        // H294/thresholds.json "30": 0.6467037200927734 is this exact f32.
        assert_eq!(WIDE_MARGIN.to_bits(), 0x3f25_8e60);
        assert_eq!(f64::from(WIDE_MARGIN), 0.646_703_720_092_773_4);
        assert_eq!(
            wide_gate(),
            CommitWideGate::new(48, WIDE_MARGIN).expect("valid gate")
        );
        let c = commit_config(false);
        assert_eq!(
            (c.commit_chunk, c.beam_size, c.tail_pad_samples),
            (32, 16, 5120)
        );
        assert_eq!(
            (c.ctc_weight, c.mbr_enabled, c.rms_norm),
            (0.0, false, false)
        );
        assert_eq!(c.max_pcm_samples, DEFAULT_MAX_PCM_SAMPLES);
        assert!(commit_config(true).rms_norm);
        assert_eq!((LM_SCALE, LODR_SCALE, TOKEN_BONUS), (0.80, 0.60, 0.0));
        assert_eq!(RESCORE_SKIP_MARGIN, 3.75);
    }

    #[test]
    fn stream_commit_keeps_the_v410_settings() {
        let s = stream_commit_config(false);
        assert_eq!(
            s,
            CommitConfig {
                commit_chunk: 16,
                ..commit_config(false)
            }
        );
    }

    #[test]
    fn missing_rescorer_files_are_named() {
        let td = tempfile::tempdir().expect("tempdir");
        let dir = ModelDir {
            root: td.path().to_path_buf(),
        };
        let err = commit_rescorer(&dir).expect_err("no files");
        assert!(err.contains("content-fit.json"), "{err}");
    }
}
