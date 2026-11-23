//! Optional endpoint-only token n-gram LM + LODR beam configuration for the H45 commit pass.
//!
//! Keeps provisional live streaming 100% greedy / low-watt while enabling bounded LM beam search
//! (`beam_chunk_with_lm`) solely during the closed-segment endpoint commit pass.

use std::path::Path;
use std::sync::Arc;

use ultra_frontend::{TrainerLogMel, MEL_BINS};
use ultra_gp::{chunk_bounds, FeatureRing};

use super::lm_gate::{decode_enc_gated, CommitTokenLm, InternalLm, LmGateConfig};
use crate::encoder::{encode_chunk, ChunkJob, EncState, EncWorkspace, Precision};
use crate::kernels::Isa;
use crate::lm::{LmConfig, NgramLm};
use crate::model::Model;
use crate::{Engine, EngineError};

/// Endpoint-only LM shallow-fusion + LODR beam settings for [`super::CommitPassSession`].
#[derive(Debug, Clone)]
pub struct CommitLmConfig {
    pub lm: Arc<NgramLm>,
    pub beam: usize,
    pub lm_scale: f32,
    pub lodr_scale: f32,
    pub token_bonus: f32,
    pub max_sym_per_frame: usize,
}

impl CommitLmConfig {
    /// Construct endpoint LM beam settings (defaults: `token_bonus = 0.0`, `max_sym_per_frame = 1`).
    pub fn new(lm: Arc<NgramLm>, beam: usize, lm_scale: f32, lodr_scale: f32) -> Self {
        Self {
            lm,
            beam,
            lm_scale,
            lodr_scale,
            token_bonus: 0.0,
            max_sym_per_frame: 1,
        }
    }

    /// Load a packed `ultra.lm-ngram/v1|v2` (`.ulm`) file for endpoint commit beam decoding.
    pub fn from_file(
        path: &Path,
        beam: usize,
        lm_scale: f32,
        lodr_scale: f32,
        token_bonus: f32,
        max_sym_per_frame: usize,
    ) -> Result<Self, EngineError> {
        let lm = Arc::new(NgramLm::load(path)?);
        let cfg = Self {
            lm,
            beam,
            lm_scale,
            lodr_scale,
            token_bonus,
            max_sym_per_frame,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), EngineError> {
        if !(2..=64).contains(&self.beam) {
            return Err(EngineError::Shape(format!(
                "commit LM beam {} not in 2..=64",
                self.beam
            )));
        }
        if !(1..=8).contains(&self.max_sym_per_frame) {
            return Err(EngineError::Shape(format!(
                "commit LM max_sym_per_frame {} not in 1..=8",
                self.max_sym_per_frame
            )));
        }
        self.lm_config().validate()
    }

    pub fn lm_config(&self) -> LmConfig {
        LmConfig {
            lm_weight: self.lm_scale,
            lodr_weight: self.lodr_scale,
            blank_penalty: self.token_bonus,
        }
    }

    pub(crate) fn apply_to_commit_engine(&self, engine: Engine) -> Result<Engine, EngineError> {
        self.validate()?;
        engine
            .with_beam(self.beam)
            .with_beam_max_symbols(self.max_sym_per_frame)
            .with_lm(self.lm.clone(), self.lm_config())
    }
}

impl super::CommitPassSession {
    /// Attach optional endpoint-only LM+LODR beam decoding to `commit_engine`.
    /// When `None`, behavior is bit-identical with zero overhead; `provisional_engine` is untouched.
    pub fn with_commit_lm(mut self, lm_cfg: Option<CommitLmConfig>) -> Result<Self, EngineError> {
        if let Some(cfg) = lm_cfg {
            self.commit_engine = cfg.apply_to_commit_engine(self.commit_engine)?;
        }
        Ok(self)
    }

    pub fn has_commit_lm(&self) -> bool {
        self.commit_engine.lm.is_some()
    }
}

/// Extended Wave-1B endpoint LM + ILME + acoustic-confidence gate bundle.
#[derive(Debug, Clone)]
pub struct CommitLmGateConfig {
    pub token_lm: Option<CommitTokenLm>,
    pub internal_lm: InternalLm,
    pub gate: LmGateConfig,
}

impl CommitLmGateConfig {
    /// Run a closed-segment commit pass on raw PCM using `CommitTokenLm` + `InternalLm` + margin gate.
    pub fn decode_pcm(
        &self,
        engine: &Engine,
        commit_chunk: usize,
        pcm: &[i16],
        tail_pad_samples: usize,
    ) -> Result<String, EngineError> {
        let (enc, n_frames) = encode_pcm_frames(
            &engine.model,
            engine.isa,
            engine.precision,
            commit_chunk,
            pcm,
            tail_pad_samples,
        )?;
        let toks = decode_enc_gated(
            &engine.model,
            engine.isa,
            engine.precision,
            &enc,
            n_frames,
            self.token_lm.as_ref(),
            &self.internal_lm,
            &self.gate,
        )?;
        let ids: Vec<i64> = toks.into_iter().map(i64::from).collect();
        Ok(engine.model.tokens.decode_ids(&ids))
    }
}

/// Encode PCM (+ optional trailing silence pad) into projected encoder frames `[T, joiner]`.
pub fn encode_pcm_frames(
    model: &Model,
    isa: Isa,
    prec: Precision,
    chunk: usize,
    pcm: &[i16],
    tail_pad_samples: usize,
) -> Result<(Vec<f32>, usize), EngineError> {
    let mut mel = TrainerLogMel::new(model.mel_cfg.clone());
    let (mut rows, mut ring) = (Vec::new(), FeatureRing::new());
    mel.push_i16_into(pcm, &mut rows);
    ring.push_rows(&rows);
    if tail_pad_samples > 0 {
        rows.clear();
        mel.push_i16_into(&vec![0i16; tail_pad_samples], &mut rows);
        ring.push_rows(&rows);
    }
    let mut enc = EncState::new_with_kv_and_chunk(model, 1, false, chunk);
    let (mut ws, mut feats, mut out, mut all) =
        (EncWorkspace::default(), Vec::new(), Vec::new(), Vec::new());
    let (mut nf, jn) = (0usize, model.arch.joiner);
    while let Some((t0, t1)) = chunk_bounds(enc.t, ring.n_feat(), chunk, true) {
        let n = t1 - t0;
        let need = (4 * n + 3) * MEL_BINS;
        if feats.len() < need {
            feats.resize(need, 0.0);
        }
        ring.extract_chunk_into(t0, n, &mut feats[..need]);
        let mut jobs = [ChunkJob {
            state: &mut enc,
            feats: &feats[..need],
            n,
        }];
        encode_chunk(model, isa, prec, &mut jobs, &mut ws, &mut out, None)?;
        all.extend_from_slice(&out[..n * jn]);
        nf += n;
    }
    Ok((all, nf))
}
