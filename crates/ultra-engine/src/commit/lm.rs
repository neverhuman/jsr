//! Optional endpoint-only token n-gram LM + LODR beam configuration for the H45 commit pass.
//!
//! Keeps provisional live streaming 100% greedy / low-watt while enabling bounded LM beam search
//! (`beam_chunk_with_lm`) solely during the closed-segment endpoint commit pass.

use std::path::Path;
use std::sync::Arc;

use crate::lm::{LmConfig, NgramLm};
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
        if !(2..=16).contains(&self.beam) {
            return Err(EngineError::Shape(format!(
                "commit LM beam {} not in 2..=16",
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
