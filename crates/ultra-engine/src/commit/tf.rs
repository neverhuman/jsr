//! Opt-in INT8 L2R + R2L Transformer N-best rescoring for the H45/D4 endpoint commit pass.
//!
//! When disabled (`None`), adds zero allocations and zero runtime cost so `v3.0.0` stays
//! byte-identical. When both L2R and R2L are active, runs a 2-stage cascade: L2R scores all
//! beam hypotheses via prefix-trie batching, then R2L scores the top `r2l_top_k` finalists
//! (default `4`; set `0` for exhaustive scoring) via suffix-trie batching.

use std::path::Path;
use std::sync::Arc;

use super::CommitContentReranker;
use crate::beam::Hyp;
use crate::lm::{NgramLm, TfDirection, TfRescorer, TfWorkspace};
use crate::{Engine, EngineError};

/// Default tied weight on L2R and R2L Transformer `total_logp` (Wave 1B-2 S-half CV row).
pub const DEFAULT_RESCORE_WEIGHT: f32 = 0.38;
/// Default M5 n-gram log-prob subtraction weight when Transformer rescoring is active.
pub const DEFAULT_RESCORE_M5_SUB: f32 = 0.50;
/// Default Stage-2 R2L finalist count when both L2R and R2L are enabled (`0` = score all).
pub const DEFAULT_RESCORE_R2L_TOP_K: usize = 4;

/// Opt-in commit-pass N-best Transformer rescorer configuration.
#[derive(Debug, Clone)]
pub struct CommitTfRescoreConfig {
    pub l2r: Option<Arc<TfRescorer>>,
    pub r2l: Option<Arc<TfRescorer>>,
    pub rescore_weight: f32,
    pub m5_sub: f32,
    pub r2l_top_k: usize,
    /// Skip Transformer rescoring when the fused-beam top-1 minus top-2 score is at least this
    /// margin (`f32::INFINITY` = always rescore, the default).
    pub skip_margin: f32,
    /// 7-bit activation codes in the rescorer's INT8 linears (`gemm_i8_a7` on AVX2); default off.
    pub act7: bool,
    content: Option<Arc<CommitContentReranker>>,
}

impl CommitTfRescoreConfig {
    pub fn new(
        l2r: Option<Arc<TfRescorer>>,
        r2l: Option<Arc<TfRescorer>>,
        rescore_weight: f32,
        m5_sub: f32,
    ) -> Result<Self, EngineError> {
        let cfg = Self {
            l2r,
            r2l,
            rescore_weight,
            m5_sub,
            r2l_top_k: DEFAULT_RESCORE_R2L_TOP_K,
            skip_margin: f32::INFINITY,
            act7: false,
            content: None,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Override Stage-2 R2L finalist count (`0` scores all beam hypotheses exhaustively).
    pub fn with_r2l_top_k(mut self, top_k: usize) -> Self {
        self.r2l_top_k = top_k;
        self
    }

    /// Only rescore utterances whose fused-beam top-1 margin is below `margin`.
    pub fn with_skip_margin(mut self, margin: f32) -> Self {
        self.skip_margin = margin;
        self
    }

    /// Use 7-bit activation codes in the rescorer's INT8 linears.
    pub fn with_act7(mut self, on: bool) -> Self {
        self.act7 = on;
        self
    }

    /// Use the frozen content feature weights with the same native L2R forward pass.
    pub fn with_content_reranker(
        mut self,
        content: CommitContentReranker,
    ) -> Result<Self, EngineError> {
        self.content = Some(Arc::new(content));
        self.validate()?;
        Ok(self)
    }

    /// Load optional `.utfr` L2R and/or R2L models. Returns `Ok(None)` when both paths are `None`.
    pub fn from_files(
        l2r_path: Option<&Path>,
        r2l_path: Option<&Path>,
        rescore_weight: f32,
        m5_sub: f32,
    ) -> Result<Option<Self>, EngineError> {
        if l2r_path.is_none() && r2l_path.is_none() {
            return Ok(None);
        }
        let l2r = l2r_path
            .map(|p| TfRescorer::load(p).map(Arc::new))
            .transpose()?;
        let r2l = r2l_path
            .map(|p| TfRescorer::load(p).map(Arc::new))
            .transpose()?;
        Self::new(l2r, r2l, rescore_weight, m5_sub).map(Some)
    }

    pub fn validate(&self) -> Result<(), EngineError> {
        if self.l2r.is_none() && self.r2l.is_none() {
            return Err(EngineError::Format(
                "CommitTfRescoreConfig requires at least one of l2r or r2l".into(),
            ));
        }
        if let Some(m) = &self.l2r {
            if m.cfg.direction != TfDirection::L2r {
                return Err(EngineError::Format(
                    "l2r rescorer model has non-L2R direction".into(),
                ));
            }
        }
        if let Some(m) = &self.r2l {
            if m.cfg.direction != TfDirection::R2l {
                return Err(EngineError::Format(
                    "r2l rescorer model has non-R2L direction".into(),
                ));
            }
        }
        if let Some(content) = &self.content {
            let l2r = self
                .l2r
                .as_ref()
                .filter(|_| self.r2l.is_none())
                .ok_or_else(|| {
                    EngineError::Format("content reranker requires L2R without R2L".into())
                })?;
            content.validate_model(l2r)?;
        }
        if self.skip_margin.is_nan() || self.skip_margin < 0.0 {
            return Err(EngineError::Format(
                "skip_margin must be >= 0 (or +inf)".into(),
            ));
        }
        if !self.rescore_weight.is_finite()
            || self.rescore_weight < 0.0
            || !self.m5_sub.is_finite()
            || self.m5_sub < 0.0
        {
            return Err(EngineError::Format(
                "rescore_weight and m5_sub must be finite and >= 0".into(),
            ));
        }
        Ok(())
    }

    /// Compute rescored objective for every hypothesis in `hyps`.
    pub(crate) fn rescore_hyps(
        &self,
        eng: &Engine,
        hyps: &[Hyp],
        ws: &mut TfWorkspace,
    ) -> Result<Vec<f32>, EngineError> {
        if let Some(content) = &self.content {
            let l2r = self
                .l2r
                .as_ref()
                .ok_or_else(|| EngineError::Format("missing content L2R model".into()))?;
            let lm = eng
                .lm
                .as_ref()
                .ok_or_else(|| EngineError::Format("missing content M5 model".into()))?;
            let candidates: Vec<_> = hyps
                .iter()
                .map(|hy| (&hy.tokens[hy.ctx_start..], hy.score))
                .collect();
            return content.score_batch(l2r, eng.isa, lm.0.as_ref(), &candidates, ws);
        }
        if self.skip_margin.is_finite() && hyps.len() > 1 {
            let (mut b1, mut b2) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
            for h in hyps {
                if h.score > b1 {
                    b2 = b1;
                    b1 = h.score;
                } else if h.score > b2 {
                    b2 = h.score;
                }
            }
            if b1 - b2 >= self.skip_margin {
                return Ok(hyps.iter().map(|h| h.score).collect());
            }
        }
        let seqs: Vec<&[u32]> = hyps.iter().map(|h| &h.tokens[h.ctx_start..]).collect();
        let ngram = eng.lm.as_ref().map(|(lm, _)| lm.as_ref());
        let l2r_sc = match &self.l2r {
            Some(m) => Some(m.score_batch(eng.isa, &seqs, ws)?),
            None => None,
        };
        let mut s1 = Vec::with_capacity(hyps.len());
        for (i, hy) in hyps.iter().enumerate() {
            let mut s = hy.score;
            if let Some(ls) = &l2r_sc {
                s += self.rescore_weight * ls[i].total_logp;
            }
            if self.m5_sub > 0.0 {
                if let Some(lm) = ngram {
                    s -= self.m5_sub * score_ngram_seq(lm, seqs[i]);
                }
            }
            s1.push(s);
        }
        let Some(r2l) = &self.r2l else {
            return Ok(s1);
        };
        let use_cascade = self.l2r.is_some() && self.r2l_top_k > 0 && hyps.len() > self.r2l_top_k;
        if !use_cascade {
            let r2l_sc = r2l.score_batch(eng.isa, &seqs, ws)?;
            for (s, rs) in s1.iter_mut().zip(&r2l_sc) {
                *s += self.rescore_weight * rs.total_logp;
            }
            return Ok(s1);
        }
        let mut order: Vec<usize> = (0..hyps.len()).collect();
        order.sort_by(|&a, &b| s1[b].total_cmp(&s1[a]).then_with(|| a.cmp(&b)));
        order.truncate(self.r2l_top_k);
        let sub_seqs: Vec<&[u32]> = order.iter().map(|&i| seqs[i]).collect();
        let r2l_sc = r2l.score_batch(eng.isa, &sub_seqs, ws)?;
        let mut out = vec![f32::NEG_INFINITY; hyps.len()];
        for (&idx, rs) in order.iter().zip(&r2l_sc) {
            out[idx] = s1[idx] + self.rescore_weight * rs.total_logp;
        }
        Ok(out)
    }
}

/// Exact total M5 n-gram log-probability of `tokens` (`BeginSentence` semantics, matching beam search).
pub fn score_ngram_seq(lm: &NgramLm, tokens: &[u32]) -> f32 {
    let mut total = 0.0f32;
    for i in 0..tokens.len() {
        let ctx = lm.ctx_state(&tokens[..i]);
        total += lm.lm_logp(&ctx, tokens[i] as usize);
    }
    total
}

impl super::CommitPassSession {
    /// Attach an opt-in INT8 L2R/R2L Transformer N-best rescorer to the endpoint commit pass.
    /// When `None`, behavior is bit-identical with zero allocation and zero overhead.
    pub fn with_commit_tf_rescorer(
        mut self,
        cfg: Option<CommitTfRescoreConfig>,
    ) -> Result<Self, EngineError> {
        if let Some(c) = cfg {
            c.validate()?;
            let mut ws = TfWorkspace::default();
            ws.set_act7(c.act7);
            self.tf = Some((c, ws));
        } else {
            self.tf = None;
        }
        Ok(self)
    }

    pub fn has_commit_tf_rescorer(&self) -> bool {
        self.tf.is_some()
    }
}
