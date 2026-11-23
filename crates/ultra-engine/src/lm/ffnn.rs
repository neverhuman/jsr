//! L2-resident INT8 fixed-context feed-forward streaming LM (`ultra.lm-ffnn/v1`).
//!
//! Per-row symmetric INT8 weights (`|q| <= 127`) with `f32` per-row scales and biases:
//! `E [V, d_e]` -> concat `ctx` context slots -> `W1 [d_h, ctx*d_e]` -> `ReLU`/`SiLU` ->
//! `W2 [V, d_h]` (or tied `E` when `d_h == d_e`) -> exact cross-ISA log-softmax.

use std::path::Path;

use super::ffnn_io;
use super::traits::TokenLm;
use crate::kernels::{self, Isa};
use crate::EngineError;

const STACK_MAX: usize = 2048;

/// Hidden activation supported by `ultra.lm-ffnn/v1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnnActivation {
    Relu,
    Silu,
}

/// Compact per-row INT8 fixed-context feed-forward language model.
#[derive(Debug, Clone, PartialEq)]
pub struct FfnnLm {
    pub vocab: usize,
    pub ctx: usize,
    pub d_e: usize,
    pub d_h: usize,
    pub tied: bool,
    pub act: FfnnActivation,
    pub(super) emb_q: Vec<i8>,
    pub(super) emb_scale: Vec<f32>,
    pub(super) w1_q: Vec<i8>,
    pub(super) w1_scale: Vec<f32>,
    pub(super) b1: Vec<f32>,
    pub(super) w2_q: Vec<i8>,
    pub(super) w2_scale: Vec<f32>,
    pub(super) b2: Vec<f32>,
}

impl FfnnLm {
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        vocab: usize,
        ctx: usize,
        d_e: usize,
        d_h: usize,
        tied: bool,
        act: FfnnActivation,
        emb_q: Vec<i8>,
        emb_scale: Vec<f32>,
        w1_q: Vec<i8>,
        w1_scale: Vec<f32>,
        b1: Vec<f32>,
        w2_q: Vec<i8>,
        w2_scale: Vec<f32>,
        b2: Vec<f32>,
    ) -> Result<Self, EngineError> {
        let dims_ok = vocab > 0 && vocab < usize::from(u16::MAX) && (1..=16).contains(&ctx);
        if !dims_ok || d_e == 0 || d_h == 0 || (tied && d_h != d_e) {
            return Err(EngineError::Shape("invalid FFNN LM dimensions".into()));
        }
        let w2_expect = if tied { (0, 0) } else { (vocab * d_h, vocab) };
        let lens_ok = emb_q.len() == vocab * d_e
            && emb_scale.len() == vocab
            && w1_q.len() == d_h * ctx * d_e
            && w1_scale.len() == d_h
            && b1.len() == d_h
            && w2_q.len() == w2_expect.0
            && w2_scale.len() == w2_expect.1
            && b2.len() == vocab;
        if !lens_ok {
            return Err(EngineError::Shape("FFNN LM tensor length mismatch".into()));
        }
        if emb_q.contains(&i8::MIN) || w1_q.contains(&i8::MIN) || w2_q.contains(&i8::MIN) {
            return Err(EngineError::Format("FFNN INT8 code -128 disallowed".into()));
        }
        let scales_ok = emb_scale
            .iter()
            .chain(&w1_scale)
            .chain(&w2_scale)
            .all(|&s| s.is_finite() && s >= 0.0);
        if !scales_ok || !b1.iter().chain(&b2).all(|&b| b.is_finite()) {
            return Err(EngineError::Format("invalid FFNN scale or bias".into()));
        }
        Ok(Self {
            vocab,
            ctx,
            d_e,
            d_h,
            tied,
            act,
            emb_q,
            emb_scale,
            w1_q,
            w1_scale,
            b1,
            w2_q,
            w2_scale,
            b2,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        ffnn_io::to_bytes(self)
    }

    pub fn from_bytes(raw: &[u8]) -> Result<Self, EngineError> {
        ffnn_io::from_bytes(raw)
    }

    pub fn save(&self, path: &Path) -> Result<(), EngineError> {
        ffnn_io::save(self, path)
    }

    pub fn load(path: &Path) -> Result<Self, EngineError> {
        ffnn_io::load(path)
    }

    /// Score all tokens using a specific ISA (for cross-ISA parity tests).
    pub fn score_all_isa(&self, isa: Isa, history: &[u32], out: &mut [f32]) {
        assert!(
            out.len() >= self.vocab,
            "FfnnLm::score_all output too small"
        );
        let mut slots = [0usize; 16];
        let take = history.len().min(self.ctx);
        let pad = self.ctx - take;
        for (i, &tok) in history[history.len() - take..].iter().enumerate() {
            slots[pad + i] = if (tok as usize) < self.vocab {
                tok as usize
            } else {
                0
            };
        }
        let (mut h_stack, mut h_heap) = ([0.0f32; STACK_MAX], Vec::new());
        let h: &mut [f32] = if self.d_h <= STACK_MAX {
            &mut h_stack[..self.d_h]
        } else {
            h_heap.resize(self.d_h, 0.0);
            &mut h_heap
        };
        let (ctx, d_e, d_h) = (self.ctx, self.d_e, self.d_h);
        for (j, hj) in h.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (s, &tok) in slots[..ctx].iter().enumerate() {
                let er = &self.emb_q[tok * d_e..(tok + 1) * d_e];
                let wr = &self.w1_q[(j * ctx + s) * d_e..(j * ctx + s + 1) * d_e];
                let dot: i32 = er
                    .iter()
                    .zip(wr)
                    .map(|(&e, &w)| i32::from(e) * i32::from(w))
                    .sum();
                acc += (dot as f32) * self.emb_scale[tok];
            }
            let pre = acc * self.w1_scale[j] + self.b1[j];
            *hj = if self.act == FfnnActivation::Relu {
                pre.max(0.0)
            } else {
                pre
            };
        }
        if self.act == FfnnActivation::Silu {
            kernels::silu_inplace(isa, h);
        }
        let (out_q, out_scale): (&[i8], &[f32]) = if self.tied {
            (&self.emb_q, &self.emb_scale)
        } else {
            (&self.w2_q, &self.w2_scale)
        };
        let logits = &mut out[..self.vocab];
        let mut max_z = f32::NEG_INFINITY;
        for (v, z_slot) in logits.iter_mut().enumerate() {
            let dot: f32 = h
                .iter()
                .zip(&out_q[v * d_h..(v + 1) * d_h])
                .map(|(&hv, &wq)| hv * f32::from(wq))
                .sum();
            *z_slot = dot * out_scale[v] + self.b2[v];
            max_z = max_z.max(*z_slot);
        }
        let lse = max_z + kernels::sum_exp_shifted(isa, logits, max_z).ln();
        for z in logits.iter_mut() {
            *z -= lse;
        }
    }
}

impl TokenLm for FfnnLm {
    fn score_next(&self, history: &[u32], token: u32) -> f32 {
        let t = token as usize;
        if t >= self.vocab {
            return f32::NEG_INFINITY;
        }
        let (mut stack, mut heap) = ([0.0f32; STACK_MAX], Vec::new());
        let buf = if self.vocab <= STACK_MAX {
            &mut stack[..self.vocab]
        } else {
            heap.resize(self.vocab, 0.0);
            &mut heap
        };
        self.score_all(history, buf);
        buf[t]
    }

    fn score_all(&self, history: &[u32], out: &mut [f32]) {
        self.score_all_isa(Isa::detect(), history, out);
    }

    #[inline]
    fn vocab_size(&self) -> usize {
        self.vocab
    }

    #[inline]
    fn size_bytes(&self) -> usize {
        let f32_elems = self.emb_scale.len()
            + self.w1_scale.len()
            + self.b1.len()
            + self.w2_scale.len()
            + self.b2.len();
        self.emb_q.len() + self.w1_q.len() + self.w2_q.len() + f32_elems * 4
    }
}
