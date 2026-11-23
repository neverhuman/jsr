//! Exact Stateless-Predictor Internal Language Model Estimation (ILME).
//!
//! For a stateless RNN-T predictor (`ctx = [prev2, prev1]`) and `tanh` joiner:
//! `z = W_out @ tanh(e_avg + pred(prev2, prev1)) + b_out`,
//! `ln P_ILM(k | prev2, prev1) = z[k] - logsumexp_{j != blank}(z[j])` for `k != blank`,
//! and `-inf` at `k == blank`.

use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::encoder::{linear, Precision};
use crate::kernels::{self, Isa, NR};
use crate::model::Model;
use crate::quant::{QuantScratch, KG};
use crate::EngineError;

/// On-disk JSON schema for stateless-predictor `e_avg` vectors.
pub const ILME_SCHEMA: &str = "ultra.ilme-stateless/v1";

/// Serializable empirical mean non-blank encoder vector artifact (`ultra.ilme-stateless/v1`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IlmeArtifact {
    pub schema: String,
    pub joiner: usize,
    pub num_utts: usize,
    pub num_emissions: u64,
    pub model_weights_sha256: String,
    pub e_avg_sha256: String,
    pub e_avg: Vec<f32>,
}

/// Exact stateless-predictor ILME scorer for a loaded [`Model`].
#[derive(Debug, Clone)]
pub struct StatelessIlme {
    pub vocab: usize,
    pub joiner: usize,
    pub blank: usize,
    pub ctx_init: [usize; 2],
    e_avg: Vec<f32>,
    pred_h0: Vec<f32>,
    pred_h1: Vec<f32>,
    join_w: Vec<f32>,
    join_b: Vec<f32>,
}

fn sha256_f32_le(vals: &[f32]) -> String {
    let mut h = Sha256::new();
    for &v in vals {
        h.update(v.to_le_bytes());
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn f32_linear(layer: &crate::model::Linear, inp: &[f32], rows: usize, out: &mut [f32]) {
    let mut q = QuantScratch::default();
    linear(Isa::Scalar, Precision::F32, layer, inp, rows, out, &mut q);
}

fn extract_pred_halves(model: &Model) -> Result<(Vec<f32>, Vec<f32>), EngineError> {
    if model.arch.ctx_init.len() != 2 {
        return Err(EngineError::Unsupported(
            "StatelessIlme needs context_len == 2".into(),
        ));
    }
    if let Some(pc) = &model.pred_cache {
        return Ok((pc.h0.clone(), pc.h1.clone()));
    }
    let (v, e, jn) = (model.arch.vocab, model.arch.pred_emb, model.arch.joiner);
    let mut bias = vec![0.0f32; jn];
    f32_linear(&model.pred_proj, &vec![0.0; 2 * e], 1, &mut bias);
    let (mut inp0, mut inp1) = (vec![0.0f32; v * 2 * e], vec![0.0f32; v * 2 * e]);
    for tok in 0..v {
        let er = &model.emb[tok * e..(tok + 1) * e];
        inp0[tok * 2 * e..tok * 2 * e + e].copy_from_slice(er);
        inp1[tok * 2 * e + e..(tok + 1) * 2 * e].copy_from_slice(er);
    }
    let (mut h0, mut h1) = (vec![0.0f32; v * jn], vec![0.0f32; v * jn]);
    f32_linear(&model.pred_proj, &inp0, v, &mut h0);
    f32_linear(&model.pred_proj, &inp1, v, &mut h1);
    for tok in 0..v {
        for j in 0..jn {
            h1[tok * jn + j] -= bias[j];
        }
    }
    Ok((h0, h1))
}

fn extract_joiner_wb(model: &Model) -> Result<(Vec<f32>, Vec<f32>), EngineError> {
    let (v, jn) = (model.arch.vocab, model.arch.joiner);
    if model.joiner_out.has_fp32() {
        let mut b = vec![0.0f32; v];
        f32_linear(&model.joiner_out, &vec![0.0; jn], 1, &mut b);
        let mut eye = vec![0.0f32; jn * jn];
        for k in 0..jn {
            eye[k * jn + k] = 1.0;
        }
        let mut cols = vec![0.0f32; jn * v];
        f32_linear(&model.joiner_out, &eye, jn, &mut cols);
        let mut w = vec![0.0f32; v * jn];
        for k in 0..jn {
            for row in 0..v {
                w[row * jn + k] = cols[k * v + row] - b[row];
            }
        }
        return Ok((w, b));
    }
    let Some(pq) = &model.joiner_out.q else {
        return Err(EngineError::Unsupported(
            "joiner_out lacks FP32 and INT8 weights".into(),
        ));
    };
    let (mut w, mut b) = (vec![0.0f32; v * jn], vec![0.0f32; v]);
    for row in 0..v {
        let (p, j, s) = (row / NR, row % NR, pq.scale[row]);
        let panel = pq.panel(p);
        b[row] = pq.bias[row];
        for kk in 0..jn {
            w[row * jn + kk] = f32::from(panel[((kk / KG) * NR + j) * KG + kk % KG]) * s;
        }
    }
    Ok((w, b))
}

impl StatelessIlme {
    /// Construct an exact stateless-predictor ILME from `model` and mean encoder vector `e_avg`.
    pub fn from_model(model: &Model, e_avg: Vec<f32>) -> Result<Self, EngineError> {
        let jn = model.arch.joiner;
        if e_avg.len() != jn {
            return Err(EngineError::Shape(format!(
                "e_avg len {} != joiner {jn}",
                e_avg.len()
            )));
        }
        if !e_avg.iter().all(|v| v.is_finite()) {
            return Err(EngineError::Format(
                "ILME e_avg contains non-finite values".into(),
            ));
        }
        let (pred_h0, pred_h1) = extract_pred_halves(model)?;
        let (join_w, join_b) = extract_joiner_wb(model)?;
        Ok(Self {
            vocab: model.arch.vocab,
            joiner: jn,
            blank: model.arch.blank,
            ctx_init: [model.arch.ctx_init[0], model.arch.ctx_init[1]],
            e_avg,
            pred_h0,
            pred_h1,
            join_w,
            join_b,
        })
    }

    /// Construct zero-encoder ILME (`e_avg = 0`, Meng et al. baseline).
    pub fn from_zero_enc(model: &Model) -> Result<Self, EngineError> {
        Self::from_model(model, vec![0.0f32; model.arch.joiner])
    }

    /// Load an `ultra.ilme-stateless/v1` JSON artifact and verify its schema and SHA-256.
    pub fn from_eavg_file(model: &Model, path: &Path) -> Result<Self, EngineError> {
        let art: IlmeArtifact = serde_json::from_str(&std::fs::read_to_string(path)?)
            .map_err(|e| EngineError::Format(format!("ILME {}: {e}", path.display())))?;
        if art.schema != ILME_SCHEMA {
            return Err(EngineError::Format(format!(
                "ILME schema {} != {ILME_SCHEMA}",
                art.schema
            )));
        }
        if art.joiner != art.e_avg.len() {
            return Err(EngineError::Shape(
                "ILME artifact joiner != e_avg.len()".into(),
            ));
        }
        let got_sha = sha256_f32_le(&art.e_avg);
        if got_sha != art.e_avg_sha256 {
            return Err(EngineError::Integrity(format!(
                "ILME sha256 mismatch: {got_sha}"
            )));
        }
        Self::from_model(model, art.e_avg)
    }

    /// Save an empirical `e_avg` vector in `ultra.ilme-stateless/v1` JSON format.
    pub fn save_eavg_json(
        path: &Path,
        e_avg: &[f32],
        num_utts: usize,
        num_emissions: u64,
        model_weights_sha256: &str,
    ) -> Result<IlmeArtifact, EngineError> {
        let art = IlmeArtifact {
            schema: ILME_SCHEMA.to_owned(),
            joiner: e_avg.len(),
            num_utts,
            num_emissions,
            model_weights_sha256: model_weights_sha256.to_owned(),
            e_avg_sha256: sha256_f32_le(e_avg),
            e_avg: e_avg.to_vec(),
        };
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(&art)
            .map_err(|e| EngineError::Format(format!("serialize ILME: {e}")))?;
        std::fs::write(path, json)?;
        Ok(art)
    }

    pub fn e_avg(&self) -> &[f32] {
        &self.e_avg
    }

    /// Map chronological token history to `(prev2, prev1)` using `ctx_init`.
    #[inline]
    pub fn ctx_pair(&self, history: &[u32]) -> (usize, usize) {
        match history.len() {
            0 => (self.ctx_init[0], self.ctx_init[1]),
            1 => (self.ctx_init[1], (history[0] as usize).min(self.vocab - 1)),
            n => (
                (history[n - 2] as usize).min(self.vocab - 1),
                (history[n - 1] as usize).min(self.vocab - 1),
            ),
        }
    }

    /// Fill `out[..self.vocab]` with exact non-blank ILME natural-log probabilities for `(prev2, prev1)`.
    pub fn score_ctx_isa(&self, isa: Isa, prev2: usize, prev1: usize, out: &mut [f32]) {
        let (v, jn, blank) = (self.vocab, self.joiner, self.blank);
        assert!(out.len() >= v, "StatelessIlme output buffer too small");
        let (p0, p1) = (prev2.min(v - 1), prev1.min(v - 1));
        let h0 = &self.pred_h0[p0 * jn..(p0 + 1) * jn];
        let h1 = &self.pred_h1[p1 * jn..(p1 + 1) * jn];
        let hid: Vec<f32> = (0..jn)
            .map(|j| (self.e_avg[j] + h0[j] + h1[j]).tanh())
            .collect();
        let logits = &mut out[..v];
        let mut max_nb = f32::NEG_INFINITY;
        for (row, z_slot) in logits.iter_mut().enumerate() {
            if row == blank {
                *z_slot = f32::NEG_INFINITY;
                continue;
            }
            let w_row = &self.join_w[row * jn..(row + 1) * jn];
            let acc: f64 = w_row
                .iter()
                .zip(&hid)
                .fold(f64::from(self.join_b[row]), |s, (&w, &h)| {
                    s + f64::from(w) * f64::from(h)
                });
            let z = acc as f32;
            *z_slot = z;
            max_nb = max_nb.max(z);
        }
        let lse = max_nb + kernels::sum_exp_shifted(isa, logits, max_nb).ln();
        for (row, z) in logits.iter_mut().enumerate() {
            *z = if row == blank {
                f32::NEG_INFINITY
            } else {
                *z - lse
            };
        }
    }

    /// Fill `out[..self.vocab]` with ILME log-probabilities for chronological `history`.
    pub fn score_history(&self, history: &[u32], out: &mut [f32]) {
        let (p0, p1) = self.ctx_pair(history);
        self.score_ctx_isa(Isa::detect(), p0, p1, out);
    }
}
