//! The model's quantisable linear layers: enumeration, ids and int8 quantisation.

use super::{Linear, Model};
use crate::quant::{PackedI8, WeightRule};
use crate::EngineError;

/// The stem layers `train.qat.apply_int8_stem` trains as int8 GEMMs.
pub const QAT_STEM_KINDS: [&str; 2] = ["stem_c2", "stem_out"];

/// `model_config.qat_stem` from arch.json: absent or `"none"` -> false, `"int8"` -> true.
pub(super) fn qat_stem_int8(v: Option<&str>) -> Result<bool, EngineError> {
    match v {
        None | Some("none") => Ok(false),
        Some("int8") => Ok(true),
        Some(o) => Err(EngineError::Unsupported(format!("qat_stem {o}"))),
    }
}

impl Model {
    /// Pack int8 copies (per-output-channel scales) of every encoder and joiner linear layer.
    pub fn quantize_int8(&mut self) {
        self.quantize_int8_where(|_, _| true);
    }

    /// Pack int8 copies of the layers for which `keep(kind, block)` is true (ablations);
    /// the others stay fp32 in the int8 engine.
    pub fn quantize_int8_where(&mut self, keep: impl Fn(&str, Option<usize>) -> bool) {
        let qat = self.arch.qat_stem_int8;
        for (kind, blk, l) in self.linears_mut() {
            if !l.has_fp32() {
                continue; // prepared int8 layer: its packing comes from the artifact
            }
            if qat && QAT_STEM_KINDS.contains(&kind) {
                continue; // trained int8: packed once by apply_qat_stem, always kept
            }
            l.q = keep(kind, blk).then(|| PackedI8::from_packed_f32(&l.f));
        }
    }

    /// For a checkpoint whose stem was trained int8 (`qat_stem = "int8"`), pack stem conv 2
    /// and the projection with the trainer's weight rule and run them int8 in every
    /// precision (the fp32 engine is then "fp32 except the int8-trained stem", as trained).
    /// A prepared artifact must already hold them (its writer used this packing).
    pub(super) fn apply_qat_stem(&mut self) -> Result<(), EngineError> {
        if !self.arch.qat_stem_int8 {
            return Ok(());
        }
        for (kind, _, l) in self.linears_mut() {
            if !QAT_STEM_KINDS.contains(&kind) {
                continue;
            }
            if l.has_fp32() {
                l.q = Some(PackedI8::from_packed_f32_with(&l.f, WeightRule::Trainer));
            } else if l.q.is_none() {
                return Err(EngineError::Integrity(format!(
                    "int8-trained {kind} has neither fp32 nor prepared int8 weights"
                )));
            }
            l.qat_i8 = true;
        }
        Ok(())
    }

    /// Post-training int8 for layers of the given kinds that are still fp32 (e.g. the stem
    /// of an R4 ternary model, kept fp32 by its training recipe). A NEW numerical candidate,
    /// off by default; returns how many layers were quantised.
    pub fn quantize_fp32_kinds_int8(&mut self, kinds: &[&str]) -> usize {
        let mut n = 0;
        for (kind, _, l) in self.linears_mut() {
            if kinds.contains(&kind) && l.q.is_none() && l.has_fp32() {
                l.q = Some(PackedI8::from_packed_f32(&l.f));
                n += 1;
            }
        }
        n
    }

    /// Read-only view of the quantisable layers with kind and block, in `linears_mut` order.
    pub fn linears_named(&self) -> Vec<(&'static str, Option<usize>, &Linear)> {
        let mut v = vec![
            ("stem_c2", None, &self.c2),
            ("stem_out", None, &self.stem_out),
        ];
        for (i, b) in self.blocks.iter().enumerate() {
            let s = Some(i);
            v.extend([
                ("ff1_w1", s, &b.ff1_w1),
                ("ff1_w2", s, &b.ff1_w2),
                ("qkv", s, &b.qkv),
                ("att_out", s, &b.att_out),
                ("pw1", s, &b.pw1),
                ("pw2", s, &b.pw2),
                ("ff2_w1", s, &b.ff2_w1),
                ("ff2_w2", s, &b.ff2_w2),
            ]);
        }
        if let Some(p) = &self.pair {
            v.push(("pair", None, p));
        }
        if let Some(r) = &self.restore {
            v.push(("restore", None, r));
        }
        v.push(("enc_proj", None, &self.enc_proj));
        v.push(("joiner_out", None, &self.joiner_out));
        v
    }

    fn linears_ref(&self) -> Vec<&Linear> {
        self.linears_named()
            .into_iter()
            .map(|(_, _, l)| l)
            .collect()
    }

    /// Stable id of a quantisable layer of this model (for cross-thread job descriptions).
    pub fn linear_id(&self, l: &Linear) -> Option<usize> {
        self.linears_ref().iter().position(|x| std::ptr::eq(*x, l))
    }

    /// Id of the layer whose int8 packing is `p` (pointer identity).
    pub fn packed_id(&self, p: &PackedI8) -> Option<usize> {
        self.linears_ref()
            .iter()
            .position(|x| x.q.as_ref().is_some_and(|q| std::ptr::eq(q, p)))
    }

    /// The layer with id `id` (see [`Model::linear_id`]).
    pub fn linear_at(&self, id: usize) -> Option<&Linear> {
        self.linears_ref().get(id).copied()
    }

    /// Every quantisable linear layer with its kind and block index.
    pub fn linears_mut(&mut self) -> Vec<(&'static str, Option<usize>, &mut Linear)> {
        let mut v: Vec<(&'static str, Option<usize>, &mut Linear)> = vec![
            ("stem_c2", None, &mut self.c2),
            ("stem_out", None, &mut self.stem_out),
        ];
        for (i, b) in self.blocks.iter_mut().enumerate() {
            let s = Some(i);
            v.extend([
                ("ff1_w1", s, &mut b.ff1_w1),
                ("ff1_w2", s, &mut b.ff1_w2),
                ("qkv", s, &mut b.qkv),
                ("att_out", s, &mut b.att_out),
                ("pw1", s, &mut b.pw1),
                ("pw2", s, &mut b.pw2),
                ("ff2_w1", s, &mut b.ff2_w1),
                ("ff2_w2", s, &mut b.ff2_w2),
            ]);
        }
        if let Some(p) = self.pair.as_mut() {
            v.push(("pair", None, p));
        }
        if let Some(r) = self.restore.as_mut() {
            v.push(("restore", None, r));
        }
        v.push(("enc_proj", None, &mut self.enc_proj));
        v.push(("joiner_out", None, &mut self.joiner_out));
        v
    }
}
