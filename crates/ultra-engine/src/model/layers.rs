//! The model's quantisable linear layers: enumeration, ids and int8 quantisation.

use super::{Linear, Model};
use crate::quant::PackedI8;

impl Model {
    /// Pack int8 copies (per-output-channel scales) of every encoder and joiner linear layer.
    pub fn quantize_int8(&mut self) {
        self.quantize_int8_where(|_, _| true);
    }

    /// Pack int8 copies of the layers for which `keep(kind, block)` is true (ablations);
    /// the others stay fp32 in the int8 engine.
    pub fn quantize_int8_where(&mut self, keep: impl Fn(&str, Option<usize>) -> bool) {
        for (kind, blk, l) in self.linears_mut() {
            if !l.has_fp32() {
                continue; // prepared int8 layer: its packing comes from the artifact
            }
            l.q = keep(kind, blk).then(|| PackedI8::from_packed_f32(&l.f));
        }
    }

    /// Read-only view of the quantisable layers, in `linears_mut` order.
    fn linears_ref(&self) -> Vec<&Linear> {
        let mut v = vec![&self.c2, &self.stem_out];
        for b in &self.blocks {
            v.extend([
                &b.ff1_w1, &b.ff1_w2, &b.qkv, &b.att_out, &b.pw1, &b.pw2, &b.ff2_w1, &b.ff2_w2,
            ]);
        }
        v.extend(self.pair.iter());
        v.extend(self.restore.iter());
        v.extend([&self.enc_proj, &self.joiner_out]);
        v
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
