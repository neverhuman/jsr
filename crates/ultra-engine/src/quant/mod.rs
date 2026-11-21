//! Int8 linear layers: per-output-channel symmetric weights and per-row (per-frame) dynamic
//! symmetric activations, with an exact int32 accumulation.
//!
//! - Weights: `w_q[n][k] = round(w[n][k] / s_w[n])`, `s_w[n] = max_k |w[n][k]| / 127`, so
//!   `|w_q| <= 127`.
//! - Activations: each row `r` of every GEMM input is quantised on its own, centred on its
//!   range: `c[r] = (max + min) / 2`, `s_a[r] = (max - min) / 254`,
//!   `x_q = round_half_even((x - c) / s_a)`, `|x_q| <= 127`. Centring makes one-sided rows
//!   (post-SiLU FFN inputs) use all 255 levels. A row's numbers never depend on which other
//!   rows (streams) share the batch.
//! - Product: `y[r][n] = f32(sum_k x_q w_q) * (s_a[r] * s_w[n]) + (c[r] * wsum[n] + b[n])`,
//!   `wsum[n] = s_w[n] * sum_k w_q[n][k]` (two FMAs).
//!
//! The AVX2 kernel uses `vpmaddubsw` on `(|x_q|, sign(w_q, x_q))`: with both magnitudes at most
//! 127 a pair sums to at most 2 * 127 * 127 = 32 258 < 32 767, so the saturating i16 step never
//! saturates and the int32 result equals the scalar reference exactly.

use crate::kernels::{Isa, PackedF32, NR};

mod ternary;
pub use ternary::{linear_t, PackedT};

/// Largest K whose worst-case int32 sum cannot overflow, even for public callers passing
/// activation code -128 to `gemm_i8` (|a| <= 128, |w| <= 127).
pub const MAX_K: usize = (i32::MAX as usize) / (128 * 127) / KG * KG;

/// K is padded to a multiple of this (the `vpmaddubsw` + `vpmaddwd` group).
pub const KG: usize = 4;

/// How a weight row becomes codes. Both use `s_w = max |w| / 127`; they differ only in the
/// float step, which can move a value sitting next to a rounding tie by one code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightRule {
    /// `round_half_even(w * (127 / max))`: the engine's post-training rule (default path).
    Engine,
    /// `round_half_even(w / max(max / 127, 1e-8))`: `train.qat.int8_weight_ste`, for layers
    /// trained as int8 GEMMs, so the engine runs exactly the codes the trainer used.
    Trainer,
}

/// Int8 weights packed as `data[p][kg][j][4]` = `w_q[p*16 + j][4 kg .. 4 kg + 4]`.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedI8 {
    pub(crate) n: usize,
    pub(crate) k: usize,
    /// `k` rounded up to a multiple of [`KG`].
    pub(crate) kp: usize,
    pub(crate) data: Vec<i8>,
    /// Per-output-channel scale, padded to whole panels (zero in padding).
    pub(crate) scale: Vec<f32>,
    /// Per-output-channel sum of dequantised weights (`scale * sum_k w_q`), padded.
    pub(crate) wsum: Vec<f32>,
    pub(crate) bias: Vec<f32>,
    /// Optional second packing of the same codes for the aarch64 i8mm `smmla` kernel
    /// (`[panel][k/8][col pair][2 cols][8 k]`); empty where unused.
    pub(crate) mm: Vec<i8>,
}

impl PackedI8 {
    pub fn n(&self) -> usize {
        self.n
    }

    pub fn k(&self) -> usize {
        self.k
    }

    /// Padded row stride of quantised activations.
    pub fn kp(&self) -> usize {
        self.kp
    }

    pub fn panels(&self) -> usize {
        self.n.div_ceil(NR)
    }

    pub fn panel(&self, p: usize) -> &[i8] {
        &self.data[p * self.kp * NR..(p + 1) * self.kp * NR]
    }

    /// Quantise row-major `w` (`n x k`) with an optional bias (engine rule).
    pub fn quantize(w: &[f32], n: usize, k: usize, bias: Option<&[f32]>) -> Self {
        Self::quantize_with(w, n, k, bias, WeightRule::Engine)
    }

    /// Quantise row-major `w` (`n x k`) with an optional bias under `rule`.
    pub fn quantize_with(
        w: &[f32],
        n: usize,
        k: usize,
        bias: Option<&[f32]>,
        rule: WeightRule,
    ) -> Self {
        assert!(n > 0 && k > 0, "PackedI8::quantize empty matrix");
        assert_eq!(
            Some(w.len()),
            n.checked_mul(k),
            "PackedI8::quantize weight length"
        );
        // |sum_k a w| <= kp * 127 * 127 must fit the int32 accumulator.
        assert!(
            k <= MAX_K,
            "PackedI8::quantize k = {k} exceeds the int32 accumulator bound {MAX_K}"
        );
        let kp = k.div_ceil(KG) * KG;
        let panels = n.div_ceil(NR);
        let len = panels
            .checked_mul(kp)
            .and_then(|v| v.checked_mul(NR))
            .expect("PackedI8::quantize size overflow");
        let mut data = vec![0i8; len];
        let mut scale = vec![0.0f32; panels * NR];
        let mut wsum = vec![0.0f32; panels * NR];
        for row in 0..n {
            let wr = &w[row * k..(row + 1) * k];
            let mx = wr.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let s = mx / 127.0;
            scale[row] = s;
            let inv = if mx > 0.0 { 127.0 / mx } else { 0.0 };
            let step = s.max(1e-8);
            let (p, j) = (row / NR, row % NR);
            let mut qsum = 0i64;
            for (kk, &v) in wr.iter().enumerate() {
                let x = match rule {
                    WeightRule::Engine => v * inv,
                    WeightRule::Trainer => v / step,
                };
                let q = x.round_ties_even().clamp(-127.0, 127.0) as i8;
                data[((p * (kp / KG) + kk / KG) * NR + j) * KG + kk % KG] = q;
                qsum += i64::from(q);
            }
            wsum[row] = qsum as f32 * s;
        }
        let mut b = vec![0.0f32; panels * NR];
        if let Some(bias) = bias {
            assert_eq!(bias.len(), n, "PackedI8::quantize bias length");
            b[..n].copy_from_slice(bias);
        }
        Self {
            n,
            k,
            kp,
            data,
            scale,
            wsum,
            bias: b,
            mm: Vec::new(),
        }
        .with_mm()
    }

    /// Rebuild from stored packed parts (a prepared int8 artifact), validating every length,
    /// the code range (no -128) and the accumulator bound. `bias` has length `n`.
    pub fn from_parts(
        n: usize,
        k: usize,
        data: Vec<i8>,
        scale: &[f32],
        wsum: &[f32],
        bias: &[f32],
    ) -> Result<Self, crate::EngineError> {
        let bad = |m: &str| {
            Err(crate::EngineError::Shape(format!(
                "prepared int8 {n}x{k}: {m}"
            )))
        };
        if n == 0 || k == 0 || k > MAX_K || bias.len() != n {
            return bad("dimensions");
        }
        let kp = k.div_ceil(KG) * KG;
        let panels = n.div_ceil(NR);
        if Some(data.len()) != panels.checked_mul(kp).and_then(|v| v.checked_mul(NR))
            || scale.len() != panels * NR
            || wsum.len() != panels * NR
        {
            return bad("packed lengths");
        }
        if data.contains(&i8::MIN) || scale.iter().chain(wsum).any(|v| !v.is_finite()) {
            return bad("code range or non-finite scale");
        }
        let mut b = vec![0.0f32; panels * NR];
        b[..n].copy_from_slice(bias);
        Ok(Self {
            n,
            k,
            kp,
            data,
            scale: scale.to_vec(),
            wsum: wsum.to_vec(),
            bias: b,
            mm: Vec::new(),
        }
        .with_mm())
    }

    /// Add the i8mm packing when this CPU runs the `smmla` kernel (aarch64 with i8mm and
    /// K a multiple of 8). Same codes, different order: results are identical.
    fn with_mm(mut self) -> Self {
        if crate::kernels::wants_mm_packing() && self.kp.is_multiple_of(8) {
            let (kp, panels) = (self.kp, self.panels());
            let mut mm = vec![0i8; self.data.len()];
            for p in 0..panels {
                for kk in 0..kp {
                    for j in 0..NR {
                        let src = ((p * (kp / KG) + kk / KG) * NR + j) * KG + kk % KG;
                        let dst = ((p * (kp / 8) + kk / 8) * (NR / 2) + j / 2) * 16
                            + (j % 2) * 8
                            + kk % 8;
                        mm[dst] = self.data[src];
                    }
                }
            }
            self.mm = mm;
        }
        self
    }

    /// Rebuild the i8mm packing after `data` was edited in place (tests).
    #[cfg(test)]
    pub(crate) fn rebuild_mm(&mut self) {
        let me = std::mem::replace(
            self,
            Self {
                mm: Vec::new(),
                ..self.clone()
            },
        );
        *self = Self {
            mm: Vec::new(),
            ..me
        }
        .with_mm();
    }

    /// The i8mm packing of panel `p` (empty if not built).
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    pub(crate) fn mm_panel(&self, p: usize) -> &[i8] {
        if self.mm.is_empty() {
            &[]
        } else {
            &self.mm[p * self.kp * NR..(p + 1) * self.kp * NR]
        }
    }

    /// Packed codes, per-channel scales and weight sums (for writing a prepared artifact).
    pub fn parts(&self) -> (&[i8], &[f32], &[f32]) {
        (&self.data, &self.scale, &self.wsum)
    }

    /// Quantise the weights held by an fp32 packing (unpacks, then quantises).
    pub fn from_packed_f32(f: &PackedF32) -> Self {
        Self::from_packed_f32_with(f, WeightRule::Engine)
    }

    /// [`Self::from_packed_f32`] under `rule`.
    pub fn from_packed_f32_with(f: &PackedF32, rule: WeightRule) -> Self {
        assert!(
            f.has_weights(),
            "PackedI8::from_packed_f32 needs fp32 weights"
        );
        let mut w = vec![0.0f32; f.n * f.k];
        for row in 0..f.n {
            let (p, j) = (row / NR, row % NR);
            for kk in 0..f.k {
                w[row * f.k + kk] = f.data[(p * f.k + kk) * NR + j];
            }
        }
        Self::quantize_with(&w, f.n, f.k, Some(&f.bias[..f.n]), rule)
    }

    /// The dequantised weight `s_w[n] * w_q[n][k]` (tests and accounting).
    pub fn dequant(&self, row: usize, kk: usize) -> f32 {
        let (p, j) = (row / NR, row % NR);
        let q = self.data[((p * (self.kp / KG) + kk / KG) * NR + j) * KG + kk % KG];
        f32::from(q) * self.scale[row]
    }
}

/// Largest row count for which the pool split is used (it targets single-stream latency).
const SPLIT_MAX_ROWS: usize = 16;

/// Reusable quantisation buffers (one per worker).
#[derive(Debug, Default)]
pub struct QuantScratch {
    pub a: Vec<i8>,
    pub scale: Vec<f32>,
    pub center: Vec<f32>,
    /// 7-bit activation codes (|q| <= 63) and the paired-accumulation kernel.
    pub act7: bool,
    /// T-LUT2 int32 outputs (weight-row major).
    pub acc: Vec<i32>,
    /// Optional panel split of int8 GEMMs across a pool (single-stream latency mode).
    pub split: Option<(
        std::sync::Arc<crate::pool::GemmPool>,
        std::sync::Arc<crate::model::Model>,
    )>,
    /// Phase accounting of the worker that owns this scratch.
    pub prof: crate::prof::Prof,
}

/// Per-row centred quantisation of `rows` rows of width `k` into `a` (row stride `kp`,
/// zero padded), `scale` and `center`.
fn quantize_rows(isa: Isa, x: &[f32], rows: usize, k: usize, kp: usize, q: &mut QuantScratch) {
    if q.a.len() < rows * kp {
        q.a.resize(rows * kp, 0);
    }
    if q.scale.len() < rows {
        q.scale.resize(rows, 0.0);
        q.center.resize(rows, 0.0);
    }
    for r in 0..rows {
        let xr = &x[r * k..(r + 1) * k];
        let ar = &mut q.a[r * kp..(r + 1) * kp];
        let qmax = if q.act7 { 63 } else { 127 };
        (q.scale[r], q.center[r]) = crate::kernels::quantize_row_q(isa, xr, &mut ar[..k], qmax);
        ar[k..].fill(0);
    }
}

/// `out[r][n] = (sum_k x_q[r][k] w_q[n][k]) * s_a[r] * s_w[n] + c[r] * wsum[n] + b[n]`.
pub fn linear_i8(
    isa: Isa,
    p: &PackedI8,
    x: &[f32],
    rows: usize,
    out: &mut [f32],
    q: &mut QuantScratch,
) {
    assert!(
        x.len() >= rows * p.k && out.len() >= rows * p.n,
        "linear_i8 lengths"
    );
    use crate::prof::Phase;
    let t = q.prof.start();
    quantize_rows(isa, x, rows, p.k, p.kp, q);
    q.prof.stop(Phase::Quant, t);
    let t = q.prof.start();
    let split_id = q.split.as_ref().and_then(|(pool, m)| {
        (rows <= SPLIT_MAX_ROWS && p.panels() >= pool.threads()).then(|| m.packed_id(p))?
    });
    match (split_id, q.split.as_ref()) {
        (Some(id), Some((pool, m))) => pool.gemm_i8(
            isa,
            m,
            id,
            &q.a[..rows * p.kp],
            &q.scale[..rows],
            &q.center[..rows],
            rows,
            q.act7,
            out,
        ),
        _ => {
            let gemm = if q.act7 {
                crate::kernels::gemm_i8_a7
            } else {
                crate::kernels::gemm_i8
            };
            gemm(
                isa,
                p,
                &q.a[..rows * p.kp],
                &q.scale[..rows],
                &q.center[..rows],
                rows,
                out,
            );
        }
    }
    q.prof.stop(Phase::GemmI8, t);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::tests::Rng;

    #[test]
    fn quantised_weights_round_trip_within_half_step() {
        let mut rng = Rng::new(9);
        let (n, k) = (37, 30);
        let w = rng.vec(n * k, 0.3);
        let b = rng.vec(n, 1.0);
        let p = PackedI8::quantize(&w, n, k, Some(&b));
        assert_eq!(p.kp, 32);
        for row in 0..n {
            for kk in 0..k {
                let err = (p.dequant(row, kk) - w[row * k + kk]).abs();
                assert!(err <= 0.5 * p.scale[row] + 1e-7);
            }
        }
        let f = PackedF32::pack(&w, n, k, Some(&b));
        assert_eq!(PackedI8::from_packed_f32(&f), p);
    }

    #[test]
    fn trainer_rule_matches_int8_weight_ste_and_differs_only_near_ties() {
        let mut rng = Rng::new(11);
        let (n, k) = (64, 512);
        let w = rng.vec(n * k, 0.4);
        let e = PackedI8::quantize_with(&w, n, k, None, WeightRule::Engine);
        let t = PackedI8::quantize_with(&w, n, k, None, WeightRule::Trainer);
        assert_eq!(e.scale, t.scale);
        let mut differ = 0;
        for row in 0..n {
            let wr = &w[row * k..(row + 1) * k];
            let s = (wr.iter().fold(0.0f32, |a, &v| a.max(v.abs())) / 127.0).max(1e-8);
            for (kk, &wv) in wr.iter().enumerate() {
                // train.qat.int8_weight_ste in float32: clamp(round(w / s), -127, 127).
                let want = (wv / s).round_ties_even().clamp(-127.0, 127.0);
                assert_eq!(t.dequant(row, kk), want * t.scale[row]);
                if e.dequant(row, kk) != t.dequant(row, kk) {
                    differ += 1;
                    assert!(((wv / s).fract().abs() - 0.5).abs() < 1e-3);
                }
            }
        }
        assert!(differ < n * k / 1000, "{differ}");
    }

    #[test]
    fn linear_i8_tracks_fp32_and_zero_rows_give_bias() {
        let mut rng = Rng::new(10);
        let (n, k, rows) = (40, 96, 5);
        let w = rng.vec(n * k, 0.2);
        let b = rng.vec(n, 1.0);
        let mut x = rng.vec(rows * k, 3.0);
        x[2 * k..3 * k].fill(0.0);
        // A one-sided (post-SiLU-like) row and a constant row.
        for v in &mut x[3 * k..4 * k] {
            *v = v.abs() + 0.5;
        }
        x[4 * k..5 * k].fill(1.25);
        let p = PackedI8::quantize(&w, n, k, Some(&b));
        let mut out = vec![0.0; rows * n];
        let mut q = QuantScratch::default();
        linear_i8(Isa::Scalar, &p, &x, rows, &mut out, &mut q);
        // A non-finite input row yields an all-NaN output row (refused downstream), others
        // are unaffected.
        let mut xb = x.clone();
        xb[k + 3] = f32::INFINITY;
        let mut ob = vec![0.0; rows * n];
        linear_i8(Isa::detect(), &p, &xb, rows, &mut ob, &mut q);
        assert!(ob[n..2 * n].iter().all(|v| v.is_nan()));
        assert_eq!(&ob[..n], &out[..n]);
        for r in 0..rows {
            for j in 0..n {
                let exact: f32 = (0..k).map(|kk| x[r * k + kk] * w[j * k + kk]).sum::<f32>() + b[j];
                let deq: f32 = (0..k)
                    .map(|kk| x[r * k + kk] * p.dequant(j, kk))
                    .sum::<f32>()
                    + b[j];
                assert!((out[r * n + j] - exact).abs() < 0.05, "{r} {j}");
                if r == 4 {
                    // Constant rows are exact up to rounding (centre carries the value).
                    assert!((out[r * n + j] - deq).abs() < 1e-4, "{r} {j}");
                }
                if r == 2 {
                    assert_eq!(out[r * n + j], b[j]);
                }
            }
        }
    }
}
