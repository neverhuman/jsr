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

/// K is padded to a multiple of this (the `vpmaddubsw` + `vpmaddwd` group).
pub const KG: usize = 4;

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

    /// Quantise row-major `w` (`n x k`) with an optional bias.
    pub fn quantize(w: &[f32], n: usize, k: usize, bias: Option<&[f32]>) -> Self {
        assert!(n > 0 && k > 0, "PackedI8::quantize empty matrix");
        assert_eq!(
            Some(w.len()),
            n.checked_mul(k),
            "PackedI8::quantize weight length"
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
            let (p, j) = (row / NR, row % NR);
            let mut qsum = 0i64;
            for (kk, &v) in wr.iter().enumerate() {
                let q = (v * inv).round_ties_even().clamp(-127.0, 127.0) as i8;
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
        }
    }

    /// Quantise the weights held by an fp32 packing (unpacks, then quantises).
    pub fn from_packed_f32(f: &PackedF32) -> Self {
        let mut w = vec![0.0f32; f.n * f.k];
        for row in 0..f.n {
            let (p, j) = (row / NR, row % NR);
            for kk in 0..f.k {
                w[row * f.k + kk] = f.data[(p * f.k + kk) * NR + j];
            }
        }
        Self::quantize(&w, f.n, f.k, Some(&f.bias[..f.n]))
    }

    /// The dequantised weight `s_w[n] * w_q[n][k]` (tests and accounting).
    pub fn dequant(&self, row: usize, kk: usize) -> f32 {
        let (p, j) = (row / NR, row % NR);
        let q = self.data[((p * (self.kp / KG) + kk / KG) * NR + j) * KG + kk % KG];
        f32::from(q) * self.scale[row]
    }
}

/// Reusable quantisation buffers (one per worker).
#[derive(Debug, Default)]
pub struct QuantScratch {
    pub a: Vec<i8>,
    pub scale: Vec<f32>,
    pub center: Vec<f32>,
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
        (q.scale[r], q.center[r]) = crate::kernels::quantize_row(isa, xr, &mut ar[..k]);
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
    crate::kernels::gemm_i8(
        isa,
        p,
        &q.a[..rows * p.kp],
        &q.scale[..rows],
        &q.center[..rows],
        rows,
        out,
    );
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
