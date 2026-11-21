//! Portable scalar references. The AVX2 kernels follow these operation for operation:
//! the GEMM and the exponential use `mul_add` exactly where the SIMD code uses FMA, so those
//! pairs agree bit for bit; reductions (LayerNorm, dot) agree within a stated tolerance.

use super::{PackedF32, NR};

pub(crate) const EXP_HI: f32 = 88.376_26;
pub(crate) const EXP_LO: f32 = -88.376_26;
pub(crate) const LOG2E: f32 = std::f32::consts::LOG2_E;
pub(crate) const LN2_HI: f32 = 0.693_359_4;
pub(crate) const LN2_LO: f32 = -2.121_944_4e-4;
pub(crate) const EXP_P: [f32; 6] = [
    1.987_569_1e-4,
    1.398_199_9e-3,
    8.333_452e-3,
    4.166_579_6e-2,
    1.666_666_5e-1,
    0.5,
];

/// Cephes-style `exp` (relative error about 2e-7 over the clamped range); the AVX2 kernel
/// performs the identical sequence of roundings.
#[inline]
pub fn exp_approx(x: f32) -> f32 {
    let x = x.clamp(EXP_LO, EXP_HI);
    let fx = (x * LOG2E).round_ties_even();
    let r = (-fx).mul_add(LN2_HI, x);
    let r = (-fx).mul_add(LN2_LO, r);
    let z = r * r;
    let mut y = EXP_P[0];
    for &c in &EXP_P[1..] {
        y = y.mul_add(r, c);
    }
    let y = y.mul_add(z, r) + 1.0;
    // fx is in [-127, 128]; 2^fx built from the exponent field (0 for fx = -127).
    let e = ((fx as i32 + 127).clamp(0, 255) as u32) << 23;
    y * f32::from_bits(e)
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp_approx(-x))
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + exp_approx(-x))
}

pub fn gemm_f32(w: &PackedF32, x: &[f32], rows: usize, out: &mut [f32]) {
    let (n, k) = (w.n, w.k);
    for p in 0..w.panels() {
        let panel = w.panel(p);
        for j in 0..NR.min(n - p * NR) {
            let col = p * NR + j;
            for r in 0..rows {
                let xr = &x[r * k..(r + 1) * k];
                let mut acc = w.bias[col];
                for (kk, &xv) in xr.iter().enumerate() {
                    acc = xv.mul_add(panel[kk * NR + j], acc);
                }
                out[r * n + col] = acc;
            }
        }
    }
}

pub fn layer_norm(x: &[f32], d: usize, g: &[f32], b: &[f32], out: &mut [f32]) {
    for (xr, or) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        let mean = xr.iter().map(|&v| f64::from(v)).sum::<f64>() / d as f64;
        let var = xr
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / d as f64;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for i in 0..d {
            or[i] = ((f64::from(xr[i]) - mean) * inv) as f32 * g[i] + b[i];
        }
    }
}

pub fn silu_inplace(x: &mut [f32]) {
    for v in x {
        *v = silu(*v);
    }
}

pub fn glu(a: &[f32], d: usize, out: &mut [f32]) {
    for (ar, or) in a.chunks_exact(2 * d).zip(out.chunks_exact_mut(d)) {
        for c in 0..d {
            or[c] = ar[c] * sigmoid(ar[d + c]);
        }
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    for (yv, &xv) in y.iter_mut().zip(x) {
        *yv += alpha * xv;
    }
}
