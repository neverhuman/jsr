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

/// Range `(min, max)` of a row; `(NaN, NaN)` if any element is NaN or infinite.
pub fn row_range(x: &[f32]) -> (f32, f32) {
    if x.iter().any(|v| !v.is_finite()) {
        return (f32::NAN, f32::NAN);
    }
    x.iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        })
}

/// Centre, half range and inverse step of a row range (shared by scalar and SIMD).
#[inline]
pub fn row_params(lo: f32, hi: f32, qmax: f32) -> Option<(f32, f32, f32)> {
    let half = 0.5 * hi - 0.5 * lo;
    let c = 0.5 * hi + 0.5 * lo;
    let inv = qmax / half;
    // A tiny (subnormal) half range would overflow the inverse step: treat it as constant.
    (half > 0.0 && half.is_finite() && c.is_finite() && inv.is_finite())
        .then(|| (c, half / qmax, inv))
}

/// Per-row centred symmetric quantisation to `[-qmax, qmax]` (see `kernels::quantize_row`).
pub fn quantize_row(x: &[f32], out: &mut [i8], qmax: f32) -> (f32, f32) {
    let (lo, hi) = row_range(x);
    finish_row(x, out, lo, hi, qmax)
}

/// Quantise with a known range; returns `(scale, centre)`.
pub fn finish_row(x: &[f32], out: &mut [i8], lo: f32, hi: f32, qmax: f32) -> (f32, f32) {
    match row_params(lo, hi, qmax) {
        Some((c, scale, inv)) => {
            for (o, &v) in out.iter_mut().zip(x) {
                *o = ((v - c) * inv).round_ties_even().clamp(-qmax, qmax) as i8;
            }
            (scale, c)
        }
        None if lo.is_nan() || hi.is_nan() => {
            // Non-finite input: zero codes and NaN scale/centre, so every output of this row is
            // NaN and the encoder's finite check refuses the chunk (EngineError::Output).
            out.fill(0);
            (f32::NAN, f32::NAN)
        }
        None => {
            // Constant, empty or subnormal-range row: all codes zero and the centre carries
            // it (error <= half the range, below 1e-37 for the subnormal case).
            out.fill(0);
            let c = if x.is_empty() {
                0.0
            } else {
                0.5 * hi + 0.5 * lo
            };
            (0.0, c)
        }
    }
}

pub fn gemm_i8(
    w: &crate::quant::PackedI8,
    a: &[i8],
    sa: &[f32],
    ca: &[f32],
    rows: usize,
    out: &mut [f32],
) {
    gemm_i8_panels(w, a, sa, ca, rows, out, 0, w.panels(), w.n);
}

/// Panels `p0..p1` only, into `out` with row stride `ldo` (column `p0 * 16` at offset 0).
#[allow(clippy::too_many_arguments)]
pub fn gemm_i8_panels(
    w: &crate::quant::PackedI8,
    a: &[i8],
    sa: &[f32],
    ca: &[f32],
    rows: usize,
    out: &mut [f32],
    p0: usize,
    p1: usize,
    ldo: usize,
) {
    use crate::quant::KG;
    let (n, kp) = (w.n, w.kp);
    let col0 = p0 * NR;
    for p in p0..p1 {
        let panel = w.panel(p);
        for j in 0..NR.min(n - p * NR) {
            let col = p * NR + j;
            for r in 0..rows {
                let ar = &a[r * kp..(r + 1) * kp];
                let mut acc = 0i32;
                for (kk, &av) in ar.iter().enumerate() {
                    let wv = panel[((kk / KG) * NR + j) * KG + kk % KG];
                    acc += i32::from(av) * i32::from(wv);
                }
                let base = ca[r].mul_add(w.wsum[col], w.bias[col]);
                out[r * ldo + col - col0] = (acc as f32).mul_add(sa[r] * w.scale[col], base);
            }
        }
    }
}

/// One attention score in the engine's fixed summation order (the order of the original
/// row-major AVX2 kernel, kept so kernel layout changes never move a bit):
/// - 16 partial FMA chains, `a[l] = sum_c q[16c + l] * k[16c + l]` over c ascending
///   (`l < 16`); when `dh % 16 == 8` the last 8 elements extend chains `l < 8`;
/// - `v[l] = a[l] + a[l + 8]` (l < 8), then the tree `(v0+v4, v1+v5, v2+v6, v3+v7)`,
///   `(u0+u2, u1+u3)`, `w0 + w1`; finally `* scale`.
#[inline]
pub fn score_fixed_order(q: &[f32], k: impl Fn(usize) -> f32, dh: usize, scale: f32) -> f32 {
    let mut a = [0.0f32; 16];
    let full = dh / 16;
    for c in 0..full {
        for (l, al) in a.iter_mut().enumerate() {
            *al = q[16 * c + l].mul_add(k(16 * c + l), *al);
        }
    }
    if dh % 16 == 8 {
        for (l, al) in a[..8].iter_mut().enumerate() {
            *al = q[dh - 8 + l].mul_add(k(dh - 8 + l), *al);
        }
    }
    let v: [f32; 8] = std::array::from_fn(|l| a[l] + a[l + 8]);
    let u: [f32; 4] = std::array::from_fn(|l| v[l] + v[l + 4]);
    let w = [u[0] + u[2], u[1] + u[3]];
    (w[0] + w[1]) * scale
}

#[allow(clippy::too_many_arguments)]
/// `scores[i * nk + j]` = [`score_fixed_order`] of query `i` and window slot `j` (tiled `kt`).
pub fn attention_scores(
    q: &[f32],
    q_stride: usize,
    n: usize,
    dh: usize,
    kt: &[f32],
    ring: usize,
    segs: [(usize, usize); 2],
    scale: f32,
    scores: &mut [f32],
) {
    debug_assert!(kt.len() >= ring * dh);
    let nk = segs[0].1 + segs[1].1;
    for i in 0..n {
        let qi = &q[i * q_stride..i * q_stride + dh];
        let mut j = 0;
        for (start, len) in segs {
            for c in start..start + len {
                scores[i * nk + j] =
                    score_fixed_order(qi, |e| kt[super::kt_index(c, e, dh)], dh, scale);
                j += 1;
            }
        }
    }
}

/// Softmax over each query's `nk` scores, then `out[i] = sum_j p_ij v_j`.
#[allow(clippy::too_many_arguments)]
pub fn attention_values(
    n: usize,
    dh: usize,
    vals: [&[f32]; 2],
    scores: &mut [f32],
    out: &mut [f32],
    o_stride: usize,
) {
    let nk = (vals[0].len() + vals[1].len()) / dh;
    for i in 0..n {
        let sc = &mut scores[i * nk..(i + 1) * nk];
        let mx = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in sc.iter_mut() {
            *s = exp_approx(*s - mx);
            sum += *s;
        }
        let inv = 1.0 / sum;
        let o = &mut out[i * o_stride..i * o_stride + dh];
        o.fill(0.0);
        let rows = vals[0].chunks_exact(dh).chain(vals[1].chunks_exact(dh));
        for (&p, vr) in sc.iter().zip(rows) {
            axpy(p * inv, vr, o);
        }
    }
}
