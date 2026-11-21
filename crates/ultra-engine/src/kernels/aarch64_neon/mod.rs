//! aarch64 NEON kernels (Apple M-series and other ARMv8 cores).
//!
//! # Safety model
//! NEON (ASIMD) is part of the aarch64 baseline, so these functions need no run-time check;
//! `unsafe` blocks only perform loads and stores inside slices whose lengths are asserted
//! before the first pointer is formed. The optional `sdot` path (int8 module) runs only after
//! `is_aarch64_feature_detected!("dotprod")`.
//!
//! # Arithmetic
//! Every kernel reproduces the x86 AVX2 kernel's exact sequence of roundings (eight-lane
//! accumulators are two four-lane halves; horizontal sums use the same tree), so fp32
//! results, int8 codes and transcripts are bit-identical across the two ISAs.

use std::arch::aarch64::*;

use super::scalar::{self, EXP_HI, EXP_LO, EXP_P, LN2_HI, LN2_LO, LOG2E};
use super::{PackedF32, NR};

mod attention;
mod int8;

pub(crate) use attention::{attention_scores, attention_values};
pub(crate) use int8::{gemm_i8_panels, has_i8mm, quantize_row};

/// Rows per fp32 GEMM register tile.
const MR: usize = 4;

/// Eight f32 lanes as two NEON registers (lanes 0-3, 4-7): the AVX2 `__m256` shape.
#[derive(Clone, Copy)]
pub(super) struct V8(pub(super) float32x4_t, pub(super) float32x4_t);

impl V8 {
    #[inline(always)]
    pub(super) fn zero() -> Self {
        // SAFETY: vdupq_n_f32 is a register-only NEON op (baseline on aarch64).
        unsafe { V8(vdupq_n_f32(0.0), vdupq_n_f32(0.0)) }
    }

    #[inline(always)]
    pub(super) fn splat(v: f32) -> Self {
        // SAFETY: register-only NEON op.
        unsafe { V8(vdupq_n_f32(v), vdupq_n_f32(v)) }
    }

    /// Load 8 floats from `s[i..i + 8]`.
    #[inline(always)]
    pub(super) fn load(s: &[f32], i: usize) -> Self {
        assert!(i + 8 <= s.len());
        // SAFETY: i + 8 <= s.len() (asserted).
        unsafe {
            V8(
                vld1q_f32(s.as_ptr().add(i)),
                vld1q_f32(s.as_ptr().add(i + 4)),
            )
        }
    }

    #[inline(always)]
    pub(super) fn store(self, s: &mut [f32], i: usize) {
        assert!(i + 8 <= s.len());
        // SAFETY: i + 8 <= s.len() (asserted).
        unsafe {
            vst1q_f32(s.as_mut_ptr().add(i), self.0);
            vst1q_f32(s.as_mut_ptr().add(i + 4), self.1);
        }
    }

    #[inline(always)]
    pub(super) fn add(self, o: Self) -> Self {
        // SAFETY: register-only NEON ops.
        unsafe { V8(vaddq_f32(self.0, o.0), vaddq_f32(self.1, o.1)) }
    }

    #[inline(always)]
    pub(super) fn sub(self, o: Self) -> Self {
        // SAFETY: register-only NEON ops.
        unsafe { V8(vsubq_f32(self.0, o.0), vsubq_f32(self.1, o.1)) }
    }

    #[inline(always)]
    pub(super) fn mul(self, o: Self) -> Self {
        // SAFETY: register-only NEON ops.
        unsafe { V8(vmulq_f32(self.0, o.0), vmulq_f32(self.1, o.1)) }
    }

    #[inline(always)]
    pub(super) fn div(self, o: Self) -> Self {
        // SAFETY: register-only NEON ops.
        unsafe { V8(vdivq_f32(self.0, o.0), vdivq_f32(self.1, o.1)) }
    }

    /// `self * b + c` with one rounding (`_mm256_fmadd_ps(self, b, c)`).
    #[inline(always)]
    pub(super) fn fma(self, b: Self, c: Self) -> Self {
        // SAFETY: register-only NEON ops (vfmaq_f32(c, a, b) = c + a * b, fused).
        unsafe { V8(vfmaq_f32(c.0, self.0, b.0), vfmaq_f32(c.1, self.1, b.1)) }
    }

    /// The AVX2 `hsum` tree: (lane j + lane j+4), then (s0 + s2, s1 + s3), then their sum.
    #[inline(always)]
    pub(super) fn hsum(self) -> f32 {
        // SAFETY: register-only NEON ops.
        unsafe {
            let s = vaddq_f32(self.0, self.1);
            let a = vgetq_lane_f32::<0>(s) + vgetq_lane_f32::<2>(s);
            let b = vgetq_lane_f32::<1>(s) + vgetq_lane_f32::<3>(s);
            a + b
        }
    }
}

/// Four-lane twin of `scalar::exp_approx` (same roundings as the AVX2 `exp8`).
#[inline(always)]
pub(super) fn exp4(x: float32x4_t) -> float32x4_t {
    // SAFETY: register-only NEON ops.
    unsafe {
        let x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(EXP_LO)), vdupq_n_f32(EXP_HI));
        let fx = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(LOG2E)));
        let r = vfmsq_f32(x, fx, vdupq_n_f32(LN2_HI));
        let r = vfmsq_f32(r, fx, vdupq_n_f32(LN2_LO));
        let z = vmulq_f32(r, r);
        let mut y = vdupq_n_f32(EXP_P[0]);
        for &c in &EXP_P[1..] {
            y = vfmaq_f32(vdupq_n_f32(c), y, r);
        }
        let y = vaddq_f32(vfmaq_f32(r, y, z), vdupq_n_f32(1.0));
        let e = vaddq_s32(vcvtq_s32_f32(fx), vdupq_n_s32(127));
        let e = vminq_s32(vmaxq_s32(e, vdupq_n_s32(0)), vdupq_n_s32(255));
        vmulq_f32(y, vreinterpretq_f32_s32(vshlq_n_s32::<23>(e)))
    }
}

#[inline(always)]
pub(super) fn exp8(x: V8) -> V8 {
    V8(exp4(x.0), exp4(x.1))
}

#[inline(always)]
fn neg8(x: V8) -> V8 {
    // SAFETY: register-only NEON ops.
    unsafe { V8(vnegq_f32(x.0), vnegq_f32(x.1)) }
}

pub(crate) fn gemm_f32(w: &PackedF32, x: &[f32], rows: usize, out: &mut [f32]) {
    let (n, k) = (w.n, w.k);
    assert!(x.len() >= rows * k && out.len() >= rows * n);
    assert!(w.data.len() == w.panels() * k * NR && w.bias.len() == w.panels() * NR);
    for p in 0..w.panels() {
        let panel = w.panel(p);
        let bias = &w.bias[p * NR..(p + 1) * NR];
        let ncols = NR.min(n - p * NR);
        let mut r = 0;
        while r < rows {
            let m = (rows - r).min(MR);
            let xs = &x[r * k..(r + m) * k];
            let os = &mut out[r * n + p * NR..];
            match m {
                4 => tile_f32::<4>(xs, k, panel, bias, os, n, ncols),
                3 => tile_f32::<3>(xs, k, panel, bias, os, n, ncols),
                2 => tile_f32::<2>(xs, k, panel, bias, os, n, ncols),
                _ => tile_f32::<1>(xs, k, panel, bias, os, n, ncols),
            }
            r += m;
        }
    }
}

/// `M` rows x 16 columns: per output one FMA chain over k ascending, starting at the bias.
fn tile_f32<const M: usize>(
    x: &[f32],
    k: usize,
    panel: &[f32],
    bias: &[f32],
    out: &mut [f32],
    ldo: usize,
    ncols: usize,
) {
    assert!(x.len() >= M * k && panel.len() >= k * NR && bias.len() >= NR);
    assert!(ncols <= NR && out.len() >= (M - 1) * ldo + ncols);
    // SAFETY: bias.len() >= 16 (asserted).
    let b = unsafe {
        [
            vld1q_f32(bias.as_ptr()),
            vld1q_f32(bias.as_ptr().add(4)),
            vld1q_f32(bias.as_ptr().add(8)),
            vld1q_f32(bias.as_ptr().add(12)),
        ]
    };
    let mut acc = [b; M];
    for kk in 0..k {
        // SAFETY: kk * 16 + 16 <= k * 16 <= panel.len() (asserted).
        let w = unsafe {
            let p = panel.as_ptr().add(kk * NR);
            [
                vld1q_f32(p),
                vld1q_f32(p.add(4)),
                vld1q_f32(p.add(8)),
                vld1q_f32(p.add(12)),
            ]
        };
        for (m, a) in acc.iter_mut().enumerate() {
            // SAFETY: register-only NEON ops; x index m * k + kk < M * k <= x.len().
            unsafe {
                let xv = vdupq_n_f32(x[m * k + kk]);
                for (av, wv) in a.iter_mut().zip(w) {
                    *av = vfmaq_f32(*av, xv, wv);
                }
            }
        }
    }
    for (m, a) in acc.iter().enumerate() {
        let mut tmp = [0.0f32; NR];
        for (c, av) in a.iter().enumerate() {
            // SAFETY: tmp holds 16 floats; c * 4 + 4 <= 16.
            unsafe { vst1q_f32(tmp.as_mut_ptr().add(4 * c), *av) };
        }
        out[m * ldo..m * ldo + ncols].copy_from_slice(&tmp[..ncols]);
    }
}

pub(crate) fn silu_inplace(x: &mut [f32]) {
    let one = V8::splat(1.0);
    let main = x.len() - x.len() % 8;
    let mut i = 0;
    while i < main {
        let v = V8::load(x, i);
        v.div(one.add(exp8(neg8(v)))).store(x, i);
        i += 8;
    }
    scalar::silu_inplace(&mut x[main..]);
}

pub(crate) fn glu(a: &[f32], d: usize, out: &mut [f32]) {
    let one = V8::splat(1.0);
    let main = d - d % 8;
    for (ar, or) in a.chunks_exact(2 * d).zip(out.chunks_exact_mut(d)) {
        let mut c = 0;
        while c < main {
            let v = V8::load(ar, c);
            let g = V8::load(ar, d + c);
            let s = one.div(one.add(exp8(neg8(g))));
            v.mul(s).store(or, c);
            c += 8;
        }
        for c in main..d {
            or[c] = ar[c] * scalar::sigmoid(ar[d + c]);
        }
    }
}

pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let main = n - n % 16;
    let (mut s0, mut s1) = (V8::zero(), V8::zero());
    let mut i = 0;
    while i < main {
        s0 = V8::load(a, i).fma(V8::load(b, i), s0);
        s1 = V8::load(a, i + 8).fma(V8::load(b, i + 8), s1);
        i += 16;
    }
    s0.add(s1).hsum() + scalar::dot(&a[main..], &b[main..])
}

pub(crate) fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    let main = n - n % 8;
    let av = V8::splat(alpha);
    let mut i = 0;
    while i < main {
        av.fma(V8::load(x, i), V8::load(y, i)).store(y, i);
        i += 8;
    }
    scalar::axpy(alpha, &x[main..], &mut y[main..]);
}

pub(crate) fn layer_norm(x: &[f32], d: usize, g: &[f32], b: &[f32], out: &mut [f32]) {
    if !d.is_multiple_of(8) {
        scalar::layer_norm(x, d, g, b, out);
        return;
    }
    assert!(g.len() == d && b.len() == d && x.len() == out.len());
    for (xr, or) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        let mut s = V8::zero();
        for i in (0..d).step_by(8) {
            s = s.add(V8::load(xr, i));
        }
        let mean = s.hsum() / d as f32;
        let mv = V8::splat(mean);
        let mut q = V8::zero();
        for i in (0..d).step_by(8) {
            let c = V8::load(xr, i).sub(mv);
            q = c.fma(c, q);
        }
        let inv = V8::splat(1.0 / (q.hsum() / d as f32 + 1e-5).sqrt());
        for i in (0..d).step_by(8) {
            let c = V8::load(xr, i).sub(mv);
            c.mul(inv).fma(V8::load(g, i), V8::load(b, i)).store(or, i);
        }
    }
}
