//! x86-64 AVX2 + FMA kernels.
//!
//! # Safety model
//! Every `#[target_feature(enable = "avx2,fma")]` function here is reached only through the
//! `pub(crate)` wrappers at the top of each section, which are called only from the
//! dispatchers in `kernels/mod.rs` after `Isa::Avx2Fma.is_available()` returned true at run
//! time. All loads and stores are unaligned (`loadu`/`storeu`) and stay inside slices whose
//! lengths are asserted before the first pointer is formed.
//!
//! # Arithmetic
//! The GEMM and the exponential perform exactly the roundings of `kernels::scalar`
//! (`fmadd` where the scalar code uses `mul_add`), so they agree bit for bit.

use std::arch::x86_64::*;

use super::scalar::{self, EXP_HI, EXP_LO, EXP_P, LN2_HI, LN2_LO, LOG2E};
use super::{PackedF32, NR};

/// Rows per GEMM register tile (6 x 16 floats = 12 accumulators).
const MR: usize = 6;

pub(crate) fn gemm_f32(w: &PackedF32, x: &[f32], rows: usize, out: &mut [f32]) {
    // SAFETY: reached only after Isa::Avx2Fma.is_available() (module docs).
    unsafe { gemm_f32_avx2(w, x, rows, out) }
}

#[target_feature(enable = "avx2,fma")]
fn gemm_f32_avx2(w: &PackedF32, x: &[f32], rows: usize, out: &mut [f32]) {
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
                6 => tile_f32::<6>(xs, k, panel, bias, os, n, ncols),
                5 => tile_f32::<5>(xs, k, panel, bias, os, n, ncols),
                4 => tile_f32::<4>(xs, k, panel, bias, os, n, ncols),
                3 => tile_f32::<3>(xs, k, panel, bias, os, n, ncols),
                2 => tile_f32::<2>(xs, k, panel, bias, os, n, ncols),
                _ => tile_f32::<1>(xs, k, panel, bias, os, n, ncols),
            }
            r += m;
        }
    }
}

/// `M` rows x 16 columns; `out` starts at the tile's first element, row stride `ldo`.
#[target_feature(enable = "avx2,fma")]
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
    let (b0, b1) = unsafe {
        (
            _mm256_loadu_ps(bias.as_ptr()),
            _mm256_loadu_ps(bias.as_ptr().add(8)),
        )
    };
    let mut acc = [[b0, b1]; M];
    let xp = x.as_ptr();
    let wp = panel.as_ptr();
    for kk in 0..k {
        // SAFETY: kk * 16 + 16 <= k * 16 <= panel.len() (asserted).
        let (w0, w1) = unsafe {
            (
                _mm256_loadu_ps(wp.add(kk * NR)),
                _mm256_loadu_ps(wp.add(kk * NR + 8)),
            )
        };
        for (m, a) in acc.iter_mut().enumerate() {
            // SAFETY: m * k + kk < M * k <= x.len() (asserted).
            let xv = unsafe { _mm256_broadcast_ss(&*xp.add(m * k + kk)) };
            a[0] = _mm256_fmadd_ps(xv, w0, a[0]);
            a[1] = _mm256_fmadd_ps(xv, w1, a[1]);
        }
    }
    for (m, a) in acc.iter().enumerate() {
        let row = &mut out[m * ldo..];
        if ncols == NR {
            // SAFETY: row.len() >= 16 when ncols == 16 (asserted bound on out).
            unsafe {
                _mm256_storeu_ps(row.as_mut_ptr(), a[0]);
                _mm256_storeu_ps(row.as_mut_ptr().add(8), a[1]);
            }
        } else {
            let mut tmp = [0.0f32; NR];
            // SAFETY: tmp holds 16 floats.
            unsafe {
                _mm256_storeu_ps(tmp.as_mut_ptr(), a[0]);
                _mm256_storeu_ps(tmp.as_mut_ptr().add(8), a[1]);
            }
            row[..ncols].copy_from_slice(&tmp[..ncols]);
        }
    }
}

/// Eight-lane twin of `scalar::exp_approx` (same roundings).
#[target_feature(enable = "avx2,fma")]
pub(crate) fn exp8(x: __m256) -> __m256 {
    let x = _mm256_min_ps(
        _mm256_max_ps(x, _mm256_set1_ps(EXP_LO)),
        _mm256_set1_ps(EXP_HI),
    );
    let fx = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_ps(
        x,
        _mm256_set1_ps(LOG2E),
    ));
    let r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(LN2_HI), x);
    let r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(LN2_LO), r);
    let z = _mm256_mul_ps(r, r);
    let mut y = _mm256_set1_ps(EXP_P[0]);
    for &c in &EXP_P[1..] {
        y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(c));
    }
    let y = _mm256_add_ps(_mm256_fmadd_ps(y, z, r), _mm256_set1_ps(1.0));
    let e = _mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(127));
    let e = _mm256_min_epi32(
        _mm256_max_epi32(e, _mm256_setzero_si256()),
        _mm256_set1_epi32(255),
    );
    _mm256_mul_ps(y, _mm256_castsi256_ps(_mm256_slli_epi32::<23>(e)))
}

#[target_feature(enable = "avx2,fma")]
pub(super) fn neg8(x: __m256) -> __m256 {
    _mm256_xor_ps(x, _mm256_set1_ps(-0.0))
}

pub(crate) fn sum_exp_shifted(x: &[f32], shift: f32) -> f32 {
    // SAFETY: reached only after Isa::Avx2Fma.is_available() (module docs).
    unsafe { sum_exp_avx2(x, shift) }
}

/// Lane `l` accumulates elements `8 j + l` (as `scalar::sum_exp_shifted`); the tail goes to
/// its lanes in scalar code with the identical exp, then the shared reduction.
#[target_feature(enable = "avx2,fma")]
fn sum_exp_avx2(x: &[f32], shift: f32) -> f32 {
    let s = _mm256_set1_ps(shift);
    let mut acc = _mm256_setzero_ps();
    let mut chunks = x.chunks_exact(8);
    for c in &mut chunks {
        // SAFETY: c.len() == 8.
        let v = unsafe { _mm256_loadu_ps(c.as_ptr()) };
        acc = _mm256_add_ps(acc, exp8(_mm256_sub_ps(v, s)));
    }
    let mut lanes = [0.0f32; 8];
    // SAFETY: lanes holds 8 floats.
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
    for (i, &v) in chunks.remainder().iter().enumerate() {
        lanes[i] += scalar::exp_approx(v - shift);
    }
    scalar::reduce8(&lanes)
}

pub(crate) fn silu_inplace(x: &mut [f32]) {
    // SAFETY: reached only after Isa::Avx2Fma.is_available() (module docs).
    unsafe { silu_avx2(x) }
}

#[target_feature(enable = "avx2,fma")]
fn silu_avx2(x: &mut [f32]) {
    let one = _mm256_set1_ps(1.0);
    let mut chunks = x.chunks_exact_mut(8);
    for c in &mut chunks {
        // SAFETY: c.len() == 8.
        unsafe {
            let v = _mm256_loadu_ps(c.as_ptr());
            let y = _mm256_div_ps(v, _mm256_add_ps(one, exp8(neg8(v))));
            _mm256_storeu_ps(c.as_mut_ptr(), y);
        }
    }
    scalar::silu_inplace(chunks.into_remainder());
}

pub(crate) fn glu(a: &[f32], d: usize, out: &mut [f32]) {
    // SAFETY: reached only after Isa::Avx2Fma.is_available() (module docs).
    unsafe { glu_avx2(a, d, out) }
}

#[target_feature(enable = "avx2,fma")]
fn glu_avx2(a: &[f32], d: usize, out: &mut [f32]) {
    let one = _mm256_set1_ps(1.0);
    let main = d - d % 8;
    for (ar, or) in a.chunks_exact(2 * d).zip(out.chunks_exact_mut(d)) {
        let mut c = 0;
        while c < main {
            // SAFETY: c + 8 <= d, d + c + 8 <= 2 d = ar.len(), c + 8 <= or.len().
            unsafe {
                let v = _mm256_loadu_ps(ar.as_ptr().add(c));
                let g = _mm256_loadu_ps(ar.as_ptr().add(d + c));
                let s = _mm256_div_ps(one, _mm256_add_ps(one, exp8(neg8(g))));
                _mm256_storeu_ps(or.as_mut_ptr().add(c), _mm256_mul_ps(v, s));
            }
            c += 8;
        }
        for c in main..d {
            or[c] = ar[c] * scalar::sigmoid(ar[d + c]);
        }
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) fn hsum(v: __m256) -> f32 {
    let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps::<0b01>(s, s));
    _mm_cvtss_f32(s)
}

pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: reached only after Isa::Avx2Fma.is_available() (module docs).
    unsafe { dot_avx2(a, b) }
}

#[target_feature(enable = "avx2,fma")]
fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let main = n - n % 16;
    let (mut s0, mut s1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let mut i = 0;
    while i < main {
        // SAFETY: i + 16 <= n for both slices.
        unsafe {
            s0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i)),
                _mm256_loadu_ps(b.as_ptr().add(i)),
                s0,
            );
            s1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i + 8)),
                _mm256_loadu_ps(b.as_ptr().add(i + 8)),
                s1,
            );
        }
        i += 16;
    }
    hsum(_mm256_add_ps(s0, s1)) + scalar::dot(&a[main..], &b[main..])
}

pub(crate) fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    // SAFETY: reached only after Isa::Avx2Fma.is_available() (module docs).
    unsafe { axpy_avx2(alpha, x, y) }
}

#[target_feature(enable = "avx2,fma")]
fn axpy_avx2(alpha: f32, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    let main = n - n % 8;
    let av = _mm256_set1_ps(alpha);
    let mut i = 0;
    while i < main {
        // SAFETY: i + 8 <= n for both slices.
        unsafe {
            let r = _mm256_fmadd_ps(
                av,
                _mm256_loadu_ps(x.as_ptr().add(i)),
                _mm256_loadu_ps(y.as_ptr().add(i)),
            );
            _mm256_storeu_ps(y.as_mut_ptr().add(i), r);
        }
        i += 8;
    }
    scalar::axpy(alpha, &x[main..], &mut y[main..]);
}

pub(crate) fn layer_norm(x: &[f32], d: usize, g: &[f32], b: &[f32], out: &mut [f32]) {
    // SAFETY: reached only after Isa::Avx2Fma.is_available() (module docs).
    unsafe { layer_norm_avx2(x, d, g, b, out) }
}

#[target_feature(enable = "avx2,fma")]
fn layer_norm_avx2(x: &[f32], d: usize, g: &[f32], b: &[f32], out: &mut [f32]) {
    if !d.is_multiple_of(8) {
        scalar::layer_norm(x, d, g, b, out);
        return;
    }
    assert!(g.len() == d && b.len() == d && x.len() == out.len());
    for (xr, or) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        let mut s = _mm256_setzero_ps();
        for i in (0..d).step_by(8) {
            // SAFETY: i + 8 <= d = xr.len().
            s = _mm256_add_ps(s, unsafe { _mm256_loadu_ps(xr.as_ptr().add(i)) });
        }
        let mean = hsum(s) / d as f32;
        let mv = _mm256_set1_ps(mean);
        let mut q = _mm256_setzero_ps();
        for i in (0..d).step_by(8) {
            // SAFETY: as above.
            let c = _mm256_sub_ps(unsafe { _mm256_loadu_ps(xr.as_ptr().add(i)) }, mv);
            q = _mm256_fmadd_ps(c, c, q);
        }
        let inv = _mm256_set1_ps(1.0 / (hsum(q) / d as f32 + 1e-5).sqrt());
        for i in (0..d).step_by(8) {
            // SAFETY: i + 8 <= d for xr, or, g and b.
            unsafe {
                let c = _mm256_sub_ps(_mm256_loadu_ps(xr.as_ptr().add(i)), mv);
                let y = _mm256_fmadd_ps(
                    _mm256_mul_ps(c, inv),
                    _mm256_loadu_ps(g.as_ptr().add(i)),
                    _mm256_loadu_ps(b.as_ptr().add(i)),
                );
                _mm256_storeu_ps(or.as_mut_ptr().add(i), y);
            }
        }
    }
}

mod attention;
mod int8;
mod int8_u7;

pub(crate) use attention::{attention_scores, attention_values};
pub(crate) use int8::{gemm_i8, gemm_i8_panels, quantize_row};
pub(crate) use int8_u7::gemm_i8_u7;
