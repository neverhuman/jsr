//! Fast AVX2 per-row int8 activation quantisation producing both `a: [i8]` and `a_abs: [u8]`
//! in a single pass, eliminating `vpabsb` from the inner INT8 GEMM loop.
//!
//! Arithmetic is 100% bit-exact with `scalar::quantize_row_q` and `int8::quantize_row`.

use std::arch::x86_64::*;

use super::super::scalar;

pub(crate) fn quantize_row_with_abs(
    x: &[f32],
    out: &mut [i8],
    out_abs: &mut [u8],
    qmax: f32,
) -> (f32, f32) {
    // SAFETY: reached only after Isa::Avx2Fma.is_available() (module docs).
    unsafe { quantize_row_with_abs_avx2(x, out, out_abs, qmax) }
}

#[target_feature(enable = "avx2,fma")]
fn q8(x: &[f32], i: usize, c: __m256, iv: __m256, qm: i32) -> __m256i {
    debug_assert!(i + 8 <= x.len());
    // SAFETY: caller checks i + 32 <= x.len().
    let v = unsafe { _mm256_loadu_ps(x.as_ptr().add(i)) };
    let r = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_sub_ps(v, c), iv));
    _mm256_min_epi32(
        _mm256_max_epi32(r, _mm256_set1_epi32(-qm)),
        _mm256_set1_epi32(qm),
    )
}

#[target_feature(enable = "avx2,fma")]
fn quantize_row_with_abs_avx2(
    x: &[f32],
    out: &mut [i8],
    out_abs: &mut [u8],
    qmax: f32,
) -> (f32, f32) {
    let qm = qmax as i32;
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len(), out_abs.len());
    let n = x.len();
    let main16 = n - n % 16;
    let main8 = n - n % 8;
    let zero = _mm256_setzero_ps();
    let (mut lo0, mut hi0) = (
        _mm256_set1_ps(f32::INFINITY),
        _mm256_set1_ps(f32::NEG_INFINITY),
    );
    let (mut lo1, mut hi1) = (lo0, hi0);
    let (mut bad0, mut bad1) = (zero, zero);
    let mut i = 0;
    while i < main16 {
        // SAFETY: i + 16 <= n.
        unsafe {
            let v0 = _mm256_loadu_ps(x.as_ptr().add(i));
            let v1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
            lo0 = _mm256_min_ps(lo0, v0);
            hi0 = _mm256_max_ps(hi0, v0);
            bad0 = _mm256_add_ps(bad0, _mm256_mul_ps(v0, zero));
            lo1 = _mm256_min_ps(lo1, v1);
            hi1 = _mm256_max_ps(hi1, v1);
            bad1 = _mm256_add_ps(bad1, _mm256_mul_ps(v1, zero));
        }
        i += 16;
    }
    let mut lo = _mm256_min_ps(lo0, lo1);
    let mut hi = _mm256_max_ps(hi0, hi1);
    let mut bad = _mm256_add_ps(bad0, bad1);
    while i < main8 {
        // SAFETY: i + 8 <= n.
        let v = unsafe { _mm256_loadu_ps(x.as_ptr().add(i)) };
        lo = _mm256_min_ps(lo, v);
        hi = _mm256_max_ps(hi, v);
        bad = _mm256_add_ps(bad, _mm256_mul_ps(v, zero));
        i += 8;
    }
    let mut b8 = [0.0f32; 8];
    // SAFETY: b8 holds 8 floats.
    unsafe { _mm256_storeu_ps(b8.as_mut_ptr(), bad) };
    if b8.iter().any(|v| v.is_nan()) {
        out_abs.fill(0);
        return scalar::finish_row(x, out, f32::NAN, f32::NAN, qmax);
    }
    let (mut l8, mut h8) = ([0.0f32; 8], [0.0f32; 8]);
    // SAFETY: both arrays hold 8 floats.
    unsafe {
        _mm256_storeu_ps(l8.as_mut_ptr(), lo);
        _mm256_storeu_ps(h8.as_mut_ptr(), hi);
    }
    let (tl, th) = scalar::row_range(&x[main8..]);
    let lo_s = l8.iter().fold(tl, |a, &v| a.min(v));
    let hi_s = h8.iter().fold(th, |a, &v| a.max(v));
    let Some((c, scale, inv)) = scalar::row_params(lo_s, hi_s, qmax) else {
        out_abs.fill(0);
        return scalar::finish_row(x, out, lo_s, hi_s, qmax);
    };
    let (cv, iv) = (_mm256_set1_ps(c), _mm256_set1_ps(inv));
    let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    let main32 = n - n % 32;
    let mut i = 0;
    while i < main32 {
        let a = _mm256_packs_epi32(q8(x, i, cv, iv, qm), q8(x, i + 8, cv, iv, qm));
        let b = _mm256_packs_epi32(q8(x, i + 16, cv, iv, qm), q8(x, i + 24, cv, iv, qm));
        let p = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(a, b), perm);
        let pa = _mm256_abs_epi8(p);
        // SAFETY: i + 32 <= n = out.len() = out_abs.len().
        unsafe {
            _mm256_storeu_si256(out.as_mut_ptr().add(i).cast(), p);
            _mm256_storeu_si256(out_abs.as_mut_ptr().add(i).cast(), pa);
        }
        i += 32;
    }
    for ((o, oa), &v) in out[main32..]
        .iter_mut()
        .zip(&mut out_abs[main32..])
        .zip(&x[main32..])
    {
        let q = ((v - c) * inv).round_ties_even().clamp(-qmax, qmax) as i8;
        *o = q;
        *oa = q.unsigned_abs();
    }
    (scale, c)
}
