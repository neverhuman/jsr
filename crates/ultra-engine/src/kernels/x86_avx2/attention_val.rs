//! Softmax normalisation and 2-query register-blocked value mixing over a ring window.
//! Safety model and dispatch: see the parent module docs.

use std::arch::x86_64::*;

use super::super::scalar;
use super::{exp8, hsum};

pub(crate) fn attention_values(
    n: usize,
    dh: usize,
    vals: [&[f32]; 2],
    scores: &mut [f32],
    out: &mut [f32],
    o_stride: usize,
) {
    // SAFETY: reached only after Isa::Avx2Fma.is_available() (module docs).
    unsafe { values_avx2(n, dh, vals, scores, out, o_stride) }
}

/// Accumulate 4 YMM vectors (32 floats) for 2 queries across one contiguous value segment.
/// Uses 8 explicit YMM accumulators + 3 temps = 11 YMMs -> zero stack spills, zero inner branches.
#[target_feature(enable = "avx2,fma")]
unsafe fn accum_v2_4(
    sc0: &[f32],
    sc1: &[f32],
    vseg: &[f32],
    dh: usize,
    off: usize,
    a0: &mut [__m256; 4],
    a1: &mut [__m256; 4],
) {
    let (mut b00, mut b01, mut b02, mut b03) = (a0[0], a0[1], a0[2], a0[3]);
    let (mut b10, mut b11, mut b12, mut b13) = (a1[0], a1[1], a1[2], a1[3]);
    // SAFETY: caller ensures sc0.len() == sc1.len() == vseg.len() / dh and off + 32 <= dh.
    unsafe {
        // Empty segments and the final unused stride may fall outside the slice.
        let mut vp = vseg.as_ptr().wrapping_add(off);
        for (&s0, &s1) in sc0.iter().zip(sc1) {
            let p0 = _mm256_set1_ps(s0);
            let p1 = _mm256_set1_ps(s1);
            let v0 = _mm256_loadu_ps(vp);
            b00 = _mm256_fmadd_ps(p0, v0, b00);
            b10 = _mm256_fmadd_ps(p1, v0, b10);
            let v1 = _mm256_loadu_ps(vp.add(8));
            b01 = _mm256_fmadd_ps(p0, v1, b01);
            b11 = _mm256_fmadd_ps(p1, v1, b11);
            let v2 = _mm256_loadu_ps(vp.add(16));
            b02 = _mm256_fmadd_ps(p0, v2, b02);
            b12 = _mm256_fmadd_ps(p1, v2, b12);
            let v3 = _mm256_loadu_ps(vp.add(24));
            b03 = _mm256_fmadd_ps(p0, v3, b03);
            b13 = _mm256_fmadd_ps(p1, v3, b13);
            vp = vp.wrapping_add(dh);
        }
    }
    *a0 = [b00, b01, b02, b03];
    *a1 = [b10, b11, b12, b13];
}

#[target_feature(enable = "avx2,fma")]
fn values_avx2(
    n: usize,
    dh: usize,
    vals: [&[f32]; 2],
    scores: &mut [f32],
    out: &mut [f32],
    o_stride: usize,
) {
    let nv = dh / 8;
    assert!(
        dh.is_multiple_of(8)
            && vals[0].len().is_multiple_of(dh)
            && vals[1].len().is_multiple_of(dh)
    );
    let n0 = vals[0].len() / dh;
    let nk = n0 + vals[1].len() / dh;
    assert!(scores.len() >= n * nk && out.len() >= (n - 1) * o_stride + dh);
    for i in 0..n {
        let sc = &mut scores[i * nk..(i + 1) * nk];
        let mx = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mv = _mm256_set1_ps(mx);
        let mut sv = _mm256_setzero_ps();
        let mut chunks = sc.chunks_exact_mut(8);
        for c in &mut chunks {
            // SAFETY: c.len() == 8.
            unsafe {
                let e = exp8(_mm256_sub_ps(_mm256_loadu_ps(c.as_ptr()), mv));
                _mm256_storeu_ps(c.as_mut_ptr(), e);
                sv = _mm256_add_ps(sv, e);
            }
        }
        let mut sum = hsum(sv);
        for s in chunks.into_remainder() {
            *s = scalar::exp_approx(*s - mx);
            sum += *s;
        }
        let inv = 1.0 / sum;
        let inv_v = _mm256_set1_ps(inv);
        let mut chunks = sc.chunks_exact_mut(8);
        for c in &mut chunks {
            // SAFETY: c.len() == 8.
            unsafe {
                let p = _mm256_mul_ps(_mm256_loadu_ps(c.as_ptr()), inv_v);
                _mm256_storeu_ps(c.as_mut_ptr(), p);
            }
        }
        for s in chunks.into_remainder() {
            *s *= inv;
        }
    }
    let mut i = 0;
    while i + 1 < n {
        let sc0 = &scores[i * nk..(i + 1) * nk];
        let sc1 = &scores[(i + 1) * nk..(i + 2) * nk];
        let mut v0 = 0;
        while v0 + 4 <= nv {
            let off = v0 * 8;
            let mut a0 = [_mm256_setzero_ps(); 4];
            let mut a1 = [_mm256_setzero_ps(); 4];
            // SAFETY: off + 32 <= dh, vals[0].len() == n0 * dh, vals[1].len() == (nk - n0) * dh.
            unsafe {
                accum_v2_4(&sc0[..n0], &sc1[..n0], vals[0], dh, off, &mut a0, &mut a1);
                accum_v2_4(&sc0[n0..], &sc1[n0..], vals[1], dh, off, &mut a0, &mut a1);
                let o0 = out.as_mut_ptr().add(i * o_stride + off);
                let o1 = out.as_mut_ptr().add((i + 1) * o_stride + off);
                for c in 0..4 {
                    _mm256_storeu_ps(o0.add(c * 8), a0[c]);
                    _mm256_storeu_ps(o1.add(c * 8), a1[c]);
                }
            }
            v0 += 4;
        }
        for v in v0..nv {
            let off = v * 8;
            let (mut a0, mut a1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
            for (seg, (s0, s1)) in vals
                .iter()
                .zip([(&sc0[..n0], &sc1[..n0]), (&sc0[n0..], &sc1[n0..])])
            {
                for ((&p0, &p1), vr) in s0.iter().zip(s1).zip(seg.chunks_exact(dh)) {
                    // SAFETY: off + 8 <= dh = vr.len().
                    let vc = unsafe { _mm256_loadu_ps(vr.as_ptr().add(off)) };
                    a0 = _mm256_fmadd_ps(_mm256_set1_ps(p0), vc, a0);
                    a1 = _mm256_fmadd_ps(_mm256_set1_ps(p1), vc, a1);
                }
            }
            // SAFETY: i * o_stride + off + 8 <= out.len() and (i+1) * o_stride + off + 8 <= out.len().
            unsafe {
                _mm256_storeu_ps(out.as_mut_ptr().add(i * o_stride + off), a0);
                _mm256_storeu_ps(out.as_mut_ptr().add((i + 1) * o_stride + off), a1);
            }
        }
        i += 2;
    }
    if i < n {
        let sc = &scores[i * nk..(i + 1) * nk];
        for v in 0..nv {
            let off = v * 8;
            let mut a = _mm256_setzero_ps();
            for (seg, ss) in vals.iter().zip([&sc[..n0], &sc[n0..]]) {
                for (&p, vr) in ss.iter().zip(seg.chunks_exact(dh)) {
                    // SAFETY: off + 8 <= dh = vr.len().
                    a = _mm256_fmadd_ps(
                        _mm256_set1_ps(p),
                        unsafe { _mm256_loadu_ps(vr.as_ptr().add(off)) },
                        a,
                    );
                }
            }
            // SAFETY: i * o_stride + off + 8 <= out.len().
            unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(i * o_stride + off), a) };
        }
    }
}
