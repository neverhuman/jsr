//! One softmax attention head over a ring window (scores, exp, weighted values).
//! Safety model and dispatch: see the parent module docs.

use std::arch::x86_64::*;

use super::super::scalar;
use super::{exp8, hsum};

#[allow(clippy::too_many_arguments)]
pub(crate) fn attend_head(
    q: &[f32],
    q_stride: usize,
    n: usize,
    dh: usize,
    keys: [&[f32]; 2],
    vals: [&[f32]; 2],
    scale: f32,
    scores: &mut [f32],
    out: &mut [f32],
    o_stride: usize,
) {
    // SAFETY: reached only after Isa::Avx2Fma.is_available() (module docs).
    unsafe { attend_head_avx2(q, q_stride, n, dh, keys, vals, scale, scores, out, o_stride) }
}

/// Up to 16 vectors (dh <= 128) of one head row, held in registers.
const MAXV: usize = 16;

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
fn attend_head_avx2(
    q: &[f32],
    q_stride: usize,
    n: usize,
    dh: usize,
    keys: [&[f32]; 2],
    vals: [&[f32]; 2],
    scale: f32,
    scores: &mut [f32],
    out: &mut [f32],
    o_stride: usize,
) {
    let nv = dh / 8;
    assert!(dh.is_multiple_of(8) && nv <= MAXV);
    let nk = (keys[0].len() + keys[1].len()) / dh;
    assert!(scores.len() >= nk && q.len() >= (n - 1) * q_stride + dh);
    assert!(out.len() >= (n - 1) * o_stride + dh);
    for i in 0..n {
        let qi = &q[i * q_stride..i * q_stride + dh];
        let mut qv = [_mm256_setzero_ps(); MAXV];
        for (v, qr) in qv.iter_mut().zip(qi.chunks_exact(8)) {
            // SAFETY: qr.len() == 8.
            *v = unsafe { _mm256_loadu_ps(qr.as_ptr()) };
        }
        let mut j = 0;
        for seg in keys {
            for kr in seg.chunks_exact(dh) {
                let (mut a0, mut a1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
                for (c, kc) in kr.chunks_exact(16).enumerate() {
                    // SAFETY: kc.len() == 16.
                    unsafe {
                        a0 = _mm256_fmadd_ps(qv[2 * c], _mm256_loadu_ps(kc.as_ptr()), a0);
                        a1 =
                            _mm256_fmadd_ps(qv[2 * c + 1], _mm256_loadu_ps(kc.as_ptr().add(8)), a1);
                    }
                }
                if !nv.is_multiple_of(2) {
                    // SAFETY: the last 8 floats of kr.
                    a0 = _mm256_fmadd_ps(
                        qv[nv - 1],
                        unsafe { _mm256_loadu_ps(kr.as_ptr().add(dh - 8)) },
                        a0,
                    );
                }
                scores[j] = hsum(_mm256_add_ps(a0, a1)) * scale;
                j += 1;
            }
        }
        let sc = &mut scores[..nk];
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
        let mut acc = [_mm256_setzero_ps(); MAXV];
        let mut j = 0;
        for seg in vals {
            for vr in seg.chunks_exact(dh) {
                let p = _mm256_set1_ps(scores[j] * inv);
                for (a, vc) in acc[..nv].iter_mut().zip(vr.chunks_exact(8)) {
                    // SAFETY: vc.len() == 8.
                    *a = _mm256_fmadd_ps(p, unsafe { _mm256_loadu_ps(vc.as_ptr()) }, *a);
                }
                j += 1;
            }
        }
        let o = &mut out[i * o_stride..i * o_stride + dh];
        for (a, oc) in acc[..nv].iter().zip(o.chunks_exact_mut(8)) {
            // SAFETY: oc.len() == 8.
            unsafe { _mm256_storeu_ps(oc.as_mut_ptr(), *a) };
        }
    }
}
