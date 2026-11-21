//! One softmax attention head over a ring window (scores, exp, weighted values).
//! Safety model and dispatch: see the parent module docs.

use std::arch::x86_64::*;

use super::super::scalar;
use super::{exp8, hsum};

#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_scores(
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
    // SAFETY: reached only after Isa::Avx2Fma.is_available() (module docs).
    unsafe { scores_avx2(q, q_stride, n, dh, kt, ring, segs, scale, scores) }
}

/// Queries handled together per 8-key column block.
const QB: usize = 4;

/// Scores for 8 keys at a time: `kt` rows are contiguous over slots, so one load serves
/// 8 keys and each query adds `q[e] * k[e][j..j+8]` with one FMA (no horizontal sums).
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
fn scores_avx2(
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
    let nk = segs[0].1 + segs[1].1;
    assert!(kt.len() >= dh * ring && scores.len() >= n * nk);
    assert!(q.len() >= (n - 1) * q_stride + dh);
    assert!(ring.is_multiple_of(8) && segs.iter().all(|&(s, l)| s + l <= ring));
    let sv = _mm256_set1_ps(scale);
    let mut j0 = 0;
    for (start, len) in segs {
        if len > 0 {
            let (t_first, t_last) = (start / 8, (start + len - 1) / 8);
            let mut i0 = 0;
            while i0 < n {
                let qn = (n - i0).min(QB);
                for t in t_first..=t_last {
                    let tile = &kt[t * dh * 8..(t + 1) * dh * 8];
                    let mut acc = [_mm256_setzero_ps(); QB];
                    for (e, kv8) in tile.chunks_exact(8).enumerate() {
                        // SAFETY: kv8.len() == 8.
                        let kv = unsafe { _mm256_loadu_ps(kv8.as_ptr()) };
                        for (a, qi) in acc[..qn].iter_mut().zip(i0..i0 + qn) {
                            let qv = _mm256_set1_ps(q[qi * q_stride + e]);
                            *a = _mm256_fmadd_ps(qv, kv, *a);
                        }
                    }
                    // Keep only the tile lanes inside the window segment.
                    let lo = (t * 8).max(start);
                    let hi = (t * 8 + 8).min(start + len);
                    for (a, qi) in acc[..qn].iter().zip(i0..i0 + qn) {
                        let mut tmp = [0.0f32; 8];
                        // SAFETY: tmp holds 8 floats.
                        unsafe { _mm256_storeu_ps(tmp.as_mut_ptr(), _mm256_mul_ps(*a, sv)) };
                        let dst = qi * nk + j0 + (lo - start);
                        scores[dst..dst + (hi - lo)].copy_from_slice(&tmp[lo - t * 8..hi - t * 8]);
                    }
                }
                i0 += qn;
            }
        }
        j0 += len;
    }
}

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

/// Up to 16 vectors (dh <= 128) of one head row, held in registers.
const MAXV: usize = 16;

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
    assert!(dh.is_multiple_of(8) && nv <= MAXV);
    let nk = (vals[0].len() + vals[1].len()) / dh;
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
        let mut acc = [_mm256_setzero_ps(); MAXV];
        let rows = vals[0].chunks_exact(dh).chain(vals[1].chunks_exact(dh));
        for (&p, vr) in sc.iter().zip(rows) {
            let pv = _mm256_set1_ps(p * inv);
            for (a, vc) in acc[..nv].iter_mut().zip(vr.chunks_exact(8)) {
                // SAFETY: vc.len() == 8.
                *a = _mm256_fmadd_ps(pv, unsafe { _mm256_loadu_ps(vc.as_ptr()) }, *a);
            }
        }
        let o = &mut out[i * o_stride..i * o_stride + dh];
        for (a, oc) in acc[..nv].iter().zip(o.chunks_exact_mut(8)) {
            // SAFETY: oc.len() == 8.
            unsafe { _mm256_storeu_ps(oc.as_mut_ptr(), *a) };
        }
    }
}
