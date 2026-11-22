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

/// Scores for 8 keys at a time (one tile; lane = key) in `scalar::score_fixed_order`'s
/// summation order, so they are bit-identical to the original row-major kernel.
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
            for t in t_first..=t_last {
                let tile = &kt[t * dh * 8..(t + 1) * dh * 8];
                let row = |e: usize| {
                    // SAFETY: e < dh, so e * 8 + 8 <= dh * 8 = tile.len().
                    unsafe { _mm256_loadu_ps(tile.as_ptr().add(e * 8)) }
                };
                let (lo, hi) = ((t * 8).max(start), (t * 8 + 8).min(start + len));
                for qi in 0..n {
                    let qr = &q[qi * q_stride..qi * q_stride + dh];
                    // score_fixed_order, lane = key: pair chains (l, l + 8) -> v[l].
                    let mut v = [_mm256_setzero_ps(); 8];
                    for (l, vl) in v.iter_mut().enumerate() {
                        let (mut a0, mut a1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
                        for c in 0..dh / 16 {
                            a0 = _mm256_fmadd_ps(
                                _mm256_set1_ps(qr[16 * c + l]),
                                row(16 * c + l),
                                a0,
                            );
                            a1 = _mm256_fmadd_ps(
                                _mm256_set1_ps(qr[16 * c + 8 + l]),
                                row(16 * c + 8 + l),
                                a1,
                            );
                        }
                        if dh % 16 == 8 {
                            a0 = _mm256_fmadd_ps(
                                _mm256_set1_ps(qr[dh - 8 + l]),
                                row(dh - 8 + l),
                                a0,
                            );
                        }
                        *vl = _mm256_add_ps(a0, a1);
                    }
                    let u: [__m256; 4] = std::array::from_fn(|l| _mm256_add_ps(v[l], v[l + 4]));
                    let w0 = _mm256_add_ps(u[0], u[2]);
                    let w1 = _mm256_add_ps(u[1], u[3]);
                    let sc = _mm256_mul_ps(_mm256_add_ps(w0, w1), sv);
                    let mut tmp = [0.0f32; 8];
                    // SAFETY: tmp holds 8 floats.
                    unsafe { _mm256_storeu_ps(tmp.as_mut_ptr(), sc) };
                    let dst = qi * nk + j0 + (lo - start);
                    scores[dst..dst + (hi - lo)].copy_from_slice(&tmp[lo - t * 8..hi - t * 8]);
                }
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
    // First normalize softmax weights for all `n` queries in-place (same exact exp/sum order).
    let mut invs = [0.0f32; 64];
    assert!(n <= invs.len(), "attention_values n > 64");
    for (i, inv_slot) in invs.iter_mut().enumerate().take(n) {
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
        *inv_slot = 1.0 / sum;
    }
    // 2-Query Attention Value Mixing (0-ULP bit-identical FMA order, halves V-ring loads!).
    // Process in 32-float (4-YMM) strips so acc0[4] + acc1[4] + loads stay <= 11/16 YMM regs.
    let mut i = 0;
    while i + 1 < n {
        let sc0 = &scores[i * nk..(i + 1) * nk];
        let sc1 = &scores[(i + 1) * nk..(i + 2) * nk];
        let (inv0, inv1) = (invs[i], invs[i + 1]);
        let mut v0 = 0;
        while v0 < nv {
            let strip = (nv - v0).min(4);
            let mut acc0 = [_mm256_setzero_ps(); 4];
            let mut acc1 = [_mm256_setzero_ps(); 4];
            let rows = vals[0].chunks_exact(dh).chain(vals[1].chunks_exact(dh));
            for ((&p0, &p1), vr) in sc0.iter().zip(sc1.iter()).zip(rows) {
                let pv0 = _mm256_set1_ps(p0 * inv0);
                let pv1 = _mm256_set1_ps(p1 * inv1);
                for c in 0..strip {
                    // SAFETY: (v0 + c) * 8 + 8 <= dh = vr.len().
                    let vc = unsafe { _mm256_loadu_ps(vr.as_ptr().add((v0 + c) * 8)) };
                    acc0[c] = _mm256_fmadd_ps(pv0, vc, acc0[c]);
                    acc1[c] = _mm256_fmadd_ps(pv1, vc, acc1[c]);
                }
            }
            let o0 = &mut out[i * o_stride + v0 * 8..i * o_stride + (v0 + strip) * 8];
            for (a, oc) in acc0[..strip].iter().zip(o0.chunks_exact_mut(8)) {
                // SAFETY: oc.len() == 8.
                unsafe { _mm256_storeu_ps(oc.as_mut_ptr(), *a) };
            }
            let o1 = &mut out[(i + 1) * o_stride + v0 * 8..(i + 1) * o_stride + (v0 + strip) * 8];
            for (a, oc) in acc1[..strip].iter().zip(o1.chunks_exact_mut(8)) {
                // SAFETY: oc.len() == 8.
                unsafe { _mm256_storeu_ps(oc.as_mut_ptr(), *a) };
            }
            v0 += strip;
        }
        i += 2;
    }
    if i < n {
        let sc = &scores[i * nk..(i + 1) * nk];
        let inv = invs[i];
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
