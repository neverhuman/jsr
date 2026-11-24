//! Softmax attention score computation over a ring window (8-key tiles, 2-query register tiling).
//! Safety model and dispatch: see the parent module docs.

use std::arch::x86_64::*;

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

/// One reduction lane `l in 0..8` for 2 queries when `dh == 64` (`c = 0..4`, zero branches).
#[target_feature(enable = "avx2,fma")]
unsafe fn lane2_d64(q0: *const f32, q1: *const f32, tp: *const f32, l: usize) -> (__m256, __m256) {
    // SAFETY: caller ensures q0, q1 have >= 64 floats and tp has >= 64 * 8 floats.
    unsafe {
        let (k0, k1) = (
            _mm256_loadu_ps(tp.add(l * 8)),
            _mm256_loadu_ps(tp.add((8 + l) * 8)),
        );
        let mut a00 = _mm256_mul_ps(_mm256_broadcast_ss(&*q0.add(l)), k0);
        let mut a10 = _mm256_mul_ps(_mm256_broadcast_ss(&*q1.add(l)), k0);
        let mut a01 = _mm256_mul_ps(_mm256_broadcast_ss(&*q0.add(8 + l)), k1);
        let mut a11 = _mm256_mul_ps(_mm256_broadcast_ss(&*q1.add(8 + l)), k1);
        for c in 1..4 {
            let (e0, e1) = (16 * c + l, 16 * c + 8 + l);
            let k0 = _mm256_loadu_ps(tp.add(e0 * 8));
            let k1 = _mm256_loadu_ps(tp.add(e1 * 8));
            a00 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q0.add(e0)), k0, a00);
            a10 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q1.add(e0)), k0, a10);
            a01 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q0.add(e1)), k1, a01);
            a11 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*q1.add(e1)), k1, a11);
        }
        (_mm256_add_ps(a00, a01), _mm256_add_ps(a10, a11))
    }
}

/// One reduction lane `l in 0..8` for 1 query (generic `dh`, 0-ULP exact).
#[target_feature(enable = "avx2,fma")]
unsafe fn score_lane1(q0: *const f32, tp: *const f32, dh: usize, l: usize) -> __m256 {
    let (mut a0, mut a1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    // SAFETY: caller ensures q0 has >= dh floats and tp has >= dh * 8 floats.
    unsafe {
        for c in 0..dh / 16 {
            let (e0, e1) = (16 * c + l, 16 * c + 8 + l);
            a0 = _mm256_fmadd_ps(
                _mm256_broadcast_ss(&*q0.add(e0)),
                _mm256_loadu_ps(tp.add(e0 * 8)),
                a0,
            );
            a1 = _mm256_fmadd_ps(
                _mm256_broadcast_ss(&*q0.add(e1)),
                _mm256_loadu_ps(tp.add(e1 * 8)),
                a1,
            );
        }
        if dh % 16 == 8 {
            let e0 = dh - 8 + l;
            a0 = _mm256_fmadd_ps(
                _mm256_broadcast_ss(&*q0.add(e0)),
                _mm256_loadu_ps(tp.add(e0 * 8)),
                a0,
            );
        }
    }
    _mm256_add_ps(a0, a1)
}

#[target_feature(enable = "avx2,fma")]
unsafe fn store_score8(
    scores: &mut [f32],
    dst: usize,
    sc: __m256,
    lo: usize,
    hi: usize,
    t8: usize,
) {
    if hi - lo == 8 {
        // SAFETY: caller ensures dst + 8 <= scores.len().
        unsafe { _mm256_storeu_ps(scores.as_mut_ptr().add(dst), sc) };
    } else {
        let mut tmp = [0.0f32; 8];
        // SAFETY: tmp holds 8 floats.
        unsafe { _mm256_storeu_ps(tmp.as_mut_ptr(), sc) };
        scores[dst..dst + (hi - lo)].copy_from_slice(&tmp[lo - t8..hi - t8]);
    }
}

/// 2-query `dh == 64` score tile kept out-of-line so LLVM allocates <= 14 YMMs cleanly with
/// zero stack spills (exact `((v0+v4)+(v2+v6)) + ((v1+v5)+(v3+v7))` tree order).
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn tile2_d64(
    q0: *const f32,
    q1: *const f32,
    tp: *const f32,
    sv: __m256,
) -> (__m256, __m256) {
    // SAFETY: caller ensures q0, q1 have >= 64 floats and tp has >= 64 * 8 floats.
    unsafe {
        let (v0a, v0b) = lane2_d64(q0, q1, tp, 0);
        let (v4a, v4b) = lane2_d64(q0, q1, tp, 4);
        let (u0a, u0b) = (_mm256_add_ps(v0a, v4a), _mm256_add_ps(v0b, v4b));
        let (v2a, v2b) = lane2_d64(q0, q1, tp, 2);
        let (v6a, v6b) = lane2_d64(q0, q1, tp, 6);
        let w0a = _mm256_add_ps(u0a, _mm256_add_ps(v2a, v6a));
        let w0b = _mm256_add_ps(u0b, _mm256_add_ps(v2b, v6b));
        let (v1a, v1b) = lane2_d64(q0, q1, tp, 1);
        let (v5a, v5b) = lane2_d64(q0, q1, tp, 5);
        let (u1a, u1b) = (_mm256_add_ps(v1a, v5a), _mm256_add_ps(v1b, v5b));
        let (v3a, v3b) = lane2_d64(q0, q1, tp, 3);
        let (v7a, v7b) = lane2_d64(q0, q1, tp, 7);
        let w1a = _mm256_add_ps(u1a, _mm256_add_ps(v3a, v7a));
        let w1b = _mm256_add_ps(u1b, _mm256_add_ps(v3b, v7b));
        (
            _mm256_mul_ps(_mm256_add_ps(w0a, w1a), sv),
            _mm256_mul_ps(_mm256_add_ps(w0b, w1b), sv),
        )
    }
}

/// Scores for 8 keys at a time in `scalar::score_fixed_order`'s exact summation order.
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
                let tp = kt[t * dh * 8..(t + 1) * dh * 8].as_ptr();
                let t8 = t * 8;
                let (lo, hi) = (t8.max(start), (t8 + 8).min(start + len));
                let col_off = j0 + (lo - start);
                let mut qi = 0;
                if dh == 64 {
                    while qi + 2 <= n {
                        let q0 = q[qi * q_stride..qi * q_stride + 64].as_ptr();
                        let q1 = q[(qi + 1) * q_stride..(qi + 1) * q_stride + 64].as_ptr();
                        // SAFETY: q0, q1 have >= 64 floats; tp has 64*8 floats; dst + (hi-lo) <= scores.len().
                        unsafe {
                            let (sca, scb) = tile2_d64(q0, q1, tp, sv);
                            store_score8(scores, qi * nk + col_off, sca, lo, hi, t8);
                            store_score8(scores, (qi + 1) * nk + col_off, scb, lo, hi, t8);
                        }
                        qi += 2;
                    }
                }
                while qi < n {
                    let q0 = q[qi * q_stride..qi * q_stride + dh].as_ptr();
                    // SAFETY: q0 has >= dh floats; tp has dh*8 floats.
                    unsafe {
                        let mut v = [_mm256_setzero_ps(); 8];
                        for (l, vl) in v.iter_mut().enumerate() {
                            *vl = score_lane1(q0, tp, dh, l);
                        }
                        let u0 = _mm256_add_ps(v[0], v[4]);
                        let u1 = _mm256_add_ps(v[1], v[5]);
                        let u2 = _mm256_add_ps(v[2], v[6]);
                        let u3 = _mm256_add_ps(v[3], v[7]);
                        let sc = _mm256_mul_ps(
                            _mm256_add_ps(_mm256_add_ps(u0, u2), _mm256_add_ps(u1, u3)),
                            sv,
                        );
                        store_score8(scores, qi * nk + col_off, sc, lo, hi, t8);
                    }
                    qi += 1;
                }
            }
        }
        j0 += len;
    }
}
