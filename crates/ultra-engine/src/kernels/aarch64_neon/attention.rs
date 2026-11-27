//! Attention scores and softmax-weighted values, in the AVX2 kernels' exact orders.
//! Safety model: see the parent module docs.

use super::super::scalar;
use super::{exp8, V8};

/// Scores in `scalar::score_fixed_order`'s order, one key per lane (a tile = 8 keys = two
/// four-lane halves), bit-identical to the AVX2 and scalar kernels.
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
    let nk = segs[0].1 + segs[1].1;
    assert!(kt.len() >= dh * ring && scores.len() >= n * nk);
    assert!(q.len() >= (n - 1) * q_stride + dh);
    assert!(ring.is_multiple_of(8) && segs.iter().all(|&(s, l)| s + l <= ring));
    let sv = V8::splat(scale);
    let mut j0 = 0;
    for (start, len) in segs {
        if len > 0 {
            let (t_first, t_last) = (start / 8, (start + len - 1) / 8);
            for t in t_first..=t_last {
                let tile = &kt[t * dh * 8..(t + 1) * dh * 8];
                let row = |e: usize| V8::load(tile, e * 8);
                let (lo, hi) = ((t * 8).max(start), (t * 8 + 8).min(start + len));
                for qi in 0..n {
                    let qr = &q[qi * q_stride..qi * q_stride + dh];
                    let mut v = [V8::zero(); 8];
                    for (l, vl) in v.iter_mut().enumerate() {
                        let (mut a0, mut a1) = (V8::zero(), V8::zero());
                        for c in 0..dh / 16 {
                            a0 = V8::splat(qr[16 * c + l]).fma(row(16 * c + l), a0);
                            a1 = V8::splat(qr[16 * c + 8 + l]).fma(row(16 * c + 8 + l), a1);
                        }
                        if dh % 16 == 8 {
                            a0 = V8::splat(qr[dh - 8 + l]).fma(row(dh - 8 + l), a0);
                        }
                        *vl = a0.add(a1);
                    }
                    let u: [V8; 4] = std::array::from_fn(|l| v[l].add(v[l + 4]));
                    let w0 = u[0].add(u[2]);
                    let w1 = u[1].add(u[3]);
                    let sc = w0.add(w1).mul(sv);
                    let mut tmp = [0.0f32; 8];
                    sc.store(&mut tmp, 0);
                    let dst = qi * nk + j0 + (lo - start);
                    scores[dst..dst + (hi - lo)].copy_from_slice(&tmp[lo - t * 8..hi - t * 8]);
                }
            }
        }
        j0 += len;
    }
}

/// Softmax and weighted values exactly as the AVX2 `values_avx2` (8-lane exp chunks with
/// the AVX2 horizontal sum, scalar remainder, FMA accumulation over keys in window order).
pub(crate) fn attention_values(
    n: usize,
    dh: usize,
    vals: [&[f32]; 2],
    scores: &mut [f32],
    out: &mut [f32],
    o_stride: usize,
) {
    assert!(dh.is_multiple_of(8) && dh <= 128);
    let nv = dh / 8;
    let nk = (vals[0].len() + vals[1].len()) / dh;
    assert!(scores.len() >= n * nk && out.len() >= (n - 1) * o_stride + dh);
    let mut invs = [0.0f32; 64];
    if n > invs.len() {
        for start in (0..n).step_by(invs.len()) {
            let end = (start + invs.len()).min(n);
            attention_values(
                end - start,
                dh,
                vals,
                &mut scores[start * nk..end * nk],
                &mut out[start * o_stride..(end - 1) * o_stride + dh],
                o_stride,
            );
        }
        return;
    }
    for (i, inv_slot) in invs.iter_mut().enumerate().take(n) {
        let sc = &mut scores[i * nk..(i + 1) * nk];
        let mx = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mv = V8::splat(mx);
        let mut sv = V8::zero();
        let main = nk - nk % 8;
        let mut c = 0;
        while c < main {
            let e = exp8(V8::load(sc, c).sub(mv));
            e.store(sc, c);
            sv = sv.add(e);
            c += 8;
        }
        let mut sum = sv.hsum();
        for s in &mut sc[main..] {
            *s = scalar::exp_approx(*s - mx);
            sum += *s;
        }
        *inv_slot = 1.0 / sum;
    }
    // 2-Query Attention Value Mixing in 32-float (4 V8 = 8 NEON regs/query) strips (0-ULP exact).
    let mut i = 0;
    while i + 1 < n {
        let sc0 = &scores[i * nk..(i + 1) * nk];
        let sc1 = &scores[(i + 1) * nk..(i + 2) * nk];
        let (inv0, inv1) = (invs[i], invs[i + 1]);
        let mut v0 = 0;
        while v0 < nv {
            let strip = (nv - v0).min(4);
            let mut acc0 = [V8::zero(); 4];
            let mut acc1 = [V8::zero(); 4];
            let rows = vals[0].chunks_exact(dh).chain(vals[1].chunks_exact(dh));
            for ((&p0, &p1), vr) in sc0.iter().zip(sc1.iter()).zip(rows) {
                let pv0 = V8::splat(p0 * inv0);
                let pv1 = V8::splat(p1 * inv1);
                for c in 0..strip {
                    let vc = V8::load(vr, 8 * (v0 + c));
                    acc0[c] = pv0.fma(vc, acc0[c]);
                    acc1[c] = pv1.fma(vc, acc1[c]);
                }
            }
            let o0 = &mut out[i * o_stride..i * o_stride + dh];
            for (c, a) in acc0[..strip].iter().enumerate() {
                a.store(o0, 8 * (v0 + c));
            }
            let o1 = &mut out[(i + 1) * o_stride..(i + 1) * o_stride + dh];
            for (c, a) in acc1[..strip].iter().enumerate() {
                a.store(o1, 8 * (v0 + c));
            }
            v0 += strip;
        }
        i += 2;
    }
    if i < n {
        let sc = &scores[i * nk..(i + 1) * nk];
        let inv = invs[i];
        let mut acc = [V8::zero(); 16];
        let rows = vals[0].chunks_exact(dh).chain(vals[1].chunks_exact(dh));
        for (&p, vr) in sc.iter().zip(rows) {
            let pv = V8::splat(p * inv);
            for (a, cidx) in acc[..nv].iter_mut().zip(0..nv) {
                *a = pv.fma(V8::load(vr, 8 * cidx), *a);
            }
        }
        let o = &mut out[i * o_stride..i * o_stride + dh];
        for (cidx, a) in acc[..nv].iter().enumerate() {
            a.store(o, 8 * cidx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::attention_values;

    #[test]
    fn values_above_stack_query_limit_match_independent_queries() {
        let (n, dh, nk, stride) = (129, 8, 11, 12);
        let values: Vec<_> = (0..nk * dh).map(|i| (i as f32 - 23.0) / 13.0).collect();
        let windows = [&values[..5 * dh], &values[5 * dh..]];
        let logits: Vec<_> = (0..n * nk).map(|i| ((i * 7) % 31) as f32 / 6.0).collect();
        let mut scores = logits.clone();
        let mut actual = vec![-1.0; n * stride];
        attention_values(n, dh, windows, &mut scores, &mut actual, stride);

        let mut expected = vec![-1.0; n * stride];
        let mut expected_scores = logits;
        for i in 0..n {
            attention_values(
                1,
                dh,
                windows,
                &mut expected_scores[i * nk..(i + 1) * nk],
                &mut expected[i * stride..(i + 1) * stride],
                stride,
            );
        }
        assert_eq!(scores, expected_scores);
        assert_eq!(actual, expected);
    }
}
