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
    for i in 0..n {
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
        let inv = 1.0 / sum;
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
