//! One Conformer block (macaron FFN, chunk-masked RoPE attention over a KV ring window,
//! causal depthwise convolution with a state ring, FFN, optional post-norm), batched over
//! the rows of several streams that share a rate group.

use super::{grow, linear, norm, EncWorkspace, Precision, View};
use crate::kernels::{self, Isa};
use crate::model::{Block, Linear, Model, Norm};
use crate::prof::Phase;

/// Time `$e` into phase `$p` of `$prof` when profiling is on.
macro_rules! timed {
    ($prof:expr, $p:expr, $e:expr) => {{
        let t = $prof.start();
        let r = $e;
        $prof.stop($p, t);
        r
    }};
}

/// The FFN on `ws.ln` (already `LN(x)`): leaves `FFN(LN(x))` in `ws.y` (the caller adds the
/// half residual, fused with the next norm).
fn ffn(isa: Isa, prec: Precision, w1: &Linear, w2: &Linear, ws: &mut EncWorkspace, rows: usize) {
    let d = w2.n();
    let h = grow(&mut ws.h, rows * w1.n());
    linear(isa, prec, w1, &ws.ln[..rows * w1.k()], rows, h, &mut ws.q);
    timed!(ws.q.prof, Phase::Act, kernels::silu_inplace(isa, h));
    let y = grow(&mut ws.y, rows * d);
    linear(isa, prec, w2, h, rows, y, &mut ws.q);
}

/// `x += alpha * y`, then `ln = LN(x)`, one row at a time so each row is normalised while it
/// is still in L1. Per element and per row this is exactly `add_scaled` followed by
/// `layer_norm` (both are row-local), so the results are bit-identical.
#[allow(clippy::too_many_arguments)]
fn add_norm(
    isa: Isa,
    x: &mut [f32],
    y: &[f32],
    alpha: f32,
    nm: &Norm,
    ln: &mut [f32],
    rows: usize,
    d: usize,
) {
    for r in 0..rows {
        let xr = &mut x[r * d..(r + 1) * d];
        kernels::add_scaled(isa, xr, &y[r * d..(r + 1) * d], alpha);
        norm(isa, nm, xr, d, &mut ln[r * d..(r + 1) * d]);
    }
}

/// One Conformer block over all views' rows; `li` is the block's cache layer within the
/// views' rate group.
///
/// `cs` holds the RoPE angle table of every row (`[cos | sin]`, `dh` values per row, rows in
/// view order), computed once per step and rate group by [`rope_table`].
///
/// Each residual add is fused with the norm that follows it ([`add_norm`]): FFN1's half
/// residual with the attention norm, attention's with the convolution norm, the
/// convolution's with FFN2's norm. The fused passes are profiled as `Norm`.
#[allow(clippy::too_many_arguments)]
pub(super) fn block(
    m: &Model,
    isa: Isa,
    prec: Precision,
    b: &Block,
    li: usize,
    views: &mut [View<'_>],
    ws: &mut EncWorkspace,
    rows: usize,
    cs: &[f32],
) {
    let a = &m.arch;
    let d = a.d;
    {
        let ln = grow(&mut ws.ln, rows * d);
        timed!(
            ws.q.prof,
            Phase::Norm,
            norm(isa, &b.ff1_norm, &ws.x[..rows * d], d, ln)
        );
    }
    ffn(isa, prec, &b.ff1_w1, &b.ff1_w2, ws, rows);

    // Attention.
    {
        let ln = grow(&mut ws.ln, rows * d);
        timed!(
            ws.q.prof,
            Phase::Norm,
            add_norm(
                isa,
                &mut ws.x[..rows * d],
                &ws.y[..rows * d],
                0.5,
                &b.att_norm,
                ln,
                rows,
                d
            )
        );
        let qkv = grow(&mut ws.h, rows * 3 * d);
        linear(isa, prec, &b.qkv, ln, rows, qkv, &mut ws.q);
        let att = grow(&mut ws.h2, rows * d);
        let t_att = ws.q.prof.start();
        let mut r0 = 0;
        for j in views.iter_mut() {
            attend(
                m,
                isa,
                li,
                j,
                &mut qkv[r0 * 3 * d..(r0 + j.n) * 3 * d],
                &mut att[r0 * d..(r0 + j.n) * d],
                &cs[r0 * a.dh..(r0 + j.n) * a.dh],
                &mut ws.scores,
                &mut ws.kv_k,
                &mut ws.kv_v,
            );
            r0 += j.n;
        }
        ws.q.prof.stop(Phase::Attention, t_att);
        let y = grow(&mut ws.y, rows * d);
        linear(isa, prec, &b.att_out, att, rows, y, &mut ws.q);
    }

    // Convolution module.
    {
        let ln = grow(&mut ws.ln, rows * d);
        timed!(
            ws.q.prof,
            Phase::Norm,
            add_norm(
                isa,
                &mut ws.x[..rows * d],
                &ws.y[..rows * d],
                1.0,
                &b.conv_norm,
                ln,
                rows,
                d
            )
        );
        let p1 = grow(&mut ws.h, rows * 2 * d);
        linear(isa, prec, &b.pw1, ln, rows, p1, &mut ws.q);
        let g = grow(&mut ws.h2, rows * d);
        timed!(ws.q.prof, Phase::Act, kernels::glu(isa, p1, d, g));
        let y = grow(&mut ws.y, rows * d);
        let km1 = a.kernel - 1;
        let t_conv = ws.q.prof.start();
        let mut r0 = 0;
        for j in views.iter_mut() {
            let hist = &mut j.conv[li * km1 * d..(li + 1) * km1 * d];
            let cur = &g[r0 * d..(r0 + j.n) * d];
            // Register-blocked tap accumulation across `hist` and `cur` (0-ULP exact).
            for i in 0..j.n {
                let o = &mut y[(r0 + i) * d..(r0 + i + 1) * d];
                kernels::depthwise_conv_frame(isa, o, &b.dw_b, hist, cur, &b.dw_w, i, a.kernel, d);
            }
            if j.n >= km1 {
                hist.copy_from_slice(&cur[(j.n - km1) * d..j.n * d]);
            } else if j.n > 0 {
                hist.copy_within(j.n * d..km1 * d, 0);
                hist[(km1 - j.n) * d..km1 * d].copy_from_slice(cur);
            }
            r0 += j.n;
        }
        ws.q.prof.stop(Phase::Conv, t_conv);
        let ln = grow(&mut ws.ln, rows * d);
        timed!(ws.q.prof, Phase::Norm, norm(isa, &b.dnorm, y, d, ln));
        timed!(ws.q.prof, Phase::Act, kernels::silu_inplace(isa, ln));
        let y2 = grow(&mut ws.y, rows * d);
        linear(isa, prec, &b.pw2, ln, rows, y2, &mut ws.q);
    }

    {
        let ln = grow(&mut ws.ln, rows * d);
        timed!(
            ws.q.prof,
            Phase::Norm,
            add_norm(
                isa,
                &mut ws.x[..rows * d],
                &ws.y[..rows * d],
                1.0,
                &b.ff2_norm,
                ln,
                rows,
                d
            )
        );
    }
    ffn(isa, prec, &b.ff2_w1, &b.ff2_w2, ws, rows);
    timed!(
        ws.q.prof,
        Phase::Act,
        kernels::add_scaled(isa, &mut ws.x[..rows * d], &ws.y[..rows * d], 0.5)
    );
    if let Some(pn) = &b.post_norm {
        let ln = grow(&mut ws.ln, rows * d);
        timed!(
            ws.q.prof,
            Phase::Norm,
            norm(isa, pn, &ws.x[..rows * d], d, ln)
        );
        ws.x[..rows * d].copy_from_slice(ln);
    }
}

/// RoPE angles (`[cos | sin]`, `dh` per row) for rows at positions `pos..pos + n` of every
/// `(pos, n)` in `spans`, appended to `out` (cleared first). Computed once per step and rate
/// group instead of once per layer: the same values, so outputs are bit-identical.
pub(super) fn rope_table(m: &Model, spans: &[(usize, usize)], out: &mut Vec<f32>) {
    let dh = m.arch.dh;
    let half = dh / 2;
    out.clear();
    for &(pos0, n) in spans {
        for i in 0..n {
            let pos_idx = pos0 + i;
            let use_f64 = pos_idx >= 16_384;
            let pos = pos_idx as f32;
            let base = out.len();
            out.resize(base + dh, 0.0);
            for f in 0..half {
                let ang = if use_f64 {
                    ((pos_idx as f64) * f64::from(m.rope_inv[f])).rem_euclid(std::f64::consts::TAU)
                } else {
                    f64::from(pos * m.rope_inv[f])
                };
                out[base + f] = ang.cos() as f32;
                out[base + half + f] = ang.sin() as f32;
            }
        }
    }
}

/// The ring rows `[start, start + first)` then `[0, nk - first)`.
fn window(r: &[f32], start: usize, first: usize, nk: usize, dh: usize) -> [&[f32]; 2] {
    [
        &r[start * dh..(start + first) * dh],
        &r[..(nk - first) * dh],
    ]
}

/// Symmetric per-head/per-slot int8 quantisation for K/V cache compression.
#[inline]
fn quantize_kv_slot(src: &[f32], dst: &mut [i8]) -> f32 {
    let mx = src.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    if mx == 0.0 || !mx.is_finite() {
        dst.fill(0);
        return 0.0;
    }
    let scale = mx / 127.0;
    let inv = 127.0 / mx;
    for (o, &v) in dst.iter_mut().zip(src) {
        *o = (v * inv).round_ties_even().clamp(-127.0, 127.0) as i8;
    }
    scale
}

/// Attention for one stream: `qkv` holds its `n` rows (`[q | k | v]`, heads contiguous);
/// writes `n` rows of concatenated head outputs to `out` and stores the chunk's keys and
/// values into the ring after use.
#[allow(clippy::too_many_arguments)]
fn attend(
    m: &Model,
    isa: Isa,
    li: usize,
    j: &mut View<'_>,
    qkv: &mut [f32],
    out: &mut [f32],
    cs: &[f32],
    scores: &mut Vec<f32>,
    kv_k: &mut Vec<f32>,
    kv_v: &mut Vec<f32>,
) {
    let a = &m.arch;
    let (d, dh, hn, left, n) = (a.d, a.dh, a.heads, j.left, j.n);
    let t0 = j.pos;
    debug_assert_eq!(cs.len(), n * dh);
    for i in 0..n {
        let row = &mut qkv[i * 3 * d..(i + 1) * 3 * d];
        let csi = &cs[i * dh..(i + 1) * dh];
        for h in 0..hn {
            kernels::rope_pair(isa, &mut row[h * dh..(h + 1) * dh], csi);
            kernels::rope_pair(isa, &mut row[d + h * dh..d + (h + 1) * dh], csi);
        }
    }
    // Store the step's keys/values first. The ring holds left + (max chunks) slots, so the
    // overwritten slots (positions <= t0 + n - 1 - ring < t0 - left) are outside every window.
    let ring = j.ring;
    let lbase = li * hn * ring * dh;
    let sbase = li * hn * ring;
    if j.kv_int8 {
        for i in 0..n {
            let slot = (t0 + i) % ring;
            let r = &qkv[i * 3 * d..(i + 1) * 3 * d];
            for h in 0..hn {
                let hb = lbase + h * ring * dh + slot * dh;
                let sb = sbase + h * ring + slot;
                j.k_s[sb] =
                    quantize_kv_slot(&r[d + h * dh..d + (h + 1) * dh], &mut j.k_q[hb..hb + dh]);
                j.v_s[sb] = quantize_kv_slot(
                    &r[2 * d + h * dh..2 * d + (h + 1) * dh],
                    &mut j.v_q[hb..hb + dh],
                );
            }
        }
    } else {
        for i in 0..n {
            let slot = (t0 + i) % ring;
            let r = &qkv[i * 3 * d..(i + 1) * 3 * d];
            for h in 0..hn {
                let hb = lbase + h * ring * dh;
                // Keys in 8-key tiles so scores read 8 keys per load from contiguous memory.
                for (e, &kv) in r[d + h * dh..d + (h + 1) * dh].iter().enumerate() {
                    j.k[hb + kernels::kt_index(slot, e, dh)] = kv;
                }
                let vb = hb + slot * dh;
                j.v[vb..vb + dh].copy_from_slice(&r[2 * d + h * dh..2 * d + (h + 1) * dh]);
            }
        }
    }
    // Sub-chunk c = [s0, s0 + m): its queries see positions [s0 - past_c, s0 + m), with
    // past_c = min(left, kv_len + (s0 - t0)) -- exactly the trained chunk mask.
    let scale = 1.0 / (dh as f32).sqrt();
    let mut off = 0;
    while off < n {
        let m = j.chunk.min(n - off);
        let s0 = t0 + off;
        let past = (j.kv_len + off).min(left);
        let nk = past + m;
        let start = (s0 - past) % ring;
        let first = nk.min(ring - start);
        let sc = grow(scores, m * nk);
        if j.kv_int8 {
            let kt_ring = nk.next_multiple_of(kernels::KT_TILE);
            let kt_buf = grow(kv_k, kt_ring * dh);
            let vr_buf = grow(kv_v, nk * dh);
            for h in 0..hn {
                let hb = lbase + h * ring * dh;
                let sb = sbase + h * ring;
                let mut dst_slot = 0usize;
                for (seg_start, seg_len) in [(start, first), (0, nk - first)] {
                    for slot in seg_start..seg_start + seg_len {
                        let ks = j.k_s[sb + slot];
                        let vs = j.v_s[sb + slot];
                        let kq = &j.k_q[hb + slot * dh..hb + (slot + 1) * dh];
                        let vq = &j.v_q[hb + slot * dh..hb + (slot + 1) * dh];
                        for (e, &q) in kq.iter().enumerate() {
                            kt_buf[kernels::kt_index(dst_slot, e, dh)] = f32::from(q) * ks;
                        }
                        let vo = &mut vr_buf[dst_slot * dh..(dst_slot + 1) * dh];
                        for (o, &q) in vo.iter_mut().zip(vq) {
                            *o = f32::from(q) * vs;
                        }
                        dst_slot += 1;
                    }
                }
                kernels::attend_head(
                    isa,
                    &qkv[off * 3 * d + h * dh..],
                    3 * d,
                    m,
                    dh,
                    kt_buf,
                    kt_ring,
                    [(0, nk), (0, 0)],
                    [vr_buf, &[]],
                    scale,
                    sc,
                    &mut out[off * d + h * dh..],
                    d,
                );
            }
        } else {
            for h in 0..hn {
                let hb = lbase + h * ring * dh;
                let kt = &j.k[hb..hb + ring * dh];
                let vr = &j.v[hb..hb + ring * dh];
                kernels::attend_head(
                    isa,
                    &qkv[off * 3 * d + h * dh..],
                    3 * d,
                    m,
                    dh,
                    kt,
                    ring,
                    [(start, first), (0, nk - first)],
                    window(vr, start, first, nk, dh),
                    scale,
                    sc,
                    &mut out[off * d + h * dh..],
                    d,
                );
            }
        }
        off += m;
    }
}
