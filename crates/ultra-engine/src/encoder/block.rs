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

#[allow(clippy::too_many_arguments)]
fn ffn(
    isa: Isa,
    prec: Precision,
    nm: &Norm,
    w1: &Linear,
    w2: &Linear,
    ws: &mut EncWorkspace,
    rows: usize,
    d: usize,
) {
    let x = &mut ws.x[..rows * d];
    let ln = grow(&mut ws.ln, rows * d);
    timed!(ws.q.prof, Phase::Norm, norm(isa, nm, x, d, ln));
    let h = grow(&mut ws.h, rows * w1.n());
    linear(isa, prec, w1, ln, rows, h, &mut ws.q);
    timed!(ws.q.prof, Phase::Act, kernels::silu_inplace(isa, h));
    let y = grow(&mut ws.y, rows * d);
    linear(isa, prec, w2, h, rows, y, &mut ws.q);
    timed!(ws.q.prof, Phase::Act, kernels::add_scaled(isa, x, y, 0.5));
}

/// One Conformer block over all views' rows; `li` is the block's cache layer within the
/// views' rate group.
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
) {
    let a = &m.arch;
    let d = a.d;
    ffn(isa, prec, &b.ff1_norm, &b.ff1_w1, &b.ff1_w2, ws, rows, d);

    // Attention.
    {
        let ln = grow(&mut ws.ln, rows * d);
        timed!(
            ws.q.prof,
            Phase::Norm,
            norm(isa, &b.att_norm, &ws.x[..rows * d], d, ln)
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
                &mut ws.rope,
                &mut ws.scores,
            );
            r0 += j.n;
        }
        ws.q.prof.stop(Phase::Attention, t_att);
        let y = grow(&mut ws.y, rows * d);
        linear(isa, prec, &b.att_out, att, rows, y, &mut ws.q);
        timed!(
            ws.q.prof,
            Phase::Act,
            kernels::add_scaled(isa, &mut ws.x[..rows * d], y, 1.0)
        );
    }

    // Convolution module.
    {
        let ln = grow(&mut ws.ln, rows * d);
        timed!(
            ws.q.prof,
            Phase::Norm,
            norm(isa, &b.conv_norm, &ws.x[..rows * d], d, ln)
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
            let hist = &mut j.g.conv[li * km1 * d..(li + 1) * km1 * d];
            let win = grow(&mut ws.win, (km1 + j.n) * d);
            win[..km1 * d].copy_from_slice(hist);
            win[km1 * d..].copy_from_slice(&g[r0 * d..(r0 + j.n) * d]);
            for i in 0..j.n {
                let o = &mut y[(r0 + i) * d..(r0 + i + 1) * d];
                o.copy_from_slice(&b.dw_b);
                for t in 0..a.kernel {
                    let src = &win[(i + t) * d..(i + t + 1) * d];
                    let w = &b.dw_w[t * d..(t + 1) * d];
                    for ((ov, &sv), &wv) in o.iter_mut().zip(src).zip(w) {
                        *ov += wv * sv;
                    }
                }
            }
            hist.copy_from_slice(&win[j.n * d..]);
            r0 += j.n;
        }
        ws.q.prof.stop(Phase::Conv, t_conv);
        let ln = grow(&mut ws.ln, rows * d);
        timed!(ws.q.prof, Phase::Norm, norm(isa, &b.dnorm, y, d, ln));
        timed!(ws.q.prof, Phase::Act, kernels::silu_inplace(isa, ln));
        let y2 = grow(&mut ws.y, rows * d);
        linear(isa, prec, &b.pw2, ln, rows, y2, &mut ws.q);
        timed!(
            ws.q.prof,
            Phase::Act,
            kernels::add_scaled(isa, &mut ws.x[..rows * d], y2, 1.0)
        );
    }

    ffn(isa, prec, &b.ff2_norm, &b.ff2_w1, &b.ff2_w2, ws, rows, d);
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

/// The ring rows `[start, start + first)` then `[0, nk - first)`.
fn window(r: &[f32], start: usize, first: usize, nk: usize, dh: usize) -> [&[f32]; 2] {
    [
        &r[start * dh..(start + first) * dh],
        &r[..(nk - first) * dh],
    ]
}

/// RoPE on pairs `(2i, 2i+1)` of one head vector at angle table `cs` (`[cos..., sin...]`),
/// with torch's roundings (`x1 * cos - x2 * sin`, `x1 * sin + x2 * cos`).
fn rope(v: &mut [f32], cs: &[f32]) {
    let half = v.len() / 2;
    for i in 0..half {
        let (c, s) = (cs[i], cs[half + i]);
        let (x1, x2) = (v[2 * i], v[2 * i + 1]);
        v[2 * i] = x1 * c - x2 * s;
        v[2 * i + 1] = x1 * s + x2 * c;
    }
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
    rope_buf: &mut Vec<f32>,
    scores: &mut Vec<f32>,
) {
    let a = &m.arch;
    let (d, dh, hn, left, n) = (a.d, a.dh, a.heads, j.g.left, j.n);
    let half = dh / 2;
    let t0 = j.pos;
    let cs = grow(rope_buf, n * dh);
    for i in 0..n {
        let pos = (t0 + i) as f32;
        for f in 0..half {
            let ang = pos * m.rope_inv[f];
            cs[i * dh + f] = f64::from(ang).cos() as f32;
            cs[i * dh + half + f] = f64::from(ang).sin() as f32;
        }
    }
    for i in 0..n {
        let row = &mut qkv[i * 3 * d..(i + 1) * 3 * d];
        for h in 0..hn {
            rope(&mut row[h * dh..(h + 1) * dh], &cs[i * dh..(i + 1) * dh]);
            rope(
                &mut row[d + h * dh..d + (h + 1) * dh],
                &cs[i * dh..(i + 1) * dh],
            );
        }
    }
    // Store the step's keys/values first. The ring holds left + (max chunks) slots, so the
    // overwritten slots (positions <= t0 + n - 1 - ring < t0 - left) are outside every window.
    let ring = j.g.ring;
    let lbase = li * hn * ring * dh;
    for i in 0..n {
        let slot = (t0 + i) % ring;
        let r = &qkv[i * 3 * d..(i + 1) * 3 * d];
        for h in 0..hn {
            let hb = lbase + h * ring * dh;
            // Keys in 8-key tiles so scores read 8 keys per load from contiguous memory.
            for (e, &kv) in r[d + h * dh..d + (h + 1) * dh].iter().enumerate() {
                j.g.k[hb + kernels::kt_index(slot, e, dh)] = kv;
            }
            let vb = hb + slot * dh;
            j.g.v[vb..vb + dh].copy_from_slice(&r[2 * d + h * dh..2 * d + (h + 1) * dh]);
        }
    }
    // Sub-chunk c = [s0, s0 + m): its queries see positions [s0 - past_c, s0 + m), with
    // past_c = min(left, kv_len + (s0 - t0)) -- exactly the trained chunk mask.
    let scale = 1.0 / (dh as f32).sqrt();
    let mut off = 0;
    while off < n {
        let m = j.g.chunk.min(n - off);
        let s0 = t0 + off;
        let past = (j.g.kv_len + off).min(left);
        let nk = past + m;
        let start = (s0 - past) % ring;
        let first = nk.min(ring - start);
        let sc = grow(scores, m * nk);
        for h in 0..hn {
            let hb = lbase + h * ring * dh;
            let kt = &j.g.k[hb..hb + ring * dh];
            let vr = &j.g.v[hb..hb + ring * dh];
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
        off += m;
    }
}
