//! Chunk forward of the dense R1 encoder for a batch of streams.
//!
//! Arithmetic follows `python/ultra/train/infer.py::ChunkEncoder` (`_stem`, `_run_blocks`):
//! - stem: the chunk `[t0, t0+n)` reads feature rows `4 t0 - 6 ..= 4 (t0+n) - 4`; layer-1
//!   outputs at negative positions (`2 t0 - 2 + j < 0`) are zeroed before layer 2;
//! - each block: `x += 0.5 FFN1(x)`; `x += Att(x)` where the chunk's queries see every valid
//!   cached key (RoPE applied at absolute positions) plus all `n` keys of the chunk;
//!   `x += Conv(x)` (causal depthwise over the last `kernel-1` GLU outputs plus the chunk);
//!   `x += 0.5 FFN2(x)`; optional post-block LayerNorm; then the final LayerNorm.
//!
//! Rows of all streams are stacked so every linear layer is one GEMM over `sum n_i` rows;
//! attention and the depthwise convolution run per stream. Streams in one call may have
//! different `n` (partial final chunks) and positions; nothing a stream computes depends on
//! another stream's rows (per-row quantisation keeps that true for int8 too).

use crate::kernels::{self, Isa};
use crate::model::{Linear, Model, Norm};
use crate::prof::Phase;
use crate::quant::{self, QuantScratch};
use crate::EngineError;

/// Execution precision of the linear layers (attention, norms and the depthwise conv stay fp32).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    F32,
    Int8,
}

impl Precision {
    pub fn parse(s: &str) -> Result<Self, EngineError> {
        match s {
            "fp32" => Ok(Self::F32),
            "int8" => Ok(Self::Int8),
            o => Err(EngineError::Unsupported(format!("precision {o}"))),
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "fp32",
            Self::Int8 => "int8",
        }
    }
}

/// Per-stream encoder caches.
#[derive(Debug, Clone)]
pub struct EncState {
    /// Absolute index of the next encoder frame.
    pub t: usize,
    /// Valid attention cache entries (saturates at `left`).
    pub kv_len: usize,
    /// Ring slots per layer and head: `left + max_chunks * chunk`.
    pub ring: usize,
    /// `[layer][head][ring][dh]`; absolute position `p` lives in slot `p % ring`, so a
    /// step's keys (up to `max_chunks` chunks) can be written before its queries run.
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    /// `[layer][kernel-1][d]`, oldest first.
    pub conv: Vec<f32>,
}

impl EncState {
    /// Caches for steps of up to `max_chunks` chunks.
    pub fn new(m: &Model, max_chunks: usize) -> Self {
        let a = &m.arch;
        let ring = a.left + a.chunk * max_chunks.max(1);
        let kv = a.n_blocks * a.heads * ring * a.dh;
        Self {
            t: 0,
            kv_len: 0,
            ring,
            k: vec![0.0; kv],
            v: vec![0.0; kv],
            conv: vec![0.0; a.n_blocks * (a.kernel - 1) * a.d],
        }
    }
}

/// One stream's step: `n` frames from `state.t` (one or more consecutive chunks; every
/// sub-chunk but the last is full), with its `(4n+3) x 80` feature slice.
pub struct ChunkJob<'a> {
    pub state: &'a mut EncState,
    pub feats: &'a [f32],
    pub n: usize,
}

/// Reusable per-worker buffers (grown on demand, never shrunk).
#[derive(Debug, Default)]
pub struct EncWorkspace {
    c1: Vec<f32>,
    col: Vec<f32>,
    c2: Vec<f32>,
    x: Vec<f32>,
    ln: Vec<f32>,
    h: Vec<f32>,
    h2: Vec<f32>,
    y: Vec<f32>,
    win: Vec<f32>,
    rope: Vec<f32>,
    scores: Vec<f32>,
    pub(crate) q: QuantScratch,
}

fn grow(v: &mut Vec<f32>, n: usize) -> &mut [f32] {
    if v.len() < n {
        v.resize(n, 0.0);
    }
    &mut v[..n]
}

/// Layer taps for parity tests: called with (layer index, rows of all jobs, row width).
/// Layer 0 is the stem, `1..=n_blocks` the blocks, then the final norm and the projection.
pub type Tap<'a> = &'a mut dyn FnMut(usize, &[f32], usize);

/// `out = lin(x)` over `rows` rows, in the requested precision.
pub fn linear(
    isa: Isa,
    prec: Precision,
    lin: &Linear,
    x: &[f32],
    rows: usize,
    out: &mut [f32],
    q: &mut QuantScratch,
) {
    match (prec, &lin.q) {
        (Precision::Int8, Some(p)) => quant::linear_i8(isa, p, x, rows, out, q),
        _ => {
            let t = q.prof.start();
            kernels::gemm_f32(isa, &lin.f, x, rows, &mut out[..rows * lin.n()]);
            q.prof.stop(Phase::GemmF32, t);
        }
    }
}

/// Time `$e` into phase `$p` of `$prof` when profiling is on.
macro_rules! timed {
    ($prof:expr, $p:expr, $e:expr) => {{
        let t = $prof.start();
        let r = $e;
        $prof.stop($p, t);
        r
    }};
}

fn norm(isa: Isa, nm: &Norm, x: &[f32], d: usize, out: &mut [f32]) {
    kernels::layer_norm(isa, x, d, &nm.g, &nm.b, out);
}

/// Run one chunk for every job; writes `sum n_i` rows of joiner-width encoder projections
/// into `out` (job order, `n_i` rows each) and advances every job's state.
pub fn encode_chunk(
    m: &Model,
    isa: Isa,
    prec: Precision,
    jobs: &mut [ChunkJob<'_>],
    ws: &mut EncWorkspace,
    out: &mut Vec<f32>,
    mut tap: Option<Tap<'_>>,
) -> Result<usize, EngineError> {
    let a = &m.arch;
    let d = a.d;
    for j in jobs.iter() {
        if j.n == 0 || j.n > j.state.ring - a.left || j.feats.len() != (4 * j.n + 3) * a.n_mels {
            return Err(EngineError::Shape(format!(
                "chunk job n={} with {} feature values",
                j.n,
                j.feats.len()
            )));
        }
    }
    let rows: usize = jobs.iter().map(|j| j.n).sum();
    stem(m, isa, prec, jobs, ws, rows);
    if let Some(t) = tap.as_mut() {
        t(0, &ws.x[..rows * d], d);
    }
    for li in 0..a.n_blocks {
        block(m, isa, prec, li, jobs, ws, rows);
        if let Some(t) = tap.as_mut() {
            t(li + 1, &ws.x[..rows * d], d);
        }
    }
    let x = &mut ws.x[..rows * d];
    if let Some(fnorm) = &m.final_norm {
        let ln = grow(&mut ws.ln, rows * d);
        timed!(ws.q.prof, Phase::Norm, norm(isa, fnorm, x, d, ln));
        x.copy_from_slice(ln);
    }
    if let Some(t) = tap.as_mut() {
        t(a.n_blocks + 1, x, d);
    }
    let o = grow(out, rows * a.joiner);
    linear(isa, prec, &m.enc_proj, x, rows, o, &mut ws.q);
    if let Some(t) = tap.as_mut() {
        t(a.n_blocks + 2, o, a.joiner);
    }
    if o.iter().any(|v| !v.is_finite()) {
        return Err(EngineError::Output("non-finite encoder projection".into()));
    }
    for j in jobs.iter_mut() {
        j.state.t += j.n;
        j.state.kv_len = (j.state.kv_len + j.n).min(a.left);
    }
    Ok(rows)
}

fn stem(
    m: &Model,
    isa: Isa,
    prec: Precision,
    jobs: &[ChunkJob<'_>],
    ws: &mut EncWorkspace,
    rows: usize,
) {
    let a = &m.arch;
    let (ch, f1, f2, nm) = (a.stem_ch, a.f1, a.f2, a.n_mels);
    let t_stem = ws.q.prof.start();
    let col = grow(&mut ws.col, rows * f2 * 9 * ch);
    let mut crow = 0;
    for j in jobs {
        let n1 = 2 * j.n + 1;
        let c1 = grow(&mut ws.c1, n1 * f1 * ch);
        // Layer 1: out[j1][f][c], channel-last.
        for r in 0..n1 {
            let zero = 2 * j.state.t + r < 2;
            for f in 0..f1 {
                let o = &mut c1[(r * f1 + f) * ch..(r * f1 + f + 1) * ch];
                if zero {
                    o.fill(0.0);
                    continue;
                }
                o.copy_from_slice(&m.c1_b);
                for kh in 0..3 {
                    for kw in 0..3 {
                        let xv = j.feats[(2 * r + kh) * nm + 2 * f + kw];
                        let t = kh * 3 + kw;
                        for (ov, &wv) in o.iter_mut().zip(&m.c1_w[t * ch..(t + 1) * ch]) {
                            *ov += wv * xv;
                        }
                    }
                }
                kernels::silu_inplace(isa, o);
            }
        }
        // im2col for layer 2: row (i, f), K = (kh, kw, c).
        for i in 0..j.n {
            for f in 0..f2 {
                let dst = &mut col[(crow + i * f2 + f) * 9 * ch..][..9 * ch];
                for kh in 0..3 {
                    for kw in 0..3 {
                        let src = ((2 * i + kh) * f1 + 2 * f + kw) * ch;
                        dst[(kh * 3 + kw) * ch..][..ch].copy_from_slice(&c1[src..src + ch]);
                    }
                }
            }
        }
        crow += j.n * f2;
    }
    ws.q.prof.stop(Phase::Stem1, t_stem);
    let c2 = grow(&mut ws.c2, rows * f2 * ch);
    linear(isa, prec, &m.c2, col, rows * f2, c2, &mut ws.q);
    timed!(ws.q.prof, Phase::Act, kernels::silu_inplace(isa, c2));
    let x = grow(&mut ws.x, rows * a.d);
    linear(isa, prec, &m.stem_out, c2, rows, x, &mut ws.q);
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

fn block(
    m: &Model,
    isa: Isa,
    prec: Precision,
    li: usize,
    jobs: &mut [ChunkJob<'_>],
    ws: &mut EncWorkspace,
    rows: usize,
) {
    let a = &m.arch;
    let b = &m.blocks[li];
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
        for j in jobs.iter_mut() {
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
        for j in jobs.iter_mut() {
            let hist = &mut j.state.conv[li * km1 * d..(li + 1) * km1 * d];
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
    j: &mut ChunkJob<'_>,
    qkv: &mut [f32],
    out: &mut [f32],
    rope_buf: &mut Vec<f32>,
    scores: &mut Vec<f32>,
) {
    let a = &m.arch;
    let (d, dh, hn, left, n) = (a.d, a.dh, a.heads, a.left, j.n);
    let half = dh / 2;
    let t0 = j.state.t;
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
    let ring = j.state.ring;
    let lbase = li * hn * ring * dh;
    for i in 0..n {
        let s = ((t0 + i) % ring) * dh;
        let r = &qkv[i * 3 * d..(i + 1) * 3 * d];
        for h in 0..hn {
            let base = lbase + h * ring * dh + s;
            j.state.k[base..base + dh].copy_from_slice(&r[d + h * dh..d + (h + 1) * dh]);
            j.state.v[base..base + dh].copy_from_slice(&r[2 * d + h * dh..2 * d + (h + 1) * dh]);
        }
    }
    // Sub-chunk c = [s0, s0 + m): its queries see positions [s0 - past_c, s0 + m), with
    // past_c = min(left, kv_len + (s0 - t0)) -- exactly the trained chunk mask.
    let scale = 1.0 / (dh as f32).sqrt();
    let mut off = 0;
    while off < n {
        let m = a.chunk.min(n - off);
        let s0 = t0 + off;
        let past = (j.state.kv_len + off).min(left);
        let nk = past + m;
        let start = (s0 - past) % ring;
        let first = nk.min(ring - start);
        let sc = grow(scores, nk);
        for h in 0..hn {
            let hb = lbase + h * ring * dh;
            let kr = &j.state.k[hb..hb + ring * dh];
            let vr = &j.state.v[hb..hb + ring * dh];
            kernels::attend_head(
                isa,
                &qkv[off * 3 * d + h * dh..],
                3 * d,
                m,
                dh,
                window(kr, start, first, nk, dh),
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
