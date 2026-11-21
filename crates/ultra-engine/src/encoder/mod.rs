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

mod block;
mod stem;

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

/// Attention and depthwise-convolution caches of one rate group: the 25 Hz blocks (all
/// blocks of a dense model; lower + final blocks of a multirate model) or the 12.5 Hz
/// middle blocks of a multirate model.
#[derive(Debug, Clone)]
pub struct GroupCache {
    /// Valid attention cache entries (saturates at `left`).
    pub kv_len: usize,
    /// Attention history and chunk length at this group's rate.
    pub left: usize,
    pub chunk: usize,
    /// Ring slots per layer and head: `left + max_chunks * chunk`.
    pub ring: usize,
    /// Keys `[layer][head][ring / 8][dh][8]` (8-key tiles, `kernels::kt_index`) and values
    /// `[layer][head][ring][dh]`; absolute position `p` lives in slot `p % ring`, so a
    /// step's keys (up to `max_chunks` chunks) can be written before its queries run.
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    /// `[layer][kernel-1][d]`, oldest first.
    pub conv: Vec<f32>,
}

impl GroupCache {
    fn new(m: &Model, layers: usize, left: usize, chunk: usize, max_chunks: usize) -> Self {
        let a = &m.arch;
        // At least left + one step of keys; a whole number of 8-key tiles.
        let ring = (left + chunk * max_chunks.max(1)).next_multiple_of(kernels::KT_TILE);
        let kv = layers * a.heads * ring * a.dh;
        Self {
            kv_len: 0,
            left,
            chunk,
            ring,
            k: vec![0.0; kv],
            v: vec![0.0; kv],
            conv: vec![0.0; layers * (a.kernel - 1) * a.d],
        }
    }
}

/// Per-stream encoder caches. The 25 Hz group (every block of a dense model; lower + final
/// blocks of a multirate model) is stored inline with the field names of the dense layout;
/// a multirate model adds the 12.5 Hz middle group in `gmid`.
#[derive(Debug, Clone)]
pub struct EncState {
    /// Absolute index of the next 25 Hz encoder frame.
    pub t: usize,
    /// Valid 25 Hz attention cache entries (saturates at `left`).
    pub kv_len: usize,
    /// 25 Hz ring slots per layer and head (a multiple of 8, at least left + one step).
    pub ring: usize,
    /// 25 Hz keys `[layer][head][ring / 8][dh][8]`, values `[layer][head][ring][dh]`,
    /// depthwise history `[layer][kernel-1][d]` (see [`GroupCache`]).
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub conv: Vec<f32>,
    left: usize,
    chunk: usize,
    /// Multirate only: the 12.5 Hz middle blocks.
    pub gmid: Option<GroupCache>,
}

impl EncState {
    /// Caches for steps of up to `max_chunks` chunks.
    pub fn new(m: &Model, max_chunks: usize) -> Self {
        let a = &m.arch;
        let g = GroupCache::new(m, a.lower + a.fin, a.left, a.chunk, max_chunks);
        Self {
            t: 0,
            kv_len: 0,
            ring: g.ring,
            k: g.k,
            v: g.v,
            conv: g.conv,
            left: g.left,
            chunk: g.chunk,
            gmid: (a.mid > 0)
                .then(|| GroupCache::new(m, a.mid, a.left / 2, a.chunk / 2, max_chunks)),
        }
    }

    /// Most frames one step may carry (cache capacity).
    pub fn max_step_frames(&self) -> usize {
        (self.ring - self.left) / self.chunk * self.chunk
    }

    /// Bytes held by every cache buffer (both rate groups).
    pub fn retained_bytes(&self) -> usize {
        let g = self
            .gmid
            .as_ref()
            .map_or(0, |g| g.k.capacity() + g.v.capacity() + g.conv.capacity());
        4 * (self.k.capacity() + self.v.capacity() + self.conv.capacity() + g)
    }

    fn view25(&mut self, n: usize) -> View<'_> {
        View {
            pos: self.t,
            n,
            kv_len: self.kv_len,
            left: self.left,
            chunk: self.chunk,
            ring: self.ring,
            k: &mut self.k,
            v: &mut self.v,
            conv: &mut self.conv,
        }
    }
}

impl GroupCache {
    fn view(&mut self, pos: usize, n: usize) -> View<'_> {
        View {
            pos,
            n,
            kv_len: self.kv_len,
            left: self.left,
            chunk: self.chunk,
            ring: self.ring,
            k: &mut self.k,
            v: &mut self.v,
            conv: &mut self.conv,
        }
    }
}

/// One stream's rows in one rate group: `n` rows from position `pos` at that rate, with
/// that group's cache geometry and buffers.
pub(crate) struct View<'a> {
    pos: usize,
    n: usize,
    kv_len: usize,
    left: usize,
    chunk: usize,
    ring: usize,
    k: &'a mut [f32],
    v: &'a mut [f32],
    conv: &'a mut [f32],
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
    skip: Vec<f32>,
    pair_in: Vec<f32>,
    pub(crate) q: QuantScratch,
}

pub(super) fn grow(v: &mut Vec<f32>, n: usize) -> &mut [f32] {
    if v.len() < n {
        v.resize(n, 0.0);
    }
    &mut v[..n]
}

/// Layer taps for parity tests: called with (layer index, rows of all jobs, row width).
/// Layer 0 is the stem, `1..=n_blocks` the blocks, then the final norm and the projection.
pub type Tap<'a> = &'a mut dyn FnMut(usize, &[f32], usize);

/// Largest row count routed to the T-LUT2 ternary kernel (measured crossover on Zen 2).
const TLUT2_MAX_ROWS: usize = 8;

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
    match &lin.q {
        // T-LUT2 is a per-frame kernel: best for few rows (2-bit weights stream from cache);
        // larger batches reuse each int8 weight panel across rows. Both are exact.
        // A layer trained as an int8 GEMM (`qat_i8`) runs int8 in every precision.
        Some(p) if prec == Precision::Int8 || lin.qat_i8 => match &lin.t {
            Some(t) if rows <= TLUT2_MAX_ROWS => quant::linear_t(isa, p, t, x, rows, out, q),
            _ => quant::linear_i8(isa, p, x, rows, out, q),
        },
        _ => {
            let t = q.prof.start();
            kernels::gemm_f32(isa, &lin.f, x, rows, &mut out[..rows * lin.n()]);
            q.prof.stop(Phase::GemmF32, t);
        }
    }
}

pub(super) fn norm(isa: Isa, nm: &Norm, x: &[f32], d: usize, out: &mut [f32]) {
    kernels::layer_norm(isa, x, d, &nm.g, &nm.b, out);
}

/// Run one step for every job; writes `sum n_i` rows of joiner-width encoder projections
/// into `out` (job order, `n_i` rows each) and advances every job's state.
///
/// Taps: 0 = stem, then one per block in execution order (multirate: lower, middle at
/// 12.5 Hz with `ceil(n/2)` rows per job, final after restoration + skip), then the final
/// norm and the joiner projection.
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
        if j.n == 0
            || j.n > j.state.max_step_frames()
            || j.feats.len() != (4 * j.n + 3) * a.n_mels
            || (a.mid > 0) != j.state.gmid.is_some()
        {
            return Err(EngineError::Shape(format!(
                "chunk job n={} with {} feature values",
                j.n,
                j.feats.len()
            )));
        }
    }
    let rows: usize = jobs.iter().map(|j| j.n).sum();
    stem::stem(m, isa, prec, jobs, ws, rows);
    let mut layer = 0;
    let mut emit = |tap: &mut Option<Tap<'_>>, v: &[f32], w: usize| {
        if let Some(t) = tap.as_mut() {
            t(layer, v, w);
        }
        layer += 1;
    };
    emit(&mut tap, &ws.x[..rows * d], d);
    // 25 Hz lower blocks (every block of a dense model).
    for li in 0..a.lower {
        let mut views: Vec<View<'_>> = jobs.iter_mut().map(|j| j.state.view25(j.n)).collect();
        block::block(m, isa, prec, &m.blocks[li], li, &mut views, ws, rows);
        emit(&mut tap, &ws.x[..rows * d], d);
    }
    if a.mid > 0 {
        let (pair, restore) = match (&m.pair, &m.restore) {
            (Some(p), Some(r)) => (p, r),
            _ => {
                return Err(EngineError::Unsupported(
                    "multirate model without pair/restore".into(),
                ))
            }
        };
        let skip = grow(&mut ws.skip, rows * d);
        skip.copy_from_slice(&ws.x[..rows * d]);
        // Pair positions (2i, 2i+1); an odd final row pairs with zeros (as in training).
        let mrows: usize = jobs.iter().map(|j| j.n.div_ceil(2)).sum();
        let pin = grow(&mut ws.pair_in, mrows * 2 * d);
        pin.fill(0.0);
        let (mut r25, mut rm) = (0, 0);
        for j in jobs.iter() {
            pin[rm * 2 * d..(rm * 2 * d + j.n * d)]
                .copy_from_slice(&ws.x[r25 * d..(r25 + j.n) * d]);
            r25 += j.n;
            rm += j.n.div_ceil(2);
        }
        let x = grow(&mut ws.x, mrows.max(rows) * d);
        linear(isa, prec, pair, pin, mrows, x, &mut ws.q);
        for li in 0..a.mid {
            let mut views: Vec<View<'_>> = jobs
                .iter_mut()
                .map(|j| {
                    let pos = j.state.t / 2;
                    let g = j.state.gmid.as_mut().expect("checked above");
                    g.view(pos, j.n.div_ceil(2))
                })
                .collect();
            block::block(
                m,
                isa,
                prec,
                &m.blocks[a.lower + li],
                li,
                &mut views,
                ws,
                mrows,
            );
            emit(&mut tap, &ws.x[..mrows * d], d);
        }
        // Restore two 25 Hz positions per middle position, drop the padding, add the skip.
        let rest = grow(&mut ws.pair_in, mrows * 2 * d);
        linear(
            isa,
            prec,
            restore,
            &ws.x[..mrows * d],
            mrows,
            rest,
            &mut ws.q,
        );
        let (mut r25, mut rm) = (0, 0);
        for j in jobs.iter() {
            let dst = &mut ws.x[r25 * d..(r25 + j.n) * d];
            dst.copy_from_slice(&rest[rm * 2 * d..rm * 2 * d + j.n * d]);
            kernels::add_scaled(isa, dst, &ws.skip[r25 * d..(r25 + j.n) * d], 1.0);
            r25 += j.n;
            rm += j.n.div_ceil(2);
        }
        for li in 0..a.fin {
            let mut views: Vec<View<'_>> = jobs.iter_mut().map(|j| j.state.view25(j.n)).collect();
            let b = &m.blocks[a.lower + a.mid + li];
            block::block(m, isa, prec, b, a.lower + li, &mut views, ws, rows);
            emit(&mut tap, &ws.x[..rows * d], d);
        }
    }
    let x = &mut ws.x[..rows * d];
    if let Some(fnorm) = &m.final_norm {
        let ln = grow(&mut ws.ln, rows * d);
        norm(isa, fnorm, x, d, ln);
        x.copy_from_slice(ln);
    }
    emit(&mut tap, x, d);
    let o = grow(out, rows * a.joiner);
    linear(isa, prec, &m.enc_proj, x, rows, o, &mut ws.q);
    emit(&mut tap, o, a.joiner);
    if o.iter().any(|v| !v.is_finite()) {
        return Err(EngineError::Output("non-finite encoder projection".into()));
    }
    for j in jobs.iter_mut() {
        let st = &mut *j.state;
        st.t += j.n;
        st.kv_len = (st.kv_len + j.n).min(st.left);
        if let Some(g) = st.gmid.as_mut() {
            g.kv_len = (g.kv_len + j.n.div_ceil(2)).min(g.left);
        }
    }
    Ok(rows)
}
