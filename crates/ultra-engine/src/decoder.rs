//! Stateless predictor, tanh joiner and frame-synchronous greedy RNN-T search, batched
//! across streams (`python/ultra/train/infer.py::StreamingRecognizer._greedy`).
//!
//! For every encoder frame each stream may emit up to `max_symbols` tokens: logits =
//! `W_out tanh(enc_t + pred(ctx)) + b`, k = first-index argmax; blank advances the frame,
//! otherwise k is emitted and `ctx = (ctx[1], k)`. A stream still emitting after
//! `max_symbols` rounds advances anyway and counts a forced advance. Streams are batched
//! only to share the weight pass; each stream's choices depend only on its own rows.

use crate::encoder::{linear, Precision};
use crate::kernels::Isa;
use crate::model::Model;
use crate::quant::QuantScratch;
use crate::EngineError;

/// Per-stream decoder state.
#[derive(Debug, Clone)]
pub struct DecState {
    pub ctx: [usize; 2],
    /// Cached predictor output for `ctx` (joiner width), valid when `dec_ok`.
    pub dec: Vec<f32>,
    pub dec_ok: bool,
    pub tokens: Vec<u32>,
    pub forced: u64,
}

impl DecState {
    pub fn new(m: &Model) -> Self {
        Self {
            ctx: m.arch.ctx_init,
            dec: vec![0.0; m.arch.joiner],
            dec_ok: false,
            tokens: Vec::new(),
            forced: 0,
        }
    }
}

#[derive(Debug, Default)]
pub struct DecWorkspace {
    emb_in: Vec<f32>,
    dec_out: Vec<f32>,
    hid: Vec<f32>,
    logits: Vec<f32>,
    active: Vec<usize>,
    need: Vec<usize>,
    q: QuantScratch,
}

fn grow(v: &mut Vec<f32>, n: usize) -> &mut [f32] {
    if v.len() < n {
        v.resize(n, 0.0);
    }
    &mut v[..n]
}

/// First-index argmax over finite logits (numpy tie-breaking); refuses NaN/Inf.
fn argmax(l: &[f32]) -> Result<usize, EngineError> {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, &v) in l.iter().enumerate() {
        if !v.is_finite() {
            return Err(EngineError::Output(format!(
                "non-finite joiner logit at {i}"
            )));
        }
        if v > best.1 {
            best = (i, v);
        }
    }
    Ok(best.0)
}

/// Recompute the cached predictor output of every stream in `ws.need`.
fn refresh_pred(m: &Model, isa: Isa, states: &mut [&mut DecState], ws: &mut DecWorkspace) {
    let (e, jn) = (m.arch.pred_emb, m.arch.joiner);
    let rows = ws.need.len();
    if rows == 0 {
        return;
    }
    let inp = grow(&mut ws.emb_in, rows * 2 * e);
    for (r, &s) in ws.need.iter().enumerate() {
        for (half, &tok) in states[s].ctx.iter().enumerate() {
            inp[(2 * r + half) * e..(2 * r + half + 1) * e]
                .copy_from_slice(&m.emb[tok * e..(tok + 1) * e]);
        }
    }
    let out = grow(&mut ws.dec_out, rows * jn);
    // The predictor is tiny and stays fp32 in every precision (as in the ONNX bundle).
    linear(isa, Precision::F32, &m.pred_proj, inp, rows, out, &mut ws.q);
    for (r, &s) in ws.need.iter().enumerate() {
        states[s].dec.copy_from_slice(&out[r * jn..(r + 1) * jn]);
        states[s].dec_ok = true;
    }
}

/// Greedy search over one chunk: stream `i` owns `ns[i]` rows of `enc` (joiner width) in
/// order. Appends emitted tokens to each state.
pub fn greedy_chunk(
    m: &Model,
    isa: Isa,
    prec: Precision,
    states: &mut [&mut DecState],
    ns: &[usize],
    enc: &[f32],
    ws: &mut DecWorkspace,
) -> Result<(), EngineError> {
    let jn = m.arch.joiner;
    let v = m.arch.vocab;
    let offs: Vec<usize> = ns
        .iter()
        .scan(0usize, |acc, &n| {
            let o = *acc;
            *acc += n;
            Some(o)
        })
        .collect();
    let tmax = ns.iter().copied().max().unwrap_or(0);
    for t in 0..tmax {
        ws.active.clear();
        ws.active.extend((0..states.len()).filter(|&s| t < ns[s]));
        let mut rounds = 0;
        while !ws.active.is_empty() && rounds < m.arch.max_symbols {
            rounds += 1;
            ws.need.clear();
            ws.need
                .extend(ws.active.iter().copied().filter(|&s| !states[s].dec_ok));
            refresh_pred(m, isa, states, ws);
            let a_n = ws.active.len();
            let hid = grow(&mut ws.hid, a_n * jn);
            for (r, &s) in ws.active.iter().enumerate() {
                let e = &enc[(offs[s] + t) * jn..(offs[s] + t + 1) * jn];
                for ((h, &ev), &dv) in hid[r * jn..(r + 1) * jn]
                    .iter_mut()
                    .zip(e)
                    .zip(&states[s].dec)
                {
                    *h = (ev + dv).tanh();
                }
            }
            let logits = grow(&mut ws.logits, a_n * v);
            linear(isa, prec, &m.joiner_out, hid, a_n, logits, &mut ws.q);
            let mut keep = 0;
            for r in 0..a_n {
                let s = ws.active[r];
                let k = argmax(&logits[r * v..(r + 1) * v])?;
                if k != m.arch.blank {
                    let st = &mut *states[s];
                    st.tokens.push(k as u32);
                    st.ctx = [st.ctx[1], k];
                    st.dec_ok = false;
                    ws.active[keep] = s;
                    keep += 1;
                }
            }
            ws.active.truncate(keep);
        }
        for &s in &ws.active {
            states[s].forced += 1;
        }
    }
    Ok(())
}
