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
    /// Beam search state (`Engine::with_beam(k)`, k > 1); `None` for greedy.
    pub beam: Option<crate::beam::BeamState>,
}

impl DecState {
    pub fn new(m: &Model) -> Self {
        Self {
            ctx: m.arch.ctx_init,
            dec: vec![0.0; m.arch.joiner],
            dec_ok: false,
            tokens: Vec::new(),
            forced: 0,
            beam: None,
        }
    }

    /// Reset predictor context (and collapse any active beam to its best hypothesis with fresh
    /// context) without discarding previously emitted tokens. Used by optional endpoint-and-reset.
    pub fn reset_context(&mut self, m: &Model) {
        self.ctx = m.arch.ctx_init;
        self.dec_ok = false;
        if let Some(b) = &mut self.beam {
            b.reset_context(m);
        }
    }
}

#[derive(Debug, Default)]
pub struct DecWorkspace {
    pub(crate) emb_in: Vec<f32>,
    pub(crate) dec_out: Vec<f32>,
    pub(crate) hid: Vec<f32>,
    pub(crate) logits: Vec<f32>,
    active: Vec<usize>,
    need: Vec<usize>,
    pub(crate) beam_offs: Vec<usize>,
    pub(crate) beam_active: Vec<usize>,
    pub(crate) beam_need: Vec<(usize, usize)>,
    pub(crate) q: QuantScratch,
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
    if let Some(pc) = &m.pred_cache {
        for &s in &ws.need {
            let [t0, t1] = states[s].ctx;
            let a = &pc.h0[t0 * jn..(t0 + 1) * jn];
            let b = &pc.h1[t1 * jn..(t1 + 1) * jn];
            for ((o, &av), &bv) in states[s].dec.iter_mut().zip(a).zip(b) {
                *o = av + bv;
            }
            states[s].dec_ok = true;
        }
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
    let (cert, int8_slack) = match prec {
        Precision::F32 => (m.blank_cert_f32.as_ref(), false),
        Precision::Int8 => (
            if m.joiner_out.q.is_some() {
                m.blank_cert_i8.as_ref()
            } else {
                m.blank_cert_f32.as_ref()
            },
            m.joiner_out.q.is_some(),
        ),
    };
    let offs: Vec<usize> = ns
        .iter()
        .scan(0usize, |acc, &n| {
            let o = *acc;
            *acc += n;
            Some(o)
        })
        .collect();
    let tmax = ns.iter().copied().max().unwrap_or(0);
    let t_all = ws.q.prof.start();
    let gemm0 = ws.q.prof.ns;
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
            let mut keep_gemm = 0;
            for r in 0..a_n {
                let s = ws.active[r];
                let e = &enc[(offs[s] + t) * jn..(offs[s] + t + 1) * jn];
                let row = &mut hid[keep_gemm * jn..(keep_gemm + 1) * jn];
                for ((h, &ev), &dv) in row.iter_mut().zip(e).zip(&states[s].dec) {
                    *h = (ev + dv).tanh();
                }
                // Exact Cauchy-Schwarz blank-first certificate (0.000% error):
                // if z_blank > upper_bound(z_{v != blank}), skip the V x J joiner GEMM.
                if cert.is_some_and(|c| c.is_certified_blank(row, int8_slack)) {
                    continue;
                }
                ws.active[keep_gemm] = s;
                keep_gemm += 1;
            }
            ws.active.truncate(keep_gemm);
            if keep_gemm == 0 {
                break;
            }
            let logits = grow(&mut ws.logits, keep_gemm * v);
            linear(
                isa,
                prec,
                &m.joiner_out,
                &ws.hid[..keep_gemm * jn],
                keep_gemm,
                logits,
                &mut ws.q,
            );
            let mut keep = 0;
            for r in 0..keep_gemm {
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
    // Greedy bookkeeping (tanh, argmax, predictor gathers) excluding the GEMM time inside.
    if let Some(t) = t_all {
        let inner: u64 = ws.q.prof.ns.iter().zip(&gemm0).map(|(a, b)| a - b).sum();
        let total = u64::try_from(t.elapsed().as_nanos()).unwrap_or(u64::MAX);
        ws.q.prof.ns[crate::prof::Phase::Greedy as usize] += total.saturating_sub(inner);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::model::BlankCert;

    #[test]
    fn blank_cert_is_sound_and_skips_confident_blank() {
        let (v, jn, blank) = (8, 16, 0);
        let mut w = vec![0.0f32; v * jn];
        let mut b = vec![-2.0f32; v];
        b[blank] = 4.0;
        for row in 0..v {
            for k in 0..jn {
                w[row * jn + k] = if row == blank {
                    0.3
                } else {
                    ((row * 7 + k * 3) % 11) as f32 * 0.02 - 0.1
                };
            }
        }
        let cert = BlankCert::from_row_major(&w, &b, v, jn, blank).expect("valid cert");
        // Confident blank vector must be certified.
        let hid_blank = vec![0.5f32; jn];
        assert!(cert.is_certified_blank(&hid_blank, false));
        assert!(cert.is_certified_blank(&hid_blank, true));
        // Adversarial non-blank vector: raise bias of token 3 above z_blank (= 6.4) so argmax != blank.
        let mut b2 = b.clone();
        b2[3] = 7.0;
        let cert2 = BlankCert::from_row_major(&w, &b2, v, jn, blank).expect("valid cert");
        assert!(!cert2.is_certified_blank(&hid_blank, false));
        assert!(!cert2.is_certified_blank(&hid_blank, true));
        // NaN must never be certified blank.
        let mut hid_nan = hid_blank.clone();
        hid_nan[4] = f32::NAN;
        assert!(!cert.is_certified_blank(&hid_nan, false));
    }
}
