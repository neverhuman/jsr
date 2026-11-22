//! Bounded frame-synchronous beam search (`python/ultra/train/beam.py::modified_beam_search`,
//! icefall's modified beam search), batched across streams.
//!
//! Per encoder frame, for every hypothesis `s` of a stream: context = `(s[-2], s[-1])` with
//! BLANK where the sequence is shorter, `logp = log_softmax(joiner(enc_t, pred(ctx)))` in f32,
//! candidate score = `score(s) + logp[k]`. The `beam` best candidates over all hypotheses
//! and the whole vocabulary are kept: blank keeps `s`, a token `k` gives `s + (k)`, and
//! candidates with equal token sequences are merged by f32 `logaddexp`. At most one token
//! per frame (no `max_symbols` loop). The answer is the highest-scoring hypothesis.
//!
//! Determinism: ties in the top-`beam` selection are broken by (score descending, then
//! hypothesis index, then token id ascending), i.e. the flat index of beam.py's `topk`
//! input; merges happen in that order; the final pick takes the first of equal maxima in
//! insertion order (Python's `max` over the dict). Batch invariance: each stream's
//! hypotheses are their own joiner rows, and the joiner GEMM quantises per row.

use crate::decoder::DecWorkspace;
use crate::encoder::{linear, Precision};
use crate::kernels::Isa;
use crate::model::Model;
use crate::EngineError;

/// One hypothesis: its token sequence, f32 log-probability, and cached predictor output for
/// its context (valid when `dec_ok`).
#[derive(Debug, Clone)]
pub struct Hyp {
    pub tokens: Vec<u32>,
    pub score: f32,
    dec: Vec<f32>,
    dec_ok: bool,
}

/// Per-stream beam state.
#[derive(Debug, Clone)]
pub struct BeamState {
    pub hyps: Vec<Hyp>,
}

impl BeamState {
    pub fn new(m: &Model) -> Self {
        Self {
            hyps: vec![Hyp {
                tokens: Vec::new(),
                score: 0.0,
                dec: vec![0.0; m.arch.joiner],
                dec_ok: false,
            }],
        }
    }

    /// The best hypothesis' tokens (first of equal maxima in insertion order).
    pub fn best(&self) -> &[u32] {
        let mut b = 0;
        for (i, h) in self.hyps.iter().enumerate() {
            if h.score > self.hyps[b].score {
                b = i;
            }
        }
        &self.hyps[b].tokens
    }
}

fn grow(v: &mut Vec<f32>, n: usize) -> &mut [f32] {
    if v.len() < n {
        v.resize(n, 0.0);
    }
    &mut v[..n]
}

/// `logsumexp` of one row in f32: `max + ln(sum exp(x - max))`, the sum in the engine's
/// fixed cross-ISA order ([`kernels::sum_exp_shifted`]). Refuses NaN/Inf logits.
fn logsumexp(isa: Isa, row: &[f32]) -> Result<f32, EngineError> {
    let mut mx = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if !v.is_finite() {
            return Err(EngineError::Output(format!(
                "non-finite joiner logit at {i}"
            )));
        }
        mx = mx.max(v);
    }
    Ok(mx + crate::kernels::sum_exp_shifted(isa, row, mx).ln())
}

/// torch's `logaddexp` for finite f32: `max + log1p(exp(-|a - b|))`.
fn logaddexp(a: f32, b: f32) -> f32 {
    let m = a.max(b);
    m + (-(a - b).abs()).exp().ln_1p()
}

/// The predictor context of a hypothesis, BLANK-padded as in beam.py.
fn context(tokens: &[u32], blank: usize) -> [usize; 2] {
    let n = tokens.len();
    [
        if n > 1 { tokens[n - 2] as usize } else { blank },
        if n > 0 { tokens[n - 1] as usize } else { blank },
    ]
}

/// A candidate: (score, hypothesis index, token).
#[derive(Clone, Copy)]
struct Cand {
    score: f32,
    h: usize,
    k: usize,
}

/// `a` before `b` in the deterministic top-`beam` order.
fn better(a: &Cand, b: &Cand) -> bool {
    a.score > b.score || (a.score == b.score && (a.h, a.k) < (b.h, b.k))
}

/// Beam search over one chunk: stream `i` owns `ns[i]` rows of `enc` (joiner width).
#[allow(clippy::too_many_arguments)]
pub fn beam_chunk(
    m: &Model,
    isa: Isa,
    prec: Precision,
    beam: usize,
    states: &mut [&mut BeamState],
    ns: &[usize],
    enc: &[f32],
    ws: &mut DecWorkspace,
) -> Result<(), EngineError> {
    let (jn, v, e, blank) = (m.arch.joiner, m.arch.vocab, m.arch.pred_emb, m.arch.blank);
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
    let mut cands: Vec<Cand> = Vec::with_capacity(beam);
    for t in 0..tmax {
        let active: Vec<usize> = (0..states.len()).filter(|&s| t < ns[s]).collect();
        // Predictor outputs for hypotheses whose context changed.
        let need: Vec<(usize, usize)> = active
            .iter()
            .flat_map(|&s| {
                let st = &states[s];
                (0..st.hyps.len())
                    .filter(move |&h| !st.hyps[h].dec_ok)
                    .map(move |h| (s, h))
            })
            .collect();
        if !need.is_empty() {
            let inp = grow(&mut ws.emb_in, need.len() * 2 * e);
            for (r, &(s, h)) in need.iter().enumerate() {
                for (half, tok) in context(&states[s].hyps[h].tokens, blank)
                    .into_iter()
                    .enumerate()
                {
                    inp[(2 * r + half) * e..(2 * r + half + 1) * e]
                        .copy_from_slice(&m.emb[tok * e..(tok + 1) * e]);
                }
            }
            let out = grow(&mut ws.dec_out, need.len() * jn);
            linear(
                isa,
                Precision::F32,
                &m.pred_proj,
                inp,
                need.len(),
                out,
                &mut ws.q,
            );
            for (r, &(s, h)) in need.iter().enumerate() {
                let hy = &mut states[s].hyps[h];
                hy.dec.copy_from_slice(&out[r * jn..(r + 1) * jn]);
                hy.dec_ok = true;
            }
        }
        // Joiner rows: every hypothesis of every active stream.
        let rows: usize = active.iter().map(|&s| states[s].hyps.len()).sum();
        let hid = grow(&mut ws.hid, rows * jn);
        let mut r = 0;
        for &s in &active {
            let ev = &enc[(offs[s] + t) * jn..(offs[s] + t + 1) * jn];
            for hy in &states[s].hyps {
                for ((o, &a), &b) in hid[r * jn..(r + 1) * jn].iter_mut().zip(ev).zip(&hy.dec) {
                    *o = (a + b).tanh();
                }
                r += 1;
            }
        }
        let logits = grow(&mut ws.logits, rows * v);
        linear(isa, prec, &m.joiner_out, hid, rows, logits, &mut ws.q);
        let mut r0 = 0;
        for &s in &active {
            let st = &mut *states[s];
            let nh = st.hyps.len();
            // Top `beam` candidates over (hypothesis, token).
            cands.clear();
            for h in 0..nh {
                let row = &logits[(r0 + h) * v..(r0 + h + 1) * v];
                let lse = logsumexp(isa, row)?;
                let base = st.hyps[h].score;
                for (k, &x) in row.iter().enumerate() {
                    // log_softmax value x - lse, then the hypothesis score (beam.py's order).
                    let score = base + (x - lse);
                    if cands.len() == beam && score < cands[beam - 1].score {
                        continue; // cannot enter the top `beam`
                    }
                    let c = Cand { score, h, k };
                    if cands.len() < beam {
                        cands.push(c);
                        let mut i = cands.len() - 1;
                        while i > 0 && better(&cands[i], &cands[i - 1]) {
                            cands.swap(i, i - 1);
                            i -= 1;
                        }
                    } else if better(&c, &cands[beam - 1]) {
                        cands[beam - 1] = c;
                        let mut i = beam - 1;
                        while i > 0 && better(&cands[i], &cands[i - 1]) {
                            cands.swap(i, i - 1);
                            i -= 1;
                        }
                    }
                }
            }
            r0 += nh;
            // New hypotheses in candidate order, merging equal sequences.
            let mut next: Vec<Hyp> = Vec::with_capacity(beam);
            for c in &cands {
                let parent = &st.hyps[c.h];
                let (tokens, dec_ok) = if c.k == blank {
                    (parent.tokens.clone(), parent.dec_ok)
                } else {
                    let mut tk = parent.tokens.clone();
                    tk.push(c.k as u32);
                    (tk, false)
                };
                if let Some(x) = next.iter_mut().find(|x| x.tokens == tokens) {
                    x.score = logaddexp(x.score, c.score);
                } else {
                    next.push(Hyp {
                        tokens,
                        score: c.score,
                        dec: parent.dec.clone(),
                        dec_ok,
                    });
                }
            }
            st.hyps = next;
        }
    }
    if let Some(t) = t_all {
        let inner: u64 = ws.q.prof.ns.iter().zip(&gemm0).map(|(a, b)| a - b).sum();
        let total = u64::try_from(t.elapsed().as_nanos()).unwrap_or(u64::MAX);
        ws.q.prof.ns[crate::prof::Phase::Greedy as usize] += total.saturating_sub(inner);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logaddexp_and_log_softmax_basics() {
        assert!((logaddexp(0.0, 0.0) - std::f32::consts::LN_2).abs() < 1e-7);
        assert_eq!(logaddexp(-1e30, 3.0), 3.0);
        let r = [1.0f32, 2.0, 3.0];
        let lse = logsumexp(Isa::Scalar, &r).expect("finite");
        let s: f32 = r.iter().map(|v| (v - lse).exp()).sum();
        assert!((s - 1.0).abs() < 1e-6);
        assert!(logsumexp(Isa::Scalar, &[f32::NAN]).is_err());
        assert_eq!(context(&[], 0), [0, 0]);
        assert_eq!(context(&[7], 0), [0, 7]);
        assert_eq!(context(&[5, 7, 9], 0), [7, 9]);
    }
}
