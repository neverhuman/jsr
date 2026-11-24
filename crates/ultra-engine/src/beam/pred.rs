//! Predictor-context helpers and deduplicated predictor-projection refresh for bounded beam search.

use std::sync::Arc;

use super::Hyp;
use crate::decoder::DecWorkspace;
use crate::encoder::{linear, Precision};
use crate::kernels::Isa;
use crate::model::Model;

pub(super) fn grow(v: &mut Vec<f32>, n: usize) -> &mut [f32] {
    if v.len() < n {
        v.resize(n, 0.0);
    }
    &mut v[..n]
}

/// The predictor context of a hypothesis, padded with the configured initial context.
pub(crate) fn context<'a>(
    tokens: &'a [u32],
    initial: &'a [usize],
) -> impl Iterator<Item = usize> + 'a {
    let kept = tokens.len().min(initial.len());
    initial[kept..]
        .iter()
        .copied()
        .chain(tokens[tokens.len() - kept..].iter().map(|&id| id as usize))
}

#[inline]
pub(super) fn ctx2(hy: &Hyp, initial: &[usize]) -> (usize, usize) {
    let s = &hy.tokens[hy.ctx_start..];
    match s.len() {
        0 => (initial[0], initial[1]),
        1 => (initial[1], s[0] as usize),
        n => (s[n - 2] as usize, s[n - 1] as usize),
    }
}

/// Exact predictor-context equality: two hypotheses with equal predictor contexts always
/// produce bit-identical `dec`, `hid = tanh(enc_t + dec)`, `joiner_out` logits, and `lse`.
#[inline]
pub(super) fn same_pred_ctx(a: &Hyp, b: &Hyp, initial: &[usize]) -> bool {
    if initial.len() == 2 {
        ctx2(a, initial) == ctx2(b, initial)
    } else {
        context(&a.tokens[a.ctx_start..], initial).eq(context(&b.tokens[b.ctx_start..], initial))
    }
}

/// Obtain a mutable slice into `hy.dec` without copying shared parent data (caller overwrites all `jn` floats).
#[inline]
fn take_fresh_dec(hy: &mut Hyp, jn: usize) -> &mut [f32] {
    if Arc::strong_count(&hy.dec) > 1 {
        hy.dec = Arc::new(vec![0.0; jn]);
    }
    let v = Arc::get_mut(&mut hy.dec).expect("unique dec buffer");
    if v.len() != jn {
        v.resize(jn, 0.0);
    }
    v.as_mut_slice()
}

/// Refresh predictor embeddings for hypotheses whose context changed, reusing any already-valid
/// `dec` within the same stream that shares the exact same predictor context.
pub(super) fn refresh_beam_pred(
    m: &Model,
    isa: Isa,
    states: &mut [&mut super::BeamState],
    ws: &mut DecWorkspace,
) {
    if ws.beam_need.is_empty() {
        return;
    }
    let (jn, e) = (m.arch.joiner, m.arch.pred_emb);
    let initial = &m.arch.ctx_init;
    if let Some(pc) = &m.pred_cache {
        for &(s, h) in &ws.beam_need {
            let hyps = &mut states[s].hyps;
            if hyps[h].dec_ok {
                continue;
            }
            let (t0, t1) = ctx2(&hyps[h], initial);
            let mut reused = None;
            for other in hyps.iter() {
                if other.dec_ok && ctx2(other, initial) == (t0, t1) {
                    reused = Some(Arc::clone(&other.dec));
                    break;
                }
            }
            if let Some(dec) = reused {
                hyps[h].dec = dec;
                hyps[h].dec_ok = true;
                continue;
            }
            let a = &pc.h0[t0 * jn..(t0 + 1) * jn];
            let b = &pc.h1[t1 * jn..(t1 + 1) * jn];
            let dec = take_fresh_dec(&mut hyps[h], jn);
            for ((o, &av), &bv) in dec.iter_mut().zip(a).zip(b) {
                *o = av + bv;
            }
            hyps[h].dec_ok = true;
        }
        return;
    }
    // Non-cached predictor: reuse any already-valid matching context in the same stream first.
    let mut keep = 0usize;
    for i in 0..ws.beam_need.len() {
        let (s, h) = ws.beam_need[i];
        let hyps = &mut states[s].hyps;
        if hyps[h].dec_ok {
            continue;
        }
        let mut reused = None;
        for other in hyps.iter() {
            if other.dec_ok && same_pred_ctx(&hyps[h], other, initial) {
                reused = Some(Arc::clone(&other.dec));
                break;
            }
        }
        if let Some(dec) = reused {
            hyps[h].dec = dec;
            hyps[h].dec_ok = true;
        } else {
            ws.beam_need[keep] = (s, h);
            keep += 1;
        }
    }
    ws.beam_need.truncate(keep);
    let need_len = ws.beam_need.len();
    if need_len == 0 {
        return;
    }
    let inp = grow(&mut ws.emb_in, need_len * initial.len() * e);
    for (r, &(s, h)) in ws.beam_need.iter().enumerate() {
        let hy = &states[s].hyps[h];
        for (position, tok) in context(&hy.tokens[hy.ctx_start..], initial).enumerate() {
            let start = (initial.len() * r + position) * e;
            inp[start..start + e].copy_from_slice(&m.emb[tok * e..(tok + 1) * e]);
        }
    }
    let out = grow(&mut ws.dec_out, need_len * jn);
    linear(
        isa,
        Precision::F32,
        &m.pred_proj,
        &ws.emb_in[..need_len * initial.len() * e],
        need_len,
        out,
        &mut ws.q,
    );
    for (r, &(s, h)) in ws.beam_need.iter().enumerate() {
        let hy = &mut states[s].hyps[h];
        let dec = take_fresh_dec(hy, jn);
        dec.copy_from_slice(&ws.dec_out[r * jn..(r + 1) * jn]);
        hy.dec_ok = true;
    }
}
