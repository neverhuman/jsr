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

use std::sync::Arc;

use crate::decoder::DecWorkspace;
use crate::encoder::{linear, Precision};
use crate::kernels::Isa;
use crate::model::Model;
use crate::EngineError;

/// One hypothesis: its token sequence, f32 log-probability, and cached predictor output for
/// its context (valid when `dec_ok`). `dec` is `Arc`-shared so blank continuations avoid
/// copying `joiner` floats on every frame.
#[derive(Debug, Clone)]
pub struct Hyp {
    pub tokens: Vec<u32>,
    /// Index in `tokens` where the current predictor/LM context begins (advanced on
    /// [`BeamState::reset_context`] so previously emitted tokens remain preserved while
    /// predictor and LM context reset to the configured start tokens and BOS).
    pub ctx_start: usize,
    pub score: f32,
    dec: Arc<Vec<f32>>,
    dec_ok: bool,
    lm_ctx: Option<crate::lm::LmCtx>,
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
                ctx_start: 0,
                score: 0.0,
                dec: Arc::new(vec![0.0; m.arch.joiner]),
                dec_ok: false,
                lm_ctx: None,
            }],
        }
    }

    /// The best hypothesis (first of equal maxima in insertion order).
    pub(crate) fn best_hyp(&self) -> &Hyp {
        let mut b = 0;
        for (i, h) in self.hyps.iter().enumerate() {
            if h.score > self.hyps[b].score {
                b = i;
            }
        }
        &self.hyps[b]
    }

    /// The best hypothesis' tokens (first of equal maxima in insertion order).
    pub fn best(&self) -> &[u32] {
        &self.best_hyp().tokens
    }

    pub(crate) fn shared_prefix(&self) -> &[u32] {
        let first = &self.hyps[0].tokens;
        let common = self.hyps[1..].iter().fold(first.len(), |len, hyp| {
            first[..len]
                .iter()
                .zip(&hyp.tokens)
                .take_while(|(a, b)| a == b)
                .count()
        });
        &first[..common]
    }

    /// Collapse the beam to a single hypothesis at the start of a fresh utterance segment,
    /// preserving previously emitted tokens while resetting predictor and LM context cleanly.
    pub fn reset_context(&mut self, m: &Model) {
        let best_tokens = self.best().to_vec();
        let ctx_start = best_tokens.len();
        self.hyps.clear();
        self.hyps.push(Hyp {
            tokens: best_tokens,
            ctx_start,
            score: 0.0,
            dec: Arc::new(vec![0.0; m.arch.joiner]),
            dec_ok: false,
            lm_ctx: None,
        });
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

/// Refresh predictor embeddings for hypotheses whose context changed.
fn refresh_beam_pred(m: &Model, isa: Isa, states: &mut [&mut BeamState], ws: &mut DecWorkspace) {
    if ws.beam_need.is_empty() {
        return;
    }
    let (jn, e) = (m.arch.joiner, m.arch.pred_emb);
    let initial = &m.arch.ctx_init;
    if let Some(pc) = &m.pred_cache {
        for &(s, h) in &ws.beam_need {
            let hy = &mut states[s].hyps[h];
            let mut tokens = context(&hy.tokens[hy.ctx_start..], initial);
            let t0 = tokens.next().expect("first cached context position");
            let t1 = tokens.next().expect("second cached context position");
            let a = &pc.h0[t0 * jn..(t0 + 1) * jn];
            let b = &pc.h1[t1 * jn..(t1 + 1) * jn];
            let dec = Arc::make_mut(&mut hy.dec);
            for ((o, &av), &bv) in dec.iter_mut().zip(a).zip(b) {
                *o = av + bv;
            }
            hy.dec_ok = true;
        }
        return;
    }
    let need_len = ws.beam_need.len();
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
        Arc::make_mut(&mut hy.dec).copy_from_slice(&ws.dec_out[r * jn..(r + 1) * jn]);
        hy.dec_ok = true;
    }
}

/// Insert `h` into `dst`, merging equal token sequences via `logaddexp`.
fn merge_hyp(dst: &mut Vec<Hyp>, hy: Hyp) {
    if let Some(x) = dst.iter_mut().find(|x| x.tokens == hy.tokens) {
        x.score = logaddexp(x.score, hy.score);
    } else {
        dst.push(hy);
    }
}

/// Beam search over one chunk (default 1 token/frame B6 parity, exact always-on beam).
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
    beam_chunk_with_opts(m, isa, prec, beam, 1, 0.0, states, ns, enc, ws)
}

/// Beam search over one chunk with configurable `max_symbols` per frame (`1..=8`) and
/// optional Entropy-Gated Micro-Beam-4 threshold (`beam_margin > 0.0`).
#[allow(clippy::too_many_arguments)]
pub fn beam_chunk_with_opts(
    m: &Model,
    isa: Isa,
    prec: Precision,
    beam: usize,
    max_symbols: usize,
    beam_margin: f32,
    states: &mut [&mut BeamState],
    ns: &[usize],
    enc: &[f32],
    ws: &mut DecWorkspace,
) -> Result<(), EngineError> {
    beam_chunk_with_lm(
        m,
        isa,
        prec,
        beam,
        max_symbols,
        beam_margin,
        None,
        states,
        ns,
        enc,
        ws,
    )
}

/// Insert candidate `c` into sorted top-`beam` `cands` using deterministic tie-breaking.
#[inline]
fn push_cand(cands: &mut Vec<Cand>, beam: usize, c: Cand) {
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

/// Beam search over one chunk with optional pure-Rust token n-gram shallow fusion + LODR
/// (`crate::lm::NgramLm`) and exact branch-and-bound pruning.
#[allow(clippy::too_many_arguments)]
pub fn beam_chunk_with_lm(
    m: &Model,
    isa: Isa,
    prec: Precision,
    beam: usize,
    max_symbols: usize,
    beam_margin: f32,
    lm: Option<(&crate::lm::NgramLm, crate::lm::LmConfig)>,
    states: &mut [&mut BeamState],
    ns: &[usize],
    enc: &[f32],
    ws: &mut DecWorkspace,
) -> Result<(), EngineError> {
    let (jn, v, blank) = (m.arch.joiner, m.arch.vocab, m.arch.blank);
    let max_sym = max_symbols.max(1);
    let use_margin = beam_margin.is_finite() && beam_margin > 0.0;
    let blank_pen = lm.map_or(0.0, |(_, c)| c.blank_penalty);
    ws.beam_offs.clear();
    let mut acc = 0usize;
    for &n in ns {
        ws.beam_offs.push(acc);
        acc += n;
    }
    let tmax = ns.iter().copied().max().unwrap_or(0);
    let t_all = ws.q.prof.start();
    let gemm0 = ws.q.prof.ns;
    let mut cands: Vec<Cand> = Vec::with_capacity(beam);
    // When `max_sym > 1`, `done[s]` accumulates hypotheses that already emitted blank on frame `t`.
    let mut done: Vec<Vec<Hyp>> = if max_sym > 1 {
        vec![Vec::new(); states.len()]
    } else {
        Vec::new()
    };
    let mut gated_only: Vec<bool> = Vec::with_capacity(states.len());
    for t in 0..tmax {
        if max_sym > 1 {
            for d in &mut done {
                d.clear();
            }
        }
        ws.beam_active.clear();
        for (s, &n) in ns.iter().enumerate().take(states.len()) {
            if t < n {
                ws.beam_active.push(s);
            }
        }
        for round in 0..max_sym {
            if ws.beam_active.is_empty() {
                break;
            }
            // Refresh cached LM contexts for any hypothesis that emitted a token.
            if let Some((ngram, _)) = lm {
                for &s in &ws.beam_active {
                    for hy in &mut states[s].hyps {
                        if hy.lm_ctx.is_none() {
                            hy.lm_ctx = Some(ngram.ctx_state(&hy.tokens[hy.ctx_start..]));
                        }
                    }
                }
            }
            // Entropy-Gated Micro-Beam-4: on round 0, if hyps[0] dominates runner-up by > beam_margin,
            // evaluate ONLY hyps[0] first; if blank dominates non-blank by > beam_margin, collapse to 1.
            gated_only.clear();
            ws.beam_need.clear();
            for &s in &ws.beam_active {
                let st = &states[s];
                let gate = use_margin
                    && round == 0
                    && st.hyps.len() > 1
                    && (st.hyps[0].score - st.hyps[1].score) > beam_margin;
                gated_only.push(gate);
                let lim = if gate { 1 } else { st.hyps.len() };
                for h in 0..lim {
                    if !st.hyps[h].dec_ok {
                        ws.beam_need.push((s, h));
                    }
                }
            }
            refresh_beam_pred(m, isa, states, ws);

            // Evaluate joiner rows (1 row for gated streams, all `hyps.len()` for others).
            let rows: usize = ws
                .beam_active
                .iter()
                .zip(&gated_only)
                .map(|(&s, &g)| if g { 1 } else { states[s].hyps.len() })
                .sum();
            let hid = grow(&mut ws.hid, rows * jn);
            let mut r = 0;
            for (&s, &g) in ws.beam_active.iter().zip(&gated_only) {
                let ev = &enc[(ws.beam_offs[s] + t) * jn..(ws.beam_offs[s] + t + 1) * jn];
                let lim = if g { 1 } else { states[s].hyps.len() };
                for hy in &states[s].hyps[..lim] {
                    for ((o, &a), &b) in hid[r * jn..(r + 1) * jn]
                        .iter_mut()
                        .zip(ev)
                        .zip(hy.dec.iter())
                    {
                        *o = (a + b).tanh();
                    }
                    r += 1;
                }
            }
            let logits = grow(&mut ws.logits, rows * v);
            linear(
                isa,
                prec,
                &m.joiner_out,
                &ws.hid[..rows * jn],
                rows,
                logits,
                &mut ws.q,
            );

            // If any gated stream's top hyp was NOT a confident blank, re-run its remaining hyps.
            let mut fallback: Vec<usize> = Vec::new();
            let mut r0 = 0;
            for (idx, (&s, &g)) in ws.beam_active.iter().zip(&gated_only).enumerate() {
                if g {
                    let row = &ws.logits[r0 * v..(r0 + 1) * v];
                    let zb = row[blank] - blank_pen;
                    let mut best_nb = f32::NEG_INFINITY;
                    if let Some((ngram, cfg)) = lm {
                        let ctx0 = states[s].hyps[0].lm_ctx.expect("lm_ctx refreshed");
                        let lm_ub = cfg.lm_weight * ctx0.max_logp_ub;
                        for (k, &x) in row.iter().enumerate() {
                            if k != blank {
                                let eff = x - cfg.lodr_weight * ngram.lodr_logp(&ctx0, k) + lm_ub;
                                if eff > best_nb {
                                    best_nb = eff;
                                }
                            }
                        }
                    } else {
                        for (k, &x) in row.iter().enumerate() {
                            if k != blank && x > best_nb {
                                best_nb = x;
                            }
                        }
                    }
                    if !(zb.is_finite() && best_nb.is_finite() && (zb - best_nb) > beam_margin) {
                        fallback.push(idx);
                    }
                    r0 += 1;
                } else {
                    r0 += states[s].hyps.len();
                }
            }
            // Fast path: if any gated stream fell back, evaluate all hyps for fallback streams cleanly.
            let mut fb_logits: Vec<Vec<f32>> = Vec::new();
            if !fallback.is_empty() {
                ws.beam_need.clear();
                for &idx in &fallback {
                    let s = ws.beam_active[idx];
                    for h in 1..states[s].hyps.len() {
                        if !states[s].hyps[h].dec_ok {
                            ws.beam_need.push((s, h));
                        }
                    }
                }
                refresh_beam_pred(m, isa, states, ws);
                let mut fb_rows = 0usize;
                for &idx in &fallback {
                    fb_rows += states[ws.beam_active[idx]].hyps.len() - 1;
                }
                let fb_hid = grow(&mut ws.hid, fb_rows * jn);
                let mut fr = 0;
                for &idx in &fallback {
                    let s = ws.beam_active[idx];
                    let ev = &enc[(ws.beam_offs[s] + t) * jn..(ws.beam_offs[s] + t + 1) * jn];
                    for hy in &states[s].hyps[1..] {
                        for ((o, &a), &b) in fb_hid[fr * jn..(fr + 1) * jn]
                            .iter_mut()
                            .zip(ev)
                            .zip(hy.dec.iter())
                        {
                            *o = (a + b).tanh();
                        }
                        fr += 1;
                    }
                }
                let mut tmp_logits = vec![0.0f32; fb_rows * v];
                linear(
                    isa,
                    prec,
                    &m.joiner_out,
                    &ws.hid[..fb_rows * jn],
                    fb_rows,
                    &mut tmp_logits,
                    &mut ws.q,
                );
                let mut off = 0;
                for &idx in &fallback {
                    let rem = states[ws.beam_active[idx]].hyps.len() - 1;
                    fb_logits.push(tmp_logits[off * v..(off + rem) * v].to_vec());
                    off += rem;
                }
            }

            let mut fb_pos = 0usize;
            let mut next_active: Vec<usize> = Vec::new();
            let mut r0 = 0;
            for (idx, (&s, &g)) in ws.beam_active.iter().zip(&gated_only).enumerate() {
                let st = &mut *states[s];
                let is_fb = fb_pos < fallback.len() && fallback[fb_pos] == idx;
                if g && !is_fb {
                    // Confident blank: collapse to width-1 Micro-Beam with exact log-softmax score.
                    let row = &ws.logits[r0 * v..(r0 + 1) * v];
                    let lse = logsumexp(isa, row)?;
                    let score = st.hyps[0].score + (row[blank] - lse) - blank_pen;
                    st.hyps.truncate(1);
                    st.hyps[0].score = score;
                    if max_sym > 1 {
                        let hy = st.hyps.remove(0);
                        merge_hyp(&mut done[s], hy);
                    }
                    r0 += 1;
                    continue;
                }
                let nh = st.hyps.len();
                let extra = if is_fb {
                    let e = &fb_logits[fb_pos];
                    fb_pos += 1;
                    Some(e.as_slice())
                } else {
                    None
                };
                // Top `beam` candidates over (hypothesis, token).
                cands.clear();
                for h in 0..nh {
                    let row = if let Some(ex) = extra {
                        if h == 0 {
                            &ws.logits[r0 * v..(r0 + 1) * v]
                        } else {
                            &ex[(h - 1) * v..h * v]
                        }
                    } else {
                        &ws.logits[(r0 + h) * v..(r0 + h + 1) * v]
                    };
                    let lse = logsumexp(isa, row)?;
                    let base = st.hyps[h].score;
                    if let Some((ngram, cfg)) = lm {
                        // 1. Score blank first (no LM/LODR term) to seed tight beam cutoff.
                        let b_score = base + (row[blank] - lse) - cfg.blank_penalty;
                        if cands.len() < beam || b_score >= cands[beam - 1].score {
                            push_cand(
                                &mut cands,
                                beam,
                                Cand {
                                    score: b_score,
                                    h,
                                    k: blank,
                                },
                            );
                        }
                        // 2. Non-blank tokens with exact branch-and-bound pruning before 5-gram lookup.
                        let ctx = st.hyps[h].lm_ctx.expect("lm_ctx refreshed");
                        let lm_w = cfg.lm_weight;
                        let lodr_w = cfg.lodr_weight;
                        let lm_ub_term = lm_w * ctx.max_logp_ub;
                        for (k, &x) in row.iter().enumerate() {
                            if k == blank {
                                continue;
                            }
                            let mut ub = base + (x - lse);
                            if lodr_w > 0.0 {
                                ub -= lodr_w * ngram.lodr_logp(&ctx, k);
                            }
                            if cands.len() == beam && (ub + lm_ub_term) < cands[beam - 1].score {
                                continue;
                            }
                            let score = if lm_w > 0.0 {
                                ub + lm_w * ngram.lm_logp(&ctx, k)
                            } else {
                                ub
                            };
                            if cands.len() == beam && score < cands[beam - 1].score {
                                continue;
                            }
                            push_cand(&mut cands, beam, Cand { score, h, k });
                        }
                    } else {
                        for (k, &x) in row.iter().enumerate() {
                            let score = base + (x - lse);
                            if cands.len() == beam && score < cands[beam - 1].score {
                                continue;
                            }
                            push_cand(&mut cands, beam, Cand { score, h, k });
                        }
                    }
                }
                r0 += if g { 1 } else { nh };

                if max_sym == 1 {
                    // Fast path (~85% of speech frames): distinct blank continuations in-place.
                    if cands.len() == nh
                        && cands.iter().all(|c| c.k == blank)
                        && (0..cands.len()).all(|i| cands[..i].iter().all(|p| p.h != cands[i].h))
                    {
                        let mut old: Vec<Option<Hyp>> = st.hyps.drain(..).map(Some).collect();
                        for c in &cands {
                            let mut hy = old[c.h].take().expect("distinct blank parent");
                            hy.score = c.score;
                            st.hyps.push(hy);
                        }
                        continue;
                    }
                    let mut next: Vec<Hyp> = Vec::with_capacity(beam);
                    for c in &cands {
                        let parent = &st.hyps[c.h];
                        let (tokens, dec_ok, lm_ctx) = if c.k == blank {
                            (parent.tokens.clone(), parent.dec_ok, parent.lm_ctx)
                        } else {
                            let mut tk = parent.tokens.clone();
                            tk.push(c.k as u32);
                            (tk, false, None)
                        };
                        merge_hyp(
                            &mut next,
                            Hyp {
                                tokens,
                                ctx_start: parent.ctx_start,
                                score: c.score,
                                dec: Arc::clone(&parent.dec),
                                dec_ok,
                                lm_ctx,
                            },
                        );
                    }
                    st.hyps = next;
                } else {
                    // Multi-symbol beam: blanks go to `done[s]`; non-blanks stay in `expand` for next round.
                    let mut expand: Vec<Hyp> = Vec::with_capacity(beam);
                    for c in &cands {
                        let parent = &st.hyps[c.h];
                        if c.k == blank {
                            merge_hyp(
                                &mut done[s],
                                Hyp {
                                    tokens: parent.tokens.clone(),
                                    ctx_start: parent.ctx_start,
                                    score: c.score,
                                    dec: Arc::clone(&parent.dec),
                                    dec_ok: parent.dec_ok,
                                    lm_ctx: parent.lm_ctx,
                                },
                            );
                        } else {
                            let mut tk = parent.tokens.clone();
                            tk.push(c.k as u32);
                            merge_hyp(
                                &mut expand,
                                Hyp {
                                    tokens: tk,
                                    ctx_start: parent.ctx_start,
                                    score: c.score,
                                    dec: Arc::clone(&parent.dec),
                                    dec_ok: false,
                                    lm_ctx: None,
                                },
                            );
                        }
                    }
                    st.hyps = expand;
                    if !st.hyps.is_empty() && round + 1 < max_sym {
                        next_active.push(s);
                    }
                }
            }
            if max_sym > 1 {
                ws.beam_active.clear();
                ws.beam_active.extend(next_active);
            }
        }
        if max_sym > 1 {
            for s in 0..states.len() {
                if t >= ns[s] {
                    continue;
                }
                let st = &mut *states[s];
                let mut pool = std::mem::take(&mut done[s]);
                for hy in st.hyps.drain(..) {
                    merge_hyp(&mut pool, hy);
                }
                pool.sort_by(|a, b| b.score.total_cmp(&a.score));
                pool.truncate(beam);
                st.hyps = pool;
            }
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
    fn predictor_uses_the_configured_initial_context_until_displaced() {
        let initial = [5, 6, 7];
        assert!(context(&[], &initial).eq([5, 6, 7]));
        assert!(context(&[8], &initial).eq([6, 7, 8]));
        assert!(context(&[8, 9, 10, 11], &initial).eq([9, 10, 11]));
    }

    #[test]
    fn logaddexp_and_log_softmax_basics() {
        assert!((logaddexp(0.0, 0.0) - std::f32::consts::LN_2).abs() < 1e-7);
        assert_eq!(logaddexp(-1e30, 3.0), 3.0);
        let r = [1.0f32, 2.0, 3.0];
        let lse = logsumexp(Isa::Scalar, &r).expect("finite");
        let s: f32 = r.iter().map(|v| (v - lse).exp()).sum();
        assert!((s - 1.0).abs() < 1e-6);
        assert!(logsumexp(Isa::Scalar, &[f32::NAN]).is_err());
        assert!(context(&[], &[0, 0]).eq([0, 0]));
        assert!(context(&[7], &[0, 0]).eq([0, 7]));
        assert!(context(&[5, 7, 9], &[0, 0]).eq([7, 9]));
    }

    #[test]
    fn context_respects_ctx_start_after_reset() {
        let tokens = vec![12u32, 34, 56];
        assert!(context(&tokens[0..], &[0, 0]).eq([34, 56]));
        assert!(context(&tokens[3..], &[0, 0]).eq([0, 0]));
        let mut extended = tokens.clone();
        extended.push(78);
        assert!(context(&extended[3..], &[0, 0]).eq([0, 78]));
        extended.push(90);
        assert!(context(&extended[3..], &[0, 0]).eq([78, 90]));
    }

    #[test]
    fn shared_prefix_stops_at_first_divergence_across_beam_hyps() {
        let mk = |tokens: Vec<u32>| Hyp {
            tokens,
            ctx_start: 0,
            score: 0.0,
            dec: Arc::new(Vec::new()),
            dec_ok: false,
            lm_ctx: None,
        };
        let single = BeamState {
            hyps: vec![mk(vec![1, 2, 3])],
        };
        assert_eq!(single.shared_prefix(), &[1, 2, 3]);

        let partial = BeamState {
            hyps: vec![mk(vec![1, 2, 3]), mk(vec![1, 2, 4, 5])],
        };
        assert_eq!(partial.shared_prefix(), &[1, 2]);

        let empty = BeamState {
            hyps: vec![mk(vec![1, 2, 3]), mk(vec![1, 2, 4]), mk(vec![9, 2, 3])],
        };
        assert_eq!(empty.shared_prefix(), &[] as &[u32]);
    }
}
