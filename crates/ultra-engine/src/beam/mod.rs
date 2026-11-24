//! Bounded frame-synchronous beam search (`python/ultra/train/beam.py::modified_beam_search`,
//! icefall's modified beam search), batched across streams.
//!
//! Per encoder frame, for every hypothesis `s` of a stream: context = `(s[-2], s[-1])` with
//! BLANK where the sequence is shorter, `logp = log_softmax(joiner(enc_t, pred(ctx)))` in f32,
//! candidate score = `score(s) + logp[k]`. The `beam` best candidates over all hypotheses
//! and the whole vocabulary are kept: blank keeps `s`, a token `k` gives `s + (k)`, and
//! candidates with equal token sequences are merged by f32 `logaddexp`. At most one token
//! per frame (no `max_symbols` loop) by default, or up to `max_symbols` per frame when configured.
//! The answer is the highest-scoring hypothesis.
//!
//! Determinism: ties in the top-`beam` selection are broken by (score descending, then
//! hypothesis index, then token id ascending), i.e. the flat index of beam.py's `topk`
//! input; merges happen in that order; the final pick takes the first of equal maxima in
//! insertion order (Python's `max` over the dict). Batch invariance: each stream's
//! hypotheses are their own joiner rows, and the joiner GEMM quantises per row.

mod pred;
mod score;

use std::sync::Arc;

pub(crate) use pred::context;
use pred::{grow, refresh_beam_pred, same_pred_ctx};
use score::{
    is_confident_blank_margin, logsumexp_and_max_nb, merge_hyp, push_cand, score_nonblank_lm,
    update_hyps_all_blank_inplace,
};

use crate::decoder::DecWorkspace;
use crate::encoder::{linear, Precision};
use crate::kernels::Isa;
use crate::model::Model;
use crate::EngineError;

/// One hypothesis: its token sequence, f32 log-probability, and cached predictor output for
/// its context (valid when `dec_ok`). `dec` is `Arc`-shared so blank continuations and
/// hypotheses sharing the same predictor context avoid copying `joiner` floats.
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

/// A candidate: (score, hypothesis index, token).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Cand {
    pub(super) score: f32,
    pub(super) h: usize,
    pub(super) k: usize,
}

#[inline]
fn fill_joiner_hid(ev: &[f32], dec: &[f32], dst: &mut [f32]) {
    for ((o, &a), &b) in dst.iter_mut().zip(ev).zip(dec) {
        *o = (a + b).tanh();
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
    let initial = &m.arch.ctx_init;
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

    if max_sym > 1 {
        if ws.beam_done.len() < states.len() {
            ws.beam_done.resize_with(states.len(), Vec::new);
        }
        ws.beam_fast_blank.resize(states.len(), false);
    }

    for t in 0..tmax {
        if max_sym > 1 {
            for s in 0..states.len() {
                ws.beam_done[s].clear();
                ws.beam_fast_blank[s] = false;
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
            ws.beam_gated.clear();
            ws.beam_need.clear();
            for &s in &ws.beam_active {
                let st = &states[s];
                let gate = use_margin
                    && round == 0
                    && st.hyps.len() > 1
                    && (st.hyps[0].score - st.hyps[1].score) > beam_margin;
                ws.beam_gated.push(gate);
                let lim = if gate { 1 } else { st.hyps.len() };
                for h in 0..lim {
                    if !st.hyps[h].dec_ok {
                        ws.beam_need.push((s, h));
                    }
                }
            }
            refresh_beam_pred(m, isa, states, ws);

            // Deduplicate joiner rows within each stream by exact predictor context.
            ws.beam_stream_row_off.clear();
            ws.beam_row_map.clear();
            let mut unique_rows = 0usize;
            for (&s, &g) in ws.beam_active.iter().zip(&ws.beam_gated) {
                let map_base = ws.beam_row_map.len();
                ws.beam_stream_row_off.push(map_base);
                let hyps = &states[s].hyps;
                let lim = if g { 1 } else { hyps.len() };
                for h in 0..lim {
                    let mut dup = None;
                    for hp in 0..h {
                        if same_pred_ctx(&hyps[h], &hyps[hp], initial) {
                            dup = Some(ws.beam_row_map[map_base + hp]);
                            break;
                        }
                    }
                    if let Some(r_idx) = dup {
                        ws.beam_row_map.push(r_idx);
                    } else {
                        ws.beam_row_map.push(unique_rows);
                        unique_rows += 1;
                    }
                }
            }
            let hid = grow(&mut ws.hid, unique_rows * jn);
            let mut next_r = 0usize;
            for (idx, (&s, &g)) in ws.beam_active.iter().zip(&ws.beam_gated).enumerate() {
                let ev = &enc[(ws.beam_offs[s] + t) * jn..(ws.beam_offs[s] + t + 1) * jn];
                let map_base = ws.beam_stream_row_off[idx];
                let hyps = &states[s].hyps;
                let lim = if g { 1 } else { hyps.len() };
                for (h, hy) in hyps[..lim].iter().enumerate() {
                    if ws.beam_row_map[map_base + h] == next_r {
                        fill_joiner_hid(ev, &hy.dec, &mut hid[next_r * jn..(next_r + 1) * jn]);
                        next_r += 1;
                    }
                }
            }
            debug_assert_eq!(next_r, unique_rows);
            let logits = grow(&mut ws.logits, unique_rows * v);
            linear(
                isa,
                prec,
                &m.joiner_out,
                &ws.hid[..unique_rows * jn],
                unique_rows,
                logits,
                &mut ws.q,
            );

            ws.beam_lse.clear();
            ws.beam_max_nb.clear();
            for r in 0..unique_rows {
                let row = &ws.logits[r * v..(r + 1) * v];
                let (lse, max_nb) = logsumexp_and_max_nb(isa, row, blank)?;
                ws.beam_lse.push(lse);
                ws.beam_max_nb.push(max_nb);
            }

            // If any gated stream's top hyp was NOT a confident blank, re-run its remaining hyps.
            ws.beam_fallback.clear();
            for (idx, (&s, &g)) in ws.beam_active.iter().zip(&ws.beam_gated).enumerate() {
                if g {
                    let r0 = ws.beam_row_map[ws.beam_stream_row_off[idx]];
                    let row = &ws.logits[r0 * v..(r0 + 1) * v];
                    if !is_confident_blank_margin(
                        row,
                        ws.beam_max_nb[r0],
                        blank,
                        blank_pen,
                        beam_margin,
                        lm,
                        states[s].hyps[0].lm_ctx,
                    ) {
                        ws.beam_fallback.push(idx);
                    }
                }
            }

            // Evaluate fallback hyps with predictor-context dedup (including matching h == 0).
            ws.beam_fb_row_map.clear();
            ws.beam_fb_stream_off.clear();
            ws.beam_fb_lse.clear();
            ws.beam_fb_max_nb.clear();
            if !ws.beam_fallback.is_empty() {
                ws.beam_need.clear();
                for &idx in &ws.beam_fallback {
                    let s = ws.beam_active[idx];
                    for h in 1..states[s].hyps.len() {
                        if !states[s].hyps[h].dec_ok {
                            ws.beam_need.push((s, h));
                        }
                    }
                }
                refresh_beam_pred(m, isa, states, ws);

                let mut fb_unique = 0usize;
                for &idx in &ws.beam_fallback {
                    let s = ws.beam_active[idx];
                    let r0 = ws.beam_row_map[ws.beam_stream_row_off[idx]];
                    let hyps = &states[s].hyps;
                    let fb_base = ws.beam_fb_row_map.len();
                    ws.beam_fb_stream_off.push(fb_base);
                    for h in 1..hyps.len() {
                        if same_pred_ctx(&hyps[h], &hyps[0], initial) {
                            ws.beam_fb_row_map.push(-1 - (r0 as isize));
                            continue;
                        }
                        let mut dup = None;
                        for hp in 1..h {
                            if same_pred_ctx(&hyps[h], &hyps[hp], initial) {
                                dup = Some(ws.beam_fb_row_map[fb_base + (hp - 1)]);
                                break;
                            }
                        }
                        if let Some(m_idx) = dup {
                            ws.beam_fb_row_map.push(m_idx);
                        } else {
                            ws.beam_fb_row_map.push(fb_unique as isize);
                            fb_unique += 1;
                        }
                    }
                }
                if fb_unique > 0 {
                    let fb_hid = grow(&mut ws.hid, fb_unique * jn);
                    let mut next_fr = 0usize;
                    for (fb_i, &idx) in ws.beam_fallback.iter().enumerate() {
                        let s = ws.beam_active[idx];
                        let ev = &enc[(ws.beam_offs[s] + t) * jn..(ws.beam_offs[s] + t + 1) * jn];
                        let fb_base = ws.beam_fb_stream_off[fb_i];
                        for (off_h, hy) in states[s].hyps[1..].iter().enumerate() {
                            if ws.beam_fb_row_map[fb_base + off_h] == next_fr as isize {
                                fill_joiner_hid(
                                    ev,
                                    &hy.dec,
                                    &mut fb_hid[next_fr * jn..(next_fr + 1) * jn],
                                );
                                next_fr += 1;
                            }
                        }
                    }
                    debug_assert_eq!(next_fr, fb_unique);
                    let fb_logits = grow(&mut ws.beam_fb_logits, fb_unique * v);
                    linear(
                        isa,
                        prec,
                        &m.joiner_out,
                        &ws.hid[..fb_unique * jn],
                        fb_unique,
                        fb_logits,
                        &mut ws.q,
                    );
                    for r in 0..fb_unique {
                        let row = &ws.beam_fb_logits[r * v..(r + 1) * v];
                        let (lse, max_nb) = logsumexp_and_max_nb(isa, row, blank)?;
                        ws.beam_fb_lse.push(lse);
                        ws.beam_fb_max_nb.push(max_nb);
                    }
                }
            }

            let mut fb_pos = 0usize;
            ws.beam_next_active.clear();
            for idx in 0..ws.beam_active.len() {
                let s = ws.beam_active[idx];
                let g = ws.beam_gated[idx];
                let st = &mut *states[s];
                let is_fb = fb_pos < ws.beam_fallback.len() && ws.beam_fallback[fb_pos] == idx;
                let map_base = ws.beam_stream_row_off[idx];
                if g && !is_fb {
                    let r0 = ws.beam_row_map[map_base];
                    let row = &ws.logits[r0 * v..(r0 + 1) * v];
                    let score = st.hyps[0].score + (row[blank] - ws.beam_lse[r0]) - blank_pen;
                    st.hyps.truncate(1);
                    st.hyps[0].score = score;
                    if max_sym > 1 {
                        ws.beam_fast_blank[s] = true;
                    }
                    continue;
                }
                let nh = st.hyps.len();
                let fb_base = if is_fb {
                    let b = ws.beam_fb_stream_off[fb_pos];
                    fb_pos += 1;
                    Some(b)
                } else {
                    None
                };

                let row_ref = |h: usize| -> (bool, usize) {
                    if let Some(fb_b) = fb_base {
                        if h == 0 {
                            (true, ws.beam_row_map[map_base])
                        } else {
                            let code = ws.beam_fb_row_map[fb_b + (h - 1)];
                            if code < 0 {
                                (true, (-1 - code) as usize)
                            } else {
                                (false, code as usize)
                            }
                        }
                    } else {
                        (true, ws.beam_row_map[map_base + h])
                    }
                };

                // Step A: Score ALL blanks across `h in 0..nh` first to seed a tight beam cutoff.
                ws.beam_cands.clear();
                for h in 0..nh {
                    let (is_pri, r_idx) = row_ref(h);
                    let (row, lse) = if is_pri {
                        (&ws.logits[r_idx * v..(r_idx + 1) * v], ws.beam_lse[r_idx])
                    } else {
                        (
                            &ws.beam_fb_logits[r_idx * v..(r_idx + 1) * v],
                            ws.beam_fb_lse[r_idx],
                        )
                    };
                    let b_score = st.hyps[h].score + (row[blank] - lse) - blank_pen;
                    if ws.beam_cands.len() < beam || b_score >= ws.beam_cands[beam - 1].score {
                        push_cand(
                            &mut ws.beam_cands,
                            beam,
                            Cand {
                                score: b_score,
                                h,
                                k: blank,
                            },
                        );
                    }
                }

                // Step B: Score non-blanks for each `h in 0..nh` using exact branch-and-bound.
                for h in 0..nh {
                    let (is_pri, r_idx) = row_ref(h);
                    let (row, lse, max_nb) = if is_pri {
                        (
                            &ws.logits[r_idx * v..(r_idx + 1) * v],
                            ws.beam_lse[r_idx],
                            ws.beam_max_nb[r_idx],
                        )
                    } else {
                        (
                            &ws.beam_fb_logits[r_idx * v..(r_idx + 1) * v],
                            ws.beam_fb_lse[r_idx],
                            ws.beam_fb_max_nb[r_idx],
                        )
                    };
                    let base = st.hyps[h].score;
                    if let Some((ngram, cfg)) = lm {
                        let ctx = st.hyps[h].lm_ctx.expect("lm_ctx refreshed");
                        score_nonblank_lm(
                            h,
                            row,
                            lse,
                            max_nb,
                            base,
                            blank,
                            beam,
                            ngram,
                            cfg,
                            &ctx,
                            &mut ws.beam_cands,
                            &mut ws.beam_shortlist,
                            &mut ws.beam_ub_buf,
                        );
                    } else {
                        let slack = 1e-3f32.max(base.abs() * 1e-5);
                        if ws.beam_cands.len() == beam
                            && base + (max_nb - lse) < ws.beam_cands[beam - 1].score - slack
                        {
                            continue;
                        }
                        for (k, &x) in row.iter().enumerate() {
                            if k == blank {
                                continue;
                            }
                            let score = base + (x - lse);
                            if ws.beam_cands.len() == beam && score < ws.beam_cands[beam - 1].score
                            {
                                continue;
                            }
                            push_cand(&mut ws.beam_cands, beam, Cand { score, h, k });
                        }
                    }
                }

                let all_blank_distinct = ws.beam_cands.len() == nh
                    && ws.beam_cands.iter().all(|c| c.k == blank)
                    && (0..ws.beam_cands.len())
                        .all(|i| ws.beam_cands[..i].iter().all(|p| p.h != ws.beam_cands[i].h));

                if max_sym == 1 {
                    if all_blank_distinct {
                        update_hyps_all_blank_inplace(
                            &mut st.hyps,
                            &ws.beam_cands,
                            &mut ws.beam_old_hyps,
                        );
                        continue;
                    }
                    expand_hyps_single(st, blank, nh, ws);
                } else {
                    if round == 0 && all_blank_distinct {
                        update_hyps_all_blank_inplace(
                            &mut st.hyps,
                            &ws.beam_cands,
                            &mut ws.beam_old_hyps,
                        );
                        ws.beam_fast_blank[s] = true;
                        continue;
                    }
                    expand_hyps_multi(st, s, blank, nh, ws);
                    if !st.hyps.is_empty() && round + 1 < max_sym {
                        ws.beam_next_active.push(s);
                    }
                }
            }
            if max_sym > 1 {
                std::mem::swap(&mut ws.beam_active, &mut ws.beam_next_active);
            }
        }
        if max_sym > 1 {
            for s in 0..states.len() {
                if t >= ns[s] || ws.beam_fast_blank[s] {
                    continue;
                }
                let st = &mut *states[s];
                for hy in st.hyps.drain(..) {
                    merge_hyp(&mut ws.beam_done[s], hy);
                }
                ws.beam_done[s].sort_by(|a, b| b.score.total_cmp(&a.score));
                ws.beam_done[s].truncate(beam);
                std::mem::swap(&mut st.hyps, &mut ws.beam_done[s]);
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

/// Apply `ws.beam_cands` into `st.hyps` for `max_sym == 1`, moving the parent's `tokens` vector
/// on its final consumer instead of cloning.
fn expand_hyps_single(st: &mut BeamState, blank: usize, nh: usize, ws: &mut DecWorkspace) {
    ws.beam_refcounts.clear();
    ws.beam_refcounts.resize(nh, 0);
    for c in &ws.beam_cands {
        ws.beam_refcounts[c.h] += 1;
    }
    ws.beam_old_hyps.clear();
    ws.beam_old_hyps.extend(st.hyps.drain(..).map(Some));
    ws.beam_scratch_hyps.clear();
    for c in &ws.beam_cands {
        let rem = ws.beam_refcounts[c.h];
        ws.beam_refcounts[c.h] = rem - 1;
        let hy = if rem == 1 {
            let mut parent = ws.beam_old_hyps[c.h].take().expect("last parent use");
            parent.score = c.score;
            if c.k != blank {
                parent.tokens.push(c.k as u32);
                parent.dec_ok = false;
                parent.lm_ctx = None;
            }
            parent
        } else {
            let parent = ws.beam_old_hyps[c.h].as_ref().expect("shared parent");
            let (tokens, dec_ok, lm_ctx) = if c.k == blank {
                (parent.tokens.clone(), parent.dec_ok, parent.lm_ctx)
            } else {
                let mut tk = Vec::with_capacity(parent.tokens.len() + 1);
                tk.extend_from_slice(&parent.tokens);
                tk.push(c.k as u32);
                (tk, false, None)
            };
            Hyp {
                tokens,
                ctx_start: parent.ctx_start,
                score: c.score,
                dec: Arc::clone(&parent.dec),
                dec_ok,
                lm_ctx,
            }
        };
        merge_hyp(&mut ws.beam_scratch_hyps, hy);
    }
    std::mem::swap(&mut st.hyps, &mut ws.beam_scratch_hyps);
}

/// Apply `ws.beam_cands` into `ws.beam_done[s]` (blanks) and `st.hyps` (non-blanks) for `max_sym > 1`.
fn expand_hyps_multi(st: &mut BeamState, s: usize, blank: usize, nh: usize, ws: &mut DecWorkspace) {
    ws.beam_refcounts.clear();
    ws.beam_refcounts.resize(nh, 0);
    for c in &ws.beam_cands {
        ws.beam_refcounts[c.h] += 1;
    }
    ws.beam_old_hyps.clear();
    ws.beam_old_hyps.extend(st.hyps.drain(..).map(Some));
    ws.beam_scratch_hyps.clear();
    for c in &ws.beam_cands {
        let rem = ws.beam_refcounts[c.h];
        ws.beam_refcounts[c.h] = rem - 1;
        if c.k == blank {
            let hy = if rem == 1 {
                let mut parent = ws.beam_old_hyps[c.h].take().expect("last parent use");
                parent.score = c.score;
                parent
            } else {
                let parent = ws.beam_old_hyps[c.h].as_ref().expect("shared parent");
                Hyp {
                    tokens: parent.tokens.clone(),
                    ctx_start: parent.ctx_start,
                    score: c.score,
                    dec: Arc::clone(&parent.dec),
                    dec_ok: parent.dec_ok,
                    lm_ctx: parent.lm_ctx,
                }
            };
            merge_hyp(&mut ws.beam_done[s], hy);
        } else {
            let hy = if rem == 1 {
                let mut parent = ws.beam_old_hyps[c.h].take().expect("last parent use");
                parent.tokens.push(c.k as u32);
                parent.score = c.score;
                parent.dec_ok = false;
                parent.lm_ctx = None;
                parent
            } else {
                let parent = ws.beam_old_hyps[c.h].as_ref().expect("shared parent");
                let mut tk = Vec::with_capacity(parent.tokens.len() + 1);
                tk.extend_from_slice(&parent.tokens);
                tk.push(c.k as u32);
                Hyp {
                    tokens: tk,
                    ctx_start: parent.ctx_start,
                    score: c.score,
                    dec: Arc::clone(&parent.dec),
                    dec_ok: false,
                    lm_ctx: None,
                }
            };
            merge_hyp(&mut ws.beam_scratch_hyps, hy);
        }
    }
    std::mem::swap(&mut st.hyps, &mut ws.beam_scratch_hyps);
}

#[cfg(test)]
mod tests {
    use super::score::logaddexp;
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
        let (lse, max_nb) = logsumexp_and_max_nb(Isa::Scalar, &r, 0).expect("finite");
        assert_eq!(max_nb, 3.0);
        let s: f32 = r.iter().map(|v| (v - lse).exp()).sum();
        assert!((s - 1.0).abs() < 1e-6);
        assert!(logsumexp_and_max_nb(Isa::Scalar, &[f32::NAN], 0).is_err());
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
