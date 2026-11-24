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
    pub ctx: Vec<usize>,
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
            ctx: m.arch.ctx_init.clone(),
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
        self.ctx.clone_from(&m.arch.ctx_init);
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
    pub(crate) beam_cands: Vec<crate::beam::Cand>,
    pub(crate) beam_gated: Vec<bool>,
    pub(crate) beam_fallback: Vec<usize>,
    pub(crate) beam_fb_logits: Vec<f32>,
    pub(crate) beam_fb_lse: Vec<f32>,
    pub(crate) beam_fb_max_nb: Vec<f32>,
    pub(crate) beam_fb_row_map: Vec<isize>,
    pub(crate) beam_fb_stream_off: Vec<usize>,
    pub(crate) beam_next_active: Vec<usize>,
    pub(crate) beam_done: Vec<Vec<crate::beam::Hyp>>,
    pub(crate) beam_fast_blank: Vec<bool>,
    pub(crate) beam_scratch_hyps: Vec<crate::beam::Hyp>,
    pub(crate) beam_old_hyps: Vec<Option<crate::beam::Hyp>>,
    pub(crate) beam_lse: Vec<f32>,
    pub(crate) beam_max_nb: Vec<f32>,
    pub(crate) beam_row_map: Vec<usize>,
    pub(crate) beam_stream_row_off: Vec<usize>,
    pub(crate) beam_shortlist: Vec<(f32, usize)>,
    pub(crate) beam_ub_buf: Vec<f32>,
    pub(crate) beam_refcounts: Vec<u16>,
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
            let (t0, t1) = (states[s].ctx[0], states[s].ctx[1]);
            let a = &pc.h0[t0 * jn..(t0 + 1) * jn];
            let b = &pc.h1[t1 * jn..(t1 + 1) * jn];
            for ((o, &av), &bv) in states[s].dec.iter_mut().zip(a).zip(b) {
                *o = av + bv;
            }
            states[s].dec_ok = true;
        }
        return;
    }
    let context = m.arch.ctx_init.len();
    let inp = grow(&mut ws.emb_in, rows * context * e);
    for (r, &s) in ws.need.iter().enumerate() {
        for (position, &tok) in states[s].ctx.iter().enumerate() {
            let start = (context * r + position) * e;
            inp[start..start + e].copy_from_slice(&m.emb[tok * e..(tok + 1) * e]);
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
                // The unsigned activation path still needs kernel-level argmax and tie parity.
                let qmax = int8_slack.then_some(127);
                if !(int8_slack && ws.q.act7)
                    && cert.is_some_and(|c| c.is_certified_blank(row, qmax))
                {
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
                    st.ctx.rotate_left(1);
                    let last = st.ctx.len() - 1;
                    st.ctx[last] = k;
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
        assert!(cert.is_certified_blank(&hid_blank, None));
        assert!(cert.is_certified_blank(&hid_blank, Some(127)));
        assert!(cert.is_certified_blank(&hid_blank, Some(63)));
        // Adversarial non-blank vector: raise bias of token 3 above z_blank (= 6.4) so argmax != blank.
        let mut b2 = b.clone();
        b2[3] = 7.0;
        let cert2 = BlankCert::from_row_major(&w, &b2, v, jn, blank).expect("valid cert");
        assert!(!cert2.is_certified_blank(&hid_blank, None));
        assert!(!cert2.is_certified_blank(&hid_blank, Some(127)));
        assert!(!cert2.is_certified_blank(&hid_blank, Some(63)));
        // 7-bit (--act7) regression: a tight margin that passes 8-bit slack (/254) MUST be
        // rejected under 7-bit slack (/126) because quantisation step is ~2x wider.
        let mut hid_wide = vec![-1.0f32; jn];
        hid_wide[0] = 1.0; // hi - lo = 2.0 -> scale_127 = 2/254, scale_63 = 2/126
        let mut w_tight = vec![0.0f32; 2 * jn];
        w_tight[0] = 1.0; // blank weight
        w_tight[jn] = -1.0; // non-blank weight -> max_w_norm = 1.0
        let z_b = 1.0f64;
        // upper_127 = -1.0 + r_max*sqrt(jn) + b_nb + margin_127; choose b_nb between 8-bit and 7-bit slack.
        let base_margin = 1e-4 * (1.0 + (jn as f64).sqrt());
        let s127 = (2.0 / 254.0) * (jn as f64).sqrt() + 1e-3;
        let s63 = (2.0 / 126.0) * (jn as f64).sqrt() + 1e-3;
        let b_nb = (z_b - (-1.0) - base_margin - 0.5 * (s127 + s63)) as f32;
        let cert_tight =
            BlankCert::from_row_major(&w_tight, &[0.0, b_nb], 2, jn, 0).expect("valid tight cert");
        assert!(cert_tight.is_certified_blank(&hid_wide, Some(127)));
        assert!(!cert_tight.is_certified_blank(&hid_wide, Some(63)));
        // NaN must never be certified blank.
        let mut hid_nan = hid_blank.clone();
        hid_nan[4] = f32::NAN;
        assert!(!cert.is_certified_blank(&hid_nan, None));
    }

    #[test]
    fn greedy_chunk_act7_disables_blank_cert_and_matches_packed_argmax_and_ties() {
        use std::path::PathBuf;
        use std::sync::Arc;

        use ultra_frontend::{GlobalCmvn, TrainerMelConfig};
        use ultra_gp::TokenTable;

        use super::*;
        use crate::model::{Arch, Linear};
        use crate::quant::PackedI8;

        let (vocab, pred_emb, joiner) = (4, 4, 16);
        let ctx_len = 3;
        let arch = Arch {
            n_mels: 80,
            d: 4,
            n_blocks: 0,
            lower: 0,
            mid: 0,
            fin: 0,
            heads: 1,
            dh: 4,
            ffn: 16,
            kernel: 3,
            stem_ch: 4,
            f1: 39,
            f2: 19,
            chunk: 4,
            left: 16,
            post_block_norm: false,
            vocab,
            pred_emb,
            joiner,
            blank: 0,
            max_symbols: 1,
            ctx_init: vec![0; ctx_len],
            log_floor: 1e-6,
            checkpoint_sha256: String::new(),
            weights_sha256: String::new(),
            qat_stem_int8: false,
        };
        let tokens = Arc::new(
            TokenTable::from_str_checked("<blk> 0\na 1\nb 2\nc 3\n", vocab).expect("tokens"),
        );
        let cmvn = GlobalCmvn::new(&[0.0; 80], &[1.0; 80]).expect("cmvn");
        let mel_cfg = Arc::new(
            TrainerMelConfig::from_fbank_le_bytes(&vec![0u8; 257 * 80 * 4], 1e-6, cmvn)
                .expect("mel_cfg"),
        );

        // Construct quantized joiner weights with exact ties:
        // - Row 0 (blank) and row 1 (token 1) are identical -> tie must resolve to index 0 (blank).
        // - Row 2 (token 2) and row 3 (token 3) are identical -> tie must resolve to index 2.
        let mut join_w = vec![0.0f32; vocab * joiner];
        for k in 0..joiner {
            let v01 = if k % 2 == 0 { 0.5 } else { -0.25 };
            let v23 = if k % 2 == 0 { -0.5 } else { 0.75 };
            join_w[k] = v01;
            join_w[joiner + k] = v01;
            join_w[2 * joiner + k] = v23;
            join_w[3 * joiner + k] = v23;
        }
        let join_b = vec![0.0f32; vocab];
        let mut joiner_out = Linear::from_fp32(&join_w, vocab, joiner, Some(&join_b));
        joiner_out.q = Some(PackedI8::quantize(&join_w, vocab, joiner, Some(&join_b)));

        // Adversarial cert that strongly (and falsely) claims blank for every vector.
        let mut fake_w = vec![0.0f32; vocab * joiner];
        fake_w[..joiner].fill(1.0);
        let fake_cert =
            BlankCert::from_row_major(&fake_w, &[50.0, -50.0, -50.0, -50.0], vocab, joiner, 0)
                .expect("fake cert");

        let dummy = Linear::from_fp32(&[0.0], 1, 1, None);
        let mut model = Model {
            arch,
            root: PathBuf::new(),
            c1_w: Vec::new(),
            c1_b: Vec::new(),
            c2: dummy.clone(),
            stem_out: dummy.clone(),
            blocks: Vec::new(),
            final_norm: None,
            pair: None,
            restore: None,
            enc_proj: dummy,
            emb: vec![0.0f32; vocab * pred_emb],
            pred_proj: Linear::from_fp32(
                &vec![0.0f32; joiner * ctx_len * pred_emb],
                joiner,
                ctx_len * pred_emb,
                Some(&vec![0.0f32; joiner]),
            ),
            pred_cache: None,
            joiner_out,
            ctc_head: None,
            blank_cert_f32: None,
            blank_cert_i8: Some(fake_cert),
            rope_inv: Vec::new(),
            mel_cfg,
            tokens,
            int8_prepared: None,
        };

        // Frame 0: favors rows 0 & 1 equally -> first-index tie breaks to blank (0).
        // Frame 1: favors rows 2 & 3 equally -> first-index tie breaks to token 2 (non-blank).
        let mut enc = vec![0.0f32; 2 * joiner];
        for k in 0..joiner {
            enc[k] = if k % 2 == 0 { 1.0 } else { -1.0 };
            enc[joiner + k] = if k % 2 == 0 { -1.0 } else { 1.0 };
        }

        let isa = Isa::detect();
        let mut st_with_cert = DecState::new(&model);
        let mut ws_cert = DecWorkspace::default();
        ws_cert.q.act7 = true;
        greedy_chunk(
            &model,
            isa,
            Precision::Int8,
            &mut [&mut st_with_cert],
            &[2],
            &enc,
            &mut ws_cert,
        )
        .expect("act7 greedy with cert");

        // Compare directly against no-certificate packed act7 execution.
        model.blank_cert_i8 = None;
        let mut st_no_cert = DecState::new(&model);
        let mut ws_no_cert = DecWorkspace::default();
        ws_no_cert.q.act7 = true;
        greedy_chunk(
            &model,
            isa,
            Precision::Int8,
            &mut [&mut st_no_cert],
            &[2],
            &enc,
            &mut ws_no_cert,
        )
        .expect("act7 greedy without cert");

        assert_eq!(st_with_cert.tokens, vec![2]);
        assert_eq!(st_with_cert.tokens, st_no_cert.tokens);
        assert_eq!(st_with_cert.ctx, vec![0, 0, 2]);
    }
}
