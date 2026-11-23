//! Exact CPU forced-alignment scoring for RNN-T hypotheses (`f64` forward recursion) plus
//! optional external n-gram LM and LODR bigram log-probabilities.
//!
//! Encodes each utterance once ([`encode_pcm`]) and scores candidate transcripts against cached
//! encoder frames ([`score_tokens`]), emitting both `rnnt_full_logp` (standard full-sum forward)
//! and `rnnt_sync_logp` (1-symbol-per-frame forward matching modified beam search).

mod dp;

use ultra_frontend::{TrainerLogMel, MEL_BINS};
use ultra_gp::{chunk_bounds, FeatureRing};

use crate::commit::SpmTokenizer;
use crate::encoder::{encode_chunk, linear, ChunkJob, EncState, EncWorkspace, Precision};
use crate::lm::NgramLm;
use crate::quant::QuantScratch;
use crate::{Engine, EngineError, Model};

pub use dp::rnnt_forward_from_log_probs;
use dp::ForwardDp;

/// Cached encoder projection frames (`[n_frames][joiner_dim]`) for one utterance.
#[derive(Debug, Clone)]
pub struct EncodedAudio {
    pub frames: Vec<f32>,
    pub n_frames: usize,
    pub joiner_dim: usize,
}

/// Exact forced-alignment and LM scores for a single candidate transcript.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForcedScore {
    pub rnnt_full_logp: f64,
    pub rnnt_sync_logp: f64,
    pub lm_logp: f64,
    pub lodr_logp: f64,
    pub n_tokens: usize,
    pub n_frames: usize,
}

/// Reusable per-thread buffers for [`encode_pcm`] and [`score_tokens`].
#[derive(Debug, Default)]
pub struct FscoreWorkspace {
    enc: EncWorkspace,
    rows: Vec<[f32; MEL_BINS]>,
    feats: Vec<f32>,
    chunk_out: Vec<f32>,
    pred: Vec<f32>,
    emb_in: Vec<f32>,
    hid: Vec<f32>,
    logits: Vec<f32>,
    q: QuantScratch,
}

/// Tokenize text via the model's SentencePiece `.model`, shifting piece IDs around `blank_id`.
pub fn tokenize_text(
    spm: &SpmTokenizer,
    text: &str,
    blank_id: usize,
) -> Result<Vec<u32>, EngineError> {
    let ids = spm
        .encode_ids(text)
        .ok_or_else(|| EngineError::Format(format!("failed to tokenize text: {text:?}")))?;
    ids.into_iter()
        .map(|p| {
            let shifted = if p < blank_id { p } else { p + 1 };
            u32::try_from(shifted)
                .map_err(|_| EngineError::Format(format!("token id overflow: {shifted}")))
        })
        .collect()
}

/// Exact `f64` sum of external n-gram LM and LODR bigram natural-log probabilities.
pub fn score_lm_tokens(lm: Option<&NgramLm>, tokens: &[u32]) -> (f64, f64) {
    let Some(lm) = lm else { return (0.0, 0.0) };
    let (mut lm_sum, mut lodr_sum) = (0.0f64, 0.0f64);
    for (i, &tok) in tokens.iter().enumerate() {
        let ctx = lm.ctx_state(&tokens[..i]);
        lm_sum += f64::from(lm.lm_logp(&ctx, tok as usize));
        lodr_sum += f64::from(lm.lodr_logp(&ctx, tok as usize));
    }
    (lm_sum, lodr_sum)
}

/// Encode 16 kHz mono PCM16 audio (plus optional trailing silence pad) once using `engine`.
pub fn encode_pcm(
    engine: &Engine,
    pcm: &[i16],
    tail_pad_samples: usize,
    ws: &mut FscoreWorkspace,
) -> Result<EncodedAudio, EngineError> {
    let (m, a) = (&engine.model, &engine.model.arch);
    let mut mel = TrainerLogMel::new(m.mel_cfg.clone());
    let mut ring = FeatureRing::new();
    ws.rows.clear();
    mel.push_i16_into(pcm, &mut ws.rows);
    ring.push_rows(&ws.rows);
    if tail_pad_samples > 0 {
        let pad = vec![0i16; tail_pad_samples];
        ws.rows.clear();
        mel.push_i16_into(&pad, &mut ws.rows);
        ring.push_rows(&ws.rows);
    }

    let mut state = EncState::new_with_kv_and_chunk(m, 1, engine.kv_int8, engine.chunk);
    let (mut frames, mut n_frames) = (Vec::new(), 0usize);
    ws.enc.q.act7 = engine.act7;

    while let Some((t0, t1)) = chunk_bounds(state.t, ring.n_feat(), engine.chunk, true) {
        let n = t1 - t0;
        let feat_len = (4 * n + 3) * a.n_mels;
        if ws.feats.len() < feat_len {
            ws.feats.resize(feat_len, 0.0);
        }
        ring.extract_chunk_into(t0, n, &mut ws.feats[..feat_len]);
        let mut jobs = [ChunkJob {
            state: &mut state,
            feats: &ws.feats[..feat_len],
            n,
        }];
        encode_chunk(
            m,
            engine.isa,
            engine.precision,
            &mut jobs,
            &mut ws.enc,
            &mut ws.chunk_out,
            None,
        )?;
        frames.extend_from_slice(&ws.chunk_out[..n * a.joiner]);
        n_frames += n;
    }
    Ok(EncodedAudio {
        frames,
        n_frames,
        joiner_dim: a.joiner,
    })
}

fn fill_predictor_states(
    m: &Model,
    isa: crate::Isa,
    tokens: &[u32],
    ws: &mut FscoreWorkspace,
) -> Result<(), EngineError> {
    let (v, blank, e, jn) = (m.arch.vocab, m.arch.blank, m.arch.pred_emb, m.arch.joiner);
    let cols = tokens.len() + 1;
    for (idx, &t) in tokens.iter().enumerate() {
        if t as usize >= v || t as usize == blank {
            return Err(EngineError::Format(format!(
                "invalid non-blank token {t} at {idx} (vocab={v}, blank={blank})"
            )));
        }
    }
    ws.pred.resize(cols * jn, 0.0);
    let ctx_len = m.arch.ctx_init.len();
    if let Some(pc) = &m.pred_cache {
        if ctx_len == 2 {
            let mut ctx = [m.arch.ctx_init[0], m.arch.ctx_init[1]];
            for u in 0..cols {
                let a = &pc.h0[ctx[0] * jn..(ctx[0] + 1) * jn];
                let b = &pc.h1[ctx[1] * jn..(ctx[1] + 1) * jn];
                for ((o, &av), &bv) in ws.pred[u * jn..(u + 1) * jn].iter_mut().zip(a).zip(b) {
                    *o = av + bv;
                }
                if u < tokens.len() {
                    ctx = [ctx[1], tokens[u] as usize];
                }
            }
            return Ok(());
        }
    }
    ws.emb_in.resize(cols * ctx_len * e, 0.0);
    let mut ctx = m.arch.ctx_init.clone();
    for u in 0..cols {
        for (pos, &tok) in ctx.iter().enumerate() {
            let dst = (u * ctx_len + pos) * e;
            ws.emb_in[dst..dst + e].copy_from_slice(&m.emb[tok * e..(tok + 1) * e]);
        }
        if u < tokens.len() && ctx_len > 0 {
            ctx.rotate_left(1);
            ctx[ctx_len - 1] = tokens[u] as usize;
        }
    }
    linear(
        isa,
        Precision::F32,
        &m.pred_proj,
        &ws.emb_in[..cols * ctx_len * e],
        cols,
        &mut ws.pred[..cols * jn],
        &mut ws.q,
    );
    Ok(())
}

/// Score one candidate token sequence against cached [`EncodedAudio`] frames and optional LM.
pub fn score_tokens(
    engine: &Engine,
    enc: &EncodedAudio,
    tokens: &[u32],
    lm: Option<&NgramLm>,
    ws: &mut FscoreWorkspace,
) -> Result<ForcedScore, EngineError> {
    let m = &engine.model;
    let (jn, v, blank) = (m.arch.joiner, m.arch.vocab, m.arch.blank);
    if enc.joiner_dim != jn || enc.frames.len() != enc.n_frames * jn {
        return Err(EngineError::Shape("encoded audio shape mismatch".into()));
    }
    fill_predictor_states(m, engine.isa, tokens, ws)?;
    let (lm_logp, lodr_logp) = score_lm_tokens(lm, tokens);
    let (n_tokens, n_frames, cols) = (tokens.len(), enc.n_frames, tokens.len() + 1);
    ws.hid.resize(cols * jn, 0.0);
    ws.logits.resize(cols * v, 0.0);
    ws.q.act7 = engine.act7;

    let mut dp = ForwardDp::new(n_tokens);
    let (mut cur_blank, mut cur_tok) = (vec![0.0f64; cols], vec![0.0f64; n_tokens]);
    for t in 0..n_frames {
        let ev = &enc.frames[t * jn..(t + 1) * jn];
        for u in 0..cols {
            let pv = &ws.pred[u * jn..(u + 1) * jn];
            for ((h, &e_v), &p_v) in ws.hid[u * jn..(u + 1) * jn].iter_mut().zip(ev).zip(pv) {
                *h = (e_v + p_v).tanh();
            }
        }
        linear(
            engine.isa,
            engine.precision,
            &m.joiner_out,
            &ws.hid[..cols * jn],
            cols,
            &mut ws.logits[..cols * v],
            &mut ws.q,
        );
        for u in 0..cols {
            let row = &ws.logits[u * v..(u + 1) * v];
            let mut mx = f64::NEG_INFINITY;
            for (k, &x) in row.iter().enumerate() {
                if !x.is_finite() {
                    return Err(EngineError::Output(format!(
                        "non-finite logit t={t} u={u} k={k}"
                    )));
                }
                mx = mx.max(f64::from(x));
            }
            let lse = mx
                + row
                    .iter()
                    .map(|&x| (f64::from(x) - mx).exp())
                    .sum::<f64>()
                    .ln();
            cur_blank[u] = f64::from(row[blank]) - lse;
            if u < n_tokens {
                cur_tok[u] = f64::from(row[tokens[u] as usize]) - lse;
            }
        }
        dp.step(&cur_blank, &cur_tok);
    }
    let (rnnt_full_logp, rnnt_sync_logp) = dp.finish(n_tokens);
    Ok(ForcedScore {
        rnnt_full_logp,
        rnnt_sync_logp,
        lm_logp,
        lodr_logp,
        n_tokens,
        n_frames,
    })
}
