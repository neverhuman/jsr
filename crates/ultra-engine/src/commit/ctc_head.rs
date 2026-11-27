//! Optional `model.ctc_head` projection into frame-level CTC log-probabilities for H55 Wave 1A.
//! Zero overhead when `--ctc-weight == 0.0` (never invoked on the default commit path).

use std::collections::BTreeMap;

use crate::kernels::{self, Isa};
use crate::lm::TfWorkspace;
use crate::model::Model;
use crate::{Engine, EngineError, EngineStream, Workspace};

use super::rescore::{logaddexp_f64, select_commit_hyp, CtcInput};
use super::tf::CommitTfRescoreConfig;
use super::types::CommitConfig;
use super::vocab::CtcVocab;
use super::wide::{beam_margin, wide_engine, CommitWideGate};

/// Aggregate distinct BPE segmentations with identical whitespace-normalized text via `logaddexp`
/// before selection or MBR. Returns `(words, best_raw_text, merged_logp, first_hyp_idx)`.
pub fn aggregate_duplicate_hyps(hyps: &[(String, f32)]) -> Vec<(Vec<String>, String, f32, usize)> {
    let mut index: BTreeMap<Vec<String>, usize> = BTreeMap::new();
    let mut groups: Vec<(Vec<String>, String, f64, f32, usize)> = Vec::with_capacity(hyps.len());
    for (i, (text, score)) in hyps.iter().enumerate() {
        let words: Vec<String> = text.split_whitespace().map(str::to_owned).collect();
        let s64 = if score.is_finite() {
            f64::from(*score)
        } else {
            f64::NEG_INFINITY
        };
        if let Some(&gidx) = index.get(&words) {
            let g = &mut groups[gidx];
            g.2 = logaddexp_f64(g.2, s64);
            if score.is_finite() && (!g.3.is_finite() || *score > g.3) {
                g.1 = text.clone();
                g.3 = *score;
            }
        } else {
            index.insert(words.clone(), groups.len());
            groups.push((words, text.clone(), s64, *score, i));
        }
    }
    groups
        .into_iter()
        .map(|(w, txt, m, _, i)| {
            let s = if m.is_finite() {
                m as f32
            } else {
                f32::NEG_INFINITY
            };
            (w, txt, s, i)
        })
        .collect()
}

/// Run the closed-segment commit decode on `eng` (using optional external or model-internal CTC
/// scoring) and return `(committed_text, selected_hyp_index, beam_width)`. Zero CTC overhead when
/// unused. With `wide`, a low-margin first beam is searched again over the cached encoder
/// projection at the gate's width (CTC paths never widen).
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_commit_segment(
    eng: &Engine,
    pcm: &[i16],
    cfg: CommitConfig,
    ws: &mut Workspace,
    ext_ctc: Option<(&CtcVocab, &[f32], usize)>,
    model_ctc_vocab: Option<&CtcVocab>,
    tf: Option<(&CommitTfRescoreConfig, &mut TfWorkspace)>,
    wide: Option<CommitWideGate>,
) -> Result<(String, usize, usize), EngineError> {
    let width = eng.beam;
    let pcm = super::types::commit_input_pcm(pcm, cfg.rms_norm);
    let mut cs = eng.create_stream();
    cs.accept_i16(&pcm)?;
    if cfg.tail_pad_samples > 0 {
        cs.accept_i16(&vec![0i16; cfg.tail_pad_samples])?;
    }
    cs.input_finished();
    if let Some((vocab, lp, frames)) = ext_ctc {
        while eng.decode(&mut [&mut cs], ws)? > 0 {}
        let (text, idx) = select_commit_hyp(
            eng,
            &cs,
            cfg,
            Some(CtcInput::External(vocab, lp, frames)),
            tf,
        )?;
        return Ok((text, idx, width));
    }
    if cfg.ctc_weight > 0.0 && eng.model.ctc_head.is_some() {
        let (ctc_lp, ctc_frames) = drain_stream_with_ctc(eng, &mut cs, ws)?;
        let ctc_v = eng.model.ctc_head.as_ref().map_or(0, |h| h.n());
        let (text, idx) = select_commit_hyp(
            eng,
            &cs,
            cfg,
            Some(CtcInput::ModelHead(
                model_ctc_vocab,
                &ctc_lp,
                ctc_frames,
                ctc_v,
            )),
            tf,
        )?;
        return Ok((text, idx, width));
    }
    let Some(gate) = wide else {
        while eng.decode(&mut [&mut cs], ws)? > 0 {}
        let (text, idx) = select_commit_hyp(eng, &cs, cfg, None, tf)?;
        return Ok((text, idx, width));
    };
    let mut projection = eng.commit_projection();
    while eng.decode(&mut [&mut cs], ws)? > 0 {
        eng.cache_commit_projection(ws, &mut projection);
    }
    if gate.routes(beam_margin(&cs)) {
        let wider = wide_engine(eng, gate);
        wider.replay_commit_projection(&mut cs, &projection, ws)?;
        let (text, idx) = select_commit_hyp(&wider, &cs, cfg, None, tf)?;
        return Ok((text, idx, gate.width));
    }
    let (text, idx) = select_commit_hyp(eng, &cs, cfg, None, tf)?;
    Ok((text, idx, width))
}

/// Project `frames` encoder `final_norm` rows (`[frames][d]`) through `model.ctc_head` (fp32)
/// and apply row-wise `log_softmax` in `f64` accumulation (`[frames][ctc_vocab]`).
pub(crate) fn compute_ctc_log_probs(
    model: &Model,
    isa: Isa,
    enc_frames: &[f32],
    frames: usize,
) -> Result<Vec<f32>, EngineError> {
    let head = model.ctc_head.as_ref().ok_or_else(|| {
        EngineError::Unsupported("model has no ctc_head weights in weights.safetensors".into())
    })?;
    let (v, d) = (head.n(), head.k());
    if enc_frames.len() != frames * d {
        return Err(EngineError::Shape(format!(
            "ctc_log_probs: expected {} floats ({frames}x{d}), got {}",
            frames * d,
            enc_frames.len()
        )));
    }
    if frames == 0 {
        return Ok(Vec::new());
    }
    let mut logits = vec![0.0f32; frames * v];
    kernels::gemm_f32(isa, &head.f, enc_frames, frames, &mut logits);
    for row in logits.chunks_exact_mut(v) {
        let mut mx = f32::NEG_INFINITY;
        for &x in row.iter() {
            if !x.is_finite() {
                return Err(EngineError::Output("non-finite ctc_head logit".into()));
            }
            mx = mx.max(x);
        }
        let mx64 = f64::from(mx);
        let mut sum = 0.0f64;
        for &x in row.iter() {
            sum += (f64::from(x) - mx64).exp();
        }
        let lse = mx64 + sum.ln();
        for x in row.iter_mut() {
            *x = (f64::from(*x) - lse) as f32;
        }
    }
    Ok(logits)
}

/// Drain `stream` on `eng` while tapping encoder `final_norm` outputs (tap index `1 + n_blocks`)
/// and projecting them through `model.ctc_head` into `[frames][ctc_vocab]` log-probabilities.
pub(crate) fn drain_stream_with_ctc(
    eng: &Engine,
    stream: &mut EngineStream,
    ws: &mut Workspace,
) -> Result<(Vec<f32>, usize), EngineError> {
    if eng.model.ctc_head.is_none() {
        return Err(EngineError::Unsupported(
            "ctc_weight > 0 requires ctc_head weights in weights.safetensors".into(),
        ));
    }
    let d = eng.model.arch.d;
    let target_layer = 1 + eng.model.arch.n_blocks;
    let mut enc_frames = Vec::<f32>::new();
    let mut total_frames = 0usize;
    loop {
        let mut tap = |layer: usize, vals: &[f32], width: usize| {
            if layer == target_layer && width == d {
                enc_frames.extend_from_slice(vals);
                total_frames += vals.len() / d;
            }
        };
        if eng.decode_tapped(&mut [stream], ws, Some(&mut tap))? == 0 {
            break;
        }
    }
    let log_probs = compute_ctc_log_probs(&eng.model, eng.isa, &enc_frames, total_frames)?;
    Ok((log_probs, total_frames))
}

/// Best-effort bind of `<model.root>/bpe.model` when `model.ctc_head` matches its vocab size.
pub(crate) fn load_model_ctc_vocab(model: &Model) -> Option<CtcVocab> {
    let head = model.ctc_head.as_ref()?;
    let bpe = model.root.join("bpe.model");
    if !bpe.is_file() {
        return None;
    }
    let vocab = CtcVocab::from_spm_model_file(&bpe, model.arch.blank).ok()?;
    (vocab.vocab_size() == head.n()).then_some(vocab)
}
