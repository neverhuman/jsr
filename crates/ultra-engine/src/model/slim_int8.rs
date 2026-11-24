//! Self-contained slim INT8 bundle helpers: enumerates and copies the small non-quantised FP32
//! tensors (norms, biases, stem layer 1, depthwise conv, RoPE, predictor, optional CTC head, and
//! any declared `fp32_kinds` weights) into `weights.int8.safetensors` so INT8 inference never
//! opens or ships the 227 MB `weights.safetensors`.

use std::path::Path;

use super::Arch;
use crate::safetensors::{LazyTensors, SafeTensors, Tensor};
use crate::EngineError;

const BLOCK_LINEARS: [(&str, &str); 8] = [
    (".ff1.w1", "ff1_w1"),
    (".ff1.w2", "ff1_w2"),
    (".att.qkv", "qkv"),
    (".att.out", "att_out"),
    (".conv.pw1", "pw1"),
    (".conv.pw2", "pw2"),
    (".ff2.w1", "ff2_w1"),
    (".ff2.w2", "ff2_w2"),
];

fn has_kind(fp32_kinds: &[String], kind: &str) -> bool {
    fp32_kinds.iter().any(|k| k == kind)
}

fn push_linear(out: &mut Vec<String>, prefix: &str, kind: &str, fp32_kinds: &[String]) {
    out.push(format!("{prefix}.bias"));
    if has_kind(fp32_kinds, kind) {
        out.push(format!("{prefix}.weight"));
    }
}

fn push_norm(out: &mut Vec<String>, prefix: &str) {
    out.push(format!("{prefix}.weight"));
    out.push(format!("{prefix}.bias"));
}

/// Exact list of non-quantised FP32 tensor names required in a self-contained INT8 artifact.
pub(super) fn aux_fp32_names(arch: &Arch, fp32_kinds: &[String], has_ctc: bool) -> Vec<String> {
    let mut names = vec![
        "cmvn.mean".into(),
        "cmvn.istd".into(),
        "encoder.stem.c1.weight".into(),
        "encoder.stem.c1.bias".into(),
    ];
    push_linear(&mut names, "encoder.stem.c2", "stem_c2", fp32_kinds);
    push_linear(&mut names, "encoder.stem.out", "stem_out", fp32_kinds);
    let block_prefixes = (0..arch.lower)
        .map(|i| format!("encoder.blocks.{i}"))
        .chain((0..arch.mid).map(|i| format!("encoder.mid_blocks.{i}")))
        .chain((0..arch.fin).map(|i| format!("encoder.final_blocks.{i}")));
    for p in block_prefixes {
        names.push(format!("{p}.conv.dw.weight"));
        names.push(format!("{p}.conv.dw.bias"));
        push_norm(&mut names, &format!("{p}.ff1.norm"));
        push_norm(&mut names, &format!("{p}.att.norm"));
        push_norm(&mut names, &format!("{p}.conv.norm"));
        push_norm(&mut names, &format!("{p}.conv.dnorm"));
        push_norm(&mut names, &format!("{p}.ff2.norm"));
        if arch.post_block_norm {
            push_norm(&mut names, &format!("{p}.norm"));
        }
        for (tail, kind) in BLOCK_LINEARS {
            push_linear(&mut names, &format!("{p}{tail}"), kind, fp32_kinds);
        }
    }
    if !arch.post_block_norm {
        push_norm(&mut names, "encoder.final_norm");
    }
    names.push("engine.rope_inv_freq".into());
    if arch.mid > 0 {
        push_linear(&mut names, "encoder.pair", "pair", fp32_kinds);
        push_linear(&mut names, "encoder.restore", "restore", fp32_kinds);
    }
    names.push("predictor.emb.weight".into());
    names.push("predictor.proj.weight".into());
    names.push("predictor.proj.bias".into());
    push_linear(&mut names, "joiner.enc_proj", "enc_proj", fp32_kinds);
    push_linear(&mut names, "joiner.out", "joiner_out", fp32_kinds);
    if has_ctc {
        names.push("ctc_head.weight".into());
        names.push("ctc_head.bias".into());
    }
    names
}

/// Copy every required non-quantised FP32 tensor byte-for-byte from `src_st_path` into `dst`.
pub(super) fn copy_aux_fp32_tensors(
    src_st_path: &Path,
    arch: &Arch,
    fp32_kinds: &[String],
    dst: &mut SafeTensors,
) -> Result<(), EngineError> {
    let src = LazyTensors::open(src_st_path)?;
    let has_ctc = src.contains("ctc_head.weight") && src.contains("ctc_head.bias");
    for name in aux_fp32_names(arch, fp32_kinds, has_ctc) {
        let shape = src
            .shape(&name)
            .ok_or_else(|| EngineError::Format(format!("missing aux tensor {name}")))?
            .to_vec();
        let data = src.f32(&name, &shape)?;
        dst.tensors.insert(name, Tensor { shape, data });
    }
    Ok(())
}
