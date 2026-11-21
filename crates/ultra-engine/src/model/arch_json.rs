//! The `arch.json` schema (`ultra.r1-weights/v1`) as read by the loader, and the file and
//! dimension checks applied before anything is allocated or opened.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use ultra_core::receipt::sha256_hex;

use crate::EngineError;

#[derive(Deserialize)]
pub(super) struct RawArch {
    pub(super) schema: String,
    pub(super) model: RawModel,
    pub(super) source: RawSource,
    pub(super) files: BTreeMap<String, RawFile>,
    pub(super) features: RawFeatures,
    pub(super) decoder: RawDecoder,
}

#[derive(Deserialize)]
pub(super) struct RawModel {
    pub(super) family: String,
    pub(super) config: RawConfig,
    pub(super) encoder: RawEncoder,
}

#[derive(Deserialize)]
pub(super) struct RawConfig {
    pub(super) vocab_rnnt: usize,
    pub(super) pred_emb: usize,
    pub(super) joiner: usize,
    pub(super) joiner_act: String,
}

#[derive(Deserialize)]
pub(super) struct RawEncoder {
    pub(super) n_mels: usize,
    pub(super) d_model: usize,
    pub(super) n_blocks: usize,
    pub(super) n_heads: usize,
    pub(super) ffn_mult: usize,
    pub(super) conv_kernel: usize,
    pub(super) stem_channels: usize,
    pub(super) chunk: usize,
    pub(super) left_context: usize,
    pub(super) post_block_norm: bool,
    pub(super) family: String,
}

#[derive(Deserialize)]
pub(super) struct RawSource {
    pub(super) checkpoint_sha256: String,
}

#[derive(Deserialize)]
pub(super) struct RawFile {
    pub(super) path: String,
    pub(super) sha256: String,
}

#[derive(Deserialize)]
pub(super) struct RawFeatures {
    pub(super) sample_rate: u32,
    pub(super) frame_length: usize,
    pub(super) frame_shift: usize,
    pub(super) n_fft: usize,
    pub(super) n_mels: usize,
    pub(super) log_floor: f32,
    pub(super) cmvn: RawCmvn,
}

#[derive(Deserialize)]
pub(super) struct RawCmvn {
    pub(super) mean: Vec<f32>,
    pub(super) istd: Vec<f32>,
}

#[derive(Deserialize)]
pub(super) struct RawDecoder {
    pub(super) kind: String,
    pub(super) context: usize,
    pub(super) vocab_size: usize,
    pub(super) joiner_activation: String,
    pub(super) blank_id: usize,
    pub(super) max_symbols: usize,
    pub(super) ctx_init: [usize; 2],
}

/// Resolve a declared file inside `root` (no separators, no escaping symlinks) and, when
/// `check` is set, verify its SHA-256 against `arch.json`.
pub(super) fn verify(root: &Path, f: &RawFile, check: bool) -> Result<PathBuf, EngineError> {
    if f.path.is_empty() || f.path.contains('/') || f.path.contains('\\') || f.path.starts_with('.')
    {
        return Err(EngineError::Format(format!(
            "file path {} escapes the directory",
            f.path
        )));
    }
    let base = root.canonicalize()?;
    let p = base.join(&f.path).canonicalize()?;
    if p.parent() != Some(base.as_path()) {
        return Err(EngineError::Format(format!(
            "file {} resolves outside the weight directory",
            f.path
        )));
    }
    if check {
        let got = sha256_hex(&std::fs::read(&p)?);
        if got != f.sha256 {
            return Err(EngineError::Integrity(format!(
                "{}: sha256 {got} != arch.json {}",
                f.path, f.sha256
            )));
        }
    }
    Ok(p)
}

/// Positive and at most `max` (guards every allocation product derived from arch.json).
pub(super) fn bounded(name: &str, v: usize, max: usize) -> Result<usize, EngineError> {
    if v == 0 || v > max {
        return Err(EngineError::Shape(format!(
            "{name} = {v} outside 1..={max}"
        )));
    }
    Ok(v)
}

pub(super) fn file<'a>(raw: &'a RawArch, key: &str) -> Result<&'a RawFile, EngineError> {
    raw.files
        .get(key)
        .ok_or_else(|| EngineError::Format(format!("arch.json has no files.{key}")))
}
