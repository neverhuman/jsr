//! The `arch.json` schema (`ultra.r1-weights/v1`) as read by the loader, and the file and
//! dimension checks applied before anything is allocated or opened.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::EngineError;

#[derive(Deserialize)]
pub(super) struct RawArch {
    pub(super) schema: String,
    pub(super) model: RawModel,
    pub(super) source: RawSource,
    pub(super) files: BTreeMap<String, RawFile>,
    pub(super) features: RawFeatures,
    pub(super) decoder: RawDecoder,
    /// Present in a prepared-INT8 directory (`ultra-engine-prepare`).
    #[serde(default)]
    pub(super) int8: Option<RawInt8>,
}

#[derive(Deserialize)]
pub(super) struct RawInt8 {
    pub(super) layout: String,
    pub(super) source_weights_sha256: String,
    pub(super) fp32_kinds: Vec<String>,
    /// `"ternary"`: every code is in {-1, 0, 1} (R4 ternary weights).
    #[serde(default)]
    pub(super) codes: Option<String>,
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
    pub(super) lower_blocks: usize,
    pub(super) middle_blocks: usize,
    pub(super) final_blocks: usize,
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
        let got = sha256_file(&p)?;
        if got != f.sha256 {
            return Err(EngineError::Integrity(format!(
                "{}: sha256 {got} != arch.json {}",
                f.path, f.sha256
            )));
        }
    }
    Ok(p)
}

/// The quantisable-layer kind (as in `Model::linears_mut`) of a torch tensor prefix.
pub(super) fn kind_of(prefix: &str) -> Option<&'static str> {
    const TAIL: [(&str, &str); 8] = [
        (".ff1.w1", "ff1_w1"),
        (".ff1.w2", "ff1_w2"),
        (".att.qkv", "qkv"),
        (".att.out", "att_out"),
        (".conv.pw1", "pw1"),
        (".conv.pw2", "pw2"),
        (".ff2.w1", "ff2_w1"),
        (".ff2.w2", "ff2_w2"),
    ];
    match prefix {
        "encoder.stem.c2" => Some("stem_c2"),
        "encoder.stem.out" => Some("stem_out"),
        "encoder.pair" => Some("pair"),
        "encoder.restore" => Some("restore"),
        "joiner.enc_proj" => Some("enc_proj"),
        "joiner.out" => Some("joiner_out"),
        _ => TAIL
            .iter()
            .find(|(t, _)| prefix.ends_with(t))
            .map(|&(_, k)| k),
    }
}

/// SHA-256 of a file, streamed in 1 MiB blocks (never the whole file in memory).
pub(super) fn sha256_file(p: &Path) -> Result<String, EngineError> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut f = std::fs::File::open(p)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
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
