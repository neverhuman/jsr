//! Strict `ultra.r1-export/v1` bundle manifest validator and loader.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;
use ultra_core::receipt::{is_sha256_hex, sha256_hex};
use ultra_frontend::{GlobalCmvn, TrainerMelConfig, MEL_BINS};

use crate::rnnt::TokenTable;
use crate::GpError;

pub const R1_EXPORT_SCHEMA: &str = "ultra.r1-export/v1";
pub const R1_EXPORT_SCHEMA_V1_1: &str = "ultra.r1-export/v1.1";

/// Explicit ONNX execution precision (never silently falls back to `fp32`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum R1Precision {
    Fp32,
    Int8,
}

impl R1Precision {
    pub fn parse(s: &str) -> Result<Self, GpError> {
        match s {
            "fp32" => Ok(Self::Fp32),
            "int8" => Ok(Self::Int8),
            other => Err(GpError::Precision(other.to_owned())),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Int8 => "int8",
        }
    }
}

/// Supported R1 encoder architecture families (`dense` `S` vs `multirate` `S_MR`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Dense,
    Multirate,
}

impl ModelFamily {
    pub fn parse(s: &str) -> Result<Self, GpError> {
        match s {
            "dense" => Ok(Self::Dense),
            "multirate" => Ok(Self::Multirate),
            other => Err(GpError::Manifest(format!(
                "unknown model.family '{other}' (expected 'dense' or 'multirate')"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Multirate => "multirate",
        }
    }
}

/// Validated state tensor specification (`k`, `v`, `kv_len`, `conv`, plus MR `km`, `vm`, ...).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct R1EncoderStateSpec {
    pub name: String,
    pub batch_axis: usize,
    /// Shape with `B = 1` (used to size per-stream state slices).
    pub per_stream_dims: Vec<usize>,
    pub per_stream_elems: usize,
    pub is_int64: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct R1StreamingSpec {
    pub chunk_frames: usize,
    pub chunk_ms: u32,
    pub left_context_frames: usize,
    pub joiner_dim: usize,
    pub states: Vec<R1EncoderStateSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct R1DecoderSpec {
    pub blank_id: i64,
    pub vocab_size: usize,
    pub ctx_init: [i64; 2],
    pub max_symbols: usize,
}

/// Verified, ready-to-load `ultra.r1-export/{v1,v1.1}` bundle on disk.
#[derive(Debug, Clone)]
pub struct R1Bundle {
    pub root: PathBuf,
    pub schema: String,
    pub int8_batch_invariant: bool,
    pub manifest_sha256: String,
    pub family: ModelFamily,
    pub precision: R1Precision,
    pub exporter_commit: String,
    pub encoder_onnx: PathBuf,
    pub decoder_onnx: PathBuf,
    pub joiner_onnx: PathBuf,
    pub mel_cfg: Arc<TrainerMelConfig>,
    pub streaming: R1StreamingSpec,
    pub decoder: R1DecoderSpec,
    pub tokens: Arc<TokenTable>,
}

#[derive(Deserialize)]
struct RawInt8Meta {
    #[serde(default)]
    batch_invariant: bool,
}

#[derive(Deserialize)]
struct RawManifest {
    schema: String,
    #[serde(default)]
    int8: Option<RawInt8Meta>,
    model: RawModel,
    files: BTreeMap<String, RawFileEntry>,
    features: RawFeatures,
    streaming: RawStreaming,
    decoder: RawDecoder,
}

#[derive(Deserialize)]
struct RawModel {
    family: String,
    exporter_commit: String,
}

#[derive(Deserialize)]
struct RawFileEntry {
    path: String,
    sha256: String,
}

#[derive(Deserialize)]
struct RawFeatures {
    sample_rate: u32,
    frame_length: usize,
    frame_shift: usize,
    n_fft: usize,
    n_mels: usize,
    log_floor: f32,
    cmvn: RawCmvn,
}

#[derive(Deserialize)]
struct RawCmvn {
    mean: Vec<f32>,
    istd: Vec<f32>,
}

#[derive(Deserialize)]
struct RawStreaming {
    chunk_frames: usize,
    chunk_ms: u32,
    left_context_frames: usize,
    states: Vec<RawState>,
    encoder_output: RawEncOut,
}

#[derive(Deserialize)]
struct RawState {
    name: String,
    shape: Vec<serde_json::Value>,
    dtype: String,
}

#[derive(Deserialize)]
struct RawEncOut {
    name: String,
    shape: Vec<serde_json::Value>,
}

#[derive(Deserialize)]
struct RawDecoder {
    blank_id: i64,
    vocab_size: usize,
    ctx_init: Vec<i64>,
    max_symbols: usize,
}

const MAX_STATE_ELEMS_PER_STREAM: usize = 64 * 1024 * 1024;

fn safe_contained_path(canon_root: &Path, root: &Path, rel: &str) -> Result<PathBuf, GpError> {
    let p = Path::new(rel);
    if rel.is_empty()
        || p.is_absolute()
        || p.components().any(|c| {
            matches!(
                c,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(GpError::Manifest(format!("unsafe bundle path '{rel}'")));
    }
    let joined = root.join(p);
    let canon = joined.canonicalize().map_err(|_| {
        GpError::Integrity(format!(
            "missing or unresolvable bundle file '{}'",
            joined.display()
        ))
    })?;
    if !canon.starts_with(canon_root) || !canon.is_file() {
        return Err(GpError::Integrity(format!(
            "bundle path '{rel}' resolves outside bundle root or is not a regular file"
        )));
    }
    Ok(canon)
}

impl R1Bundle {
    /// Load and strictly validate an `ultra.r1-export/{v1,v1.1}` bundle directory.
    pub fn load(root: &Path, precision: R1Precision, verify_hashes: bool) -> Result<Self, GpError> {
        let canon_root = root
            .canonicalize()
            .map_err(|e| GpError::Io(format!("canonicalize {}: {e}", root.display())))?;
        let manifest_path = safe_contained_path(&canon_root, root, "manifest.json")?;
        let manifest_bytes = std::fs::read(&manifest_path)?;
        let manifest_sha256 = sha256_hex(&manifest_bytes);
        let raw: RawManifest = serde_json::from_slice(&manifest_bytes)
            .map_err(|e| GpError::Manifest(format!("manifest.json parse error: {e}")))?;

        if raw.schema != R1_EXPORT_SCHEMA && raw.schema != R1_EXPORT_SCHEMA_V1_1 {
            return Err(GpError::Manifest(format!(
                "schema '{}' not in ['{R1_EXPORT_SCHEMA}', '{R1_EXPORT_SCHEMA_V1_1}']",
                raw.schema
            )));
        }
        let int8_batch_invariant = raw.int8.as_ref().is_some_and(|m| m.batch_invariant);
        let family = ModelFamily::parse(&raw.model.family)?;

        let mut resolved_files = BTreeMap::new();
        for (key, entry) in &raw.files {
            if !is_sha256_hex(&entry.sha256) {
                return Err(GpError::Integrity(format!(
                    "invalid sha256 hex for files.{key}"
                )));
            }
            let full = safe_contained_path(&canon_root, root, &entry.path)?;
            if verify_hashes {
                let bytes = std::fs::read(&full)?;
                let got = sha256_hex(&bytes);
                if got != entry.sha256 {
                    return Err(GpError::Integrity(format!(
                        "sha256 mismatch for files.{key}: expected {}, got {got}",
                        entry.sha256
                    )));
                }
            }
            resolved_files.insert(key.clone(), full);
        }

        let req_file = |k: &str| -> Result<PathBuf, GpError> {
            resolved_files
                .get(k)
                .cloned()
                .ok_or_else(|| GpError::Manifest(format!("missing required files.{k} entry")))
        };

        let (enc_key, joi_key) = match precision {
            R1Precision::Fp32 => ("encoder.fp32", "joiner.fp32"),
            R1Precision::Int8 => ("encoder.int8", "joiner.int8"),
        };
        let encoder_onnx = req_file(enc_key)?;
        let decoder_onnx = req_file("decoder")?;
        let joiner_onnx = req_file(joi_key)?;
        let tokens_path = req_file("tokens")?;
        // Always load mel_fbank from checksummed `files.mel_fbank` (never unverified features.mel.file).
        let fbank_path = req_file("mel_fbank")?;

        let fe = &raw.features;
        if (
            fe.sample_rate,
            fe.frame_length,
            fe.frame_shift,
            fe.n_fft,
            fe.n_mels,
        ) != (16_000, 400, 160, 512, MEL_BINS)
        {
            return Err(GpError::Manifest(format!(
                "unsupported feature geometry sr={} win={} hop={} fft={} mels={}",
                fe.sample_rate, fe.frame_length, fe.frame_shift, fe.n_fft, fe.n_mels
            )));
        }
        let cmvn = GlobalCmvn::new(&fe.cmvn.mean, &fe.cmvn.istd)?;
        let mel_cfg = Arc::new(TrainerMelConfig::from_fbank_file(
            &fbank_path,
            fe.log_floor,
            cmvn,
        )?);

        let st = &raw.streaming;
        if st.chunk_frames == 0 || st.left_context_frames == 0 {
            return Err(GpError::Manifest(
                "chunk_frames and left_context_frames must be > 0".into(),
            ));
        }
        let expected_names: &[&str] = match family {
            ModelFamily::Dense => &["k", "v", "kv_len", "conv"],
            ModelFamily::Multirate => &["k", "v", "kv_len", "conv", "km", "vm", "kvm_len", "convm"],
        };
        let got_names: Vec<&str> = st.states.iter().map(|s| s.name.as_str()).collect();
        if got_names != expected_names {
            return Err(GpError::Manifest(format!(
                "family '{}' expects states {expected_names:?}, got {got_names:?}",
                family.as_str()
            )));
        }

        let mut states = Vec::with_capacity(st.states.len());
        for s in &st.states {
            let is_int64 = match s.dtype.as_str() {
                "float32" => false,
                "int64" => true,
                other => {
                    return Err(GpError::Manifest(format!(
                        "unsupported state dtype '{other}' for {}",
                        s.name
                    )))
                }
            };
            let mut batch_axis = None;
            let mut per_stream_dims = Vec::with_capacity(s.shape.len());
            for (ax, dim) in s.shape.iter().enumerate() {
                if dim.as_str() == Some("B") {
                    if batch_axis.replace(ax).is_some() {
                        return Err(GpError::Manifest(format!(
                            "multiple 'B' axes in state {}",
                            s.name
                        )));
                    }
                    per_stream_dims.push(1);
                } else if let Some(v) = dim
                    .as_u64()
                    .filter(|&v| v > 0)
                    .and_then(|v| usize::try_from(v).ok())
                {
                    per_stream_dims.push(v);
                } else {
                    return Err(GpError::Manifest(format!(
                        "invalid dim {dim} in state {}",
                        s.name
                    )));
                }
            }
            let batch_axis = batch_axis
                .ok_or_else(|| GpError::Manifest(format!("state {} has no 'B' axis", s.name)))?;
            let per_stream_elems = per_stream_dims
                .iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                .filter(|&n| n > 0 && n <= MAX_STATE_ELEMS_PER_STREAM)
                .ok_or_else(|| {
                    GpError::Manifest(format!(
                        "state {} element count overflows or exceeds {MAX_STATE_ELEMS_PER_STREAM}",
                        s.name
                    ))
                })?;
            states.push(R1EncoderStateSpec {
                name: s.name.clone(),
                batch_axis,
                per_stream_dims,
                per_stream_elems,
                is_int64,
            });
        }

        if st.encoder_output.name != "enc" || st.encoder_output.shape.len() != 3 {
            return Err(GpError::Manifest(
                "encoder_output must be named 'enc' with rank 3 [B, n, D]".into(),
            ));
        }
        let joiner_dim = st.encoder_output.shape[2]
            .as_u64()
            .filter(|&d| d > 0)
            .ok_or_else(|| GpError::Manifest("invalid joiner_dim in encoder_output".into()))?
            as usize;

        let de = &raw.decoder;
        if de.blank_id != 0 || de.vocab_size < 2 || de.max_symbols == 0 || de.ctx_init != [0, 0] {
            return Err(GpError::Manifest(format!(
                "invalid decoder spec: blank={} vocab={} max_symbols={} ctx_init={:?}",
                de.blank_id, de.vocab_size, de.max_symbols, de.ctx_init
            )));
        }
        let tokens = Arc::new(TokenTable::from_file(&tokens_path, de.vocab_size)?);

        Ok(Self {
            root: root.to_path_buf(),
            schema: raw.schema,
            int8_batch_invariant,
            manifest_sha256,
            family,
            precision,
            exporter_commit: raw.model.exporter_commit,
            encoder_onnx,
            decoder_onnx,
            joiner_onnx,
            mel_cfg,
            streaming: R1StreamingSpec {
                chunk_frames: st.chunk_frames,
                chunk_ms: st.chunk_ms,
                left_context_frames: st.left_context_frames,
                joiner_dim,
                states,
            },
            decoder: R1DecoderSpec {
                blank_id: de.blank_id,
                vocab_size: de.vocab_size,
                ctx_init: [de.ctx_init[0], de.ctx_init[1]],
                max_symbols: de.max_symbols,
            },
            tokens,
        })
    }
}
