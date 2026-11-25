//! `ultra.r1-weights/v1` loader: `arch.json` + `weights.safetensors` into packed weights.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ultra_frontend::{GlobalCmvn, TrainerMelConfig, MEL_BINS};
use ultra_gp::TokenTable;

use crate::kernels::PackedF32;
use crate::quant::PackedI8;
use crate::safetensors::LazyTensors;
use crate::EngineError;

mod arch_json;
mod layers;
pub use layers::QAT_STEM_KINDS;
pub mod prepared;
mod slim_int8;
use arch_json::{bounded, file, verify, RawArch};

pub const WEIGHTS_SCHEMA: &str = "ultra.r1-weights/v1";

/// Validated architecture. Dense: `lower = n_blocks`, `mid = fin = 0`. Multirate (S_MR):
/// `lower` blocks at 25 Hz, a pair projection to 12.5 Hz, `mid` blocks there, a restoration
/// plus the lower-block skip, then `fin` blocks at 25 Hz.
#[derive(Debug, Clone, PartialEq)]
pub struct Arch {
    pub n_mels: usize,
    pub d: usize,
    /// Total blocks in execution order (`lower + mid + fin`).
    pub n_blocks: usize,
    pub lower: usize,
    pub mid: usize,
    pub fin: usize,
    pub heads: usize,
    pub dh: usize,
    pub ffn: usize,
    pub kernel: usize,
    pub stem_ch: usize,
    /// Frequency bins after the first / second stride-2 convolution (39 / 19 for 80 mels).
    pub f1: usize,
    pub f2: usize,
    pub chunk: usize,
    pub left: usize,
    pub post_block_norm: bool,
    pub vocab: usize,
    pub pred_emb: usize,
    pub joiner: usize,
    pub blank: usize,
    pub max_symbols: usize,
    pub ctx_init: Vec<usize>,
    pub log_floor: f32,
    pub checkpoint_sha256: String,
    pub weights_sha256: String,
    /// Stem conv 2 + projection trained as int8 GEMMs (`qat_stem = "int8"`): they run int8
    /// in every precision, with the trainer's weight rule (see `Model::apply_qat_stem`).
    pub qat_stem_int8: bool,
}

/// A linear layer `y = W x + b`: the fp32 packing always, the int8 packing when requested.
#[derive(Debug, Clone)]
pub struct Linear {
    pub f: PackedF32,
    pub q: Option<PackedI8>,
    /// Ternary layers on x86: the same codes as 2-bit T-LUT2 weights (4x smaller stream).
    pub t: Option<crate::quant::PackedT>,
    /// Trained as an int8 GEMM (QAT): runs int8 even in the fp32 engine.
    pub qat_i8: bool,
}

impl Linear {
    pub fn from_fp32(w: &[f32], n: usize, k: usize, b: Option<&[f32]>) -> Self {
        Self::new(w, n, k, b)
    }

    fn new(w: &[f32], n: usize, k: usize, b: Option<&[f32]>) -> Self {
        Self {
            f: PackedF32::pack(w, n, k, b),
            q: None,
            t: None,
            qat_i8: false,
        }
    }

    /// Shape and bias only (its weights come from a prepared int8 artifact): the fp32
    /// weights are never materialised.
    fn int8_only(n: usize, k: usize, b: &[f32]) -> Self {
        Self {
            f: PackedF32::bias_only(n, k, b),
            q: None,
            t: None,
            qat_i8: false,
        }
    }

    /// Whether the fp32 weights are resident (false for prepared-int8 layers).
    pub fn has_fp32(&self) -> bool {
        self.f.has_weights()
    }
    pub fn n(&self) -> usize {
        self.f.n
    }
    pub fn k(&self) -> usize {
        self.f.k
    }
}

#[derive(Debug, Clone)]
pub struct Norm {
    pub g: Vec<f32>,
    pub b: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct Block {
    pub ff1_norm: Norm,
    pub ff1_w1: Linear,
    pub ff1_w2: Linear,
    pub att_norm: Norm,
    pub qkv: Linear,
    pub att_out: Linear,
    pub conv_norm: Norm,
    pub pw1: Linear,
    /// Depthwise taps transposed to `[kernel][d]` so one tap is a contiguous channel vector.
    pub dw_w: Vec<f32>,
    pub dw_b: Vec<f32>,
    pub dnorm: Norm,
    pub pw2: Linear,
    pub ff2_norm: Norm,
    pub ff2_w1: Linear,
    pub ff2_w2: Linear,
    pub post_norm: Option<Norm>,
}

/// Precomputed predictor projection halves for compact vocabularies (`vocab <= 1024`, e.g.
/// BPE-500): `h0[t0] + h1[t1]` equals `W_pred @ [emb[t0], emb[t1]] + bias`, turning each
/// predictor step into a single `joiner`-float vector add (~15 ns vs a `320x512` GEMM).
#[derive(Debug, Clone)]
pub struct PredCache {
    /// `W[:, :pred_emb] @ emb[tok] + bias`, shape `[vocab][joiner]`.
    pub h0: Vec<f32>,
    /// `W[:, pred_emb:] @ emb[tok]`, shape `[vocab][joiner]`.
    pub h1: Vec<f32>,
}

/// Exact Cauchy-Schwarz blank-first certificate for the joiner output layer (`0.000%` error).
///
/// For any hidden vector `h` of length `joiner`:
/// `z_v = (w_bar . h) + (w_v - w_bar) . h + b_v <= (w_bar . h) + r_max * ||h||_2 + b_max_nonblank`
/// for all `v != blank`. Whenever `z_blank > upper` (plus FP/quant slack), `argmax == blank`
/// is mathematically certified without running the `V x joiner` joiner GEMM.
#[derive(Debug, Clone)]
pub struct BlankCert {
    pub w_blank: Vec<f32>,
    pub b_blank: f32,
    pub w_bar: Vec<f32>,
    pub r_max: f32,
    pub b_max_nonblank: f32,
    pub max_w_norm: f32,
}

impl BlankCert {
    /// Build from row-major fp32 joiner weights `w[v][j]` and biases `b[v]`.
    pub fn from_row_major(
        w: &[f32],
        b: &[f32],
        vocab: usize,
        jn: usize,
        blank: usize,
    ) -> Option<Self> {
        if vocab < 2 || jn == 0 || blank >= vocab || w.len() != vocab * jn || b.len() != vocab {
            return None;
        }
        let w_blank = w[blank * jn..(blank + 1) * jn].to_vec();
        let b_blank = b[blank];
        let mut w_bar = vec![0.0f32; jn];
        let inv_n = 1.0 / (vocab - 1) as f32;
        let mut b_max_nonblank = f32::NEG_INFINITY;
        for v in 0..vocab {
            if v == blank {
                continue;
            }
            let row = &w[v * jn..(v + 1) * jn];
            for (m, &val) in w_bar.iter_mut().zip(row) {
                *m += val * inv_n;
            }
            b_max_nonblank = b_max_nonblank.max(b[v]);
        }
        let mut r_max = 0.0f32;
        let mut max_w_norm = 0.0f32;
        for v in 0..vocab {
            let row = &w[v * jn..(v + 1) * jn];
            let mut n2 = 0.0f64;
            for &val in row {
                n2 += f64::from(val) * f64::from(val);
            }
            max_w_norm = max_w_norm.max(n2.sqrt() as f32);
            if v == blank {
                continue;
            }
            let mut d2 = 0.0f64;
            for (&val, &m) in row.iter().zip(&w_bar) {
                let d = f64::from(val) - f64::from(m);
                d2 += d * d;
            }
            r_max = r_max.max(d2.sqrt() as f32);
        }
        Some(Self {
            w_blank,
            b_blank,
            w_bar,
            r_max,
            b_max_nonblank,
            max_w_norm,
        })
    }

    /// Build from a packed int8 joiner matrix using its exact dequantised weights `w_q * scale`,
    /// streaming one row (`jn` floats) at a time to avoid a `V x jn` temporary peak allocation.
    pub fn from_packed_i8(q: &PackedI8, blank: usize) -> Option<Self> {
        use crate::kernels::NR;
        use crate::quant::KG;
        let (v, jn) = (q.n(), q.k());
        if v < 2 || jn == 0 || blank >= v {
            return None;
        }
        let unpack_row = |row: usize, dst: &mut [f32]| {
            let (p, j) = (row / NR, row % NR);
            let panel = q.panel(p);
            let s = q.scale[row];
            for (kk, slot) in dst.iter_mut().enumerate() {
                let code = panel[((kk / KG) * NR + j) * KG + kk % KG];
                *slot = f32::from(code) * s;
            }
        };
        let mut w_blank = vec![0.0f32; jn];
        unpack_row(blank, &mut w_blank);
        let b_blank = q.bias[blank];
        let mut w_bar = vec![0.0f32; jn];
        let mut row_buf = vec![0.0f32; jn];
        let inv_n = 1.0 / (v - 1) as f32;
        let mut b_max_nonblank = f32::NEG_INFINITY;
        for row in 0..v {
            if row == blank {
                continue;
            }
            unpack_row(row, &mut row_buf);
            for (m, &val) in w_bar.iter_mut().zip(&row_buf) {
                *m += val * inv_n;
            }
            b_max_nonblank = b_max_nonblank.max(q.bias[row]);
        }
        let mut r_max = 0.0f32;
        let mut max_w_norm = 0.0f32;
        for row in 0..v {
            unpack_row(row, &mut row_buf);
            let mut n2 = 0.0f64;
            for &val in &row_buf {
                n2 += f64::from(val) * f64::from(val);
            }
            max_w_norm = max_w_norm.max(n2.sqrt() as f32);
            if row == blank {
                continue;
            }
            let mut d2 = 0.0f64;
            for (&val, &m) in row_buf.iter().zip(&w_bar) {
                let d = f64::from(val) - f64::from(m);
                d2 += d * d;
            }
            r_max = r_max.max(d2.sqrt() as f32);
        }
        Some(Self {
            w_blank,
            b_blank,
            w_bar,
            r_max,
            b_max_nonblank,
            max_w_norm,
        })
    }

    /// Returns `true` ONLY when `z_blank > z_v` is mathematically certified for every `v != blank`.
    /// `int8_qmax`: `None` for FP32, `Some(127)` for 8-bit int8, `Some(63)` for 7-bit (`--act7`).
    pub fn is_certified_blank(&self, hid: &[f32], int8_qmax: Option<u8>) -> bool {
        let jn = self.w_blank.len();
        debug_assert_eq!(hid.len(), jn);
        let mut z_blank = f64::from(self.b_blank);
        let mut mu = 0.0f64;
        let mut h2 = 0.0f64;
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for ((&h, &wb), &wm) in hid.iter().zip(&self.w_blank).zip(&self.w_bar) {
            if !h.is_finite() {
                return false;
            }
            lo = lo.min(h);
            hi = hi.max(h);
            let hf = f64::from(h);
            z_blank += f64::from(wb) * hf;
            mu += f64::from(wm) * hf;
            h2 += hf * hf;
        }
        let h_norm = h2.sqrt();
        let mut margin = 1e-4f64 * (1.0 + h_norm * f64::from(self.max_w_norm));
        if let Some(qmax) = int8_qmax {
            if qmax == 0 {
                return false;
            }
            // Per-row centred int8 quantisation perturbs each element by <= scale / 2,
            // so ||x_hat - hid||_2 <= (scale / 2) * sqrt(jn) and |(w_v - w_blank).(x_hat - hid)|
            // <= scale * sqrt(jn) * max_w_norm. Exact denominator is 2 * qmax (254 or 126).
            let scale = (f64::from(hi) - f64::from(lo)) / (2.0 * f64::from(qmax));
            margin += scale * (jn as f64).sqrt() * f64::from(self.max_w_norm) + 1e-3;
        }
        let upper = mu + f64::from(self.r_max) * h_norm + f64::from(self.b_max_nonblank) + margin;
        z_blank > upper
    }
}

/// Content SHA-256 digests of the actual `arch.json`-resolved asset files loaded by
/// [`Model::load_with_mode`], independent of on-disk filenames or staging directories.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LoadedAssetsIdentity {
    /// Actual SHA-256 of the loaded `files.tokens` file (regardless of filename).
    pub tokens_sha256: Option<String>,
    /// Actual SHA-256 of the loaded `files.bpe` file (when declared/loaded).
    pub bpe_sha256: Option<String>,
    /// Actual SHA-256 of the loaded `files.mel_fbank` file (regardless of filename).
    pub mel_fbank_sha256: Option<String>,
}

/// Everything the engine needs, loaded and packed once, shared read-only by all workers.
#[derive(Debug)]
pub struct Model {
    pub arch: Arch,
    pub root: PathBuf,
    /// Layer-1 stem conv: `[9][stem_ch]` taps (kh * 3 + kw, channel contiguous) and bias.
    pub c1_w: Vec<f32>,
    pub c1_b: Vec<f32>,
    /// Layer-2 stem conv as a GEMM with K ordered (kh, kw, c_in).
    pub c2: Linear,
    /// Stem projection with K ordered (f, c) to match the channel-last layer-2 output.
    pub stem_out: Linear,
    pub blocks: Vec<Block>,
    pub final_norm: Option<Norm>,
    /// Multirate: 25 Hz pair -> 12.5 Hz projection (`d x 2d`) and its restoration (`2d x d`).
    pub pair: Option<Linear>,
    pub restore: Option<Linear>,
    pub enc_proj: Linear,
    /// Predictor embedding table `[vocab][pred_emb]`.
    pub emb: Vec<f32>,
    pub pred_proj: Linear,
    /// Precomputed predictor halves for compact vocabularies (`vocab <= 1024`).
    pub pred_cache: Option<PredCache>,
    pub joiner_out: Linear,
    /// Optional auxiliary CTC linear projection `[ctc_vocab, d]` (fp32, loaded when present in
    /// `weights.safetensors`; zero overhead when unused).
    pub ctc_head: Option<Linear>,
    /// Exact Cauchy-Schwarz blank-first certificates for fp32 and int8 joiner paths.
    pub blank_cert_f32: Option<BlankCert>,
    pub blank_cert_i8: Option<BlankCert>,
    pub rope_inv: Vec<f32>,
    pub mel_cfg: Arc<TrainerMelConfig>,
    pub tokens: Arc<TokenTable>,
    /// Content SHA-256 digests of the actual `arch.json`-resolved asset files loaded from disk.
    pub assets: LoadedAssetsIdentity,
    /// Set when the int8 weights were read from a prepared `weights.int8.safetensors`
    /// (no weight quantisation at load); `None` for an fp32 directory.
    pub int8_prepared: Option<prepared::PreparedInt8>,
}

impl Model {
    /// Load and pack an `ultra.r1-weights/v1` directory. `verify_hashes` checks every file's
    /// SHA-256 against `arch.json` first.
    pub fn load(root: &Path, verify_hashes: bool) -> Result<Self, EngineError> {
        Self::load_with_mode(root, verify_hashes, false)
    }

    /// Load an `ultra.r1-weights/v1` directory for live streaming RNN-T decode, skipping auxiliary
    /// FP32 tensors that live decode never reads (`ctc_head`, plus `emb`/`pred_proj` when
    /// `pred_cache` is precomputed) so they never enter cold process `VmHWM`.
    /// Callers requiring CTC rescoring (`ctc_weight > 0`) must use [`Model::load`] instead.
    pub fn load_live(root: &Path, verify_hashes: bool) -> Result<Self, EngineError> {
        Self::load_with_mode(root, verify_hashes, true)
    }

    fn load_with_mode(
        root: &Path,
        verify_hashes: bool,
        live_only: bool,
    ) -> Result<Self, EngineError> {
        let text = std::fs::read_to_string(root.join("arch.json"))?;
        let raw: RawArch = serde_json::from_str(&text)
            .map_err(|e| EngineError::Format(format!("arch.json: {e}")))?;
        if raw.schema != WEIGHTS_SCHEMA {
            return Err(EngineError::Format(format!(
                "schema {} != {WEIGHTS_SCHEMA}",
                raw.schema
            )));
        }
        let e = &raw.model.encoder;
        let multirate = match (raw.model.family.as_str(), e.family.as_str()) {
            ("dense", "dense") => false,
            ("multirate", "multirate") => true,
            _ => {
                return Err(EngineError::Unsupported(format!(
                    "encoder family {} (the engine implements dense and multirate)",
                    e.family
                )))
            }
        };
        let (lower, mid, fin) = if multirate {
            (e.lower_blocks, e.middle_blocks, e.final_blocks)
        } else {
            (e.n_blocks, 0, 0)
        };
        if multirate
            && (lower == 0
                || mid == 0
                || !e.chunk.is_multiple_of(2)
                || !e.left_context.is_multiple_of(2))
        {
            return Err(EngineError::Shape(
                "multirate needs blocks at both rates and even chunk/left".into(),
            ));
        }
        bounded("total blocks", lower + mid + fin, 256)?;
        if raw.model.config.joiner_act != "tanh" {
            return Err(EngineError::Unsupported(format!(
                "joiner activation {}",
                raw.model.config.joiner_act
            )));
        }
        bounded("d_model", e.d_model, 4096)?;
        bounded("n_heads", e.n_heads, 64)?;
        bounded("n_blocks", e.n_blocks, 256)?;
        bounded("ffn_mult", e.ffn_mult, 16)?;
        bounded("conv_kernel", e.conv_kernel, 255)?;
        bounded("stem_channels", e.stem_channels, 1024)?;
        bounded("chunk", e.chunk, 64)?;
        bounded("left_context", e.left_context, 4096)?;
        bounded("vocab_rnnt", raw.model.config.vocab_rnnt, 1 << 20)?;
        bounded("pred_emb", raw.model.config.pred_emb, 4096)?;
        bounded("joiner", raw.model.config.joiner, 4096)?;
        bounded("max_symbols", raw.decoder.max_symbols, 64)?;
        if e.n_mels != MEL_BINS
            || !e.d_model.is_multiple_of(e.n_heads)
            || !(e.d_model / e.n_heads).is_multiple_of(2)
        {
            return Err(EngineError::Shape("n_mels/d_model/n_heads".into()));
        }
        let (fe, de) = (&raw.features, &raw.decoder);
        // The frontend (ultra-frontend TrainerLogMel) and greedy rule the engine implements.
        if (
            fe.sample_rate,
            fe.frame_length,
            fe.frame_shift,
            fe.n_fft,
            fe.n_mels,
        ) != (16_000, 400, 160, 512, MEL_BINS)
            || !(fe.log_floor > 0.0 && fe.log_floor.is_finite())
        {
            return Err(EngineError::Unsupported(
                "feature spec differs from TrainerLogMel".into(),
            ));
        }
        if de.kind != "stateless"
            || de.context == 0
            || de.context != de.ctx_init.len()
            || de.context.checked_mul(raw.model.config.pred_emb).is_none()
            || de.vocab_size != raw.model.config.vocab_rnnt
            || de.joiner_activation != "tanh"
        {
            return Err(EngineError::Unsupported(
                "decoder spec needs stateless tanh and a nonempty context matching ctx_init".into(),
            ));
        }
        if e.chunk == 0 || e.left_context == 0 || e.conv_kernel < 2 {
            return Err(EngineError::Shape("chunk/left_context/conv_kernel".into()));
        }
        let slim_int8 = raw.int8.as_ref().is_some_and(|i8| i8.self_contained);
        let weights_sha256 = if slim_int8 {
            raw.int8
                .as_ref()
                .map(|i8| i8.source_weights_sha256.clone())
                .unwrap_or_default()
        } else {
            file(&raw, "weights")?.sha256.clone()
        };
        let f1 = (e.n_mels - 3) / 2 + 1;
        let arch = Arch {
            n_mels: e.n_mels,
            d: e.d_model,
            n_blocks: lower + mid + fin,
            lower,
            mid,
            fin,
            heads: e.n_heads,
            dh: e.d_model / e.n_heads,
            ffn: e.d_model * e.ffn_mult,
            kernel: e.conv_kernel,
            stem_ch: e.stem_channels,
            f1,
            f2: (f1 - 3) / 2 + 1,
            chunk: e.chunk,
            left: e.left_context,
            post_block_norm: e.post_block_norm,
            vocab: raw.model.config.vocab_rnnt,
            pred_emb: raw.model.config.pred_emb,
            joiner: raw.model.config.joiner,
            blank: raw.decoder.blank_id,
            max_symbols: raw.decoder.max_symbols,
            ctx_init: de.ctx_init.clone(),
            log_floor: raw.features.log_floor,
            checkpoint_sha256: raw.source.checkpoint_sha256.clone(),
            weights_sha256,
            qat_stem_int8: layers::qat_stem_int8(raw.model.config.qat_stem.as_deref())?,
        };
        if arch.blank >= arch.vocab || arch.ctx_init.iter().any(|&c| c >= arch.vocab) {
            return Err(EngineError::Format(
                "blank_id / ctx_init out of vocabulary".into(),
            ));
        }
        // Every declared payload verified exactly once (no duplicate SHA-256 passes).
        let required_keys: &[&str] = if slim_int8 {
            &["weights_int8", "tokens", "mel_fbank", "bpe"]
        } else {
            &["weights", "tokens", "mel_fbank", "bpe"]
        };
        for &key in required_keys {
            file(&raw, key)?;
        }
        let mut verified = std::collections::BTreeMap::new();
        let mut verified_sha256 = std::collections::BTreeMap::new();
        for (key, f) in &raw.files {
            if slim_int8 && key == "weights" {
                continue;
            }
            verified.insert(key.as_str(), verify(root, f, verify_hashes)?);
            if verify_hashes {
                verified_sha256.insert(key.as_str(), f.sha256.clone());
            }
        }
        let asset_sha256 = |key: &str| -> Result<String, EngineError> {
            if let Some(hex) = verified_sha256.get(key) {
                return Ok(hex.clone());
            }
            arch_json::sha256_file(&verified[key])
        };
        let assets = LoadedAssetsIdentity {
            tokens_sha256: Some(asset_sha256("tokens")?),
            bpe_sha256: Some(asset_sha256("bpe")?),
            mel_fbank_sha256: Some(asset_sha256("mel_fbank")?),
        };
        let st_path = if slim_int8 {
            &verified["weights_int8"]
        } else {
            &verified["weights"]
        };
        let st = LazyTensors::open(st_path)?;
        if st.metadata.get("checkpoint_sha256") != Some(&arch.checkpoint_sha256) {
            return Err(EngineError::Integrity(
                "safetensors metadata checkpoint_sha256 differs from arch.json".into(),
            ));
        }
        let tokens = Arc::new(TokenTable::from_file(&verified["tokens"], arch.vocab)?);
        let cmvn = GlobalCmvn::new(&raw.features.cmvn.mean, &raw.features.cmvn.istd)?;
        let mel_cfg = Arc::new(TrainerMelConfig::from_fbank_file(
            &verified["mel_fbank"],
            arch.log_floor,
            cmvn,
        )?);
        let ckpt_mean = st.f32("cmvn.mean", &[arch.n_mels])?;
        let ckpt_istd = st.f32("cmvn.istd", &[arch.n_mels])?;
        if ckpt_mean != raw.features.cmvn.mean || ckpt_istd != raw.features.cmvn.istd {
            return Err(EngineError::Integrity(
                "arch.json cmvn (mean or istd) differs from weights".into(),
            ));
        }
        // Prepared int8: linear layers stored in the artifact never get fp32 weights in memory.
        let int8_only = |prefix: &str| -> bool {
            raw.int8.as_ref().is_some_and(|i8| {
                arch_json::kind_of(prefix).is_some_and(|k| !i8.fp32_kinds.iter().any(|f| f == k))
            })
        };
        let has_ctc_in_file = st.contains("ctc_head.weight") && st.contains("ctc_head.bias");
        let mut model =
            Self::from_tensors(arch, &st, root, mel_cfg, tokens, &int8_only, live_only)?;
        model.assets = assets;
        if let Some(i8) = &raw.int8 {
            let wf = file(&raw, "weights_int8")?;
            let path = verified
                .get("weights_int8")
                .ok_or_else(|| EngineError::Format("arch.json has no files.weights_int8".into()))?;
            let sha = if verify_hashes {
                wf.sha256.clone()
            } else {
                arch_json::sha256_file(path)?
            };
            if slim_int8 {
                let aux =
                    slim_int8::aux_fp32_names(&model.arch, &i8.fp32_kinds, has_ctc_in_file).len();
                model.load_prepared_int8_from(&st, sha, i8, aux)?;
            } else {
                let st_i8 = LazyTensors::open(path)?;
                model.load_prepared_int8_from(&st_i8, sha, i8, 0)?;
            }
        }
        drop(st);
        model.refresh_blank_cert_i8();
        model.apply_qat_stem()?;
        Ok(model)
    }

    fn from_tensors(
        arch: Arch,
        st: &LazyTensors,
        root: &Path,
        mel_cfg: Arc<TrainerMelConfig>,
        tokens: Arc<TokenTable>,
        int8_only: &dyn Fn(&str) -> bool,
        live_only: bool,
    ) -> Result<Self, EngineError> {
        let a = &arch;
        let (d, ch, ffn) = (a.d, a.stem_ch, a.ffn);
        let lin = |name: &str, n: usize, k: usize| -> Result<Linear, EngineError> {
            let b = st.f32(&format!("{name}.bias"), &[n])?;
            if int8_only(name) {
                return Ok(Linear::int8_only(n, k, &b));
            }
            let w = st.f32(&format!("{name}.weight"), &[n, k])?;
            Ok(Linear::new(&w, n, k, Some(&b)))
        };
        let norm = |name: &str| -> Result<Norm, EngineError> {
            Ok(Norm {
                g: st.f32(&format!("{name}.weight"), &[d])?,
                b: st.f32(&format!("{name}.bias"), &[d])?,
            })
        };
        // Stem layer 1: (ch, 1, 3, 3) -> [ch][9].
        let c1 = st.f32("encoder.stem.c1.weight", &[ch, 1, 3, 3])?;
        let mut c1_w = vec![0.0f32; 9 * ch];
        for c in 0..ch {
            for t in 0..9 {
                c1_w[t * ch + c] = c1[c * 9 + t];
            }
        }
        let c1_b = st.f32("encoder.stem.c1.bias", &[ch])?;
        // Stem layer 2: (ch_out, ch_in, 3, 3) -> rows ch_out, K index (kh*3+kw)*ch + c_in.
        let c2_b = st.f32("encoder.stem.c2.bias", &[ch])?;
        let c2 = if int8_only("encoder.stem.c2") {
            Linear::int8_only(ch, 9 * ch, &c2_b)
        } else {
            let w2 = st.f32("encoder.stem.c2.weight", &[ch, ch, 3, 3])?;
            let mut w2p = vec![0.0f32; ch * 9 * ch];
            for o in 0..ch {
                for i in 0..ch {
                    for t in 0..9 {
                        w2p[o * 9 * ch + t * ch + i] = w2[(o * ch + i) * 9 + t];
                    }
                }
            }
            Linear::new(&w2p, ch, 9 * ch, Some(&c2_b))
        };
        // Stem projection: K index c * f2 + f in torch; f * ch + c here.
        let kin = ch * a.f2;
        let so_b = st.f32("encoder.stem.out.bias", &[d])?;
        let stem_out = if int8_only("encoder.stem.out") {
            Linear::int8_only(d, kin, &so_b)
        } else {
            let wo = st.f32("encoder.stem.out.weight", &[d, kin])?;
            let mut wop = vec![0.0f32; d * kin];
            for o in 0..d {
                for c in 0..ch {
                    for f in 0..a.f2 {
                        wop[o * kin + f * ch + c] = wo[o * kin + c * a.f2 + f];
                    }
                }
            }
            Linear::new(&wop, d, kin, Some(&so_b))
        };
        let mut blocks = Vec::with_capacity(a.n_blocks);
        let names = (0..a.lower)
            .map(|i| format!("encoder.blocks.{i}"))
            .chain((0..a.mid).map(|i| format!("encoder.mid_blocks.{i}")))
            .chain((0..a.fin).map(|i| format!("encoder.final_blocks.{i}")));
        for p in names {
            let dw = st.f32(&format!("{p}.conv.dw.weight"), &[d, 1, a.kernel])?;
            let mut dw_w = vec![0.0f32; a.kernel * d];
            for c in 0..d {
                for t in 0..a.kernel {
                    dw_w[t * d + c] = dw[c * a.kernel + t];
                }
            }
            blocks.push(Block {
                ff1_norm: norm(&format!("{p}.ff1.norm"))?,
                ff1_w1: lin(&format!("{p}.ff1.w1"), ffn, d)?,
                ff1_w2: lin(&format!("{p}.ff1.w2"), d, ffn)?,
                att_norm: norm(&format!("{p}.att.norm"))?,
                qkv: lin(&format!("{p}.att.qkv"), 3 * d, d)?,
                att_out: lin(&format!("{p}.att.out"), d, d)?,
                conv_norm: norm(&format!("{p}.conv.norm"))?,
                pw1: lin(&format!("{p}.conv.pw1"), 2 * d, d)?,
                dw_w,
                dw_b: st.f32(&format!("{p}.conv.dw.bias"), &[d])?,
                dnorm: norm(&format!("{p}.conv.dnorm"))?,
                pw2: lin(&format!("{p}.conv.pw2"), d, d)?,
                ff2_norm: norm(&format!("{p}.ff2.norm"))?,
                ff2_w1: lin(&format!("{p}.ff2.w1"), ffn, d)?,
                ff2_w2: lin(&format!("{p}.ff2.w2"), d, ffn)?,
                post_norm: if a.post_block_norm {
                    Some(norm(&format!("{p}.norm"))?)
                } else {
                    None
                },
            });
        }
        let final_norm = if a.post_block_norm {
            None
        } else {
            Some(norm("encoder.final_norm")?)
        };
        let rope_inv = st.f32("engine.rope_inv_freq", &[a.dh / 2])?;
        let (pair, restore) = if a.mid > 0 {
            (
                Some(lin("encoder.pair", d, 2 * d)?),
                Some(lin("encoder.restore", 2 * d, d)?),
            )
        } else {
            (None, None)
        };
        let emb_raw = st.f32("predictor.emb.weight", &[a.vocab, a.pred_emb])?;
        let pred_in = a.ctx_init.len() * a.pred_emb;
        let pred_w = st.f32("predictor.proj.weight", &[a.joiner, pred_in])?;
        let pred_b = st.f32("predictor.proj.bias", &[a.joiner])?;
        let pred_cache = (a.vocab <= 1024 && a.ctx_init.len() == 2).then(|| {
            let (v, e, jn) = (a.vocab, a.pred_emb, a.joiner);
            let mut h0 = vec![0.0f32; v * jn];
            let mut h1 = vec![0.0f32; v * jn];
            for tok in 0..v {
                let er = &emb_raw[tok * e..(tok + 1) * e];
                for j in 0..jn {
                    let wr = &pred_w[j * 2 * e..(j + 1) * 2 * e];
                    let mut s0 = f64::from(pred_b[j]);
                    let mut s1 = 0.0f64;
                    for kk in 0..e {
                        let ev = f64::from(er[kk]);
                        s0 += f64::from(wr[kk]) * ev;
                        s1 += f64::from(wr[e + kk]) * ev;
                    }
                    h0[tok * jn + j] = s0 as f32;
                    h1[tok * jn + j] = s1 as f32;
                }
            }
            PredCache { h0, h1 }
        });
        let (emb, pred_proj) = if live_only && pred_cache.is_some() {
            drop(emb_raw);
            drop(pred_w);
            drop(pred_b);
            (Vec::new(), Linear::int8_only(1, 1, &[0.0]))
        } else {
            let proj = Linear::new(&pred_w, a.joiner, pred_in, Some(&pred_b));
            (emb_raw, proj)
        };
        let joiner_b = st.f32("joiner.out.bias", &[a.vocab])?;
        let (joiner_out, blank_cert_f32) = if int8_only("joiner.out") {
            (Linear::int8_only(a.vocab, a.joiner, &joiner_b), None)
        } else {
            let joiner_w = st.f32("joiner.out.weight", &[a.vocab, a.joiner])?;
            let cert = BlankCert::from_row_major(&joiner_w, &joiner_b, a.vocab, a.joiner, a.blank);
            (
                Linear::new(&joiner_w, a.vocab, a.joiner, Some(&joiner_b)),
                cert,
            )
        };
        let ctc_head =
            if !live_only && st.contains("ctc_head.weight") && st.contains("ctc_head.bias") {
                let b_shape = st.shape("ctc_head.bias").unwrap_or(&[]);
                if b_shape.len() != 1 || b_shape[0] < 2 {
                    return Err(EngineError::Shape(format!(
                        "ctc_head.bias: invalid shape {b_shape:?}"
                    )));
                }
                let cv = b_shape[0];
                bounded("ctc_vocab", cv, 1 << 20)?;
                let cw = st.f32("ctc_head.weight", &[cv, d])?;
                let cb = st.f32("ctc_head.bias", &[cv])?;
                Some(Linear::new(&cw, cv, d, Some(&cb)))
            } else {
                None
            };
        Ok(Self {
            pair,
            restore,
            c1_w,
            c1_b,
            c2,
            stem_out,
            blocks,
            final_norm,
            enc_proj: lin("joiner.enc_proj", a.joiner, d)?,
            emb,
            pred_proj,
            pred_cache,
            joiner_out,
            ctc_head,
            blank_cert_f32,
            blank_cert_i8: None,
            rope_inv,
            mel_cfg,
            tokens,
            assets: LoadedAssetsIdentity::default(),
            int8_prepared: None,
            root: root.to_path_buf(),
            arch,
        })
    }
}
