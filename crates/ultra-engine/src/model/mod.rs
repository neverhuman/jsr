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
pub mod prepared;
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
    pub ctx_init: [usize; 2],
    pub log_floor: f32,
    pub checkpoint_sha256: String,
    pub weights_sha256: String,
}

/// A linear layer `y = W x + b`: the fp32 packing always, the int8 packing when requested.
#[derive(Debug, Clone)]
pub struct Linear {
    pub f: PackedF32,
    pub q: Option<PackedI8>,
}

impl Linear {
    fn new(w: &[f32], n: usize, k: usize, b: Option<&[f32]>) -> Self {
        Self {
            f: PackedF32::pack(w, n, k, b),
            q: None,
        }
    }

    /// Shape and bias only (its weights come from a prepared int8 artifact): the fp32
    /// weights are never materialised.
    fn int8_only(n: usize, k: usize, b: &[f32]) -> Self {
        Self {
            f: PackedF32::bias_only(n, k, b),
            q: None,
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
    pub joiner_out: Linear,
    pub rope_inv: Vec<f32>,
    pub mel_cfg: Arc<TrainerMelConfig>,
    pub tokens: Arc<TokenTable>,
    /// Set when the int8 weights were read from a prepared `weights.int8.safetensors`
    /// (no weight quantisation at load); `None` for an fp32 directory.
    pub int8_prepared: Option<prepared::PreparedInt8>,
}

impl Model {
    /// Load and pack an `ultra.r1-weights/v1` directory. `verify_hashes` checks every file's
    /// SHA-256 against `arch.json` first.
    pub fn load(root: &Path, verify_hashes: bool) -> Result<Self, EngineError> {
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
            || de.context != 2
            || de.vocab_size != raw.model.config.vocab_rnnt
            || de.joiner_activation != "tanh"
        {
            return Err(EngineError::Unsupported(
                "decoder spec (stateless, context 2, tanh)".into(),
            ));
        }
        if e.chunk == 0 || e.left_context == 0 || e.conv_kernel < 2 {
            return Err(EngineError::Shape("chunk/left_context/conv_kernel".into()));
        }
        let wfile = file(&raw, "weights")?;
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
            ctx_init: raw.decoder.ctx_init,
            log_floor: raw.features.log_floor,
            checkpoint_sha256: raw.source.checkpoint_sha256.clone(),
            weights_sha256: wfile.sha256.clone(),
        };
        if arch.blank >= arch.vocab || arch.ctx_init.iter().any(|&c| c >= arch.vocab) {
            return Err(EngineError::Format(
                "blank_id / ctx_init out of vocabulary".into(),
            ));
        }
        let st = LazyTensors::open(&verify(root, wfile, verify_hashes)?)?;
        if st.metadata.get("checkpoint_sha256") != Some(&arch.checkpoint_sha256) {
            return Err(EngineError::Integrity(
                "weights.safetensors metadata checkpoint_sha256 differs from arch.json".into(),
            ));
        }
        let tokens = Arc::new(TokenTable::from_file(
            &verify(root, file(&raw, "tokens")?, verify_hashes)?,
            arch.vocab,
        )?);
        let cmvn = GlobalCmvn::new(&raw.features.cmvn.mean, &raw.features.cmvn.istd)?;
        let mel_cfg = Arc::new(TrainerMelConfig::from_fbank_file(
            &verify(root, file(&raw, "mel_fbank")?, verify_hashes)?,
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
        // Every declared payload, including ones the engine never opens (bpe.model).
        for key in ["weights", "tokens", "mel_fbank", "bpe"] {
            file(&raw, key)?;
        }
        for f in raw.files.values() {
            verify(root, f, verify_hashes)?;
        }
        // Prepared int8: linear layers stored in the artifact never get fp32 weights in memory.
        let int8_only = |prefix: &str| -> bool {
            raw.int8.as_ref().is_some_and(|i8| {
                arch_json::kind_of(prefix).is_some_and(|k| !i8.fp32_kinds.iter().any(|f| f == k))
            })
        };
        let mut model = Self::from_tensors(arch, &st, root, mel_cfg, tokens, &int8_only)?;
        if let Some(i8) = &raw.int8 {
            let path = verify(root, file(&raw, "weights_int8")?, verify_hashes)?;
            model.load_prepared_int8(&path, i8)?;
        }
        Ok(model)
    }

    fn from_tensors(
        arch: Arch,
        st: &LazyTensors,
        root: &Path,
        mel_cfg: Arc<TrainerMelConfig>,
        tokens: Arc<TokenTable>,
        int8_only: &dyn Fn(&str) -> bool,
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
            emb: st.f32("predictor.emb.weight", &[a.vocab, a.pred_emb])?,
            pred_proj: lin("predictor.proj", a.joiner, 2 * a.pred_emb)?,
            joiner_out: lin("joiner.out", a.vocab, a.joiner)?,
            rope_inv,
            mel_cfg,
            tokens,
            int8_prepared: None,
            root: root.to_path_buf(),
            arch,
        })
    }
}
